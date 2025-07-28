import torch
from omegaconf import DictConfig, OmegaConf  # type: ignore
from typing import Optional, Tuple, List, Dict, Any
from transformers import AutoTokenizer
from verl import DataProto
from verl.workers.reward_manager.registry import register
from verl.workers.autorater.autorater_utils import extract_solution
from verl.workers.code_evaluator import CodeEvaluator
from verl.utils.debug.performance import _timer
from verl.utils.autorater_client import call_autorater_service

@register("reward_manager")
class RewardManager:
    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.code_evaluator = CodeEvaluator(
            config=config.code_evaluator,
            tokenizer=self.tokenizer
        )

    def start_epoch(self, epoch: int):
        pass 

    def end_epoch(self):
        pass 

    def save_epoch_metadata(self, epoch: int, log_to_wandb: bool = False):
        pass

    def _decode_batch(self, data: DataProto) -> Tuple[List[str], List[str]]:
        prompts = [self.tokenizer.decode(p_ids, skip_special_tokens=True) for p_ids in data.batch["prompts"]]
        model_responses = [self.tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in data.batch["responses"]]
        
        return prompts, model_responses

    def compute_rewards(self, data: DataProto, timing_raw: Optional[Dict[str, float]] = {}) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.config.template_type == "interleave":
            return self.compute_rewards_interleave(data, timing_raw)
        elif self.config.template_type == "default":
            return self.compute_rewards_think_answer(data, timing_raw)
        else:
            raise ValueError(f"Invalid template type: {self.config.template_type}")

    def compute_rewards_think_answer(self, data: DataProto, timing_raw: Optional[Dict[str, float]] = {}) -> Tuple[torch.Tensor, Dict[str, Any]]:
        prompts, model_responses = self._decode_batch(data)

        answers = [extract_solution(r, template_type="default") for r in model_responses]

        # figure out which evaluator to use base on data source
        data_sources = data.non_tensor_batch["data_source"]

        code_indices = []
        text_indices = []
        for i, ds in enumerate(data_sources):
            if ds and ("code" in str(ds).lower() or "mbpp" in str(ds).lower()):
                code_indices.append(i)
            else:
                text_indices.append(i)
        
        batch_size = len(data)
        # index of the example in the batch
        batch_indices = data.non_tensor_batch["index"] 

        print(f"number of code_indices: {len(code_indices)}")
        print(f"number of text_indices: {len(text_indices)}")

        # Run code evaluator on code indices
        rm_infos = data.non_tensor_batch["reward_model"]
        code_rm_infos = [rm_infos[i] for i in code_indices]
        code_batch_indices = [batch_indices[i] for i in code_indices]
        code_prompts = [prompts[i] for i in code_indices]
        code_answers = [answers[i] for i in code_indices]

        if len(code_indices) > 0:
            with _timer("code_evaluator", timing_raw):
                code_rewards = self.code_evaluator.evaluate_code(
                    code_answers,
                    code_prompts,
                    code_rm_infos,
                    batch_indices=code_batch_indices,
                )
        else:
            code_rewards = {}

        text_gt_answers = [rm_infos[i]["ground_truth"] for i in text_indices]
        text_prompts = [prompts[i] for i in text_indices]
        text_answers = [answers[i] for i in text_indices]


        if len(text_indices) > 0:
            text_decisions, text_explanations, text_raw_responses = self.compute_reward_text(
                text_prompts, text_answers, text_gt_answers
            )
            text_extras = {"autorater_scores": text_decisions}
        else:
            text_decisions = []
            text_extras = {}

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        code_count = 0
        text_count = 0

        final_code_extras = {k: [0 for _ in range(batch_size)] for k in code_rewards.keys()}
        final_text_extras = {k: [0 for _ in range(batch_size)] for k in text_extras.keys()}

        for i in range(batch_size):
            data_item = data[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            if i in code_indices:
                current_final_score = code_rewards["unit_test_pass_rate"][code_count]
                for k, v in code_rewards.items():
                    final_code_extras[k][i] = v[code_count]
                code_count += 1
            elif i in text_indices:
                current_final_score = text_decisions[text_count]
                for k, v in text_extras.items():
                    final_text_extras[k][i] = v[text_count]
                text_count += 1
            else:
                raise ValueError(f"Invalid index: {i}")

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = current_final_score
        
        extras = {}
        extras.update(final_code_extras)
        extras.update(final_text_extras)
        return reward_tensor, extras

    def compute_reward_text(self, prompts: List[str], answers: List[str], gt_answers: List[str]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        template_types = ["standard"] * len(prompts)

        # if gt rewards is a list of lists, then we combine the answers with
        # 1) ..., 2) ... , n)"
        if isinstance(gt_answers[0], list):
            updated_gt_answers = []
            for i in range(len(gt_answers)):
                merged_answer = ""
                for j in range(len(gt_answers[i])):
                    merged_answer += f"{i+1}) " + gt_answers[i][j] + ", "
                updated_gt_answers.append(merged_answer[:-2])
            gt_answers = updated_gt_answers
            
        autorater_payload = {
            "prompts": prompts,
            "responses": answers,
            "gt_answers": gt_answers,
            "template_types": template_types,
        }

        autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(
            self.config.autorater_service_url, autorater_payload, batch_size=len(prompts)
        )

        return autorater_decisions, autorater_explanations, autorater_raw_responses

    def compute_reward_text_interleave(self, prompts: List[str], answers: List[List[str]], gt_answers: List[List[str]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        template_types = ["standard"] * len(prompts)
        
        # make flat list of answers and prompts
        all_prompts = []
        all_answers = []
        all_gt_answers = []
        counts = []

        # if we have more answers than gt answers only take the first len(gt_answers) answers
        for i, interleave_answers in enumerate(answers):
            if len(interleave_answers) > len(gt_answers[i]):
                answers[i] = interleave_answers[:len(gt_answers[i])]

        for i, interleave_answers in enumerate(answers):
            
            gt_answer = gt_answers[i]

            for j, answer in enumerate(interleave_answers):
                all_prompts.append(prompts[i])
                all_answers.append(answer)
                all_gt_answers.append(gt_answer[j])

            counts.append(len(interleave_answers))

        autorater_payload = {
            "prompts": all_prompts,
            "responses": all_answers,
            "gt_answers": all_gt_answers,
            "template_types": template_types,
        }
        
        autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(
            self.config.autorater_service_url, autorater_payload, batch_size=len(all_prompts)
        )
        # convert to list of lists based on counts
        # but average the scores for the interleaved answers
        final_decisions = []
        for i in range(len(counts)):
            final_decisions.append(sum(autorater_decisions[i:i+counts[i]]) / counts[i])
            i += counts[i]
        
        return final_decisions, autorater_explanations, autorater_raw_responses

    def compute_rewards_interleave(self, data: DataProto, timing_raw: Optional[Dict[str, float]] = {}) -> Tuple[torch.Tensor, Dict[str, Any]]:
        prompts, model_responses = self._decode_batch(data)

        # this should be a list of lists of answers
        answers = [extract_solution(r, template_type="interleave") for r in model_responses]

        batch_size = len(model_responses)

        # First compute the interleave format reward
        interleave_answer_counts = [len(answer) for answer in answers]
        interleave_format_rewards = [1.0 if count >= 2 else 0.0 for count in interleave_answer_counts]

        # Then compute the other rewards for the indices with answer counts >= 2
        valid_indices = [i for i, count in enumerate(interleave_answer_counts) if count >= 2]

        if len(valid_indices) == 0:
            return torch.zeros_like(data.batch["responses"], dtype=torch.float32), {"unit_test_pass_rate": [0.0] * batch_size, "autorater_scores": [0.0] * batch_size}

        valid_outline_code_test_indices = [i for i, count in enumerate(interleave_answer_counts) if count == 3]
        print(f"number of valid_indices: {len(valid_indices)}")

        # figure out which evaluator to use base on data source
        data_sources = data.non_tensor_batch["data_source"]

        code_indices = []
        text_indices = []
        outline_code_test_indices = []

        for i, ds in enumerate(data_sources):
            if i not in valid_indices:
                continue
            if ds and ds == "bcb_outline_code_test_interleave":
                outline_code_test_indices.append(i)
            elif ds and ("code" in str(ds).lower() or "mbpp" in str(ds).lower()):
                code_indices.append(i)
            else:
                text_indices.append(i)

        # index of the example in the batch
        batch_indices = data.non_tensor_batch["index"] 

        print(f"number of outline_code_test_indices: {len(outline_code_test_indices)}")
        print(f"number of code_indices: {len(code_indices)}")
        print(f"number of text_indices: {len(text_indices)}")

        # Run outline code test evaluator on outline code test indices
        rm_infos = data.non_tensor_batch["reward_model"]
        outline_code_test_indices = set(outline_code_test_indices) & set(valid_indices)
        outline_code_test_rm_infos = [rm_infos[i] for i in outline_code_test_indices]
        outline_code_test_batch_indices = [batch_indices[i] for i in outline_code_test_indices]
        outline_code_test_prompts = [prompts[i] for i in outline_code_test_indices]
        outline_code_test_answers = [answers[i] for i in outline_code_test_indices]

        if len(outline_code_test_indices) > 0:
            with _timer("outline_code_test_evaluator", timing_raw):
                outline_code_test_rewards = self.code_evaluator.evaluate_interleaved_outline_code_test(
                    outline_code_test_answers,
                    outline_code_test_prompts,
                    outline_code_test_rm_infos,
                    outline_code_test_batch_indices,
                )
        else:
            outline_code_test_rewards = {}

        # Run regular code evaluator on code indices
        code_rm_infos = [rm_infos[i] for i in code_indices]
        code_batch_indices = [batch_indices[i] for i in code_indices]

        code_prompts = [prompts[i] for i in code_indices]
        code_answers = [answers[i] for i in code_indices]

        if len(code_indices) > 0:
            with _timer("code_evaluator", timing_raw):
                code_rewards = self.code_evaluator.evaluate_code(
                    code_answers,
                    code_prompts,
                    code_rm_infos,
                    batch_indices=code_batch_indices
                )
        else:
            code_rewards = {}

        # Run text evaluator on text indices

        # filter only indices where answers and gt_answers have the same length
        valid_text_indices = []
        for indx in text_indices:
            # if we are interleaving, sometimes we might have more answers than gt_answers
            if len(answers[indx]) >= len(rm_infos[indx]["ground_truth"]):
                valid_text_indices.append(indx)

        print(f"number of valid_text_indices: {len(valid_text_indices)}")
        text_gt_answers = [rm_infos[i]["ground_truth"] for i in valid_text_indices]
        text_prompts = [prompts[i] for i in valid_text_indices]
        text_answers = [answers[i] for i in valid_text_indices]

        if len(valid_text_indices) > 0:
            text_decisions, text_explanations, text_raw_responses = self.compute_reward_text_interleave(
                text_prompts, text_answers, text_gt_answers
            )
            text_extras = {"autorater_scores": text_decisions}
        else:
            text_decisions = []
            text_extras = {}

        # construct reward_tensor
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        code_count = 0
        text_count = 0
        outline_code_test_count = 0
        final_code_extras = {k: [0 for _ in range(batch_size)] for k in code_rewards.keys()}
        final_text_extras = {k: [0 for _ in range(batch_size)] for k in text_extras.keys()}
        final_outline_code_test_extras = {k: [0 for _ in range(batch_size)] for k in outline_code_test_rewards.keys()}

        for i in range(batch_size):
            # Retrieve the correct length for storing the reward
            data_item = data[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            if i in code_indices:
                current_final_score = interleave_format_rewards[code_count] + code_rewards["unit_test_pass_rate"][code_count]
                for k, v in code_rewards.items():
                    final_code_extras[k][i] = v[code_count]
                code_count += 1
            elif i in valid_text_indices:
                current_final_score = interleave_format_rewards[text_count] + text_decisions[text_count]
                for k, v in text_extras.items():
                    final_text_extras[k][i] = v[text_count]
                text_count += 1
            elif i in outline_code_test_indices:
                current_final_score = interleave_format_rewards[outline_code_test_count] + outline_code_test_rewards["unit_test_pass_rate"][outline_code_test_count]
                for k, v in outline_code_test_rewards.items():
                    final_outline_code_test_extras[k][i] = v[outline_code_test_count]
                outline_code_test_count += 1
            else:
                current_final_score = 0.0

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = current_final_score

        extras = {"format_rewards": interleave_format_rewards}     
        extras.update(final_code_extras)
        extras.update(final_text_extras)
        extras.update(final_outline_code_test_extras)

        return reward_tensor, extras