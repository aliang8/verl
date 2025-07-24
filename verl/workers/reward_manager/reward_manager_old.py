import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Union, Tuple, Optional
import requests # Added this import

import numpy as np  # type: ignore
import torch  # type: ignore
from omegaconf import DictConfig, OmegaConf  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from verl import DataProto # type: ignore
from verl.single_controller.base.decorator import register as base_register
from verl.workers.reward_manager.registry import register
from verl.utils.reward_score.autorater_reward import AutoRaterReward
from verl.trainer.ppo.reward_fns import format_check_reward
from verl.workers.autorater.autorater_utils import extract_solution # Added extract_solution and format_autorater_prompt
from verl.workers.code_evaluator import CodeEvaluator # Import the new CodeEvaluator
from verl.trainer.ppo.reward_fns import count_interleaved_answers, interleaved_format_reward # Import interleaved functions
from verl.utils.autorater_client import call_autorater_service  # New modular AutoRater client
from verl.utils.debug.performance import _timer  # Add timing support

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("default_reward_manager")
class RewardManager:
    """
    Manages the computation of various reward scores, including AutoRater and format-based rewards.
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        autorater_service_url: Optional[str] = None,
        use_autorater: bool = False,
        template_type: Optional[str] = None,
    ):
        """
        Initializes the RewardManager.

        Args:
            config: Configuration object for the reward manager.
            tokenizer: The tokenizer instance to use for decoding.
            autorater_service_url: URL of the remote AutoRater FastAPI service.
            use_autorater: Whether to use the remote AutoRater service.
            template_type: The template type used for generation (e.g., "interleave").
        """
        self.config = config
        self.tokenizer = tokenizer
        # Accept base URL (host:port) without endpoint path; we will append proper path dynamically.
        # Examples: "http://127.0.0.1:8000"
        self.autorater_base_url = autorater_service_url.rstrip("/") if autorater_service_url else None
        self.use_autorater = use_autorater
        self.template_type = template_type

        self.enable_format_reward = self.config.get("enable_format_reward", True)
        self.format_reward_weight = self.config.get("format_reward_weight", 1.0) # Default to 1.0 for now, can be adjusted
        
        # Interleaved format reward configuration
        self.interleaved_format_reward_weight = self.config.get("interleaved_format_reward_weight", 1.0)
        self.min_answer_count_for_interleaved = self.config.get("min_answer_count_for_interleaved", 3)
        
        # Initialize CodeEvaluator for code-related evaluation
        code_evaluator_config = self.config.get("code_evaluator", {})
        self.code_evaluator = CodeEvaluator(
            config=OmegaConf.create(code_evaluator_config),
            tokenizer=self.tokenizer,
            template_type=template_type,
            autorater_service_url=self.autorater_base_url,
        )
        
        # Error tracking configuration
        self.enable_error_tracking = self.config.get("enable_error_tracking", True)
        self.error_tracking_output_dir = self.config.get("error_tracking_output_dir", "./error_tracking_logs")

    def start_epoch(self, epoch: int):
        """Start error tracking for a new epoch"""
        if self.enable_error_tracking:
            print(f"Starting error tracking for epoch {epoch}")
            self.code_evaluator.start_epoch(epoch)
            
    def save_epoch_metadata(self, epoch: int, output_dir: Optional[str] = None, log_to_wandb: bool = True):
        """Save error tracking metadata for the completed epoch and log to wandb"""
        if self.enable_error_tracking:
            save_dir = output_dir or self.error_tracking_output_dir
            print(f"Saving error tracking metadata for epoch {epoch} to {save_dir}")
            self.code_evaluator.save_epoch_metadata(save_dir, epoch)
            
            # Print summary to console
            summary = self.code_evaluator.get_error_summary()
            print(f"Epoch {epoch} Error Summary:")
            print(f"  Total prompts: {summary['total_prompts']}")
            print(f"  Successful: {summary['successful_prompts']} ({summary['success_rate']:.1%})")
            print(f"  Failed: {summary['failed_prompts']}")
            print(f"  Error types: {summary['error_counts']}")
            print(f"  Duration: {summary['epoch_duration']:.1f}s")
            
            # Log to wandb if enabled
            if log_to_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb_metrics = self.code_evaluator.get_wandb_metrics()
                        wandb.log(wandb_metrics)
                        print(f"  Logged {len(wandb_metrics)} error tracking metrics to wandb")
                    else:
                        print("  wandb not initialized, skipping wandb logging")
                except ImportError:
                    print("  wandb not available, skipping wandb logging")
                except Exception as e:
                    print(f"  Error logging to wandb: {e}")

    def compute_rewards(
        self,
        data: DataProto,
        return_dict: bool = False,
        timing_raw: Optional[Dict[str, float]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Computes AutoRater and format rewards for a batch of data.
        Routes to either standard or interleaved reward computation based on configuration.

        Args:
            data: DataProto containing batch data (prompts, responses, reward_model_info).
            return_dict: If True, returns a dictionary with detailed reward info.
            timing_raw: Dictionary to store timing information.

        Returns:
            A tensor of combined rewards or a tuple containing a tensor and a dictionary with detailed reward information.
        """
        # Initialize timing_raw if not provided
        if timing_raw is None:
            timing_raw = {}
            
        # Check if we're doing interleaved reasoning
        is_interleaved = self.template_type and "interleave" in self.template_type.lower()

        if is_interleaved:
            return self.compute_interleaved_rewards(data, return_dict, timing_raw)
        else:
            return self.compute_rewards_flat(data, return_dict, timing_raw)

    def compute_rewards_flat(
        self,
        data: DataProto,
        return_dict: bool = False,
        timing_raw: Optional[Dict[str, float]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Computes standard (non-interleaved) rewards for a batch of data.

        Args:
            data: DataProto containing batch data (prompts, responses, reward_model_info).
            return_dict: If True, returns a dictionary with detailed reward info.
            timing_raw: Dictionary to store timing information.

        Returns:
            A tensor of combined rewards or a tuple containing a tensor and a dictionary with detailed reward information.
        """
        if timing_raw is None:
            timing_raw = {}
        
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        autorater_scores = [0.0] * batch_size
        autorater_decisions = [-1] * batch_size
        autorater_explanations = ["N/A"] * batch_size
        autorater_raw_responses = ["N/A"] * batch_size
        component_rewards = defaultdict(lambda: [0.0] * batch_size)
        extracted_pred_answers = ["" for _ in range(batch_size)]
        extracted_gt_answers = ["" for _ in range(batch_size)]

        data_sources = data.non_tensor_batch.get("data_source", [None] * batch_size)
        ground_truth_infos = data.non_tensor_batch.get("reward_model", [{} for _ in range(batch_size)])

        # Group indices by data_source type
        code_indices = []
        text_indices = []
        for i, ds in enumerate(data_sources):
            if ds and "code" in str(ds).lower():
                code_indices.append(i)
            else:
                text_indices.append(i)

        print(f"number of code_indices: {len(code_indices)}")
        print(f"number of text_indices: {len(text_indices)}")

        # Evaluate code samples in a batch
        if code_indices:
            code_data = data.select_idxs(code_indices)
            code_gt_infos = [ground_truth_infos[i] for i in code_indices]
            eval_scores, eval_decisions, eval_explanations, eval_raw, comp_rewards, pred_ans, gt_ans = self._evaluate_code(
                code_data, code_gt_infos, len(code_indices), timing_raw
            )
            for idx, i in enumerate(code_indices):
                autorater_scores[i] = eval_scores[idx]
                autorater_decisions[i] = eval_decisions[idx]
                autorater_explanations[i] = eval_explanations[idx]
                autorater_raw_responses[i] = eval_raw[idx]
                for k, v in comp_rewards.items():
                    if k not in component_rewards:
                        component_rewards[k] = [0.0] * batch_size
                    component_rewards[k][i] = v[idx] if v[idx] is not None else 0.0
                extracted_pred_answers[i] = pred_ans[idx] if pred_ans[idx] is not None else ""
                extracted_gt_answers[i] = gt_ans[idx] if gt_ans[idx] is not None else ""

        # Evaluate text samples in a batch
        if text_indices:
            text_data = data.select_idxs(text_indices)
            text_gt_infos = [ground_truth_infos[i] for i in text_indices]
            eval_scores, eval_decisions, eval_explanations, eval_raw, pred_ans, gt_ans = self._evaluate_text_responses(
                text_data, text_gt_infos, len(text_indices), timing_raw
            )
            for idx, i in enumerate(text_indices):
                autorater_scores[i] = eval_scores[idx]
                autorater_decisions[i] = eval_decisions[idx]
                autorater_explanations[i] = eval_explanations[idx]
                autorater_raw_responses[i] = eval_raw[idx]
                extracted_pred_answers[i] = pred_ans[idx] if pred_ans[idx] is not None else ""
                extracted_gt_answers[i] = gt_ans[idx] if gt_ans[idx] is not None else ""

        # --- Compute Format Rewards ---
        format_scores = self._compute_standard_format_scores(data, batch_size)

        # --- Combine scores and populate reward_tensor and reward_extra_info ---
        final_scores = []
        for i in range(batch_size):
            # Retrieve the correct length for storing the reward
            data_item = data[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            # Calculate combined score
            base_score = autorater_scores[i]
            # format_score = self.format_reward_weight * format_scores[i]
            format_score = 0.0
            current_final_score = base_score + format_score

            # Ensure we have a valid position to place the reward
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = current_final_score
                final_scores.append(current_final_score)
            else:
                logger.warning(f"Response length is 0 for sample {i}, no reward applied to token.")
                final_scores.append(0.0)
            reward_extra_info["autorater_scores"].append(autorater_scores[i])
            reward_extra_info["autorater_decisions"].append(autorater_decisions[i])
            reward_extra_info["autorater_explanations"].append(autorater_explanations[i])
            reward_extra_info["autorater_raw_responses"].append(autorater_raw_responses[i])
            reward_extra_info["format_scores"].append(format_scores[i])
            reward_extra_info["final_scores"].append(final_scores[i])
            for k, v in component_rewards.items():
                reward_extra_info[k].append(v[i])

        reward_extra_info["extracted_pred"].extend(extracted_pred_answers)
        reward_extra_info["extracted_gt"].extend(extracted_gt_answers)

        if return_dict:
            return reward_tensor, reward_extra_info
        return reward_tensor

    def compute_interleaved_rewards(
        self,
        data: DataProto,
        return_dict: bool = False,
        timing_raw: Optional[Dict[str, float]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Computes interleaved reasoning rewards for a batch of data.
        Uses interleaved format reward and calls code evaluator with interleaving enabled.
        Only performs interleaved evaluation if answer count > 3.

        Args:
            data: DataProto containing batch data (prompts, responses, reward_model_info).
            return_dict: If True, returns a dictionary with detailed reward info.
            timing_raw: Dictionary to store timing information.

        Returns:
            A tensor of combined rewards or a tuple containing a tensor and a dictionary with detailed reward information.
        """
        if timing_raw is None:
            timing_raw = {}
        
        batch_indices = data.non_tensor_batch["index"]

        print("Computing interleaved reasoning rewards")
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        logger.info("Computing interleaved reasoning rewards")
        ground_truth_infos = data.non_tensor_batch.get("reward_model", [{} for _ in range(batch_size)])
        data_sources = data.non_tensor_batch.get("data_source", [None] * batch_size)
        decoded_pred_answers = [self.tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in data.batch["responses"]]
        decoded_prompts = [self.tokenizer.decode(p_ids, skip_special_tokens=True) for p_ids in data.batch["prompts"]]

        # --- First, compute format scores and answer counts ---
        interleaved_format_scores, answer_counts = self._compute_interleaved_format_scores(data, batch_size)
        # Group indices by data_source type
        code_indices = []
        text_indices = []
        for i, ds in enumerate(data_sources):
            if ds and "code" in str(ds).lower():
                code_indices.append(i)
            else:
                text_indices.append(i)

        # Only do interleaved evaluation if answer count > 3
        interleaved_indices = [i for i in range(batch_size) if answer_counts[i] >= 3]
        # Intersect with code_indices and text_indices
        code_interleaved_indices = [i for i in interleaved_indices if i in code_indices]
        text_interleaved_indices = [i for i in interleaved_indices if i in text_indices]
        
        print(f"number of code_interleaved_indices: {len(code_interleaved_indices)}")
        print(f"number of text_interleaved_indices: {len(text_interleaved_indices)}")
        print(f"number of interleaved_indices: {len(interleaved_indices)}")
        
        autorater_scores = [0.0] * batch_size
        autorater_decisions = [0] * batch_size
        autorater_explanations = ["No evaluation - insufficient answer count (<3)"] * batch_size
        autorater_raw_responses = ["No evaluation - answer count <3"] * batch_size
        component_rewards_all = {k: [0.0] * batch_size for k in ["description_scores", "code_scores", "unit_test_scores", "pass@1"]}
        # Evaluate code interleaved samples in a batch
        if code_interleaved_indices:
            code_data = data.select_idxs(code_interleaved_indices)
            code_gt_infos = [ground_truth_infos[i] for i in code_interleaved_indices]
            interleaved_scores, interleaved_decisions, interleaved_explanations, interleaved_raw, component_rewards = self.code_evaluator.evaluate_code(
                decoded_pred_answers=[decoded_pred_answers[i] for i in code_interleaved_indices],
                original_prompts=[decoded_prompts[i] for i in code_interleaved_indices],
                ground_truth_infos=code_gt_infos,
                batch_size=len(code_interleaved_indices),
                batch_indices=[batch_indices[i] for i in code_interleaved_indices],
                timing_raw=timing_raw,
            )
            for idx, i in enumerate(code_interleaved_indices):
                autorater_scores[i] = interleaved_scores[idx]
                autorater_decisions[i] = interleaved_decisions[idx]
                autorater_explanations[i] = interleaved_explanations[idx]
                autorater_raw_responses[i] = interleaved_raw[idx]
                for k, v in component_rewards.items():
                    if k not in component_rewards_all:
                        component_rewards_all[k] = [0.0] * batch_size
                    component_rewards_all[k][i] = v[idx] if v[idx] is not None else 0.0
        
        # Evaluate text interleaved samples in a batch
        if text_interleaved_indices:
            text_data = data.select_idxs(text_interleaved_indices)
            text_gt_infos = [ground_truth_infos[i] for i in text_interleaved_indices]
            interleaved_scores, interleaved_decisions, interleaved_explanations, interleaved_raw, component_rewards, _ = self._evaluate_text_responses(
                text_data, text_gt_infos, len(text_interleaved_indices), timing_raw
            )
            for idx, i in enumerate(text_interleaved_indices):
                autorater_scores[i] = interleaved_scores[idx]
                autorater_decisions[i] = interleaved_decisions[idx]
                autorater_explanations[i] = interleaved_explanations[idx]
                autorater_raw_responses[i] = interleaved_raw[idx]
        
        final_scores = []
        for i in range(batch_size):
            # Retrieve the correct length for storing the reward
            data_item = data[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            base_score = autorater_scores[i]
            format_score = self.interleaved_format_reward_weight * interleaved_format_scores[i]
            used_interleaved_eval = i in interleaved_indices
            current_final_score = base_score + format_score
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = current_final_score
                final_scores.append(current_final_score)
            else:
                logger.warning(f"Response length is 0 for sample {i}, no reward applied to token.")
                final_scores.append(0.0)
            reward_extra_info["autorater_scores"].append(autorater_scores[i])
            reward_extra_info["autorater_decisions"].append(autorater_decisions[i])
            reward_extra_info["autorater_explanations"].append(autorater_explanations[i])
            reward_extra_info["autorater_raw_responses"].append(autorater_raw_responses[i])
            reward_extra_info["interleaved_format_scores"].append(interleaved_format_scores[i])
            reward_extra_info["answer_counts"].append(answer_counts[i])
            reward_extra_info["used_interleaved_eval"].append(used_interleaved_eval)
            reward_extra_info["final_scores"].append(final_scores[i])
            for k, v in component_rewards_all.items():
                reward_extra_info[k].append(v[i])

        # Extract answers for logging
        extracted_pred_answers = []
        extracted_gt_answers = []
        for pred_ans, gt_info in zip(decoded_pred_answers, ground_truth_infos):
            all_answers = extract_solution(pred_ans, template_type=self.template_type)
            extracted_pred_answers.append(all_answers if all_answers else "No answers extracted")
            if isinstance(gt_info, dict) and "ground_truth" in gt_info:
                extracted_gt_answers.append(str(gt_info["ground_truth"]))
            else:
                extracted_gt_answers.append(str(gt_info))
            reward_extra_info["extracted_pred"].append(extracted_pred_answers)
            reward_extra_info["extracted_gt"].append(extracted_gt_answers)
        if return_dict:
            return reward_tensor, reward_extra_info
        return reward_tensor

    def compute_helpfulness_scores(self, prompts: list[str], answers: list[str]) -> tuple[list[float], list[int]]:
        """
        Compute helpfulness scores for a list of (prompt, answer) pairs using the AutoRater service.
        If interleaving is enabled, extract all intermediate answers and compute helpfulness for each.
        Only non-empty extracted answers are sent to AutoRater; empty ones default to 0.0 score and 0 decision.
        Returns a tuple of (autorater_scores, autorater_decisions), both lists of length equal to the number of extracted answers.
        """
        if not self.use_autorater or not self.autorater_base_url:
            raise RuntimeError("AutoRater service is not enabled or URL is not set.")
        batch_size = len(answers)
        assert len(prompts) == batch_size, "prompts and answers must have the same length"
        extracted_prompts = []
        extracted_answers = []
        reward_model_info = []
        # Interleaving: extract all intermediate answers
        if self.template_type and "interleave" in self.template_type.lower():
            for p, a in zip(prompts, answers):
                all_answers = extract_solution(a, template_type=self.template_type)
                if all_answers:
                    context = []
                    for ans in all_answers:
                        extracted_prompts.append(p)
                        extracted_answers.append(ans)
                        # Add context with previous answers
                        reward_model_info.append({"template": "helpfulness", "context": context.copy()})
                        context.append(ans)  # Accumulate context for next answer
                else:
                    extracted_prompts.append(p)
                    extracted_answers.append("")
                    reward_model_info.append({"template": "helpfulness", "context": []})
        else:
            for p, a in zip(prompts, answers):
                ans = extract_solution(a, template_type=self.template_type)
                extracted_prompts.append(p)
                extracted_answers.append(ans if ans is not None else "")
                reward_model_info.append({"template": "helpfulness", "context": []})
        
        # Filter out empty answers and prepare for AutoRater
        non_empty_indices = []
        non_empty_prompts = []
        non_empty_answers = []
        non_empty_reward_model_info = []
        
        for i, (prompt, answer, rm_info) in enumerate(zip(extracted_prompts, extracted_answers, reward_model_info)):
            if answer and answer.strip():  # Check if answer is non-empty
                non_empty_indices.append(i)
                non_empty_prompts.append(prompt)
                non_empty_answers.append(answer)
                non_empty_reward_model_info.append(rm_info)
        
        # Initialize results with defaults (0.0 score, 0decision for all)
        autorater_scores = [0.0] * len(extracted_answers)
        autorater_decisions = [0] * len(extracted_answers)
        
        # Only call AutoRater if we have non-empty answers
        if non_empty_answers:
            responses = [self.tokenizer.encode(ans, add_special_tokens=False) for ans in non_empty_answers]
            attention_mask = [[1] * len(r) for r in responses]
            position_ids = [[i for i in range(len(r))] for r in responses]
            payload = {
                "prompts": non_empty_prompts,
                "responses": responses,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "reward_model_info": non_empty_reward_model_info,
            }
            non_empty_scores, non_empty_decisions, *_ = call_autorater_service(self.autorater_base_url, payload, len(non_empty_answers))
            
            # Update the results for non-empty answers
            for idx, score, decision in zip(non_empty_indices, non_empty_scores, non_empty_decisions):
                autorater_scores[idx] = score
                autorater_decisions[idx] = decision
        
        return autorater_scores, autorater_decisions

    def _get_tokenizer(self):
        return self.tokenizer

    # ------------------------------------------------------------------
    # Helper methods for remote evaluation
    # ------------------------------------------------------------------

    def _should_use_code_evaluation(self, ground_truth_infos: List[Dict[str, Any]]) -> bool:
        """Determine if code evaluation should be used based on ground truth info."""
        def _has_tests(info: Dict[str, Any]):
            return bool(
                isinstance(info, dict)
                and (
                    ("unit_tests" in info and info["unit_tests"])
                    or ("tests" in info and info["tests"])
                )
            )

        use_code_evaluator = any(_has_tests(info) for info in ground_truth_infos)
        
        # Check if we're doing interleaved reasoning
        is_interleaved = (
            (self.template_type and "interleave" in self.template_type.lower())
        )
        
        return is_interleaved or use_code_evaluator

    def _evaluate_code(
        self, 
        data: DataProto, 
        ground_truth_infos: List[Dict[str, Any]], 
        batch_size: int,
        timing_raw: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[float], List[int], List[str], List[str], Dict[str, List[float]], List[str], List[str]]:
        """Evaluate code responses using CodeEvaluator."""
        if timing_raw is None:
            timing_raw = {}
            
        logger.info("Using CodeEvaluator for evaluation")
        batch_indices = data.non_tensor_batch["index"]
        
        # Extract ground truth information
        decoded_ground_truth_answers = []
        for gt in ground_truth_infos:
            if isinstance(gt, dict) and "ground_truth" in gt:
                decoded_ground_truth_answers.append(str(gt["ground_truth"]))
            else:
                decoded_ground_truth_answers.append(str(gt))

        # Prepare decoded responses
        decoded_pred_answers = [self.tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in data.batch["responses"]]
        decoded_prompts = [self.tokenizer.decode(p_ids, skip_special_tokens=True) for p_ids in data.batch["prompts"]]

        # Evaluate using CodeEvaluator
        with _timer("code_evaluator_evaluate", timing_raw):
            autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses, component_rewards = self.code_evaluator.evaluate_code(
                decoded_pred_answers,
                decoded_prompts,
                ground_truth_infos,
                batch_size,
                batch_indices=batch_indices,
                timing_raw=timing_raw,
            )
        
        is_interleaved = self.code_evaluator.is_interleaved

        # Extract answers for logging
        extracted_pred_answers = []
        extracted_gt_answers = []
        for pred_ans, gt_ans in zip(decoded_pred_answers, decoded_ground_truth_answers):
            if is_interleaved:
                all_answers = extract_solution(pred_ans, template_type=self.template_type)
                extracted_pred_answers.append(all_answers if all_answers else "No answers extracted")
            else:
                single_answer = extract_solution(pred_ans, template_type=self.template_type)
                extracted_pred_answers.append(single_answer if single_answer else "No answer extracted")
            extracted_gt_answers.append(gt_ans)
        
        return autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses, component_rewards, extracted_pred_answers, extracted_gt_answers

    def _evaluate_text_responses(
        self, 
        data: DataProto, 
        ground_truth_infos: List[Dict[str, Any]], 
        batch_size: int,
        timing_raw: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[float], List[int], List[str], List[str], List[str], List[str]]:
        """Evaluate text responses using remote AutoRater service."""
        if timing_raw is None:
            timing_raw = {}
            
        logger.info("Using remote AutoRater service for text evaluation")
        
        # Extract ground truth information
        decoded_ground_truth_answers = []
        for gt in ground_truth_infos:
            if isinstance(gt, dict) and "ground_truth" in gt:
                decoded_ground_truth_answers.append(str(gt["ground_truth"]))
            else:
                decoded_ground_truth_answers.append(str(gt))

        # Prepare decoded responses
        decoded_pred_answers = [self.tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in data.batch["responses"]]
        decoded_prompts = [self.tokenizer.decode(p_ids, skip_special_tokens=True) for p_ids in data.batch["prompts"]]

        # Extract solutions (predicted answers only)
        processed_pred_answers = []
        processed_gt_answers = []
        extracted_pred_answers = []
        extracted_gt_answers = []

        for pred_ans, gt_ans in zip(decoded_pred_answers, decoded_ground_truth_answers):
            # Attempt to parse predicted answer inside <answer> tags
            extr_pred_raw = extract_solution(pred_ans, template_type=self.template_type)
            extracted_pred_answers.append(extr_pred_raw)

            # Ground-truth answer stays as-is
            extr_gt_raw = gt_ans
            extracted_gt_answers.append(extr_gt_raw)

            if extr_pred_raw is None:
                processed_pred_answers.append(pred_ans)  # fallback to full string
            else:
                processed_pred_answers.append(extr_pred_raw)

            processed_gt_answers.append(extr_gt_raw)

        # Re-tokenize processed answers
        pred_answers_token_ids = [self.tokenizer.encode(ans, add_special_tokens=False) for ans in processed_pred_answers]

        # Rebuild reward_model_info with ground_truth and pass through unit_tests/libs when available
        new_reward_model_info = []
        for orig_info, gt in zip(ground_truth_infos, processed_gt_answers):
            info_dict: Dict[str, Any] = {"ground_truth": gt}
            if isinstance(orig_info, dict):
                # pass unit tests if present so code evaluator can run them
                if orig_info.get("unit_tests"):
                    info_dict["unit_tests"] = orig_info["unit_tests"]
                elif orig_info.get("tests"):
                    info_dict["unit_tests"] = orig_info["tests"]
                # include optional libs
                if orig_info.get("libs"):
                    info_dict["libs"] = orig_info["libs"]
            new_reward_model_info.append(info_dict)
        
        payload_common = {
            "prompts": data.batch["prompts"].cpu().tolist(),
            "responses": pred_answers_token_ids,
            "attention_mask": data.batch["attention_mask"].cpu().tolist(),
            "position_ids": data.batch["position_ids"].cpu().tolist(),
            "reward_model_info": new_reward_model_info,
        }

        # mypy: self.autorater_base_url is ensured non-None by caller when use_autorater is True
        assert self.autorater_base_url is not None, "autorater_base_url must be provided when calling AutoRater service"
        
        with _timer("call_autorater_service", timing_raw):
            autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(  # type: ignore[arg-type]
                self.autorater_base_url, payload_common, batch_size
            )
        
        return autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses, extracted_pred_answers, extracted_gt_answers

    def _compute_interleaved_format_scores(self, data: DataProto, batch_size: int) -> Tuple[List[float], List[int]]:
        """Compute interleaved format scores and answer counts for responses."""
        format_scores = []
        answer_counts = []
        
        for i in range(batch_size):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            predicted_answer = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Count answers and compute interleaved format reward
            answer_count = count_interleaved_answers(predicted_answer)
            interleaved_score = interleaved_format_reward(predicted_answer, self.min_answer_count_for_interleaved)
            format_scores.append(interleaved_score)
            answer_counts.append(answer_count)
            
            # Interleaved format reward
            if self.enable_format_reward:
                score = interleaved_format_reward(predicted_answer, answer_count)
                format_scores.append(score)
            else:
                format_scores.append(0.0)
        
        return format_scores, answer_counts