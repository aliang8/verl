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
from verl.workers.autorater.autorater_utils import extract_solution, format_autorater_prompt # Added extract_solution and format_autorater_prompt
from verl.workers.code_evaluator import CodeEvaluator # Import the new CodeEvaluator
from verl.trainer.ppo.reward_fns import count_interleaved_answers, interleaved_format_reward # Import interleaved functions
from verl.utils.autorater_client import call_autorater_service  # New modular AutoRater client

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
        )

    def compute_rewards(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Computes AutoRater and format rewards for a batch of data.
        Routes to either standard or interleaved reward computation based on configuration.

        Args:
            data: DataProto containing batch data (prompts, responses, reward_model_info).
            return_dict: If True, returns a dictionary with detailed reward info.

        Returns:
            A tensor of combined rewards or a tuple containing a tensor and a dictionary with detailed reward information.
        """
        # Check if we're doing interleaved reasoning
        is_interleaved = self.template_type and "interleave" in self.template_type.lower()

        if is_interleaved:
            return self.compute_interleaved_rewards(data, return_dict)
        else:
            return self.compute_rewards_flat(data, return_dict)

    def compute_rewards_flat(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Computes standard (non-interleaved) rewards for a batch of data.

        Args:
            data: DataProto containing batch data (prompts, responses, reward_model_info).
            return_dict: If True, returns a dictionary with detailed reward info.

        Returns:
            A tensor of combined rewards or a tuple containing a tensor and a dictionary with detailed reward information.
        """
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        autorater_scores = [0.0] * batch_size
        autorater_decisions = [-1] * batch_size
        autorater_explanations = ["N/A"] * batch_size
        autorater_raw_responses = ["N/A"] * batch_size

        # --- Call remote AutoRater service or CodeEvaluator if enabled ---
        if self.use_autorater and self.autorater_base_url:
            ground_truth_infos = data.non_tensor_batch.get("reward_model", [{} for _ in range(batch_size)])
            
            # Determine evaluation type and call appropriate method
            if self._should_use_code_evaluation(ground_truth_infos):
                autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses, extracted_pred_answers, extracted_gt_answers = self._evaluate_code(
                    data, ground_truth_infos, batch_size
                )
            else:
                autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses, extracted_pred_answers, extracted_gt_answers = self._evaluate_text_responses(
                    data, ground_truth_infos, batch_size
                )

            # Append extracted answers to extra info so that they can be dumped later
            reward_extra_info["extracted_pred"].extend(extracted_pred_answers)
            reward_extra_info["extracted_gt"].extend(extracted_gt_answers)
        else:
            logger.info("Remote AutoRater service not enabled or base URL not provided in RewardManager.")

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
            current_final_score = autorater_scores[i] + (self.format_reward_weight * format_scores[i])

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

        if return_dict:
            return reward_tensor, reward_extra_info
        
        return reward_tensor

    def compute_interleaved_rewards(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Computes interleaved reasoning rewards for a batch of data.
        Uses interleaved format reward and calls code evaluator with interleaving enabled.
        Only performs interleaved evaluation if answer count > 3.

        Args:
            data: DataProto containing batch data (prompts, responses, reward_model_info).
            return_dict: If True, returns a dictionary with detailed reward info.

        Returns:
            A tensor of combined rewards or a tuple containing a tensor and a dictionary with detailed reward information.
        """
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        logger.info("Computing interleaved reasoning rewards")
        ground_truth_infos = data.non_tensor_batch.get("reward_model", [{} for _ in range(batch_size)])
        
        # Prepare decoded responses
        decoded_pred_answers = [self.tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in data.batch["responses"]]
        decoded_prompts = [self.tokenizer.decode(p_ids, skip_special_tokens=True) for p_ids in data.batch["prompts"]]

        # --- First, compute format scores and answer counts ---
        interleaved_format_scores, answer_counts = self._compute_interleaved_format_scores(data, batch_size)

        # --- Filter samples that meet the interleaved criteria (answer count > 3) ---
        interleaved_indices = []
        
        for i in range(batch_size):
            if answer_counts[i] >= 3:  # Only do interleaved evaluation if answer count > 3
                interleaved_indices.append(i)

        # Initialize scores arrays - samples with insufficient answers get no reward
        autorater_scores = [0.0] * batch_size
        autorater_decisions = [0] * batch_size  # 0 = failed/no evaluation
        autorater_explanations = ["No evaluation - insufficient answer count (<3)"] * batch_size
        autorater_raw_responses = ["No evaluation - answer count <3"] * batch_size

        # --- Evaluate only interleaved samples using CodeEvaluator ---
        if interleaved_indices:
            logger.info(f"Evaluating {len(interleaved_indices)} samples with interleaved reasoning (answer count > 3)")
            logger.info(f"Skipping {batch_size - len(interleaved_indices)} samples with insufficient answer count (≤3)")
            
            # Prepare data for interleaved samples
            interleaved_responses = [decoded_pred_answers[i] for i in interleaved_indices]
            interleaved_prompts = [decoded_prompts[i] for i in interleaved_indices]
            interleaved_ground_truths = [ground_truth_infos[i] for i in interleaved_indices]
            
            # Use CodeEvaluator for interleaved evaluation
            interleaved_scores, interleaved_decisions, interleaved_explanations, interleaved_raw = self.code_evaluator.evaluate_code(
                decoded_pred_answers=interleaved_responses,
                original_prompts=interleaved_prompts,
                ground_truth_infos=interleaved_ground_truths,
                batch_size=len(interleaved_indices),
            )
            
            # Assign back to main arrays (only for qualified samples)
            for idx, i in enumerate(interleaved_indices):
                autorater_scores[i] = interleaved_scores[idx]
                autorater_decisions[i] = interleaved_decisions[idx]
                autorater_explanations[i] = interleaved_explanations[idx]
                autorater_raw_responses[i] = interleaved_raw[idx]
        else:
            logger.info("No samples qualified for interleaved evaluation (all had answer count ≤3)")

        # --- Combine scores and populate reward_tensor and reward_extra_info ---
        final_scores = []
        for i in range(batch_size):
            # Retrieve the correct length for storing the reward
            data_item = data[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            # Calculate combined score
            base_score = autorater_scores[i]  # 0.0 for samples with insufficient answer count
            format_score = self.interleaved_format_reward_weight * interleaved_format_scores[i]
            
            # Determine evaluation type used
            used_interleaved_eval = i in interleaved_indices
            
            # Simple addition - no penalty system needed since insufficient samples get 0 base score
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
            reward_extra_info["interleaved_format_scores"].append(interleaved_format_scores[i])
            reward_extra_info["answer_counts"].append(answer_counts[i])
            reward_extra_info["used_interleaved_eval"].append(used_interleaved_eval)
            reward_extra_info["final_scores"].append(final_scores[i])

        # Extract answers for logging
        extracted_pred_answers = []
        extracted_gt_answers = []
        for pred_ans, gt_info in zip(decoded_pred_answers, ground_truth_infos):
            all_answers = extract_solution(pred_ans, extract_all=True)
            extracted_pred_answers.append(all_answers if all_answers else "No answers extracted")
            
            if isinstance(gt_info, dict) and "ground_truth" in gt_info:
                extracted_gt_answers.append(str(gt_info["ground_truth"]))
            else:
                extracted_gt_answers.append(str(gt_info))

        reward_extra_info["extracted_pred"].extend(extracted_pred_answers)
        reward_extra_info["extracted_gt"].extend(extracted_gt_answers)

        if return_dict:
            return reward_tensor, reward_extra_info
        
        return reward_tensor

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
        batch_size: int
    ) -> Tuple[List[float], List[int], List[str], List[str], List[str], List[str]]:
        """Evaluate code responses using CodeEvaluator."""
        logger.info("Using CodeEvaluator for evaluation")
        
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
        autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses = self.code_evaluator.evaluate_code(
            decoded_pred_answers,
            decoded_prompts,
            ground_truth_infos,
            batch_size,
        )

        import ipdb; ipdb.set_trace()
        
        # Check if we're doing interleaved reasoning for logging
        is_interleaved = (
            self.code_evaluator.enable_interleaved_reasoning or 
            (self.template_type and "interleave" in self.template_type.lower())
        )
        
        # Extract answers for logging
        extracted_pred_answers = []
        extracted_gt_answers = []
        for pred_ans, gt_ans in zip(decoded_pred_answers, decoded_ground_truth_answers):
            if is_interleaved:
                all_answers = extract_solution(pred_ans, extract_all=True)
                extracted_pred_answers.append(all_answers if all_answers else "No answers extracted")
            else:
                single_answer = extract_solution(pred_ans)
                extracted_pred_answers.append(single_answer if single_answer else "No answer extracted")
            extracted_gt_answers.append(gt_ans)
        
        return autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses, extracted_pred_answers, extracted_gt_answers

    def _evaluate_text_responses(
        self, 
        data: DataProto, 
        ground_truth_infos: List[Dict[str, Any]], 
        batch_size: int
    ) -> Tuple[List[float], List[int], List[str], List[str], List[str], List[str]]:
        """Evaluate text responses using remote AutoRater service."""
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
            extr_pred_raw = extract_solution(pred_ans)
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
        autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(  # type: ignore[arg-type]
            self.autorater_base_url, payload_common, batch_size
        )
        
        return autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses, extracted_pred_answers, extracted_gt_answers

    def _compute_standard_format_scores(self, data: DataProto, batch_size: int) -> List[float]:
        """Compute standard format scores for responses."""
        format_scores = []
        
        for i in range(batch_size):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            predicted_answer = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Standard format reward
            if self.enable_format_reward:
                score = format_check_reward(predicted_answer)
                format_scores.append(score)
            else:
                format_scores.append(0.0)
        
        return format_scores

    def _compute_interleaved_format_scores(self, data: DataProto, batch_size: int) -> Tuple[List[float], List[int]]:
        """Compute interleaved format scores and answer counts for responses."""
        interleaved_format_scores = []
        answer_counts = []
        
        for i in range(batch_size):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            predicted_answer = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            answer_count = count_interleaved_answers(predicted_answer)
            interleaved_score = interleaved_format_reward(predicted_answer, self.min_answer_count_for_interleaved)
            interleaved_format_scores.append(interleaved_score)
            answer_counts.append(answer_count)
            
        return interleaved_format_scores, answer_counts