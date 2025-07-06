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
        # Examples: "http://127.0.0.1:8000" or "https://autorater.mycorp.com"
        self.autorater_base_url = autorater_service_url.rstrip("/") if autorater_service_url else None
        self.use_autorater = use_autorater
        self.template_type = template_type

        self.enable_format_reward = self.config.get("enable_format_reward", True)
        self.format_reward_weight = self.config.get("format_reward_weight", 1.0) # Default to 1.0 for now, can be adjusted
        
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
            # Determine if any unit tests are present → choose evaluation type
            ground_truth_infos = data.non_tensor_batch.get("reward_model", [{} for _ in range(batch_size)])

            def _has_tests(info: Dict[str, Any]):
                return bool(
                    isinstance(info, dict)
                    and (
                        ("unit_tests" in info and info["unit_tests"])
                        or ("tests" in info and info["tests"])
                    )
                )

            use_code_evaluator = any(_has_tests(info) for info in ground_truth_infos)

            # Extract ground truth information from data.non_tensor_batch["reward_model"]
            decoded_ground_truth_answers = []
            for gt in ground_truth_infos:
                if isinstance(gt, dict) and "ground_truth" in gt:
                    decoded_ground_truth_answers.append(str(gt["ground_truth"]))
                else:
                    decoded_ground_truth_answers.append(str(gt))

            # Prepare decoded questions and responses
            decoded_questions = [self.tokenizer.decode(p_ids, skip_special_tokens=True) for p_ids in data.batch["prompts"]]
            decoded_pred_answers = [self.tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in data.batch["responses"]]

            # Check if we're doing interleaved reasoning or code evaluation
            is_interleaved = (
                self.code_evaluator.enable_interleaved_reasoning or 
                (self.template_type and "interleave" in self.template_type.lower())
            )
            
            if is_interleaved or use_code_evaluator:
                # Use CodeEvaluator for code-related evaluation (including interleaved reasoning)
                logger.info("Using CodeEvaluator for evaluation")
                autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses = self.code_evaluator.evaluate_code(
                    decoded_pred_answers, ground_truth_infos, batch_size
                )
                
                # For interleaved reasoning, we extract all answers for logging
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
                    
            else:
                # Standard text evaluation logic
                # Extract solutions (predicted answers only)
                processed_pred_answers = []
                processed_gt_answers = []
                parse_fail_flags = []
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
                        parse_fail_flags.append(True)
                        processed_pred_answers.append(pred_ans)  # fallback to full string
                    else:
                        parse_fail_flags.append(False)
                        processed_pred_answers.append(extr_pred_raw)

                    processed_gt_answers.append(extr_gt_raw)

                # Re-tokenize processed answers
                pred_answers_token_ids = [self.tokenizer.encode(ans, add_special_tokens=False) for ans in processed_pred_answers]
                gt_answers_token_ids = [self.tokenizer.encode(ans, add_special_tokens=False) for ans in processed_gt_answers]

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

                autorater_scores, autorater_decisions, autorater_explanations, autorater_raw_responses = self._evaluate_text(payload_common, batch_size)

            # Append extracted answers to extra info so that they can be dumped later
            reward_extra_info["extracted_pred"].extend(extracted_pred_answers)
            reward_extra_info["extracted_gt"].extend(extracted_gt_answers)
        else:
            logger.info("Remote AutoRater service not enabled or base URL not provided in RewardManager.")

        # --- Compute Format Rewards ---
        format_scores = []
        if self.enable_format_reward:
            predicted_answers_decoded = []
            for i in range(batch_size):
                data_item = data[i]
                response_ids = data_item.batch["responses"]
                predicted_answers_decoded.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))

            for pred_answer in predicted_answers_decoded:
                score = format_check_reward(pred_answer)
                format_scores.append(score)
        else:
            format_scores = [0.0] * batch_size # If disabled, format score is 0

        # --- Combine scores and populate reward_tensor and reward_extra_info ---
        final_scores = []
        for i in range(batch_size):
            # Retrieve the correct length for storing the reward
            data_item = data[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            # Ensure we have a valid position to place the reward
            if valid_response_length > 0:
                current_final_score = autorater_scores[i] + (self.format_reward_weight * format_scores[i])
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

    def _get_tokenizer(self):
        return self.tokenizer 

    # ------------------------------------------------------------------
    # Helper methods for remote evaluation
    # ------------------------------------------------------------------

    def _evaluate_text(self, payload: Dict[str, Any], batch_size: int):
        """Call /evaluate_autorater endpoint and return extracted fields."""
        full_url = f"{self.autorater_base_url}/evaluate_autorater"
        response = requests.post(full_url, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()

        scores = data.get("autorater_scores", [0.0] * batch_size)
        decisions = data.get("autorater_decisions", [-1] * batch_size)
        explanations = data.get("autorater_explanations", ["N/A"] * batch_size)
        raw = data.get("autorater_raw_responses", ["N/A"] * batch_size)

        # Shape scores using decision labels
        shaped_scores = []
        
        for dec, raw_score in zip(decisions, scores):
            if dec == 1:
                shaped_scores.append(2.0)
            elif dec == 0:
                shaped_scores.append(-1.5)
            else:
                shaped_scores.append(raw_score)

        return shaped_scores, decisions, explanations, raw