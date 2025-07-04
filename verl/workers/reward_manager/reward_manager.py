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

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("default_reward_manager")
class RewardManager:
    """
    Manages the computation of various reward scores, including AutoRater and format-based rewards.
    """

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer, autorater_service_url: Optional[str] = None, use_autorater: bool = False):
        """
        Initializes the RewardManager.

        Args:
            config: Configuration object for the reward manager.
            tokenizer: The tokenizer instance to use for decoding.
            autorater_service_url: URL of the remote AutoRater FastAPI service.
            use_autorater: Whether to use the remote AutoRater service.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.autorater_service_url = autorater_service_url
        self.use_autorater = use_autorater

        # No longer initializing local AutoRaterReward fn here, it will be called via HTTP if self.use_autorater is True
        self.autorater_reward_fn = None

        self.enable_format_reward = self.config.get("enable_format_reward", True)
        self.format_reward_weight = self.config.get("format_reward_weight", 0.5) # Default to 0.5 for now, can be adjusted

    def compute_rewards(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
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

        # --- Call remote AutoRater service if enabled ---
        if self.use_autorater and self.autorater_service_url:
            print(f"Calling remote AutoRater service at {self.autorater_service_url} from RewardManager")

            # Prepare payload for remote service with extracted solutions
            decoded_questions = [self.tokenizer.decode(p_ids, skip_special_tokens=True) for p_ids in data.batch["prompts"]]
            decoded_pred_answers = [self.tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in data.batch["responses"]]
            # Extract ground truth information from data.non_tensor_batch["reward_model"]
            ground_truth_infos = data.non_tensor_batch.get("reward_model", [{} for _ in range(batch_size)])
            decoded_ground_truth_answers = []
            for gt in ground_truth_infos:
                if isinstance(gt, dict) and "ground_truth" in gt:
                    decoded_ground_truth_answers.append(str(gt["ground_truth"]))
                else:
                    decoded_ground_truth_answers.append(str(gt))

            # Extract solutions
            extraction_method = self.config.autorater_config.get("extraction_method", "flexible") if "autorater_config" in self.config else "flexible"
            answer_formats = self.config.autorater_config.get("answer_formats", ["boxed", "hash"]) if "autorater_config" in self.config else ["boxed", "hash"]
            processed_pred_answers = []
            processed_gt_answers = []
            parse_fail_flags = []
            for pred_ans, gt_ans in zip(decoded_pred_answers, decoded_ground_truth_answers):
                extr_pred_raw = extract_solution(pred_ans, method=extraction_method, answer_formats=answer_formats)
                extr_gt_raw = extract_solution(gt_ans, method=extraction_method, answer_formats=answer_formats)
                if extr_pred_raw is None or extr_gt_raw is None:
                    # Mark parse failure
                    parse_fail_flags.append(True)
                    # Use original strings when sending to remote (will override later)
                    extr_pred = pred_ans
                    extr_gt = gt_ans
                else:
                    parse_fail_flags.append(False)
                    extr_pred = extr_pred_raw
                    extr_gt = extr_gt_raw
                processed_pred_answers.append(extr_pred)
                processed_gt_answers.append(extr_gt)

            # Re-tokenize processed answers
            pred_answers_token_ids = [self.tokenizer.encode(ans, add_special_tokens=False) for ans in processed_pred_answers]
            gt_answers_token_ids = [self.tokenizer.encode(ans, add_special_tokens=False) for ans in processed_gt_answers]

            # Rebuild reward_model_info with ground_truth string
            new_reward_model_info = [{"ground_truth": gt} for gt in processed_gt_answers]
            
            payload = {
                "prompts": data.batch["prompts"].cpu().tolist(),
                "responses": pred_answers_token_ids,
                "attention_mask": data.batch["attention_mask"].cpu().tolist(),
                "position_ids": data.batch["position_ids"].cpu().tolist(),
                "reward_model_info": new_reward_model_info
            }

            response = requests.post(self.autorater_service_url, json=payload, timeout=600)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            autorater_response_data = response.json()

            autorater_scores = autorater_response_data["autorater_scores"]
            autorater_decisions = autorater_response_data["autorater_decisions"]
            autorater_explanations = autorater_response_data.get("autorater_explanations", ["N/A"] * batch_size)
            autorater_raw_responses = autorater_response_data.get("autorater_raw_responses", ["N/A"] * batch_size)

            # Override results for parse failures
            for idx, fail in enumerate(parse_fail_flags):
                if fail:
                    autorater_scores[idx] = -2.0
                    autorater_decisions[idx] = -1
                    autorater_explanations[idx] = "Failed to parse response"
                    autorater_raw_responses[idx] = processed_pred_answers[idx]

            # Shape AutoRater scores based on decisions
            shaped_autorater_scores = []
            for dec, raw_score in zip(autorater_decisions, autorater_scores):
                if dec == 1:
                    shaped_autorater_scores.append(2.0)
                elif dec == 0:
                    shaped_autorater_scores.append(-1.5)
                else:
                    shaped_autorater_scores.append(raw_score)  # keep existing (-2 or 0.5 etc.)

            autorater_scores = shaped_autorater_scores  # replace with shaped values
        else:
            logger.info("Remote AutoRater service not enabled or URL not provided in RewardManager.")

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