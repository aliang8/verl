import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import torch # type: ignore
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer # type: ignore

from verl import DataProto # type: ignore
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.reward_score.autorater_reward import AutoRaterReward
from verl.utils.reward_score.http_autorater_reward import HTTPAutoRaterReward
from verl.trainer.ppo.reward_fns import format_check_reward

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("default_reward_manager")
class RewardManager:
    """
    Manages the computation of various reward scores, including AutoRater and format-based rewards.
    """

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer):
        """
        Initializes the RewardManager.

        Args:
            config: Configuration object for the reward manager.
            tokenizer: The tokenizer instance to use for decoding.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.autorater_reward_fn = None
        self.http_autorater_reward_fn = None

        self._initialize_autorater_fns()

        self.enable_format_reward = self.config.get("enable_format_reward", True)
        self.format_reward_weight = self.config.get("format_reward_weight", 0.5) # Default to 0.5 for now, can be adjusted

    def _initialize_autorater_fns(self):
        """Initializes the AutoRater reward functions based on configuration."""
        # Check if AutoRater config is provided
        if "autorater_config" in self.config:
            autorater_config = self.config.autorater_config
            autorater_type = autorater_config.get("type", "vllm")

            if autorater_type == "vllm":
                # Assuming AutoRaterReward expects a worker group
                # For now, we'll create a dummy worker group or expect it to be passed
                # This part might need adjustment based on how Ray Trainer passes the worker
                # For direct integration, AutoRaterReward might need refactoring or a direct LLM init
                # For this class, we assume the Ray worker handles the LLM
                # We'll need the actual AutoRaterWorker instance to pass here,
                # or modify AutoRaterReward to accept configuration directly.

                # Temporarily, we will set it to None, and it will be initialized by ray_trainer directly if needed
                # Or, if this manager is only for processing *after* the autorater has produced scores,
                # then it doesn't need to initialize the LLM directly.
                logger.warning("VLLM AutoRater integration for RewardManager is a placeholder. "
                               "Ensure AutoRaterReward can be initialized without a worker group or passed one.")
                self.autorater_reward_fn = AutoRaterReward(
                    autorater_worker_group=None,  # Placeholder, will be replaced by direct call in compute_rewards
                    config=OmegaConf.to_container(autorater_config)
                )
            elif autorater_type == "http":
                # Correct parameters for HTTPAutoRaterReward
                http_config = {
                    "http_timeout": autorater_config.get("timeout", 60),
                    "max_retries": autorater_config.get("max_retries", 3)
                }
                self.http_autorater_reward_fn = HTTPAutoRaterReward(
                    autorater_service_url=autorater_config.get("url"),
                    config=http_config,
                    tokenizer=self.tokenizer
                )
            else:
                raise ValueError(f"Unknown autorater type: {autorater_type}")
        else:
            logger.info("No autorater_config found, AutoRater rewards will not be computed by this manager.")

    def compute_rewards(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Computes AutoRater and format rewards for a batch of data.

        Args:
            data: DataProto containing batch data (prompts, responses, reward_model_info).
            return_dict: If True, returns a dictionary with detailed reward info.

        Returns:
            A tensor of combined rewards or a dictionary with detailed reward information.
        """
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # --- Decode inputs for AutoRater and format checking ---
        questions = []
        predicted_answers = []
        ground_truth_answers = []

        for i in range(batch_size):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            reward_info = data_item.non_tensor_batch["reward_model"]

            question = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            predicted_answer = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Extract ground truth from reward_model_info
            if isinstance(reward_info, dict) and "ground_truth" in reward_info:
                ground_truth = str(reward_info["ground_truth"])
            else:
                ground_truth = str(reward_info) # Fallback if not a dict

            questions.append(question)
            predicted_answers.append(predicted_answer)
            ground_truth_answers.append(ground_truth)

        # --- Compute AutoRater scores ---
        autorater_scores = [0.0] * batch_size
        autorater_decisions = [-1] * batch_size # -1 for unknown/error
        autorater_explanations = ["N/A"] * batch_size
        autorater_raw_responses = ["N/A"] * batch_size

        if self.http_autorater_reward_fn:
            # Prepare a request similar to what HTTPAutoRaterReward would do
            # Note: This is a simplification; in a real scenario, HTTPAutoRaterReward's
            # __call__ method handles the DataProto directly.
            # Here we're mimicking the input it expects for its internal call to the service.
            request_data = {
                "prompts": [p.tolist() for p in data.batch["prompts"]],
                "responses": [r.tolist() for r in data.batch["responses"]],
                "attention_mask": [am.tolist() for am in data.batch["attention_mask"]],
                "position_ids": [pid.tolist() for pid in data.batch["position_ids"]],
                "reward_model_info": data.non_tensor_batch["reward_model"]
            }
            try:
                autorater_response = self.http_autorater_reward_fn._call_autorater_service(request_data)
                if autorater_response.get("success", False):
                    autorater_scores = autorater_response["autorater_scores"]
                    autorater_decisions = autorater_response["autorater_decisions"]
                    autorater_explanations = autorater_response.get("autorater_explanations", autorater_explanations)
                    autorater_raw_responses = autorater_response.get("autorater_raw_responses", autorater_raw_responses)
                else:
                    logger.error(f"HTTP AutoRater service error: {autorater_response.get('error_message', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error calling HTTP AutoRater service: {e}")
        elif self.autorater_reward_fn:
            # If using VLLM AutoRaterReward directly (without HTTP service)
            # This path is more complex as AutoRaterReward expects a worker group for distributed calls.
            # For simplicity here, we assume if VLLM is used, it's via a pre-initialized worker,
            # and we'd directly call its compute_autorater_score method.
            # This part will require the Ray worker to pass the actual `compute_autorater_score` method or the worker itself.
            logger.warning("Direct VLLM AutoRaterReward integration is not fully implemented in RewardManager. "
                           "It needs the actual worker's `compute_autorater_score` or an equivalent direct LLM call.")
            # For now, we'll keep autorater_scores at 0.0 if not using HTTP
            pass


        # --- Compute Format Rewards ---
        format_scores = []
        if self.enable_format_reward:
            for pred_answer in predicted_answers:
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
                # Sum autorater score and format score
                # The format_reward_weight scales the format score.
                # If format_reward_weight is 0.5, then it's 0.5 * autorater_score + 0.5 * format_score
                # If format_reward_weight is 1.0, then it's only format_score (assuming autorater_score is added at other places)
                # If format_reward_weight is 0.0, then it's only autorater_score
                
                # Here we are adding the format score directly to the autorater score
                # The problem statement: "make sure the final reward sums the format reward + autorater_scores"
                current_final_score = autorater_scores[i] + (self.format_reward_weight * format_scores[i])
                
                reward_tensor[i, valid_response_length - 1] = current_final_score
                final_scores.append(current_final_score)
            else:
                # If response length is 0, no reward can be placed on a token.
                # This case might need specific handling depending on desired behavior.
                # For now, we'll add 0 and log a warning.
                logger.warning(f"Response length is 0 for sample {i}, no reward applied to token.")
                final_scores.append(0.0) # Or -1.0 if it's considered an error/penalty
            
            # Collect extra information
            reward_extra_info["autorater_scores"].append(autorater_scores[i])
            reward_extra_info["autorater_decisions"].append(autorater_decisions[i])
            reward_extra_info["autorater_explanations"].append(autorater_explanations[i])
            reward_extra_info["autorater_raw_responses"].append(autorater_raw_responses[i])
            reward_extra_info["format_scores"].append(format_scores[i])
            reward_extra_info["final_scores"].append(final_scores[i])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        
        return reward_tensor

    def _get_tokenizer(self):
        return self.tokenizer 