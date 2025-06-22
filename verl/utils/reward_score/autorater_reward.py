#!/usr/bin/env python3
"""
AutoRater-based reward function for VERL framework.
"""

import torch
from typing import Dict, Any, Tuple, Union, List
import numpy as np
from collections import defaultdict

from verl import DataProto
from verl.trainer.ppo.reward_fns import format_check_reward

class AutoRaterReward:
    """
    AutoRater-based reward function that uses a distributed LLM worker for evaluation.
    
    This reward function evaluates responses by comparing them against ground truth answers
    using a small LLM model that makes TRUE/FALSE decisions about correctness.
    """
    
    def __init__(self, autorater_worker_group=None, config=None, tokenizer=None):
        """
        Initialize the AutoRater reward function.
        
        Args:
            autorater_worker_group: The distributed worker group for AutoRater evaluation
            config: Configuration for the reward function
            tokenizer: Tokenizer for decoding text (optional, will try to get from data if not provided)
        """
        super().__init__()
        self.autorater_wg = autorater_worker_group
        self.config = config or {}
        self.tokenizer = tokenizer

        if self.autorater_wg is not None:
            self.tokenizer = self.autorater_wg.get_tokenizer()[0]
        
        # Reward weights
        self.autorater_weight = self.config.get("autorater_weight", 1.0)
        
    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Compute rewards using the AutoRater worker.
        
        Args:
            data: DataProto containing batch data
            return_dict: Whether to return additional information
            
        Returns:
            Reward tensor or dictionary with reward and extra info
        """
        if self.autorater_wg is None:
            raise ValueError("AutoRater worker group not initialized")
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # # Extract all questions, responses, and ground truths for batch processing
        # questions = []
        # predicted_answers = []
        # ground_truth_answers = []
        
        # for i in range(len(data)):
        #     data_item = data[i]  # DataProtoItem
            
        #     # Extract prompt and response for this item
        #     prompt_ids = data_item.batch["prompts"]
        #     prompt_length = prompt_ids.shape[-1]
            
        #     valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        #     valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
        #     response_ids = data_item.batch["responses"]
        #     valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        #     valid_response_ids = response_ids[:valid_response_length]
          
        #     question = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        #     predicted_answer = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        #     # Get ground truth
        #     ground_truth_answer = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            
        #     questions.append(question)
        #     predicted_answers.append(predicted_answer)
        #     ground_truth_answers.append(ground_truth_answer)
        
        # Create a single batch DataProto for all items
        # This will be distributed across workers
        batch_data = DataProto.from_dict(
            tensors={
                "prompts": data.batch["prompts"],
                "responses": data.batch["responses"],
                "attention_mask": data.batch["attention_mask"]
            },
            non_tensors={
                "reward_model": np.array([item.non_tensor_batch["reward_model"] for item in [data[i] for i in range(len(data))]], dtype=object)
            }
        )
        
        # Split the batch into chunks divisible by the number of workers (8)
        # Calculate the number of workers from the autorater worker group
        num_workers = 8  # Default to 8, but we could get this from the worker group
        batch_size = len(batch_data)
        
        # Calculate the chunk size that's divisible by num_workers
        # We'll pad the batch to make it divisible
        remainder = batch_size % num_workers
        if remainder > 0:
            pad_size = num_workers - remainder
            # Pad by repeating the last item
            padded_batch_data = DataProto.from_dict(
                tensors={
                    "prompts": torch.cat([batch_data.batch["prompts"], batch_data.batch["prompts"][-pad_size:]], dim=0),
                    "responses": torch.cat([batch_data.batch["responses"], batch_data.batch["responses"][-pad_size:]], dim=0),
                    "attention_mask": torch.cat([batch_data.batch["attention_mask"], batch_data.batch["attention_mask"][-pad_size:]], dim=0)
                },
                non_tensors={
                    "reward_model": np.concatenate([batch_data.non_tensor_batch["reward_model"], batch_data.non_tensor_batch["reward_model"][-pad_size:]])
                }
            )
            batch_data = padded_batch_data
            print(f"Padded batch from {batch_size} to {len(batch_data)} items to make it divisible by {num_workers}")
        
        # Evaluate using the AutoRater worker (will be distributed across workers)
        autorater_output = self.autorater_wg.compute_autorater_score(batch_data)
                
        # Extract scores and decisions
        autorater_scores = autorater_output.batch["autorater_scores"]  # Shape: (batch_size,)
        autorater_decisions = autorater_output.batch["autorater_decisions"]  # Shape: (batch_size,)

        # Remove padding from results if we padded
        if remainder > 0:
            autorater_scores = autorater_scores[:batch_size]
            autorater_decisions = autorater_decisions[:batch_size]
            if "autorater_explanations" in autorater_output.non_tensor_batch:
                autorater_output.non_tensor_batch["autorater_explanations"] = autorater_output.non_tensor_batch["autorater_explanations"][:batch_size]
            if "autorater_raw_responses" in autorater_output.non_tensor_batch:
                autorater_output.non_tensor_batch["autorater_raw_responses"] = autorater_output.non_tensor_batch["autorater_raw_responses"][:batch_size]
        
        # Process each item to apply format penalty and store rewards
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            # Extract response for format checking
            response_ids = data_item.batch["responses"]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Decode for format checking
            predicted_answer = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            autorater_score = autorater_scores[i].item()
            autorater_decision = autorater_decisions[i].item()
            
            # Apply format penalty using the format_check_reward function
            format_score = format_check_reward(predicted_answer)
            
            # Combine autorater and format scores
            final_score = self.autorater_weight * autorater_score
            
            # Store the reward at the last token of the response
            reward_tensor[i, valid_response_length - 1] = final_score
            
            # Collect extra information
            reward_extra_info["autorater_scores"].append(autorater_score)
            reward_extra_info["autorater_decisions"].append(autorater_decision)
            reward_extra_info["format_scores"].append(format_score)
            reward_extra_info["final_scores"].append(final_score)
            
            # Add explanations if available
            if "autorater_explanations" in autorater_output.non_tensor_batch:
                reward_extra_info["autorater_explanations"].append(
                    autorater_output.non_tensor_batch["autorater_explanations"][i]
                )
            if "autorater_raw_responses" in autorater_output.non_tensor_batch:
                reward_extra_info["autorater_raw_responses"].append(
                    autorater_output.non_tensor_batch["autorater_raw_responses"][i]
                )
        
        if return_dict:
            # Add summary statistics - make sure all values are lists for extending
            extra_info = {
                # Individual sample data (lists)
                "autorater_decisions": reward_extra_info["autorater_decisions"],
                "autorater_explanations": reward_extra_info.get("autorater_explanations", []),
                "autorater_raw_responses": reward_extra_info.get("autorater_raw_responses", []),
                "autorater_scores": reward_extra_info["autorater_scores"],
                "format_scores": reward_extra_info["format_scores"],
                "final_scores": reward_extra_info["final_scores"],
            }

            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": extra_info
            }
        
        return reward_tensor


def create_autorater_reward_fn(autorater_worker_group, config: Dict[str, Any] = None):
    """
    Factory function to create an AutoRater reward function.
    
    Args:
        autorater_worker_group: The distributed AutoRater worker group
        config: Configuration dictionary for the reward function
        
    Returns:
        Configured AutoRater reward function
    """
    return AutoRaterReward(autorater_worker_group=autorater_worker_group, config=config or {})