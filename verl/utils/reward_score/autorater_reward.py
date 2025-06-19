#!/usr/bin/env python3
"""
AutoRater-based reward function for VERL framework.
"""

import torch
from typing import Dict, Any, Tuple, Union, List
import numpy as np

from verl import DataProto
from verl.utils.reward_score.base import BaseRewardModel


class AutoRaterReward(BaseRewardModel):
    """
    AutoRater-based reward function that uses a distributed LLM worker for evaluation.
    
    This reward function evaluates responses by comparing them against ground truth answers
    using a small LLM model that makes TRUE/FALSE decisions about correctness.
    """
    
    def __init__(self, autorater_worker_group=None, config=None):
        """
        Initialize the AutoRater reward function.
        
        Args:
            autorater_worker_group: The distributed worker group for AutoRater evaluation
            config: Configuration for the reward function
        """
        super().__init__()
        self.autorater_wg = autorater_worker_group
        self.config = config or {}
        
        # Reward weights
        self.autorater_weight = self.config.get("autorater_weight", 1.0)
        self.format_penalty = self.config.get("format_penalty", 0.0)
        
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
        
        # Use the AutoRater worker to compute scores
        autorater_output = self.autorater_wg.compute_autorater_scores(data)
        
        # Extract scores and decisions
        autorater_scores = autorater_output.batch["autorater_scores"]  # Shape: (batch_size,)
        autorater_decisions = autorater_output.batch["autorater_decisions"]  # Shape: (batch_size,)
        
        # Convert to token-level rewards (replicate across response length)
        responses = data.batch["responses"]
        response_length = responses.shape[1]
        batch_size = responses.shape[0]
        
        # Expand autorater scores to token level
        token_level_autorater_scores = autorater_scores.unsqueeze(1).expand(batch_size, response_length)
        
        # Apply format penalty if no valid format detected
        # (This could be extended to check for specific format requirements)
        format_scores = torch.ones_like(autorater_scores)
        if self.format_penalty > 0:
            # Check if responses follow expected format (could be customized)
            responses_text = [data.tokenizer.decode(resp, skip_special_tokens=True) 
                            for resp in responses] if hasattr(data, 'tokenizer') else None
            if responses_text:
                for i, resp_text in enumerate(responses_text):
                    # Simple format check - could be made more sophisticated
                    if not any(marker in resp_text for marker in ["####", "Answer:", "Final answer:"]):
                        format_scores[i] = 1.0 - self.format_penalty
        
        format_scores_expanded = format_scores.unsqueeze(1).expand(batch_size, response_length)
        
        # Combine autorater and format scores
        final_scores = self.autorater_weight * token_level_autorater_scores * format_scores_expanded
        
        if return_dict:
            # Collect extra information
            extra_info = {
                "autorater_accuracy": autorater_decisions.float().mean().item(),
                "autorater_true_rate": (autorater_decisions == 1).float().mean().item(),
                "autorater_false_rate": (autorater_decisions == 0).float().mean().item(),
                "format_score": format_scores.mean().item(),
                "autorater_decisions": autorater_decisions.cpu().tolist(),
                "autorater_explanations": autorater_output.non_tensor_batch.get("autorater_explanations", []),
                "autorater_raw_responses": autorater_output.non_tensor_batch.get("autorater_raw_responses", []),
            }
            
            # Add per-sample breakdown if requested
            if self.config.get("detailed_breakdown", False):
                extra_info.update({
                    "autorater_scores_per_sample": autorater_scores.cpu().tolist(),
                    "format_scores_per_sample": format_scores.cpu().tolist(),
                })
            
            return {
                "reward_tensor": final_scores,
                "reward_extra_info": extra_info
            }
        
        return final_scores


def create_autorater_reward_fn(autorater_worker_group, config: Dict[str, Any] = None):
    """
    Factory function to create an AutoRater reward function.
    
    Args:
        autorater_worker_group: The distributed AutoRater worker group
        config: Configuration dictionary for the reward function
        
    Returns:
        Configured AutoRater reward function
    """
    return AutoRaterReward(autorater_worker_group=autorater_worker_group, config=config)