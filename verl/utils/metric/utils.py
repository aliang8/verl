# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics utils.
"""

from typing import Any, Dict, List, Optional
import re

import numpy as np


def compute_ttft_ratio(responses: List[str]) -> Dict[str, float]:
    """
    Compute Time to First Token (TTFT) ratio - tokens to first <answer> over total response length.
    
    TTFT measures how quickly the model gets to the <answer> tag within its response.
    Lower values indicate the model gets to the answer faster.
    
    Args:
        responses: List of response strings to analyze
    
    Returns:
        Dictionary containing TTFT mean, max, and min (values between 0 and 1)
    """
    if not responses:
        return {
            "ttft_ratio_mean": 0.0,
            "ttft_ratio_max": 0.0,
            "ttft_ratio_min": 0.0,
        }
    
    ttft_ratios = []
    
    for response in responses:
        if not response.strip():
            continue
            
        # Tokenize by splitting on whitespace (simple approximation)
        tokens = response.strip().split()
        total_length = len(tokens)
        
        if total_length == 0:
            continue
        
        # Look for <answer> tag
        answer_tag_pos = response.find('<answer>')
        
        if answer_tag_pos == -1:
            # <answer> not found, set TTFT to 1.0 (worst case)
            ttft_ratio = 1.0
        else:
            # Find the first token after <answer>
            text_before_answer = response[:answer_tag_pos]
            tokens_to_answer = len(text_before_answer.split())
            
            # Normalize: tokens to answer / total tokens
            ttft_ratio = tokens_to_answer / total_length if total_length > 0 else 1.0
            
            # Ensure it's between 0 and 1
            ttft_ratio = max(0.0, min(ttft_ratio, 1.0))
        
        ttft_ratios.append(ttft_ratio)
    
    if not ttft_ratios:
        return {
            "ttft_ratio_mean": 0.0,
            "ttft_ratio_max": 0.0,
            "ttft_ratio_min": 0.0,
        }
    
    ratios = np.array(ttft_ratios)
    return {
        "ttft_ratio_mean": float(np.mean(ratios)),
        "ttft_ratio_max": float(np.max(ratios)),
        "ttft_ratio_min": float(np.min(ratios)),
    }


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics
