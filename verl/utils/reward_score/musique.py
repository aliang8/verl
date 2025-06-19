# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
import string
from typing import Union, List


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute token-level F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(ground_truth_tokens) == 0:
        return 1.0 if len(prediction_tokens) == 0 else 0.0
    
    if len(prediction_tokens) == 0:
        return 0.0
    
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match_score(prediction, ground_truth):
    """Compute exact match score between prediction and ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def extract_solution(response, method="strict"):
    """
    Extract the answer from the model's response.
    
    Args:
        response: The model's response text
        method: If "strict", prioritize #### format and require specific patterns.
                If "flexible", use broader patterns as fallback.
        
    Returns:
        Extracted answer string or None if not found
    """
    if not response or not isinstance(response, str):
        return None
    
    response = response.strip()
    
    if method == "strict":
        # Prioritize GSM8K style #### format (as per instruction following requirement)
        gsm8k_pattern = r"####\s*(.+?)(?:\n|$)"
        match = re.search(gsm8k_pattern, response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Clean up common endings
            answer = re.sub(r'\.$', '', answer)
            return answer
        
        # If no #### format found, return None for strict mode
        # This enforces the instruction following format requirement
        return None
        
    else:  # flexible mode
        # First try GSM8K style #### format
        gsm8k_pattern = r"####\s*(.+?)(?:\n|$)"
        match = re.search(gsm8k_pattern, response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'\.$', '', answer)
            return answer
        
        # Fallback to other patterns for flexible extraction
        patterns = [
            # Look for answers after various markers
            r"(?:final answer|answer|conclusion|result|solution):\s*(.+?)(?:\n|$)",
            r"the answer is\s*(.+?)(?:\n|\.|\n|$)",
            r"therefore,?\s*(.+?)(?:\n|\.|\n|$)",
            r"so,?\s*(.+?)(?:\n|\.|\n|$)",
            r"thus,?\s*(.+?)(?:\n|\.|\n|$)",
            r"hence,?\s*(.+?)(?:\n|\.|\n|$)",
        ]
        
        # Try patterns in order
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean up common endings
                answer = re.sub(r'\.$', '', answer)
                if answer and len(answer) <= 200:  # Reasonable length check
                    return answer
        
        # Try the last line pattern as final fallback
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if last_line and len(last_line) <= 100 and not last_line.startswith('Note:'):
                # Clean up common endings
                last_line = re.sub(r'\.$', '', last_line)
                return last_line
    
    return None


def compute_score(response: str, ground_truth: Union[str, List[str]], method: str = "strict", format_score: float = 0.0, score: float = 1.0):
    """
    Compute the score for MuSiQue multi-hop question answering with GSM8K-style format enforcement.
    
    Args:
        response: The model's response
        ground_truth: The correct answer (string or list of possible answers)
        method: "strict" (requires #### format) or "flexible" (allows fallback patterns)
        format_score: Score given for wrong format/extraction failure (default: 0.0)
        score: Score given for correct answer (default: 1.0)
        
    Returns:
        Score between 0.0 and score value
    """
    # Extract the predicted answer from the response
    predicted_answer = extract_solution(response, method=method)
    
    # If strict mode and no #### format found, return format_score (like GSM8K)
    if method == "strict" and predicted_answer is None:
        return format_score
    
    # If flexible mode and still no answer could be extracted, return format_score
    if predicted_answer is None:
        return format_score
    
    # Handle multiple possible ground truth answers
    if isinstance(ground_truth, list):
        ground_truth_answers = ground_truth
    else:
        ground_truth_answers = [ground_truth]
    
    # Compute the best score against all possible ground truth answers
    best_score = 0.0
    
    for gt_answer in ground_truth_answers:
        if not isinstance(gt_answer, str):
            continue
            
        # Check for exact match first
        if exact_match_score(predicted_answer, gt_answer):
            return score  # Perfect match gets full score
        
        # Compute F1 score for partial credit
        f1 = f1_score(predicted_answer, gt_answer)
        best_score = max(best_score, f1)
    
    # Scale the F1 score by the maximum possible score
    return best_score * score if best_score > 0 else format_score 