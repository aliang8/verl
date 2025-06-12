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


def extract_solution(response, strict=True):
    """
    Extract the answer from the model's response.
    
    Args:
        response: The model's response text
        strict: If True, use strict patterns; if False, use flexible patterns
        
    Returns:
        Extracted answer string or None if not found
    """
    if not response or not isinstance(response, str):
        return None
    
    response = response.strip()
    
    if strict:
        # Look for answers after specific markers
        patterns = [
            r"(?:final answer|answer|conclusion):\s*(.+?)(?:\n|$)",
            r"####\s*(.+?)(?:\n|$)",  # GSM8K style
            r"the answer is\s*(.+?)(?:\n|\.|\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean up common endings
                answer = re.sub(r'\.$', '', answer)
                return answer
    else:
        # More flexible extraction patterns
        patterns = [
            # Look for answers after various markers
            r"(?:final answer|answer|conclusion|result|solution):\s*(.+?)(?:\n|$)",
            r"####\s*(.+?)(?:\n|$)",
            r"the answer is\s*(.+?)(?:\n|\.|\n|$)",
            r"therefore,?\s*(.+?)(?:\n|\.|\n|$)",
            r"so,?\s*(.+?)(?:\n|\.|\n|$)",
            r"thus,?\s*(.+?)(?:\n|\.|\n|$)",
            r"hence,?\s*(.+?)(?:\n|\.|\n|$)",
            
            # Look for standalone answers (last line that's not too long)
            r"^(.{1,100})$",  # Last line under 100 chars
        ]
        
        # Try patterns in order
        for pattern in patterns[:-1]:  # Skip the last line pattern for now
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean up common endings
                answer = re.sub(r'\.$', '', answer)
                if answer and len(answer) <= 200:  # Reasonable length check
                    return answer
        
        # Try the last line pattern
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if last_line and len(last_line) <= 100 and not last_line.startswith('Note:'):
                # Clean up common endings
                last_line = re.sub(r'\.$', '', last_line)
                return last_line
    
    return None


def compute_score(response: str, ground_truth: Union[str, List[str]], format_score: float = 0.1):
    """
    Compute the score for MuSiQue multi-hop question answering.
    
    Args:
        response: The model's response
        ground_truth: The correct answer (string or list of possible answers)
        format_score: Score given for wrong format/extraction failure (default: 0.1)
        
    Returns:
        Score between 0.0 and 1.0
    """
    # Extract the predicted answer from the response
    predicted_answer = extract_solution(response, strict=True)
    
    # If strict extraction fails, try flexible extraction
    if predicted_answer is None:
        predicted_answer = extract_solution(response, strict=False)
    
    # If no answer could be extracted, return format_score
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
            
        # Compute both exact match and F1 score
        em_score = exact_match_score(predicted_answer, gt_answer)
        f1 = f1_score(predicted_answer, gt_answer)
        
        # Use F1 score as the primary metric (common in QA evaluation)
        score = f1
        
        # If we get a perfect exact match, give it a slight boost
        if em_score == 1.0:
            score = 1.0
        
        best_score = max(best_score, score)
    
    return best_score 