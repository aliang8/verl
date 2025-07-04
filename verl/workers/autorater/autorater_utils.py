#!/usr/bin/env python3
"""
Shared utilities for AutoRater workers in VERL framework.
"""

import re
from typing import Union, List


# AutoRater template for evaluation
AUTO_RATER_TEMPLATE = """===Task===
I need your help in evaluating an answer provided by an LLM against a ground truth
answer. Your task is to determine if the ground truth answer is present in the LLM's response.
Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers – look for equivalent information or correct answers. Do
not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground
Truth Answer" are present in the "Predicted Answer:"
===Input Data===
- Predicted Answer: {predicted_answer}
- Ground Truth Answer: {ground_truth_answer}
===Output Format===
Provide your final evaluation in the following format:
"Decision:" ("TRUE" or "FALSE")

Please proceed with the evaluation.
Decision: """


def extract_solution(solution_str: str, method: str = "any", answer_formats: Union[List[str], None] = None) -> Union[str, None]:
    """Extract content inside <answer>...</answer> tags.

    If the tags are missing or empty, returns None. Additional parameters are
    kept for backward compatibility but currently ignored.
    """

    # Regex to capture text between <answer> and </answer>, non-greedy, case-insensitive, spanning lines.
    match = re.search(r"<answer>(.*?)</answer>", solution_str, re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    extracted = match.group(1).strip()
    return extracted if extracted else None


def format_autorater_prompt(question: str, predicted_answer: str, ground_truth_answer: str, template: Union[str, None] = None) -> str:
    """
    Format the auto-rater prompt with the given inputs.
    
    Args:
        question: The original question
        predicted_answer: The predicted answer to evaluate
        ground_truth_answer: The ground truth answer
        template: Custom template to use (defaults to AUTO_RATER_TEMPLATE)
    
    Returns:
        Formatted prompt string
    """
    if template is None:
        template = AUTO_RATER_TEMPLATE
    
    return template.format(
        question=question,
        predicted_answer=predicted_answer,
        ground_truth_answer=ground_truth_answer
    )


def parse_autorater_response(response: str) -> tuple[str, str]:
    """
    Parse the model's response to extract explanation and decision.
    
    Args:
        response: The raw response from the autorater model
    
    Returns:
        Tuple of (explanation, decision)
    """
    # Multiple parsing patterns to catch TRUE/FALSE decisions
    decision_patterns = [
        r'Decision:\s*["\']?(TRUE|FALSE)["\']?',
        r'\b(TRUE|FALSE)\b',
        r'(true|false)',
        r'answer is\s+(TRUE|FALSE)',
        r'decision is\s+(TRUE|FALSE)',
    ]
    
    explanation = response.strip()
    decision = "UNKNOWN"
    
    # Try to find decision
    for pattern in decision_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            decision = match.group(1).upper()
            break
    
    return explanation, decision 