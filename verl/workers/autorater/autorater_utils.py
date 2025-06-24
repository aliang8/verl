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


def extract_solution(solution_str: str, method: str = "flexible", answer_formats: Union[List[str], None] = None) -> Union[str, None]:
    """
    Extract the answer from a solution string.
    Supports #### format, \boxed{} format, and Knights & Knaves CONCLUSION format.
    
    Args:
        solution_str: The solution text
        method: "strict", "flexible", or "both"
        answer_formats: List of supported formats (["boxed", "hash"] by default)
        
    Returns:
        Extracted answer as string, or None if not found
    """
    if answer_formats is None:
        answer_formats = ["boxed", "hash"]
    
    if method == "strict":
        # Try Knights and Knaves CONCLUSION format first
        conclusion_match = re.search(r"CONCLUSION\s*\n((?:\(\d+\).*\n?)+)", solution_str, re.MULTILINE)
        if conclusion_match is not None:
            conclusion_text = conclusion_match.group(1)
            # Extract the numbered points and combine them
            points = re.findall(r"\(\d+\)\s*([^\n]+)", conclusion_text)
            if points:
                return ", ".join(points).strip()
        
        # Try #### format (GSM8K style)
        if "hash" in answer_formats:
            solution = re.search(r"#### (\\-?[0-9\\.\\,]+)", solution_str)
            if solution is not None:
                final_answer = solution.group(0)
                final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
                return final_answer
        
        # Try \boxed{} format
        if "boxed" in answer_formats:
            boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution_str)
            if boxed_match is not None:
                final_answer = boxed_match.group(1).replace(",", "").replace("$", "")
                return final_answer
        
        return None
        
    elif method == "flexible":
        # Try Knights and Knaves CONCLUSION format first
        conclusion_match = re.search(r"CONCLUSION\s*\n((?:\(\d+\).*\n?)+)", solution_str, re.MULTILINE)
        if conclusion_match is not None:
            conclusion_text = conclusion_match.group(1)
            # Extract the numbered points and combine them
            points = re.findall(r"\(\d+\)\s*([^\n]+)", conclusion_text)
            if points:
                return ", ".join(points).strip()
        
        # Try structured formats
        if "hash" in answer_formats:
            solution = re.search(r"#### (\\-?[0-9\\.\\,]+)", solution_str)
            if solution is not None:
                final_answer = solution.group(0)
                final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
                return final_answer
        
        if "boxed" in answer_formats:
            boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution_str)
            if boxed_match is not None:
                final_answer = boxed_match.group(1).replace(",", "").replace("$", "")
                return final_answer
        
        # Fallback: find any numbers in the text
        answer = re.findall(r"(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            return None
        else:
            invalid_str = ["", "."]
            # Find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
            return final_answer.replace(",", "").replace("$", "") if final_answer else None
            
    elif method == "both":
        # Try strict first, then flexible
        strict_result = extract_solution(solution_str, "strict", answer_formats)
        if strict_result is not None:
            return strict_result
        return extract_solution(solution_str, "flexible", answer_formats)
    
    return None


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