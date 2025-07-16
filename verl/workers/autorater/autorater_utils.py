#!/usr/bin/env python3
"""
Shared utilities for AutoRater workers in VERL framework.
"""

import re
from typing import Union, List, Optional


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


# AutoRater template for evaluating a concise code-solution outline
CODE_OUTLINE_RATER_TEMPLATE = """===Task===
You are given a short outline that supposedly explains how to solve a programming problem.
Your job is to judge whether the outline covers ALL critical steps concisely and correctly.

===Problem Description===
{problem_description}

===Proposed Outline===
{outline_answer}

===Evaluation Instructions===
1. Verify the outline includes the high-level algorithm, key data-structures, complexity analysis, and edge-case handling (if relevant).  
2. Ignore trivial phrasing issues; focus on technical completeness and accuracy.  
3. The outline should remain concise (≈1-2 short paragraphs or a numbered list ≤8 items).  
4. Decide if the outline is a GOOD solution description.  Good == "yes, a competent programmer could implement the solution from this outline with minimal extra research".

===Output Format===
Respond with exactly one line in this format:
"Decision: TRUE"  (if the outline is good)
or
"Decision: FALSE" (if the outline misses important aspects or is wrong).

Please proceed with the evaluation.
Decision:  """


# AutoRater template for evaluating helpfulness of a response
HELPFULNESS_RATER_TEMPLATE = """===Task===
You are given a user question and a response from an AI assistant.
Your job is to judge whether the response is helpful, relevant, and addresses the user's question.

===User Question===
{question}

===AI Response===
{predicted_answer}

===Evaluation Instructions===
1. Consider if the response provides useful, accurate, and relevant information for the user's question.
2. Ignore minor phrasing or style issues; focus on substance and helpfulness.
3. If the response is off-topic, incorrect, or unhelpful, mark as FALSE.
4. If the response is generally helpful and addresses the question, mark as TRUE.

===Output Format===
Respond with exactly one line in this format:
"Decision: TRUE"  (if the response is helpful)
or
"Decision: FALSE" (if the response is not helpful).

Please proceed with the evaluation.
Decision: """


def extract_solution(
    solution_str: str,
    method: str = "any",
    answer_formats: Optional[List[str]] = None,
    extract_all: bool = False,
) -> Union[str, List[str], None]:
    """Extract content inside <answer>...</answer> tags.

    Args:
        solution_str: The input string containing answer tags
        method: Extraction method (kept for backward compatibility, currently ignored)
        answer_formats: Answer formats (kept for backward compatibility, currently ignored)
        extract_all: If True, extracts all <answer> tags and returns as comma-separated list
    
    Returns:
        If extract_all=False: Content of first <answer> tag or None if missing/empty
        If extract_all=True: All <answer> tag contents joined by commas, or None if no tags found
    """

    if extract_all:
        # Find all matches between <answer> and </answer> tags
        matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        
        # Strip whitespace from each match and filter out empty ones
        extracted_items = [match.strip() for match in matches if match.strip()]
        return extracted_items
    else:
        # Original behavior: extract first match only
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


def format_code_outline_prompt(
    problem_description: str,
    outline_answer: str,
    template: Optional[str] = None,
) -> str:
    """Create a prompt for the AutoRater to evaluate a solution outline.

    Args:
        problem_description: The programming problem statement.
        outline_answer: The outline produced by the LLM to be evaluated.
        template: Optional custom template; defaults to CODE_OUTLINE_RATER_TEMPLATE.

    Returns:
        A formatted string to feed into the LLM AutoRater.
    """
    if template is None:
        template = CODE_OUTLINE_RATER_TEMPLATE

    return template.format(
        problem_description=problem_description.strip(),
        outline_answer=outline_answer.strip(),
    ) 


def format_helpfulness_prompt(question: str, predicted_answer: str, template: Union[str, None] = None) -> str:
    """
    Format the helpfulness rater prompt with the given inputs.
    Args:
        question: The original user question
        predicted_answer: The AI's response to evaluate
        template: Custom template to use (defaults to HELPFULNESS_RATER_TEMPLATE)
    Returns:
        Formatted prompt string
    """
    if template is None:
        template = HELPFULNESS_RATER_TEMPLATE
    return template.format(
        question=question,
        predicted_answer=predicted_answer
    ) 