#!/usr/bin/env python3
"""
Shared utilities for AutoRater workers in VERL framework.
"""

import re
from typing import Union, List, Optional

HELPFULNESS_RATER_TEMPLATE = """
## Primary Goal
You are an expert evaluator judging the "helpfulness" of an AI assistant's intermediate responses. The assistant is designed to think and respond in interleaved steps (<think><answer><think><answer>). Your task is to determine if a given intermediate answer provides tangible value to the user, or if the assistant should have continued thinking before providing that response.
## Core Principle
An intermediate response is HELPFUL if it provides a self-contained, usable piece of information that the user can understand and act on without waiting for a future revision. It should feel like a deliberate, incremental delivery of the final answer.
An intermediate response is NOT HELPFUL if it's a "work-in-progress" artifact, a trivial fragment, or a low-quality draft that the AI intends to replace or substantially revise later. It should not feel like the user is watching the AI "type and delete."
## Inputs
### User Query:
{question}
### Previous Turns:
{context}
### AI Intermediate Response:
{predicted_answer}
## Evaluation Criteria
Carefully evaluate the AI Intermediate Response based on the criteria below.
### A response is HELPFUL (Decision: TRUE) if it:
Delivers a Complete Sub-Part: Provides a finished, polished part of the final answer.
Example: For a request of "10 suggestions," responding with "Here are the first 3 fully-formed suggestions: [1], [2], [3]." is HELPFUL.
Asks a Necessary Clarifying Question: Poses a question to the user that is essential for refining the final answer.
Example: "To give you better backpacking ideas, could you tell me your budget and experience level?" is HELPFUL.
States a Concrete, Informative Action: Announces a specific action (like a tool use or search query) that informs the user about the process and manages expectations.
Example: "I'm now searching for pet-friendly trails within a 2-hour drive of NYC that are accessible by train." is HELPFUL.
Summarizes Findings and Presents a Choice: Synthesizes information gathered so far and offers the user a decision point.
Example: "I found 5 options, but none are open in the winter. Should I look for summer options instead, or give you the 5 I found?" is HELPFUL.
### A response is NOT HELPFUL (Decision: FALSE) if it:
Is a Trivial Fragment: Consists of a single word, an incomplete sentence, or conversational filler that adds no new information.
Example: "Okay, I can help with that." is NOT HELPFUL.
Example: "Here are some" (and then stops) is NOT HELPFUL.
Is "Drafting in Public": Presents incomplete fragments from multiple different parts of the final answer, instead of finishing one part completely. The response feels like a low-quality skeleton or a rough draft that the user cannot yet use.
Example: For a request to write a three-paragraph email, responding with one rough sentence for each of the three paragraphs is NOT HELPFUL. The model should have finished the first paragraph completely.
Is a Vague or Redundant Statement of Intent: Repeats the user's request or states a vague plan without providing new information or a concrete action.
Example: "I will now give you 10 suggestions for a backpacking trip." is NOT HELPFUL.
Example: "Thinking about it..." is NOT HELPFUL.
Merely Corrects a Previous Minor Error: Fixes a typo or grammar mistake from its own previous turn without adding substantive new information. Corrections should be integrated into the next helpful chunk.
## Final Check
Ask yourself: "Could the user gain real value from this response right now, or is it just noise they have to wait through?" If it delivers real, incremental progress, it's TRUE. Otherwise, it's FALSE.
## Output Format
Respond with exactly one line in this format:
Decision: TRUE
or
Decision: FALSE
Please proceed with the evaluation.
Decision: """

# AutoRater template for evaluation
AUTO_RATER_TEMPLATE = """===Task===
I need your help in evaluating an answer provided by an LLM against a ground truth
answer. Your task is to determine if the ground truth answer is present in the LLM's response.
Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers – look for equivalent information or correct answers. Do
not focus on exact wording unless the exact wording is crucial to the meaning.
3. If there are multiple questions/answers, evaluate each one and return a scalar between 0 and 1 representing the fraction of correct answers. For example, if there are two questions, output 0, 0.5, or 1. If only one question, output 0 or 1.
4. Your final decision should be based on whether the meaning and the vital facts of the "Ground
Truth Answer" are present in the "Predicted Answer" for each question.
===Input Data===
- Predicted Answer: {predicted_answer}
- Ground Truth Answer: {ground_truth_answer}
===Output Format===
Provide your final evaluation as a single scalar between 0 and 1, representing the fraction of correct answers. For example, output 1 if all are correct, 0.5 if half are correct, or 0 if none are correct.

Please proceed with the evaluation.
Score: """


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


# # AutoRater template for evaluating helpfulness of a response
# # explain what intermediate response is  
# # meta eval set for the autorater prompt, maybe as in-context examples too
# HELPFULNESS_RATER_TEMPLATE = """===Task===
# You are given a user question, previous context and a new intermediate response from an AI assistant.
# In the context of interleaved reasoning, an 'intermediate response' refers to a partial output or a visible step presented by an AI assistant during its reasoning process. 
# A good partial answer should directly address some part of a user query. 
# For example, in a trip planning prompt an intermediate answer should concretely mention hotel options, or flight options rather than provide half-baked information.
# Importantly, a good partial response goes beyond reasoning and gives actual relevant and actionable output to the user.
# Your job is to judge whether the intermediate response is a good intermediate response.

# ===User Question===
# {question}

# ===Previous Context===
# {context}

# ===AI Response===
# {predicted_answer}

# ===Evaluation Instructions===
# 1. Consider if the response provides information that is useful, relevant and actionable to the user's question.
# 2. Determine if the information presented is novel and not merely a rephrasing or repetition of what's already known or implied by the user's question.
# 3. Consider the context of previous responses - if this response builds upon or adds to previous helpful information.

# ===Output Format===
# Respond with exactly one line in this format:
# "Decision: TRUE"  (if the response is a good intermediate response)
# or
# "Decision: FALSE" (if the response is not a good intermediate response).

# Please proceed with the evaluation.
# Decision: """

HELPFULNESS_RATER_TEMPLATE_RATING = """
===Task===
You are given a user question, previous context and a new intermediate response from an AI assistant.
In the context of interleaved reasoning, an 'intermediate response' refers to a partial output or a visible step presented by an AI assistant during its reasoning process.
A good partial answer should directly address some part of a user query.
For example, in a trip planning prompt an intermediate answer should concretely mention hotel options, or flight options rather than provide half-baked information.
Importantly, a good partial response goes beyond reasoning and gives actual relevant and actionable output to the user.
Your job is to judge how good the intermediate response is.

===User Question===
{question}

===Previous Context===
{context}

===AI Response===
{predicted_answer}

===Evaluation Instructions===
Rate the quality of the AI Response as an intermediate step on a scale of 1 to 5, where:
*   **1 - Very Poor:** The response is irrelevant, incorrect, or provides no useful/actionable information. It does not contribute to the user's progress.
*   **2 - Poor:** The response provides very little useful or actionable information, or it's mostly redundant with previous context. It barely moves the user forward.
*   **3 - Fair:** The response offers some useful or actionable information, but it might be incomplete, slightly vague, or not as concrete as it could be. It contributes moderately to progress.
*   **4 - Good:** The response is clearly useful, relevant, and provides concrete, actionable information that directly addresses part of the user's query. It builds well on previous context if applicable and moves the user significantly forward.
*   **5 - Excellent:** The response is highly relevant, provides crucial, concrete, and actionable information, and is novel. It represents a significant and effective step towards fulfilling the user's request, demonstrating clear progress.

Consider the following points when assigning your score:
*   Does the response provide information that is useful, relevant, and actionable to the user's question?
*   Is the information presented novel and not merely a rephrasing or repetition of what's already known or implied by the user's question?
*   Does the response build upon or add to previous helpful information in the context?

===Output Format===
Respond with exactly one line in this format:
"Decision: [SCORE]" (where [SCORE] is an integer from 1 to 5)

Please proceed with the evaluation.
Decision: """

def extract_solution(
    solution_str: str,
    template_type: str = "default",
) -> Union[str, List[str]]:
    """
    Extract answer(s) from a solution string based on template type.

    Args:
        solution_str: The input string containing answer tags or think tags
        template_type: The template type (e.g., 'interleave' or other)
        method: Extraction method (kept for backward compatibility, currently ignored)
    Returns:
        If template_type contains 'interleave': List of all <answer>...</answer> contents (stripped), or None if no tags found
        Otherwise: String after the last </think> tag (stripped), or the whole string if no </think> tag is found
    """
    if template_type and "interleave" in template_type.lower():
        # Find all matches between <answer> and </answer> tags
        matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.IGNORECASE | re.DOTALL)
        if not matches:
            return ""
        extracted_items = [match.strip() for match in matches if match.strip()]
        return extracted_items
    else:
        # Find the last </think> tag and return everything after it
        think_match = list(re.finditer(r"</think>", solution_str, re.IGNORECASE))
        if think_match:
            last = think_match[-1]
            after = solution_str[last.end():].strip()
            return after if after else ""
        
        # Also just try to get things between <answer> and </answer>
        answer_match = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.IGNORECASE))
        if answer_match:
            return answer_match[0].group(1).strip()

    return ""


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


def parse_autorater_response_scalar(response: str) -> tuple[str, str]:
    """
    Parse the model's response to extract explanation and decision/score.
    
    Args:
        response: The raw response from the autorater model
    
    Returns:
        Tuple of (explanation, score/decision as string)
    """
    # Try to find a scalar score (float between 0 and 1) on any line
    score_pattern = r'^(0(\.\d+)?|1(\.0+)?)$'
    for line in response.strip().splitlines():
        line = line.strip()
        if re.match(score_pattern, line):
            return response.strip(), float(line)
    return response.strip(), 0.0

def parse_autorater_response_boolean(response: str) -> tuple[str, str]:
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


def format_helpfulness_prompt(question: str, predicted_answer: str, context: Optional[List[str]] = None, template: Union[str, None] = None) -> str:
    """
    Format the helpfulness rater prompt with the given inputs.
    Args:
        question: The original user question
        predicted_answer: The AI's response to evaluate
        context: Optional list of previous answers for context
        template: Custom template to use (defaults to HELPFULNESS_RATER_TEMPLATE)
    Returns:
        Formatted prompt string
    """
    if template == "helpfulness":
        template = HELPFULNESS_RATER_TEMPLATE
    elif template == "helpfulness_rating":
        template = HELPFULNESS_RATER_TEMPLATE_RATING

    # Format context as numbered list if provided
    if context and isinstance(context, list):
        context_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(context)])
    else:
        context_str = "None"
    
    return template.format(
        question=question,
        predicted_answer=predicted_answer,
        context=context_str
    ) 


# Plan evaluation template for selecting best plan from N options
PLAN_EVALUATION_TEMPLATE = """===Task===
You are an expert plan evaluator. Given a prompt and N different plans, your task is to evaluate which plan is the best.

===Evaluation Criteria===
- **Completeness**: Does the plan address all aspects of the prompt?
- **Feasibility**: Is the plan realistic and implementable?
- **Efficiency**: Is the plan optimal in terms of time, resources, or complexity?
- **Clarity**: Is the plan clear and well-structured?
- **Robustness**: Does the plan account for potential issues or edge cases?

===Input Data===
- Prompt: {prompt}
- Plans to evaluate:
{plans_text}

===Output Format===
After evaluating all plans, output ONLY a single number (1 to {num_plans}) indicating which plan is the best.
For example, if plan 3 is the best, output: 3

===Important===
- Do not provide explanations or reasoning in your final output
- Do not include any text, just the number
- If multiple plans are equally good, choose the one with the lowest number

Please proceed with the evaluation.
Score: """


def format_plan_evaluation_prompt(prompt: str, plans: List[str]) -> str:
    """
    Format a prompt for plan evaluation with a prompt and list of plans.
    
    Args:
        prompt: The original prompt/question
        plans: List of plans to evaluate (each plan should be a string)
    
    Returns:
        str: Formatted prompt string for plan evaluation
    
    Example:
        >>> prompt = "How should I implement a web scraper?"
        >>> plans = [
        ...     "Use BeautifulSoup with requests library",
        ...     "Use Selenium for dynamic content",
        ...     "Use Scrapy framework"
        ... ]
        >>> formatted_prompt = format_plan_evaluation_prompt(prompt, plans)
    """
    if not plans:
        raise ValueError("Plans list cannot be empty")
    
    # Format the plans as numbered list
    plans_text = ""
    for i, plan in enumerate(plans, 1):
        plans_text += f"**Plan {i}:**\n{plan}\n\n"
    
    return PLAN_EVALUATION_TEMPLATE.format(
        prompt=prompt,
        plans_text=plans_text.strip(),
        num_plans=len(plans)
    )


def parse_plan_evaluation_response(response: str, num_plans: int) -> int:
    """
    Parse the plan evaluation response to extract the selected plan number.
    
    Args:
        response: The raw response from the plan evaluation model
        num_plans: The total number of plans that were evaluated
    
    Returns:
        int: The selected plan number (1-based index), or 1 if parsing fails
    
    Example:
        >>> response = "3"
        >>> selected_plan = parse_plan_evaluation_response(response, 5)
        >>> print(selected_plan)  # Output: 3
    """
    # Look for numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response.strip())
    
    if numbers:
        # Get the first number found
        selected_plan = int(numbers[0])
        
        # Validate that the number is within the valid range
        if 1 <= selected_plan <= num_plans:
            return selected_plan
    
    # Fallback: return 1 if no valid number found
    return 1

