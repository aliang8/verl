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


def extract_solution(solution_str, method="strict"):
    """Extract the answer letter (A, B, C, D) from the solution string."""
    assert method in ["strict", "flexible"]

    if method == "strict":
        # Look for the exact format: "Answer: X" where X is A, B, C, or D
        solution = re.search(r"Answer:\s*([ABCD])", solution_str, re.IGNORECASE)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(1).upper()
    elif method == "flexible":
        # More flexible extraction - look for any occurrence of A, B, C, D patterns
        # Look for patterns like "Answer: A", "answer is B", "the answer is C", etc.
        patterns = [
            r"Answer:\s*([ABCD])",  # "Answer: A"
            r"answer\s+is\s*([ABCD])",  # "answer is A"
            r"the\s+answer\s+is\s*([ABCD])",  # "the answer is A"
            r"I\s+choose\s*([ABCD])",  # "I choose A"
            r"option\s*([ABCD])",  # "option A"
            r"choice\s*([ABCD])",  # "choice A"
            r"\b([ABCD])\s*is\s+correct",  # "A is correct"
            r"\b([ABCD])\s*is\s+the\s+answer",  # "A is the answer"
        ]
        
        final_answer = None
        for pattern in patterns:
            match = re.search(pattern, solution_str, re.IGNORECASE)
            if match:
                final_answer = match.group(1).upper()
                break
        
        # If no pattern matches, look for the last occurrence of A, B, C, or D
        if final_answer is None:
            matches = re.findall(r"\b([ABCD])\b", solution_str, re.IGNORECASE)
            if matches:
                final_answer = matches[-1].upper()
    
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GPQA.

    Args:
        solution_str: the solution text containing the answer
        ground_truth: the ground truth answer (A, B, C, or D)
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for having some answer format but wrong answer
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    
    if answer is None:
        return 0.0
    else:
        if answer == ground_truth.upper():
            return score
        else:
            return format_score 