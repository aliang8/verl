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
    """Extract the conclusion answers from the Knights and Knaves solution string."""
    assert method in ["strict", "flexible"]

    if method == "strict":
        # Look for the exact format with CONCLUSION: followed by numbered items
        conclusion_pattern = r"CONCLUSION:\s*((?:\([^)]+\)[^\n]*\n?)+)"
        match = re.search(conclusion_pattern, solution_str, re.IGNORECASE | re.MULTILINE)
        if match:
            conclusion_text = match.group(1).strip()
            # Extract individual conclusion lines
            lines = re.findall(r"\((\d+)\)\s*(.+)", conclusion_text)
            if lines:
                # Sort by the number and extract just the conclusion part
                sorted_lines = sorted(lines, key=lambda x: int(x[0]))
                conclusions = [line[1].strip() for line in sorted_lines]
                return conclusions
        return None
    
    elif method == "flexible":
        # More flexible extraction patterns
        patterns = [
            r"CONCLUSION:\s*((?:\([^)]+\)[^\n]*\n?)+)",  # Standard CONCLUSION format
            r"(?:Therefore|Thus|So):\s*((?:\([^)]+\)[^\n]*\n?)+)",  # Alternative markers
            r"Answer:\s*((?:\([^)]+\)[^\n]*\n?)+)",  # Answer format
            r"Solution:\s*((?:\([^)]+\)[^\n]*\n?)+)",  # Solution format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution_str, re.IGNORECASE | re.MULTILINE)
            if match:
                conclusion_text = match.group(1).strip()
                lines = re.findall(r"\((\d+)\)\s*(.+)", conclusion_text)
                if lines:
                    sorted_lines = sorted(lines, key=lambda x: int(x[0]))
                    conclusions = [line[1].strip() for line in sorted_lines]
                    return conclusions
        
        # If no structured format found, look for knight/knave statements anywhere
        knight_knave_pattern = r"(\w+)\s+is\s+a\s+(knight|knave)"
        matches = re.findall(knight_knave_pattern, solution_str, re.IGNORECASE)
        if matches:
            # Extract conclusions in the order they appear
            conclusions = [f"{name.title()} is a {role.lower()}" for name, role in matches]
            return conclusions
            
        return None


def normalize_conclusion(conclusion):
    """Normalize a conclusion string for comparison."""
    # Remove extra whitespace and convert to lowercase
    conclusion = conclusion.strip().lower()
    
    # Standardize the format: "name is a knight/knave"
    # Handle various formats like "name is knight", "name: knight", etc.
    patterns = [
        (r"(\w+)\s+is\s+a\s+(knight|knave)", r"\1 is a \2"),
        (r"(\w+)\s+is\s+(knight|knave)", r"\1 is a \2"),
        (r"(\w+):\s*(knight|knave)", r"\1 is a \2"),
        (r"(\w+)\s*-\s*(knight|knave)", r"\1 is a \2"),
    ]
    
    for pattern, replacement in patterns:
        match = re.search(pattern, conclusion, re.IGNORECASE)
        if match:
            name, role = match.groups()
            return f"{name.lower()} is a {role.lower()}"
    
    return conclusion


def compare_conclusions(predicted_conclusions, ground_truth_conclusions):
    """Compare predicted conclusions with ground truth conclusions."""
    if predicted_conclusions is None or ground_truth_conclusions is None:
        return False
    
    if len(predicted_conclusions) != len(ground_truth_conclusions):
        return False
    
    # Normalize both sets of conclusions
    pred_normalized = [normalize_conclusion(conc) for conc in predicted_conclusions]
    gt_normalized = [normalize_conclusion(conc) for conc in ground_truth_conclusions]
    
    # Check if all conclusions match (order should be maintained)
    return pred_normalized == gt_normalized


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for Knights and Knaves.

    Args:
        solution_str: the solution text containing the conclusions
        ground_truth: the ground truth conclusions (list of strings or string)
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for having some answer format but wrong conclusions
        score: the score for the correct conclusions
    """
    predicted_conclusions = extract_solution(solution_str=solution_str, method=method)
    
    # Handle ground truth format - could be string or list
    if isinstance(ground_truth, str):
        # If ground truth is a string, try to parse it
        gt_conclusions = extract_solution(ground_truth, method="flexible")
        if gt_conclusions is None:
            # Fallback: split by lines and clean up
            gt_lines = [line.strip() for line in ground_truth.split('\n') if line.strip()]
            gt_conclusions = gt_lines
    else:
        gt_conclusions = ground_truth
    
    if predicted_conclusions is None:
        return 0.0
    else:
        if compare_conclusions(predicted_conclusions, gt_conclusions):
            return score
        else:
            return format_score 