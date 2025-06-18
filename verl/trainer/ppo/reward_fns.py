import re
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig, StringExtractionConfig
from math_verify.metric import math_metric

class ConditionalRewardTracker:
    def __init__(self, epsilon=0.05):
        self.previous_batch_accuracy = 0.0
        self.current_batch_accuracy = 0.0
        self.epsilon = epsilon
        self.batch_count = 0
    
    def update_batch_accuracy(self, accuracy):
        self.previous_batch_accuracy = self.current_batch_accuracy
        self.current_batch_accuracy = accuracy
        self.batch_count += 1
    
    def should_apply_intermediate_rewards(self):
        if self.batch_count <= 1:
            return True
        return self.current_batch_accuracy > (self.previous_batch_accuracy - self.epsilon)

reward_tracker = ConditionalRewardTracker()

def extract_thinking_and_answer(text):
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'
    
    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    thinking = think_match.group(1).strip() if think_match else ""
    final_answer = answer_match.group(1).strip() if answer_match else ""
    
    return thinking, final_answer

def extract_intermediate_answers(thinking_text):
    step_patterns = [
        r'Step \d+:?(.*?)(?=Step \d+|$)',
        r'\d+\.(.*?)(?=\d+\.|$)',
        r'First,?(.*?)(?=Second|Next|Then|Finally|$)',
        r'Second,?(.*?)(?=Third|Next|Then|Finally|$)',
        r'Then,?(.*?)(?=Next|Finally|$)',
        r'Finally,?(.*?)$'
    ]
    
    answers = []
    for pattern in step_patterns:
        matches = re.findall(pattern, thinking_text, re.DOTALL | re.IGNORECASE)
        if matches:
            answers.extend([answer.strip() for answer in matches if answer.strip()])
            break
    
    return answers if answers else [thinking_text.strip()]

def format_check_reward(generated_text):
    has_think = '<think>' in generated_text and '</think>' in generated_text
    has_answer = '<answer>' in generated_text and '</answer>' in generated_text
    return 1.0 if (has_think and has_answer) else 0.0

def final_answer_reward(generated_text, ground_truth, lambda_a=1.0):
    """
    Final answer reward function as per specification:
    rfinal(x, y) = λa · {
        2.0 if y^(N)_answer = g_N
        -1.5 if y^(N)_answer ≠ g_N  
        -2.0 if answer is not parseable
    }
    """
    try:
        thinking, final_answer = extract_thinking_and_answer(generated_text)
        
        if not final_answer or final_answer.strip() == "":
            return lambda_a * (-2.0)
        
        # Check if final answer matches ground truth (exact match)
        if final_answer.lower().strip() == ground_truth.lower().strip():
            return lambda_a * 2.0
        else:
            return lambda_a * (-1.5)
            
    except Exception:
        return lambda_a * (-2.0)

def check_answer_correctness(answer_text, ground_truth_answer):
    answer_lower = answer_text.lower().strip()
    gt_lower = ground_truth_answer.lower().strip()
    return gt_lower in answer_lower

def conditional_intermediate_reward_all_or_none(generated_text, ground_truth, intermediate_truths, base_reward=1.0):
    thinking, final_answer = extract_thinking_and_answer(generated_text)
    
    intermediate_answers = extract_intermediate_answers(thinking)
    if not intermediate_answers or not intermediate_truths:
        return 0.0
    
    all_correct = True
    for k in range(len(intermediate_truths)):
        found_correct = False
        for answer in intermediate_answers:
            if check_answer_correctness(answer, intermediate_truths[k]):
                found_correct = True
                break
        if not found_correct:
            all_correct = False
            break
    
    return base_reward if all_correct else 0.0

def conditional_intermediate_reward_partial_credit(generated_text, ground_truth, intermediate_truths, base_reward=1.0):
    thinking, final_answer = extract_thinking_and_answer(generated_text)
    
    intermediate_answers = extract_intermediate_answers(thinking)
    if not intermediate_answers or not intermediate_truths:
        return 0.0
    
    N = len(intermediate_truths)
    reward_sum = 0.0
    
    for k in range(N):
        for answer in intermediate_answers:
            if check_answer_correctness(answer, intermediate_truths[k]):
                reward_sum += base_reward / N
                break
    
    return reward_sum

def conditional_intermediate_reward_time_discounted(generated_text, ground_truth, intermediate_truths, base_reward=1.0):
    thinking, final_answer = extract_thinking_and_answer(generated_text)
    
    intermediate_answers = extract_intermediate_answers(thinking)
    if not intermediate_answers or not intermediate_truths:
        return 0.0
    
    correct_step = {}
    
    for step_idx, answer in enumerate(intermediate_answers, 1):
        for gt_idx, gt_answer in enumerate(intermediate_truths):
            if gt_idx not in correct_step and check_answer_correctness(answer, gt_answer):
                correct_step[gt_idx] = step_idx
    
    if len(correct_step) == len(intermediate_truths):
        return base_reward
    else:
        if not correct_step:
            return 0.0
        sum_weights = sum(1.0 / step for step in correct_step.values())
        return (sum_weights / len(intermediate_truths)) * base_reward

def conditional_reward_function(completions, reward_type="partial_credit", base_reward=1.0, lambda_a=1.0, **kwargs):
    generated_texts = [completion[0]["content"] for completion in completions]
    ground_truths = kwargs.get('solution', [])
    intermediate_truths_list = kwargs.get('intermediate_truths', [[] for _ in generated_texts])
    
    batch_size = len(generated_texts)
    rewards = []
    
    # Calculate batch accuracy for progression tracking (based on correct final answers)
    batch_final_accuracy = sum(
        1 for gen, gt in zip(generated_texts, ground_truths)
        if final_answer_reward(gen, gt, lambda_a) == lambda_a * 2.0  # Only count correct answers
    ) / batch_size
    
    reward_tracker.update_batch_accuracy(batch_final_accuracy)
    
    for generated_text, ground_truth, intermediate_truths in zip(generated_texts, ground_truths, intermediate_truths_list):
        r_format = format_check_reward(generated_text)
        r_final = final_answer_reward(generated_text, ground_truth, lambda_a)
        
        # Check the three conditions from Algorithm 1
        is_final_correct = r_final == lambda_a * 2.0  # Only positive (correct) final answers
        is_format_valid = r_format > 0
        is_progressing = reward_tracker.should_apply_intermediate_rewards()
        
        # Only compute intermediate rewards if all conditions are met
        if is_final_correct and is_format_valid and is_progressing:
            if reward_type == "all_or_none":
                r_intermediate = conditional_intermediate_reward_all_or_none(
                    generated_text, ground_truth, intermediate_truths, base_reward
                )
            elif reward_type == "time_discounted":
                r_intermediate = conditional_intermediate_reward_time_discounted(
                    generated_text, ground_truth, intermediate_truths, base_reward
                )
            else:
                r_intermediate = conditional_intermediate_reward_partial_credit(
                    generated_text, ground_truth, intermediate_truths, base_reward
                )
        else:
            r_intermediate = 0.0
        
        total_reward = r_format + r_final + r_intermediate
        rewards.append(total_reward)
    
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]
    
def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    # Create the verification function using math_metric (more robust than direct parse/verify)
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
        precision=6
    )
    
    for content, solution in zip(completion_contents, solutions):
        # First extract the answer from <answer></answer> tags if present
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if answer_match:
            extracted_answer = answer_match.group(1).strip()
        else:
            extracted_answer = content.strip()
        
        try:
            grade, _ = verify_func([solution], [extracted_answer])
            rewards.append(float(grade))
        except Exception as e:
            rewards.append(0.0)

    return rewards

def parse_kk_assignments(text):
    """
    Parse Knights and Knaves assignments from text.
    Expected format: "David is a knight, Isabella is a knight, Evelyn is a knave, etc."
    
    Returns:
        dict: {person_name: 'knight'/'knave'}
    """
    assignments = {}
    
    # Extract answer from <answer> tags if present
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        text = answer_match.group(1).strip()
    
    # Clean up the text
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Split by common delimiters
    # Handle both "X and Y" and "X, Y, and Z" patterns
    parts = re.split(r',\s*and\s+|,\s+|\s+and\s+', text)
    
    for part in parts:
        part = part.strip().rstrip('.')
        
        # Look for pattern: "Name is a knight/knave"
        match = re.search(r'(\w+)\s+is\s+a\s+(knight|knave)', part, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            role = match.group(2).strip().lower()
            assignments[name.lower()] = role
    
    return assignments

def kk_reward_function(completions, exact_match=True, **kwargs):
    """
    Knights and Knaves reward function.
    
    Args:
        completions: List of completion dictionaries
        exact_match: If True, returns 1.0 only if all assignments are correct.
                    If False, returns ratio of correct assignments.
        **kwargs: Must contain 'solution' key with ground truth
    
    Returns:
        List of reward scores (0.0 to 1.0)
    """
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, solution in zip(completion_contents, solutions):
        try:
            # Parse the predicted assignments
            pred_assignments = parse_kk_assignments(content)
            
            # Parse the ground truth assignments
            gold_assignments = parse_kk_assignments(solution)
            
            if not gold_assignments:
                # If we can't parse ground truth, give 0 reward
                rewards.append(0.0)
                continue
            
            if not pred_assignments:
                # If we can't parse prediction, give 0 reward
                rewards.append(0.0)
                continue
            
            # Count correct assignments
            correct_count = 0
            total_count = len(gold_assignments)
            
            for name, expected_role in gold_assignments.items():
                if name in pred_assignments and pred_assignments[name] == expected_role:
                    correct_count += 1
            
            if exact_match:
                # Exact match: only give reward if all assignments are correct
                reward = 1.0 if correct_count == total_count else 0.0
            else:
                # Partial match: give ratio of correct assignments
                reward = correct_count / total_count if total_count > 0 else 0.0
            
            rewards.append(reward)
            
        except Exception as e:
            # If parsing fails, give 0 reward
            rewards.append(0.0)
    
    return rewards

def kk_exact_reward(completions, **kwargs):
    """Knights and Knaves exact match reward function."""
    return kk_reward_function(completions, exact_match=True, **kwargs)

def kk_partial_reward(completions, **kwargs):
    """Knights and Knaves partial match reward function."""
    return kk_reward_function(completions, exact_match=False, **kwargs)