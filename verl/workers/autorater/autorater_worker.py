#!/usr/bin/env python3
"""
AutoRater Worker for distributed LLM-based evaluation in VERL framework.
"""

import re
import torch
from typing import Dict, List, Tuple, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from verl import DataProto
from verl.single_controller.base import Worker
from verl.utils.model import compute_position_id_with_mask


class AutoRaterWorker(Worker):
    """
    AutoRater worker for distributed LLM-based evaluation.
    
    This worker loads a small LLM model and uses it to evaluate whether predicted answers
    contain the ground truth information, providing a more nuanced evaluation than simple
    rule-based metrics.
    """

    def __init__(self, config, **kwargs):
        """
        Initialize the AutoRater worker.
        
        Args:
            config: Configuration object containing autorater settings
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # AutoRater template for evaluation
        self.auto_rater_template = """===Task===
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
- Question: {question}
- Predicted Answer: {predicted_answer}
- Ground Truth Answer: {ground_truth_answer}
===Output Format===
Provide your final evaluation in the following format:
"Decision:" ("TRUE" or "FALSE")

Please proceed with the evaluation.
Decision: """

    def init_model(self):
        """Initialize the AutoRater model and tokenizer."""
        model_name = self.config.model.get("path", "Qwen/Qwen2.5-7B-Instruct")
        
        print(f"Loading AutoRater model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            **self.config.model.get("tokenizer_kwargs", {})
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            **self.config.model.get("model_kwargs", {})
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.get("max_new_tokens", 10),
            do_sample=self.config.get("do_sample", False),
            temperature=self.config.get("temperature", 0.0),
            top_p=self.config.get("top_p", 1.0),
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        print(f"✅ AutoRater model loaded successfully!")

    def format_prompt(self, question: str, predicted_answer: str, ground_truth_answer: str) -> str:
        """Format the auto-rater prompt with the given inputs."""
        return self.auto_rater_template.format(
            question=question,
            predicted_answer=predicted_answer,
            ground_truth_answer=ground_truth_answer
        )

    def parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the model's response to extract explanation and decision.
        
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

    def evaluate_batch(self, 
                      questions: List[str], 
                      predicted_answers: List[str], 
                      ground_truth_answers: List[str]) -> Dict[str, List[Any]]:
        """
        Evaluate a batch of responses using the AutoRater LLM.
        
        Args:
            questions: List of questions
            predicted_answers: List of predicted answers
            ground_truth_answers: List of ground truth answers
            
        Returns:
            Dictionary containing evaluation results
        """
        batch_size = len(questions)
        assert len(predicted_answers) == batch_size
        assert len(ground_truth_answers) == batch_size
        
        decisions = []
        explanations = []
        raw_responses = []
        
        # Process each item in the batch
        for i in range(batch_size):
            try:
                # Format prompt
                prompt = self.format_prompt(
                    questions[i], 
                    predicted_answers[i], 
                    ground_truth_answers[i]
                )
                
                # Generate response
                response = self.generator(prompt)[0]['generated_text']
                raw_responses.append(response)
                
                # Parse response
                explanation, decision = self.parse_response(response)
                explanations.append(explanation)
                decisions.append(decision)
                
            except Exception as e:
                print(f"Error evaluating item {i}: {e}")
                decisions.append("ERROR")
                explanations.append(f"Error: {str(e)}")
                raw_responses.append("")
        
        return {
            "decisions": decisions,
            "explanations": explanations,
            "raw_responses": raw_responses
        }

    def compute_autorater_scores(self, data: DataProto) -> DataProto:
        """
        Compute auto-rater scores for the given data batch.
        
        Args:
            data: DataProto containing the batch data
            
        Returns:
            DataProto with auto-rater evaluation results
        """
        # Extract necessary information from the batch
        responses = data.batch["responses"]
        batch_size = responses.shape[0]
        
        # Decode questions and responses
        if "prompts" in data.batch:
            questions = [self.tokenizer.decode(prompt, skip_special_tokens=True) 
                        for prompt in data.batch["prompts"]]
        else:
            # Fallback: extract from input_ids if prompts not available
            input_ids = data.batch.get("input_ids", data.batch.get("prompt_ids", None))
            if input_ids is not None:
                questions = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                           for ids in input_ids]
            else:
                questions = [""] * batch_size

        predicted_answers = [self.tokenizer.decode(response, skip_special_tokens=True) 
                           for response in responses]
        
        # Get ground truth from reward_model metadata
        ground_truth_answers = []
        if "reward_model" in data.non_tensor_batch:
            for rm_info in data.non_tensor_batch["reward_model"]:
                if isinstance(rm_info, dict) and "ground_truth" in rm_info:
                    ground_truth_answers.append(str(rm_info["ground_truth"]))
                else:
                    ground_truth_answers.append("Unknown")
        else:
            ground_truth_answers = ["Unknown"] * batch_size

        # Evaluate using the AutoRater
        evaluation_results = self.evaluate_batch(
            questions=questions,
            predicted_answers=predicted_answers,
            ground_truth_answers=ground_truth_answers
        )
        
        # Convert decisions to scores (TRUE=1.0, FALSE=0.0, ERROR/UNKNOWN=0.0)
        scores = []
        for decision in evaluation_results["decisions"]:
            if decision == "TRUE":
                scores.append(1.0)
            else:  # FALSE, ERROR, UNKNOWN
                scores.append(0.0)
        
        # Create output DataProto
        output_data = DataProto(
            batch={
                "autorater_scores": torch.tensor(scores, dtype=torch.float32, device=self.device),
                "autorater_decisions": torch.tensor([1 if d == "TRUE" else 0 for d in evaluation_results["decisions"]], 
                                                  dtype=torch.long, device=self.device)
            },
            non_tensor_batch={
                "autorater_explanations": evaluation_results["explanations"],
                "autorater_raw_responses": evaluation_results["raw_responses"]
            }
        )
        
        return output_data 