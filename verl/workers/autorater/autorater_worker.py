#!/usr/bin/env python3
"""
AutoRater Worker for distributed LLM-based evaluation in VERL framework.
"""

import re
import torch
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.model import compute_position_id_with_mask
from .autorater_utils import AUTO_RATER_TEMPLATE, extract_solution, format_autorater_prompt, parse_autorater_response

class AutoRaterWorker(Worker):
    """
    AutoRater worker for distributed LLM-based evaluation.
    
    This worker loads a small LLM model and uses it to evaluate whether predicted answers
    contain the ground truth information, providing a more nuanced evaluation than simple
    rule-based metrics.
    """

    def __init__(self, config):
        """
        Initialize the AutoRater worker.
        
        Args:
            config: Configuration object containing autorater settings
        """
        super().__init__()
        import torch.distributed
        
        # Import device utilities
        from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
        
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl" if is_cuda_available else "hccl")
        self.config = config
        
        # build device mesh for distributed processing
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        from verl.workers.fsdp_workers import create_device_mesh
        
        # Use simple device mesh for AutoRater (no complex sharding needed)
        fsdp_size = self.config.model.get("fsdp_size", -1)
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        
        # AutoRater uses simple processing - no Ulysses sequence parallel needed
        self.ulysses_sequence_parallel_size = 1
        
        # normalize config
        if self.config.get("micro_batch_size") is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size
        
        # Solution extraction settings
        self.extraction_method = self.config.get("extraction_method", "flexible")  # "strict", "flexible", or "both"
        self.answer_formats = self.config.get("answer_formats", ["boxed", "hash"])  # Support both \boxed{} and ####
        
        # Use shared template from utils
        self.auto_rater_template = AUTO_RATER_TEMPLATE

    def _build_model(self, config):
        """Build the AutoRater model following the same pattern as RewardModelWorker."""
        from transformers import AutoConfig, AutoModelForCausalLM
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer
        from verl.utils.device import get_torch_device
        
        use_shm = config.model.get("use_shm", False)
        # download the checkpoint
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        
        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        
        # Load tokenizer
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Create model - use simple initialization for AutoRater without device_map
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            autorater_module = AutoModelForCausalLM.from_pretrained(
                config.model.path
            )
            
            # Ensure model is in the correct dtype and move to device
            autorater_module.to(torch.bfloat16)
            autorater_module.to(get_torch_device().current_device())
        
        return autorater_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_tokenizer(self):
        return self.tokenizer

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the AutoRater model."""
        # This is used to import external_lib into the huggingface systems
        from verl.utils.import_utils import import_external_libs
        import_external_libs(self.config.model.get("external_lib", None))
        
        self.autorater_module = self._build_model(config=self.config)
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.autorater_module,
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
        return format_autorater_prompt(question, predicted_answer, ground_truth_answer, self.auto_rater_template)

    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model's response to extract explanation and decision."""
        return parse_autorater_response(response)

    def evaluate_batch(self, 
                      questions: List[str], 
                      predicted_answers: List[str], 
                      ground_truth_answers: List[str]) -> Dict[str, List[Any]]:
        """
        Evaluate a batch of responses using the AutoRater LLM in parallel.
        
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
        
        # Generate all evaluation prompts at once
        evaluation_prompts = []
        for i in range(batch_size):
            prompt = self.format_prompt(
                questions[i], 
                predicted_answers[i], 
                ground_truth_answers[i]
            )
            evaluation_prompts.append(prompt)
        
        decisions = []
        explanations = []
        raw_responses = []
        
        # Process the entire batch at once using the pipeline
        # The pipeline will handle batching internally for efficiency
        batch_responses = self.generator(
            evaluation_prompts,
            batch_size=min(batch_size, 8),  # Process in chunks to avoid memory issues
            truncation=True,
            padding=True,
            repetition_penalty=1.1,
            num_return_sequences=1,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=10,
        )
        
        # Process results
        for i, response_data in enumerate(batch_responses):
            response = response_data[0]['generated_text']
            raw_responses.append(response)
            
            # Parse response
            explanation, decision = self.parse_response(response)
            explanations.append(explanation)
            decisions.append(decision)
        
        return {
            "decisions": decisions,
            "explanations": explanations,
            "raw_responses": raw_responses
        }

    def extract_solution(self, solution_str: str, method: str = "flexible") -> Union[str, None]:
        """Extract the answer from a solution string using shared utilities."""
        return extract_solution(solution_str, method, self.answer_formats)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_autorater_score(self, data: DataProto) -> DataProto:
        """
        Compute auto-rater scores for a batch of data items.
        
        Args:
            data: DataProto containing batch data
            
        Returns:
            DataProto with auto-rater evaluation results
        """
        from verl.utils.device import get_torch_device
        
        # Support all hardwares
        data = data.to(get_torch_device().current_device())
        
        # Extract necessary information from the batch
        responses = data.batch["responses"]  # Shape: (batch_size, seq_len)
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
        
        # Extract solutions from predicted answers
        extracted_predictions = []
        for pred_answer in predicted_answers:
            extracted = self.extract_solution(pred_answer, method=self.extraction_method)
            extracted_predictions.append(extracted if extracted is not None else pred_answer)
            # extracted_predictions.append(extracted)
        
        # Get ground truth from reward_model metadata
        ground_truth_answers = []
        if "reward_model" in data.non_tensor_batch:
            for rm_info in data.non_tensor_batch["reward_model"]:
                if isinstance(rm_info, dict) and "ground_truth" in rm_info:
                    # Also try to extract from ground truth if it's in a formatted form
                    gt_raw = str(rm_info["ground_truth"])
                    gt_extracted = self.extract_solution(gt_raw, method=self.extraction_method)
                    ground_truth_answers.append(gt_extracted if gt_extracted is not None else gt_raw)
                else:
                    ground_truth_answers.append("Unknown")
        else:
            ground_truth_answers = ["Unknown"] * batch_size

        # Evaluate using the AutoRater with extracted solutions
        evaluation_results = self.evaluate_batch(
            questions=questions,
            predicted_answers=extracted_predictions,  # Use extracted solutions
            ground_truth_answers=ground_truth_answers
        )
        
        # Convert decisions to scores (TRUE=1.0, FALSE=0.0, ERROR/UNKNOWN=0.0)
        scores = []
        for decision in evaluation_results["decisions"]:
            if decision == "TRUE":
                scores.append(2.0)
            elif decision == "FALSE":
                scores.append(-1.5)
            else:  # ERROR, UNKNOWN
                scores.append(-2.0)
        
        # Convert decisions to numerical values (TRUE=1, FALSE=0, ERROR/UNKNOWN=-1)
        autorater_decisions = []
        for decision in evaluation_results["decisions"]:
            if decision == "TRUE":
                autorater_decisions.append(1)
            elif decision == "FALSE":
                autorater_decisions.append(0)
            else:  # ERROR, UNKNOWN
                autorater_decisions.append(-1)

        # Create output DataProto with batch dimension
        output_data = DataProto.from_dict(
            tensors={
                "autorater_scores": torch.tensor(scores, dtype=torch.float32),
                "autorater_decisions": torch.tensor(autorater_decisions, dtype=torch.long)
            },
            non_tensors={
                "autorater_explanations": np.array(evaluation_results["explanations"], dtype=object),
                "autorater_raw_responses": np.array(evaluation_results["raw_responses"], dtype=object),
                "extracted_predictions": np.array(extracted_predictions, dtype=object),  # Include extracted solutions for debugging
                "original_predictions": np.array(predicted_answers, dtype=object)  # Keep original for reference
            }
        )
        
        output_data = output_data.to("cpu")
        return output_data 