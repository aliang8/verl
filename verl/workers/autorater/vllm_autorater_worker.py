#!/usr/bin/env python3
"""
vLLM-based AutoRater Worker for memory-efficient distributed LLM evaluation in VERL framework.
"""

import re
import torch
import numpy as np
import warnings
import os
import logging
from typing import Dict, List, Tuple, Any, Union
from contextlib import contextmanager

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import GPUMemoryLogger
from .autorater_utils import AUTO_RATER_TEMPLATE, extract_solution, format_autorater_prompt, parse_autorater_response

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMAutoRaterWorker(Worker):
    """
    Memory-efficient vLLM-based AutoRater worker for larger models.
    
    This worker uses vLLM for efficient inference with tensor parallelism support,
    making it suitable for larger models like 7B Qwen while maintaining memory efficiency.
    """

    def __init__(self, config):
        """
        Initialize the vLLM AutoRater worker.
        
        Args:
            config: Configuration object containing autorater settings
        """
        super().__init__()
        import torch.distributed
        
        # Import device utilities
        from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
        
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl", 
                rank=rank, 
                world_size=world_size
            )
        
        self.config = config
        
        # Setup tensor parallelism for vLLM
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        world_size = torch.distributed.get_world_size()
        assert self.tensor_parallel_size <= world_size, \
            f"tensor parallel size {self.tensor_parallel_size} should be <= world size {world_size}"
        
        # Calculate DP and TP ranks
        self.dp_size = world_size // self.tensor_parallel_size
        self.dp_rank = torch.distributed.get_rank() // self.tensor_parallel_size
        self.tp_rank = torch.distributed.get_rank() % self.tensor_parallel_size
        
        # vLLM inference engine will be initialized in init_model
        self.inference_engine = None
        self.tokenizer = None
        
        # Solution extraction settings
        self.extraction_method = self.config.get("extraction_method", "flexible")
        self.answer_formats = self.config.get("answer_formats", ["boxed", "hash"])
        
        # Use shared template from utils
        self.auto_rater_template = AUTO_RATER_TEMPLATE

    @GPUMemoryLogger(role="vllm autorater init", logger=logger)
    def _build_vllm_engine(self):
        """Build the vLLM inference engine with memory efficiency."""
        from verl.third_party.vllm import LLM
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer
        from vllm import SamplingParams
        from omegaconf import OmegaConf
        
        # Download model to local path
        use_shm = self.config.model.get("use_shm", False)
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
        
        # Load tokenizer
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # vLLM engine configuration with conservative memory settings
        vllm_config = self.config.get("vllm_config", {})
        max_model_len = vllm_config.get("max_model_len", self.config.get("max_model_len", 1024))
        gpu_memory_utilization = vllm_config.get("gpu_memory_utilization", self.config.get("gpu_memory_utilization", 0.1))
        max_num_batched_tokens = vllm_config.get("max_num_batched_tokens", self.config.get("max_num_batched_tokens", 2048))
        
        # Engine kwargs with memory optimizations
        engine_kwargs = vllm_config.get("engine_kwargs", self.config.get("engine_kwargs", {}))
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        
        # Create vLLM engine with memory-optimized settings
        self.inference_engine = LLM(
            model=local_path,
            tokenizer=local_path,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=vllm_config.get("dtype", self.config.get("dtype", "bfloat16")),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=vllm_config.get("enforce_eager", self.config.get("enforce_eager", True)),
            disable_log_stats=vllm_config.get("disable_log_stats", self.config.get("disable_log_stats", True)),
            trust_remote_code=trust_remote_code,
            **engine_kwargs
        )
        
        # Sampling parameters for evaluation (deterministic)
        self.sampling_params = SamplingParams(
            n=1,
            max_tokens=self.config.get("max_new_tokens", 10),
            temperature=0.0,  # Deterministic for consistent evaluation
            top_p=1.0,
            stop=None,
            include_stop_str_in_output=False,
        )
        
        # vLLM handles memory internally, no manual offloading needed

    @contextmanager
    def _vllm_inference_context(self):
        """Context manager for vLLM inference with proper memory management."""
        try:
            yield
        finally:
            # Clear GPU cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_tokenizer(self):
        return self.tokenizer

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @GPUMemoryLogger(role="vllm autorater init_model", logger=logger)
    def init_model(self):
        """Initialize the vLLM AutoRater model."""
        from verl.utils.import_utils import import_external_libs
        import_external_libs(self.config.model.get("external_lib", None))
        
        self._build_vllm_engine()
        
        print(f"✅ vLLM AutoRater model loaded successfully on TP rank {self.tp_rank}/{self.tensor_parallel_size}")

    def format_prompt(self, question: str, predicted_answer: str, ground_truth_answer: str) -> str:
        """Format the auto-rater prompt with the given inputs."""
        return format_autorater_prompt(question, predicted_answer, ground_truth_answer, self.auto_rater_template)

    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model's response to extract explanation and decision."""
        return parse_autorater_response(response)

    @GPUMemoryLogger(role="vllm autorater evaluation", logger=logger)
    def evaluate_batch(self, 
                      questions: List[str], 
                      predicted_answers: List[str], 
                      ground_truth_answers: List[str],
                      micro_batch_size: int = 4) -> Dict[str, List[Any]]:
        """
        Evaluate a batch of responses using vLLM for memory-efficient inference.
        
        Args:
            questions: List of questions
            predicted_answers: List of predicted answers
            ground_truth_answers: List of ground truth answers
            micro_batch_size: Size of micro-batches for memory efficiency
            
        Returns:
            Dictionary containing evaluation results
        """
        batch_size = len(questions)
        assert len(predicted_answers) == batch_size
        assert len(ground_truth_answers) == batch_size
        
        decisions = []
        explanations = []
        raw_responses = []
        
        # Process in micro-batches to reduce memory usage
        for start_idx in range(0, batch_size, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, batch_size)
            
            # Generate evaluation prompts for this micro-batch
            evaluation_prompts = []
            for i in range(start_idx, end_idx):
                prompt = self.format_prompt(
                    questions[i], 
                    predicted_answers[i], 
                    ground_truth_answers[i]
                )
                evaluation_prompts.append(prompt)
            
            # Use vLLM for micro-batch inference with memory management
            with self._vllm_inference_context():
                # Generate responses using vLLM
                outputs = self.inference_engine.generate(
                    prompts=evaluation_prompts,
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )
                
                # Process results for this micro-batch
                for output in outputs:
                    # Get the generated text (excluding the prompt)
                    response = output.outputs[0].text
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
    @GPUMemoryLogger(role="vllm autorater compute", logger=logger)
    def compute_autorater_score(self, data: DataProto) -> DataProto:
        """
        Compute auto-rater scores for a batch of data items using vLLM.
        
        Args:
            data: DataProto containing batch data
            
        Returns:
            DataProto with auto-rater evaluation results
        """
        from verl.utils.device import get_torch_device
        
        # Only process data on TP rank 0 to avoid duplication
        if self.tp_rank != 0:
            # Return dummy results for non-primary TP ranks
            batch_size = data.batch["responses"].shape[0]
            dummy_scores = torch.zeros(batch_size, dtype=torch.float32)
            dummy_decisions = torch.zeros(batch_size, dtype=torch.long)
            dummy_explanations = np.array([""] * batch_size, dtype=object)
            dummy_responses = np.array([""] * batch_size, dtype=object)
            
            output_data = DataProto.from_dict(
                tensors={
                    "autorater_scores": dummy_scores,
                    "autorater_decisions": dummy_decisions
                },
                non_tensors={
                    "autorater_explanations": dummy_explanations,
                    "autorater_raw_responses": dummy_responses,
                }
            )
            return output_data.to("cpu")
        
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

        # Get micro batch size from config (similar to how dp_actor does it)
        micro_batch_size = self.config.get("micro_batch_size_per_gpu", 4)
        
        # Evaluate using the vLLM AutoRater with micro-batching
        evaluation_results = self.evaluate_batch(
            questions=questions,
            predicted_answers=extracted_predictions,
            ground_truth_answers=ground_truth_answers,
            micro_batch_size=micro_batch_size
        )
        
        # Convert decisions to scores (TRUE=1.0, FALSE=0.0, ERROR/UNKNOWN=-1.0)
        scores = []
        for decision in evaluation_results["decisions"]:
            if decision == "TRUE":
                scores.append(1.0)
            elif decision == "FALSE":
                scores.append(0.0)
            else:  # ERROR, UNKNOWN
                scores.append(-1.0)

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
                "extracted_predictions": np.array(extracted_predictions, dtype=object),
                "original_predictions": np.array(predicted_answers, dtype=object)
            }
        )
        
        output_data = output_data.to("cpu")
        return output_data


def create_vllm_autorater_reward_fn(autorater_worker_group, config: Dict[str, Any] = None):
    """
    Factory function to create a vLLM-based AutoRater reward function.
    
    Args:
        autorater_worker_group: The distributed vLLM AutoRater worker group
        config: Configuration dictionary for the reward function
        
    Returns:
        Configured vLLM AutoRater reward function
    """
    # Import here to avoid circular imports
    from verl.utils.reward_score.autorater_reward import AutoRaterReward
    return AutoRaterReward(autorater_worker_group=autorater_worker_group, config=config or {}) 