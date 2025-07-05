#!/usr/bin/env python3
"""
FastAPI AutoRater Service with Ray Distributed Processing

This service runs on a separate VM and provides AutoRater functionality
via HTTP API endpoints using Ray for distributed GPU processing.
"""

import asyncio
import logging
import os
import time
import traceback
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Libraries from standard python or installed via pip (assumed to be installed)
import ray  # type: ignore
import numpy as np  # type: ignore
from fastapi import FastAPI, HTTPException, BackgroundTasks  # type: ignore
from pydantic import BaseModel  # type: ignore
import uvicorn  # type: ignore

from omegaconf import DictConfig, OmegaConf  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from verl.workers.autorater.autorater_utils import format_autorater_prompt, parse_autorater_response

# Sandbox for secure code execution
from llm_sandbox import SandboxSession  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoRater Service", version="1.0.0")

# Initialize app state
app.state.autorater_actors = []
app.state.autorater_config = None
app.state.num_gpus = 0

# Global SandboxSession reused across requests to minimise startup overhead
app.state.sandbox_session = None  # Initialized lazily on first use


@ray.remote(num_gpus=1)
class AutoRaterActor:
    """Ray actor for distributed AutoRater processing on individual GPUs"""
    
    def __init__(self, config: Dict[str, Any], gpu_id: int):
        """Initialize AutoRater actor on specific GPU"""
        import os
        
        # Set GPU for this actor
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Disable distributed training since Ray handles distribution
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        
        # Use a unique port for each actor to avoid conflicts
        unique_port = 29500 + gpu_id
        os.environ["MASTER_PORT"] = str(unique_port)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        
        self.gpu_id = gpu_id
        self.config = OmegaConf.create(config)
        self.inference_engine = None
        self.tokenizer = None
        
    def initialize(self):
        """Initialize the vLLM engine directly on this GPU"""
        from verl.utils.fs import copy_to_local

        
        # Download model to local path
        use_shm = self.config.model.get("use_shm", False)
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
        
        # Load tokenizer
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create vLLM engine with single GPU configuration
        engine_config = {
            "model": local_path,
            "tokenizer": local_path,
            "tensor_parallel_size": 1,  # Single GPU per actor
            "dtype": self.config.get("dtype", "bfloat16"),
            "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0.95),
            "max_model_len": self.config.get("max_model_len", 4096),
            "max_num_batched_tokens": self.config.get("max_num_batched_tokens", 8192),
            "enforce_eager": self.config.get("enforce_eager", True),
            "disable_log_stats": self.config.get("disable_log_stats", True),
            "trust_remote_code": trust_remote_code,
        }
        
        self.inference_engine = LLM(**engine_config)
        
        # Sampling parameters for evaluation
        self.sampling_params = SamplingParams(
            n=1,
            max_tokens=128,
            temperature=0.0,  # Deterministic
            top_p=1.0,
            stop=None,
            include_stop_str_in_output=False,
        )
        
        return f"AutoRater initialized on GPU {self.gpu_id}"
    
    def evaluate_batch(self, questions: List[str], predicted_answers: List[str], ground_truth_answers: List[str]):
        """Evaluate a batch of responses using AutoRater template"""
        if self.inference_engine is None:
            raise RuntimeError("AutoRater not initialized")
                    
        # Format evaluation prompts
        evaluation_prompts = []
        for question, predicted_answer, ground_truth in zip(questions, predicted_answers, ground_truth_answers):
            prompt = format_autorater_prompt(
                question=question,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth
            )
            evaluation_prompts.append(prompt)
        
        # Generate responses using vLLM
        outputs = self.inference_engine.generate(
            prompts=evaluation_prompts,
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        
        # Parse results
        decisions = []
        explanations = []
        raw_responses = []
        
        for output in outputs:
            response = output.outputs[0].text
            raw_responses.append(response)
            
            # Parse response for decision
            explanation, decision = parse_autorater_response(response)
            explanations.append(explanation)
            decisions.append(decision)
        
        return {
            "decisions": decisions,
            "explanations": explanations,
            "raw_responses": raw_responses
        }

    async def get_tokenizer(self):
        """Return the tokenizer associated with this AutoRater actor"""
        return self.tokenizer


class AutoRaterRequest(BaseModel):
    """Request model for AutoRater evaluation"""
    prompts: List[List[int]]  # List of tokenized prompts
    responses: List[List[int]]  # List of tokenized responses
    attention_mask: List[List[int]]  # Attention masks (needed for decoding)
    position_ids: List[List[int]]  # Position IDs (might not be directly used for text decoding but part of the original DataProto)
    reward_model_info: List[Dict[str, Any]]  # Ground truth and metadata
    
    class Config:
        arbitrary_types_allowed = True


class AutoRaterResponse(BaseModel):
    """Response model for AutoRater evaluation"""
    autorater_scores: List[float]
    autorater_decisions: List[int]
    autorater_explanations: Optional[List[str]] = None
    autorater_raw_responses: Optional[List[str]] = None
    # Code execution details (optional)
    code_scores: Optional[List[float]] = None  # aggregate score (points per passed test)
    code_tests_passed: Optional[List[int]] = None
    code_total_tests: Optional[List[int]] = None
    code_stdout: Optional[List[str]] = None
    code_stderr: Optional[List[str]] = None
    code_error: Optional[List[str]] = None
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    autorater_initialized: bool
    gpu_available: bool
    memory_usage: Optional[Dict[str, float]] = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import subprocess
    import os
    
    # Check GPU availability using nvidia-smi
    gpu_available = False
    memory_usage = None
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        gpu_available = result.returncode == 0
        
        if gpu_available:
            # Parse memory usage from nvidia-smi
            try:
                smi_output = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                                          capture_output=True, text=True, timeout=5)
                if smi_output.returncode == 0:
                    lines = smi_output.stdout.strip().split('\n')
                    if lines:
                        used, total = lines[0].split(', ')
                        memory_usage = {
                            "allocated_gb": float(used) / 1024,
                            "total_gb": float(total) / 1024,
                            "available_gpus": app.state.num_gpus
                        }
            except Exception:
                pass
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy",
        autorater_initialized=(len(app.state.autorater_actors) > 0),
        gpu_available=gpu_available,
        memory_usage=memory_usage
    )


class InitializeRequest(BaseModel):
    """Request model for AutoRater initialization"""
    config: Dict[str, Any]
    num_gpus: int = 1
    gpu_ids: List[int] = [0]
    world_size: Optional[int] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    master_addr: str = "127.0.0.1"
    master_port: int = 29500


@app.post("/initialize")
async def initialize_autorater(request: InitializeRequest):
    """Initialize distributed AutoRater actors using Ray"""
    logger.info("Initializing distributed AutoRater service with Ray...")
    logger.info(f"GPU configuration: {request.num_gpus} GPUs, IDs: {request.gpu_ids}")
    
    # Store configuration
    app.state.autorater_config = OmegaConf.create(request.config)
    app.state.num_gpus = request.num_gpus
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("Ray initialized")
    
    # Validate configuration
    if "model" not in app.state.autorater_config:
        raise HTTPException(status_code=400, detail="Config must contain 'model' section")
    
    if "rollout" not in app.state.autorater_config:
        raise HTTPException(status_code=400, detail="Config must contain 'rollout' section")
    
    # Create Ray actors for each GPU
    app.state.autorater_actors = []
    initialization_futures = []
    
    for gpu_id in request.gpu_ids:
        # Create actor for this GPU
        actor = AutoRaterActor.remote( # type: ignore
            config=OmegaConf.to_container(app.state.autorater_config), # type: ignore
            gpu_id=gpu_id
        )
        app.state.autorater_actors.append(actor)
        
        # Initialize the actor asynchronously
        init_future = actor.initialize.remote()
        initialization_futures.append(init_future)
    
    # Wait for all actors to initialize
    initialization_results = ray.get(initialization_futures)
    
    # Initialize global SandboxSession once during setup
    if getattr(app.state, "sandbox_session", None) is None:
        try:
            _sess = SandboxSession(lang="python")
            try:
                _sess.open()
            except Exception:
                # If open() is not available, rely on implicit open in __enter__ via our run calls
                pass
            app.state.sandbox_session = _sess
            logger.info("Global SandboxSession created during /initialize")
        except Exception as se:
            logger.warning(f"Failed to create SandboxSession during initialization: {se}. Will fallback to lazy creation.")

    logger.info("All AutoRater actors initialized successfully:")
    for result in initialization_results:
        logger.info(f"  {result}")
    
    return {
        "status": "success", 
        "message": f"AutoRater initialized with {len(app.state.autorater_actors)} GPU actors",
        "gpu_config": {
            "num_gpus": request.num_gpus,
            "gpu_ids": request.gpu_ids,
            "num_actors": len(app.state.autorater_actors)
        }
    }


@app.post("/evaluate", response_model=AutoRaterResponse)
async def evaluate_responses(request: AutoRaterRequest):
    """Evaluate responses using distributed AutoRater actors"""
    if len(app.state.autorater_actors) == 0:
        raise HTTPException(status_code=400, detail="AutoRater not initialized. Call /initialize first.")

    start_time = time.time()
    batch_size = len(request.prompts)
    logger.info(
        f"Processing AutoRater request with {batch_size} samples using {len(app.state.autorater_actors)} actors"
    )

    tokenizer = _get_tokenizer()

    # --- Decode ---
    questions, predicted_answers, ground_truth_answers = _decode_request(request, tokenizer)

    # --- LLM AutoRater ---
    autorater_scores, autorater_decisions, autorater_explanations, autorater_raw = _run_llm_autorater(
        questions, predicted_answers, ground_truth_answers
    )

    # --- Unit Tests ---
    (
        code_scores,
        code_tests_passed,
        code_total_tests,
        code_stdout,
        code_stderr,
        code_error,
    ) = _run_unit_tests(predicted_answers, request.reward_model_info)

    # Combine scores
    all_scores = [a + c for a, c in zip(autorater_scores, code_scores)]

    processing_time = time.time() - start_time

    return AutoRaterResponse(
        autorater_scores=all_scores,
        autorater_decisions=autorater_decisions,
        autorater_explanations=autorater_explanations,
        autorater_raw_responses=autorater_raw,
        code_scores=code_scores,
        code_tests_passed=code_tests_passed,
        code_total_tests=code_total_tests,
        code_stdout=code_stdout,
        code_stderr=code_stderr,
        code_error=code_error,
        processing_time=processing_time,
        success=True,
    )


@app.post("/shutdown")
async def shutdown_service(background_tasks: BackgroundTasks):
    """Gracefully shutdown the service"""
    def cleanup():
        # Clean up Ray actors
        for actor in app.state.autorater_actors:
            ray.kill(actor)
        app.state.autorater_actors.clear()

        # Close global SandboxSession if exists
        ss = getattr(app.state, "sandbox_session", None)
        if ss is not None:
            try:
                if hasattr(ss, "close"):
                    ss.close()
                else:
                    ss.__exit__(None, None, None)
                logger.info("SandboxSession closed")
            except Exception as e:
                logger.warning(f"Error closing SandboxSession: {e}")
            app.state.sandbox_session = None
        
        # Shutdown Ray if we initialized it
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("AutoRater actors cleaned up")
    
    background_tasks.add_task(cleanup)
    return {"status": "success", "message": "Shutdown initiated"}


@app.on_event("startup")
async def startup_event():
    """Auto-initialize AutoRater if config is provided via environment"""
    import sys
    
    # Check if --config was provided as command line argument
    config_path = None
    if "--config" in sys.argv:
        config_index = sys.argv.index("--config")
        if config_index + 1 < len(sys.argv):
            config_path = sys.argv[config_index + 1]
    
    if config_path:
        try:
            logger.info(f"Auto-initializing AutoRater with config: {config_path}")
            
            config = OmegaConf.load(config_path)
            
            # Get GPU configuration from environment variables
            num_gpus = int(os.environ.get("NUM_GPUS", "1"))
            gpu_ids_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]
            
            # Create proper InitializeRequest object
            init_request = InitializeRequest(
                config=OmegaConf.to_container(config),
                num_gpus=num_gpus,
                gpu_ids=gpu_ids,
                world_size=int(os.environ.get("WORLD_SIZE", "1")),
                rank=int(os.environ.get("RANK", "0")),
                local_rank=int(os.environ.get("LOCAL_RANK", "0")),
                master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
                master_port=int(os.environ.get("MASTER_PORT", "29500"))
            )
            
            await initialize_autorater(init_request)
            logger.info("Auto-initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Auto-initialization failed: {e}")
            logger.error(traceback.format_exc())
            # Don't raise the exception - let the service start anyway
    else:
        logger.info("No config provided, skipping auto-initialization")


# ================= Helper Functions =================


def _get_tokenizer():
    """Retrieve the tokenizer from the first AutoRater actor or fall back to a default."""
    try:
        tokenizer = ray.get(app.state.autorater_actors[0].get_tokenizer.remote())  # type: ignore
    except Exception as e:
        logger.warning(
            f"Could not retrieve tokenizer from actor, using default Qwen/Qwen2.5-7B-Instruct. Error: {e}"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False)  # type: ignore
    return tokenizer


def _decode_request(request: "AutoRaterRequest", tokenizer):
    """Decode token IDs back to human-readable text lists."""
    questions: List[str] = []
    predicted_answers: List[str] = []
    ground_truth_answers: List[str] = []

    for p_ids, r_ids, rm_info in zip(request.prompts, request.responses, request.reward_model_info):
        questions.append(tokenizer.decode(p_ids, skip_special_tokens=True))
        predicted_answers.append(tokenizer.decode(r_ids, skip_special_tokens=True))

        if isinstance(rm_info, dict) and "ground_truth" in rm_info:
            gt = str(rm_info["ground_truth"])
        else:
            gt = str(rm_info)
        ground_truth_answers.append(gt)

    return questions, predicted_answers, ground_truth_answers


def _run_llm_autorater(
    questions: List[str],
    predicted_answers: List[str],
    ground_truth_answers: List[str],
) -> Tuple[List[float], List[int], List[str], List[str]]:
    """Run LLM-based AutoRater on the full batch and return results."""
    batch_size = len(questions)

    # Build prompts
    prompts = [
        format_autorater_prompt(q, pa, gt)  # type: ignore
        for q, pa, gt in zip(questions, predicted_answers, ground_truth_answers)
    ]

    # Dispatch to Ray actors
    num_actors = len(app.state.autorater_actors)
    chunk_size = max(1, batch_size // num_actors)

    futures = []
    for i in range(0, batch_size, chunk_size):
        actor_idx = (i // chunk_size) % num_actors
        actor = app.state.autorater_actors[actor_idx]
        fut = actor.evaluate_batch.remote(
            questions[i : i + chunk_size],
            predicted_answers[i : i + chunk_size],
            ground_truth_answers[i : i + chunk_size],
        )  # type: ignore
        futures.append(fut)

    results = ray.get(futures)

    # Flatten keeping order – we appended sequentially so order is preserved
    autorater_scores: List[float] = []
    autorater_decisions: List[int] = []
    autorater_explanations: List[str] = []
    autorater_raw: List[str] = []

    for res in results:
        for decision in res["decisions"]:
            if decision == "TRUE":
                autorater_scores.append(1.0)
                autorater_decisions.append(1)
            elif decision == "FALSE":
                autorater_scores.append(0.0)
                autorater_decisions.append(0)
            else:
                autorater_scores.append(0.5)
                autorater_decisions.append(0)

        autorater_explanations.extend(res["explanations"])
        autorater_raw.extend(res["raw_responses"])

    return autorater_scores, autorater_decisions, autorater_explanations, autorater_raw


def _run_unit_tests(
    predicted_answers: List[str],
    reward_model_info: List[Dict[str, Any]],
) -> Tuple[List[float], List[int], List[int], List[str], List[str], List[str]]:
    """Execute unit tests in SandboxSession and aggregate scores and outputs."""
    batch_size = len(predicted_answers)

    # Ensure sandbox session exists
    if getattr(app.state, "sandbox_session", None) is None:
        _sess = SandboxSession(lang="python")
        try:
            _sess.open()
        except Exception:
            pass
        app.state.sandbox_session = _sess

    sess = app.state.sandbox_session
    import re as _re

    code_scores: List[float] = [0.0] * batch_size
    tests_passed: List[int] = [0] * batch_size
    total_tests: List[int] = [0] * batch_size
    stdout_list: List[str] = [""] * batch_size
    stderr_list: List[str] = [""] * batch_size
    error_list: List[str] = [""] * batch_size

    for idx, rm_info in enumerate(reward_model_info):
        tests: List[str] = []
        if isinstance(rm_info, dict):
            if isinstance(rm_info.get("unit_tests"), list):
                tests = rm_info["unit_tests"]
            elif isinstance(rm_info.get("tests"), list):
                tests = rm_info["tests"]
            else:
                tc = rm_info.get("unit_tests") or rm_info.get("tests")
                if tc:
                    tests = [tc]

        if not tests:
            continue

        code_match = _re.search(r"```[\w]*\n(.*?)```", predicted_answers[idx], _re.DOTALL)
        pred_code_block = code_match.group(1) if code_match else predicted_answers[idx]

        passes = 0
        for test_snippet in tests:
            exec_code = f"{pred_code_block}\n\n{test_snippet}"
            try:
                res = sess.run(exec_code, libraries=None)
                if res.exit_code == 0:
                    passes += 1
                stdout_list[idx] += res.stdout + "\n"
                stderr_list[idx] += res.stderr + "\n"
            except Exception as exec_e:
                error_list[idx] += str(exec_e) + "\n"

        total_tests[idx] = len(tests)
        tests_passed[idx] = passes
        code_scores[idx] = float(passes)  # 1 point per passed test

    return code_scores, tests_passed, total_tests, stdout_list, stderr_list, error_list


# -------------------- New Lightweight Endpoints --------------------


@app.post("/evaluate_autorater", response_model=AutoRaterResponse)
async def evaluate_autorater_only(request: AutoRaterRequest):
    """Evaluate only using LLM AutoRater (no unit-test execution)."""
    if len(app.state.autorater_actors) == 0:
        raise HTTPException(status_code=400, detail="AutoRater not initialized. Call /initialize first.")

    start_time = time.time()
    tokenizer = _get_tokenizer()

    questions, predicted_answers, ground_truth_answers = _decode_request(request, tokenizer)

    autorater_scores, autorater_decisions, autorater_explanations, autorater_raw = _run_llm_autorater(
        questions, predicted_answers, ground_truth_answers
    )

    processing_time = time.time() - start_time

    zero_array_float = [0.0] * len(autorater_scores)
    zero_array_int = [0] * len(autorater_scores)
    empty_str_arr = [""] * len(autorater_scores)

    return AutoRaterResponse(
        autorater_scores=autorater_scores,
        autorater_decisions=autorater_decisions,
        autorater_explanations=autorater_explanations,
        autorater_raw_responses=autorater_raw,
        code_scores=zero_array_float,
        code_tests_passed=zero_array_int,
        code_total_tests=zero_array_int,
        code_stdout=empty_str_arr,
        code_stderr=empty_str_arr,
        code_error=empty_str_arr,
        processing_time=processing_time,
        success=True,
    )


@app.post("/evaluate_tests", response_model=AutoRaterResponse)
async def evaluate_unit_tests_only(request: AutoRaterRequest):
    """Evaluate only unit tests and skip LLM AutoRater."""
    start_time = time.time()

    # We still need predicted_answers decoded for extracting code blocks
    tokenizer = _get_tokenizer()
    _, predicted_answers, _ = _decode_request(request, tokenizer)

    (
        code_scores,
        code_tests_passed,
        code_total_tests,
        code_stdout,
        code_stderr,
        code_error,
    ) = _run_unit_tests(predicted_answers, request.reward_model_info)

    processing_time = time.time() - start_time

    zero_array_float = [0.0] * len(code_scores)
    zero_array_int = [0] * len(code_scores)
    empty_str_arr = [""] * len(code_scores)

    return AutoRaterResponse(
        autorater_scores=code_scores,  # Overall score equals code score when only tests run
        autorater_decisions=zero_array_int,
        autorater_explanations=[],
        autorater_raw_responses=[],
        code_scores=code_scores,
        code_tests_passed=code_tests_passed,
        code_total_tests=code_total_tests,
        code_stdout=code_stdout,
        code_stderr=code_stderr,
        code_error=code_error,
        processing_time=processing_time,
        success=True,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoRater FastAPI Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=80, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--config", type=str, help="Path to AutoRater config file")
    
    args = parser.parse_args()
    
    # Start the service (auto-initialization happens in startup event)
    uvicorn.run(
        "fastapi_autorater_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    ) 