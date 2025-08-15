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
from typing import Dict, List, Any, Optional, Tuple, Union

# Libraries from standard python or installed via pip (assumed to be installed)
import ray  # type: ignore
import numpy as np  # type: ignore
from fastapi import FastAPI, HTTPException, BackgroundTasks  # type: ignore
from pydantic import BaseModel  # type: ignore
import uvicorn  # type: ignore

from omegaconf import DictConfig, OmegaConf  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from verl.workers.autorater.autorater_utils import (
    format_autorater_prompt,
    parse_autorater_response_scalar,
    parse_autorater_response_boolean,
    format_code_outline_prompt,
    format_helpfulness_prompt,
    format_plan_evaluation_prompt,
    format_plan_quality_evaluation_prompt,
    parse_plan_evaluation_response,
    format_coding_answer_correctness_prompt,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoRater Service", version="1.0.0")

# Initialize app state
app.state.autorater_actors = []
app.state.autorater_config = None
app.state.num_gpus = 0


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create vLLM engine with single GPU configuration
        engine_config = {
            "model": local_path,
            "tokenizer": local_path,
            "tensor_parallel_size": 1,  # Single GPU per actor
            "dtype": self.config.get("dtype", "bfloat16"),
            "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0.95),
            "max_model_len": self.config.get("max_model_len", 8192),
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

    def evaluate_batch(
        self,
        prompts: List[str],
        responses: List[str],
        gt_answers: Optional[List[str]] = None,
        context: Optional[List[str]] = None,
        template_types: Optional[List[str]] = None,
    ):
        """Evaluate a batch of responses using AutoRater template"""
        if self.inference_engine is None:
            raise RuntimeError("AutoRater not initialized")

        # Format evaluation prompts
        evaluation_prompts = []
        if not template_types:
            template_types = ["standard"] * len(prompts)
        
        # Handle None context by creating a list of None values
        if gt_answers is None:
            gt_answers = [None] * len(prompts)
        if context is None:
            context = [None] * len(prompts)

        print(f"template_types: {template_types}")

        for i, (prompt, response, gt_answer, tmpl, ctx) in enumerate(
            zip(prompts, responses, gt_answers, template_types, context)
        ):
            if tmpl == "outline":
                autorater_prompt = format_code_outline_prompt(
                    problem_description=prompt, outline_answer=response
                )
            elif "helpfulness" in tmpl:
                autorater_prompt = format_helpfulness_prompt(
                    question=prompt,
                    predicted_answer=response,
                    context=ctx,
                    template=tmpl,
                )
            elif tmpl == "plan_quality_evaluation":
                autorater_prompt = format_plan_quality_evaluation_prompt(prompt, response)
            elif tmpl == "plan_evaluation":
                if not isinstance(response, list):
                    raise ValueError("Plan evaluation requires a list of plans")
                
                autorater_prompt = format_plan_evaluation_prompt(prompt, response)
            elif tmpl == "coding_answer_correctness":
                autorater_prompt = format_coding_answer_correctness_prompt(prompt, response)
            else:
                autorater_prompt = format_autorater_prompt(
                    question=prompt,
                    predicted_answer=response,
                    ground_truth_answer=gt_answer,
                )
            evaluation_prompts.append(autorater_prompt)

        # Generate responses using vLLM
        outputs = self.inference_engine.generate(
            prompts=evaluation_prompts,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

        # Parse results
        decisions = []
        explanations = []
        raw_responses = []

        for i, output in enumerate(outputs):
            response = output.outputs[0].text
            raw_responses.append(response)

            if template_types[i] == "outline" or template_types[i] == "standard" or template_types[i] == "helpfulness" or template_types[i] == "plan_quality_evaluation":
                explanation, decision = parse_autorater_response_boolean(response)
            elif template_types[i] == "plan_evaluation":
                # For plan evaluation, we need to parse the response to get the selected plan number
                # We'll use a dummy explanation and return the parsed plan number as the decision
                explanation = "Plan evaluation completed"
                # Parse the plan number from the response
                # We can get the number of plans from the original response parameter
                try:
                    num_plans = len(response)
                    print(f"Plan evaluation: {num_plans} plans, response: '{response}'")
                    selected_plan = parse_plan_evaluation_response(response, num_plans)
                    decision = selected_plan  # Return the plan number as the decision
                    print(f"Plan evaluation: selected plan {selected_plan}")
                except Exception as e:
                    print(f"Failed to parse plan evaluation response: {e}")
                    decision = 1  # Fallback to first plan
            elif template_types[i] == "coding_answer_correctness":
                explanation, decision = parse_autorater_response_boolean(response)
            else:
                explanation, decision = parse_autorater_response_scalar(response)

            explanations.append(explanation)
            decisions.append(decision)

        return {
            "decisions": decisions,
            "explanations": explanations,
            "raw_responses": raw_responses,
        }

    async def get_tokenizer(self):
        """Return the tokenizer associated with this AutoRater actor"""
        return self.tokenizer


class AutoRaterRequest(BaseModel):
    """Request model for AutoRater evaluation"""

    prompts: List[str]  # string prompts
    responses: Union[List[str], List[List[str]]]  # string responses
    gt_answers: Optional[Union[List[str], List[List[str]]]] = None  # Ground truth answers
    template_types: Optional[List[str]] = None  # template types
    context: Optional[Union[List[str], List[List[str]]]] = None

    class Config:
        arbitrary_types_allowed = True


class AutoRaterResponse(BaseModel):
    """Response model for AutoRater evaluation (no autorater_scores)"""

    autorater_decisions: List[Union[float, str]]
    autorater_explanations: Optional[List[str]] = None
    autorater_raw_responses: Optional[List[str]] = None
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
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        gpu_available = result.returncode == 0

        if gpu_available:
            # Parse memory usage from nvidia-smi
            try:
                smi_output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total",
                        "--format=csv,nounits,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if smi_output.returncode == 0:
                    lines = smi_output.stdout.strip().split("\n")
                    if lines:
                        used, total = lines[0].split(", ")
                        memory_usage = {
                            "allocated_gb": float(used) / 1024,
                            "total_gb": float(total) / 1024,
                            "available_gpus": app.state.num_gpus,
                        }
            except Exception:
                pass
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        autorater_initialized=(len(app.state.autorater_actors) > 0),
        gpu_available=gpu_available,
        memory_usage=memory_usage,
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
        raise HTTPException(
            status_code=400, detail="Config must contain 'model' section"
        )

    if "rollout" not in app.state.autorater_config:
        raise HTTPException(
            status_code=400, detail="Config must contain 'rollout' section"
        )

    # Create Ray actors for each GPU
    app.state.autorater_actors = []
    initialization_futures = []

    for gpu_id in request.gpu_ids:
        # Create actor for this GPU
        actor = AutoRaterActor.remote(  # type: ignore
            config=OmegaConf.to_container(app.state.autorater_config),  # type: ignore
            gpu_id=gpu_id,
        )
        app.state.autorater_actors.append(actor)

        # Initialize the actor asynchronously
        init_future = actor.initialize.remote()
        initialization_futures.append(init_future)

    # Wait for all actors to initialize
    initialization_results = ray.get(initialization_futures)

    # No SandboxSession needed for AutoRater-only service

    logger.info("All AutoRater actors initialized successfully:")
    for result in initialization_results:
        logger.info(f"  {result}")

    return {
        "status": "success",
        "message": f"AutoRater initialized with {len(app.state.autorater_actors)} GPU actors",
        "gpu_config": {
            "num_gpus": request.num_gpus,
            "gpu_ids": request.gpu_ids,
            "num_actors": len(app.state.autorater_actors),
        },
    }


@app.post("/evaluate", response_model=AutoRaterResponse)
async def evaluate_responses(request: AutoRaterRequest):
    """Evaluate responses using distributed AutoRater actors"""
    if len(app.state.autorater_actors) == 0:
        raise HTTPException(
            status_code=400, detail="AutoRater not initialized. Call /initialize first."
        )

    start_time = time.time()
    batch_size = len(request.prompts)
    logger.info(
        f"Processing AutoRater request with {batch_size} samples using {len(app.state.autorater_actors)} actors"
    )

    # --- LLM AutoRater ---
    autorater_decisions, autorater_explanations, autorater_raw = _run_llm_autorater(
        prompts=request.prompts,
        responses=request.responses,
        gt_answers=request.gt_answers,
        template_types=request.template_types,
        context=request.context,
    )

    processing_time = time.time() - start_time

    return AutoRaterResponse(
        autorater_decisions=autorater_decisions,
        autorater_explanations=autorater_explanations,
        autorater_raw_responses=autorater_raw,
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
                master_port=int(os.environ.get("MASTER_PORT", "29500")),
            )

            await initialize_autorater(init_request)
            logger.info("Auto-initialization completed successfully")

        except Exception as e:
            logger.error(f"Auto-initialization failed: {e}")
            logger.error(traceback.format_exc())
            # Don't raise the exception - let the service start anyway
    else:
        logger.info("No config provided, skipping auto-initialization")


def _run_llm_autorater(
    prompts: List[str],
    responses: List[str],
    gt_answers: List[str],
    template_types: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
) -> Tuple[List[float], List[str], List[str]]:
    """Run LLM-based AutoRater on the full batch and return results (no autorater_scores)."""
    batch_size = len(prompts)

    # Dispatch to Ray actors
    num_actors = len(app.state.autorater_actors)
    chunk_size = max(1, batch_size // num_actors)

    futures = []
    for i in range(0, batch_size, chunk_size):
        actor_idx = (i // chunk_size) % num_actors
        actor = app.state.autorater_actors[actor_idx]
        fut = actor.evaluate_batch.remote(
            prompts[i : i + chunk_size],
            responses[i : i + chunk_size],
            gt_answers[i : i + chunk_size] if gt_answers else None,
            context=context[i : i + chunk_size] if context else None,
            template_types=template_types[i : i + chunk_size] if template_types else None,
        )
        futures.append(fut)

    results = ray.get(futures)
    autorater_decisions = []
    autorater_explanations = []
    autorater_raw = []

    for res in results:
        autorater_decisions.extend(res["decisions"])
        autorater_explanations.extend(res["explanations"])
        autorater_raw.extend(res["raw_responses"])

    return autorater_decisions, autorater_explanations, autorater_raw


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AutoRater FastAPI Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=80, help="Port to bind to")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument("--config", type=str, help="Path to AutoRater config file")

    args = parser.parse_args()

    # Start the service (auto-initialization happens in startup event)
    uvicorn.run(
        "fastapi_autorater_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )
