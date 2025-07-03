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
from typing import Dict, List, Any, Optional

import ray
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

from omegaconf import DictConfig, OmegaConf

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
        from verl.third_party.vllm import LLM
        from verl.utils.fs import copy_to_local
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        
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
            
        from verl.workers.autorater.autorater_utils import format_autorater_prompt, parse_autorater_response
        
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
        actor = AutoRaterActor.remote(
            config=OmegaConf.to_container(app.state.autorater_config),
            gpu_id=gpu_id
        )
        app.state.autorater_actors.append(actor)
        
        # Initialize the actor asynchronously
        init_future = actor.initialize.remote()
        initialization_futures.append(init_future)
    
    # Wait for all actors to initialize
    initialization_results = ray.get(initialization_futures)
    
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
    
    # --- Logging request for debugging ---
    log_dir = "autorater_requests"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_filename = os.path.join(log_dir, f"request_{timestamp}.json")
    
    try:
        with open(log_filename, "w", encoding="utf-8") as f:
            # Convert Pydantic request object to dict for JSON serialization
            json.dump(request.dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Logged request to {log_filename}")
    except Exception as e:
        logger.error(f"Failed to log request to {log_filename}: {e}")
    # --- End logging request ---

    start_time = time.time()
    batch_size = len(request.prompts)
    logger.info(f"Processing AutoRater request with {batch_size} samples using {len(app.state.autorater_actors)} actors")
    
    # Load tokenizer to decode prompts and responses
    # Use the tokenizer from the initialized AutoRaterActor or a default one if not available
    # For simplicity, let's use a default for now if actors aren't fully initialized (though they should be)
    try:
        # Try to get tokenizer from an actor. All actors should have the same tokenizer.
        # This assumes at least one actor is initialized.
        tokenizer = await app.state.autorater_actors[0].get_tokenizer.remote() # New method to get tokenizer
    except Exception:
        logger.warning("Could not retrieve tokenizer from actor, using default Qwen/Qwen2.5-7B-Instruct. This might cause issues if models differ.")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False)

    # Decode prompts and responses to text
    questions = []
    predicted_answers = []
    ground_truth_answers = []
    
    for i in range(batch_size):
        # Decode question from prompt IDs
        question = tokenizer.decode(request.prompts[i], skip_special_tokens=True)
        questions.append(question)
        
        # Decode predicted answer from response IDs
        predicted_answer = tokenizer.decode(request.responses[i], skip_special_tokens=True)
        predicted_answers.append(predicted_answer)
        
        # Extract ground truth from reward_model_info
        reward_info = request.reward_model_info[i]
        if isinstance(reward_info, dict) and "ground_truth" in reward_info:
            ground_truth = str(reward_info["ground_truth"])
        else:
            ground_truth = str(reward_info)
        ground_truth_answers.append(ground_truth)
    
    # Split work across available actors
    num_actors = len(app.state.autorater_actors)
    chunk_size = max(1, batch_size // num_actors)
    
    # Create chunks for parallel processing
    evaluation_futures = []
    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        
        # Select actor (round robin)
        actor_idx = (i // chunk_size) % num_actors
        actor = app.state.autorater_actors[actor_idx]
        
        # Submit evaluation task to actor
        future = actor.evaluate_batch.remote(
            questions[i:end_idx],
            predicted_answers[i:end_idx],
            ground_truth_answers[i:end_idx]
        )
        evaluation_futures.append(future)
    
    # Wait for all evaluations to complete
    results = ray.get(evaluation_futures)
    
    # Combine results from all actors
    all_scores = []
    all_decisions = []
    all_explanations = []
    all_raw_responses = []
    
    for result in results:
        # Convert decisions to scores and numerical decisions
        for decision in result["decisions"]:
            if decision == "TRUE":
                all_scores.append(1.0)
                all_decisions.append(1)
            elif decision == "FALSE":
                all_scores.append(0.0)
                all_decisions.append(0)
            else:
                all_scores.append(0.5)
                all_decisions.append(0)
        
        all_explanations.extend(result["explanations"])
        all_raw_responses.extend(result["raw_responses"])
    
    processing_time = time.time() - start_time
    
    logger.info(f"AutoRater evaluation completed in {processing_time:.2f}s")
    logger.info(f"Total samples processed: {len(all_raw_responses)}")
    logger.info(f"Average score: {np.mean(all_scores):.3f}")
    
    return AutoRaterResponse(
        autorater_scores=all_scores,
        autorater_decisions=all_decisions,
        autorater_explanations=all_explanations,
        autorater_raw_responses=all_raw_responses,
        processing_time=processing_time,
        success=True
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