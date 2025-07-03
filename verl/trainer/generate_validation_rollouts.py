#!/usr/bin/env python3
"""
Standalone script for generating validation rollouts using vLLM.
Can be used to generate rollouts from any checkpoint independently of training.

Usage:
    # Single validation file:
    python generate_validation_rollouts.py --checkpoint_dir /path/to/checkpoint --val_data file.parquet
    
    # Multiple validation files:
    python generate_validation_rollouts.py --checkpoint_dir /path/to/checkpoint --val_data "['file1.parquet','file2.parquet']"
    
    # Using config file:
    python generate_validation_rollouts.py --checkpoint_dir /path/to/checkpoint --config /path/to/config.yaml
"""

import os
import json
import argparse
import logging
import ast
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("ERROR: vLLM not available. Please install vLLM to use this script.")
    exit(1)

from verl.utils.fs import copy_to_local
import verl.utils.hdfs_io as hdfs_io
from verl.trainer.fsdp_sft_trainer import create_sft_dataset
from verl.utils.templates import get_system_template


logger = logging.getLogger(__name__)


class ValidationRolloutGenerator:
    """Generate validation rollouts from model checkpoints using vLLM."""
    
    def __init__(self, 
                 checkpoint_path: str,
                 base_model_path: Optional[str] = None,
                 config: Optional[DictConfig] = None,
                 vllm_config: Optional[Dict] = None,
                 rollout_config: Optional[Dict] = None,
                 skip_model_loading: Optional[bool] = None):
        """
        Initialize the rollout generator.
        
        Args:
            checkpoint_path: Path to model checkpoint (or LoRA adapter)
            base_model_path: Path to base model (required for LoRA checkpoints)
            config: Training configuration (optional)
            vllm_config: vLLM engine configuration
            rollout_config: Rollout generation configuration
            skip_model_loading: If True, skip model loading. If None, auto-detect based on LoRA config.
        """
        self.checkpoint_path = checkpoint_path
        self.base_model_path = base_model_path
        self.config = config
        self.vllm_config = vllm_config or {}
        self.rollout_config = rollout_config or {}
        
        # Determine whether to skip model loading
        if skip_model_loading is None:
            self.skip_model_loading = self._should_skip_model_loading()
        else:
            self.skip_model_loading = skip_model_loading
        
        # Initialize components (lazy loading)
        self.vllm_engine = None
        self.tokenizer = None
        self.merged_model_path = None
        
        # Auto-detect tensor parallel size if not specified (only if loading model)
        if not self.skip_model_loading and 'tensor_parallel_size' not in self.vllm_config:
            available_gpus = torch.cuda.device_count()
            # Use up to 4 GPUs by default
            self.vllm_config['tensor_parallel_size'] = min(available_gpus, 4)
        
        print(f"Initialized ValidationRolloutGenerator:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Base model: {base_model_path}")
        print(f"  Skip model loading: {self.skip_model_loading}")
        if not self.skip_model_loading:
            print(f"  Tensor parallel size: {self.vllm_config.get('tensor_parallel_size', 1)}")
            print(f"  Available GPUs: {torch.cuda.device_count()}")
        else:
            print("  Model loading skipped (LoRA training detected or explicitly disabled)")
    
    def _should_skip_model_loading(self) -> bool:
        """Check if model loading should be skipped based on LoRA configuration."""
        if self.config is None:
            return False
        
        try:
            # Check if LoRA is enabled in the training config
            if hasattr(self.config, 'model') and hasattr(self.config.model, 'lora_rank'):
                lora_rank = self.config.model.lora_rank
                if isinstance(lora_rank, (int, float)) and lora_rank > 0:
                    print(f"  LoRA training detected (lora_rank={lora_rank}), skipping model loading")
                    return True
            return False
        except Exception as e:
            print(f"  Warning: Could not check LoRA config: {e}")
            return False
    
    def _is_lora_checkpoint(self, checkpoint_path: str) -> bool:
        """Check if checkpoint is a LoRA adapter."""
        try:
            import os
            # Check for LoRA-specific files
            checkpoint_dir = Path(checkpoint_path)
            if checkpoint_dir.is_dir():
                files = os.listdir(checkpoint_path)
                # Look for adapter files
                has_adapter_config = any('adapter_config.json' in f for f in files)
                has_adapter_model = any('adapter_model' in f for f in files)
                return has_adapter_config or has_adapter_model
            return False
        except:
            return False
    
    def _merge_lora_with_base(self, base_model_path: str, lora_path: str) -> str:
        """Merge LoRA adapter with base model and return path to merged model."""
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import tempfile
            import os
            
            print("Merging LoRA adapter with base model...")
            
            # Create temporary directory for merged model
            merged_dir = tempfile.mkdtemp(prefix="merged_model_")
            self.merged_model_path = merged_dir
            
            print(f"  Loading base model from: {base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",  # Load on CPU for merging
                trust_remote_code=True
            )
            
            print(f"  Loading LoRA adapter from: {lora_path}")
            # Load the LoRA model
            model = PeftModel.from_pretrained(base_model, lora_path)
            
            print("  Merging LoRA weights...")
            # Merge the LoRA weights
            merged_model = model.merge_and_unload()
            
            print(f"  Saving merged model to: {merged_dir}")
            # Save merged model
            merged_model.save_pretrained(merged_dir)
            
            # Also copy tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            tokenizer.save_pretrained(merged_dir)
            
            print("✓ Successfully merged LoRA with base model")
            return merged_dir
            
        except Exception as e:
            print(f"✗ Failed to merge LoRA with base model: {e}")
            raise
    
    def _cleanup_merged_model(self):
        """Clean up temporary merged model directory."""
        if self.merged_model_path and Path(self.merged_model_path).exists():
            import shutil
            try:
                shutil.rmtree(self.merged_model_path)
                print(f"✓ Cleaned up temporary merged model at {self.merged_model_path}")
            except Exception as e:
                print(f"Warning: Failed to cleanup merged model: {e}")
    
    def _is_huggingface_model_id(self, model_path: str) -> bool:
        """Check if the model path is a Hugging Face model identifier."""
        # HF model IDs typically follow the pattern "organization/model-name"
        # and don't exist as local paths
        return (
            "/" in model_path and
            not Path(model_path).exists() and
            not model_path.startswith("./") and
            not model_path.startswith("../") and
            not model_path.startswith("/")
        )
    
    def _get_model_path_for_vllm(self) -> str:
        """Get the model path to use for vLLM initialization."""
        print("Determining model path for vLLM...")
        
        # Check if this is a Hugging Face model identifier
        is_hf_model = self._is_huggingface_model_id(self.checkpoint_path)
        if is_hf_model:
            print(f"  Detected Hugging Face model identifier: {self.checkpoint_path}")
            print(f"  ✓ vLLM will download this model automatically")
            return self.checkpoint_path
        
        # Check if this is a LoRA checkpoint (only for local paths)
        is_lora = self._is_lora_checkpoint(self.checkpoint_path)
        print(f"  LoRA checkpoint detected: {is_lora}")
        
        if is_lora:
            if not self.base_model_path:
                raise ValueError(
                    "LoRA checkpoint detected but no base_model_path provided. "
                    "Please specify --base_model_path when using LoRA checkpoints."
                )
            
            print(f"  Merging LoRA adapter with base model...")
            print(f"    Base model: {self.base_model_path}")
            print(f"    LoRA adapter: {self.checkpoint_path}")
            
            # Merge LoRA with base model
            merged_path = self._merge_lora_with_base(self.base_model_path, self.checkpoint_path)
            print(f"  ✓ Using merged model at: {merged_path}")
            return merged_path
        else:
            # Regular local checkpoint, use as-is
            print(f"  Using local checkpoint weights: {self.checkpoint_path}")
            
            # Validate checkpoint exists and has model files
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint path does not exist: {self.checkpoint_path}")
            
            # Look for model files to confirm this is a valid checkpoint
            if checkpoint_path.is_dir():
                model_files = list(checkpoint_path.glob("*.bin")) + list(checkpoint_path.glob("*.safetensors")) + list(checkpoint_path.glob("model.safetensors.index.json"))
                if not model_files:
                    print(f"  Warning: No model weight files found in {self.checkpoint_path}")
                    print(f"    This might be a base model directory rather than a checkpoint")
                else:
                    print(f"  ✓ Found {len(model_files)} model weight file(s)")
            
            return self.checkpoint_path
    
    def _init_vllm_engine(self):
        """Initialize vLLM engine with checkpoint."""
        if self.vllm_engine is not None:
            print("✓ vLLM engine already initialized")
            return
        
        if self.skip_model_loading:
            print("Skipping vLLM engine initialization (model loading disabled)")
            return
        
        print("Initializing vLLM engine...")
        
        try:
            model_path = self._get_model_path_for_vllm()
            
            # Default vLLM configuration optimized for multi-GPU inference and batching
            default_config = {
                'model': model_path,
                'tensor_parallel_size': 1,
                'dtype': 'bfloat16',
                'max_model_len': 4096,
                'gpu_memory_utilization': 0.9,
                'enforce_eager': False,  # Enable CUDA graphs for better performance
                'disable_custom_all_reduce': False,  # Enable custom all-reduce for multi-GPU
                'max_num_batched_tokens': None,  # Will be set based on model size
                'distributed_executor_backend': 'ray',
                'enable_chunked_prefill': True,  # Better memory efficiency for long sequences
                'enable_prefix_caching': True,  # Cache common prefixes for better throughput
                'disable_log_stats': True,  # Reduce logging overhead
                'trust_remote_code': True,
            }
            
            # Apply user configuration
            engine_config = {**default_config, **self.vllm_config}
            
            # BATCHING OPTIMIZATIONS
            
            # Auto-configure max_num_batched_tokens for optimal throughput
            if engine_config['max_num_batched_tokens'] is None:
                model_len = engine_config['max_model_len']
                tp_size = engine_config['tensor_parallel_size']
                
                # Conservative estimate: balance memory usage and throughput
                # For larger models with TP, reduce batched tokens to avoid OOM
                if tp_size >= 4:
                    # Large model - conservative batching
                    engine_config['max_num_batched_tokens'] = min(32768, model_len * 8)
                elif tp_size >= 2:
                    # Medium model - moderate batching
                    engine_config['max_num_batched_tokens'] = min(65536, model_len * 16)
                else:
                    # Single GPU - aggressive batching
                    engine_config['max_num_batched_tokens'] = min(131072, model_len * 32)
                    
                print(f"Auto-configured max_num_batched_tokens: {engine_config['max_num_batched_tokens']}")
            
            # Optimize memory utilization for batching workloads
            if engine_config['gpu_memory_utilization'] > 0.85:
                print(f"✓ High GPU memory utilization ({engine_config['gpu_memory_utilization']}) - optimized for large batches")
            
            # Enable performance optimizations
            if not engine_config['enforce_eager']:
                print("✓ CUDA graphs enabled for optimized inference")
            
            if engine_config['enable_chunked_prefill']:
                print("✓ Chunked prefill enabled for memory efficiency")
                
            if engine_config['enable_prefix_caching']:
                print("✓ Prefix caching enabled for throughput optimization")
            
            print(f"vLLM config: {engine_config}")
            
            # Configure swap space for large batch workloads
            swap_space = 4  # Default 4GB swap space per GPU for better batch handling
            
            # Initialize vLLM engine
            print("Initializing vLLM LLM engine...")
            self.vllm_engine = LLM(**engine_config)
            print("✓ vLLM engine initialized successfully")
            
            # Verify model loading and show optimization info
            print("\n🚀 vLLM BATCHING OPTIMIZATION STATUS:")
            model_config = self.vllm_engine.llm_engine.model_config
            print(f"  Model: {model_config.model}")
            print(f"  Max model length: {model_config.max_model_len}")
            print(f"  Tensor parallel size: {engine_config['tensor_parallel_size']}")
            print(f"  Max batched tokens: {engine_config['max_num_batched_tokens']}")
            print(f"  GPU memory utilization: {engine_config['gpu_memory_utilization']}")
            print(f"  CUDA graphs: {'Enabled' if not engine_config['enforce_eager'] else 'Disabled'}")
            print(f"  Chunked prefill: {'Enabled' if engine_config['enable_chunked_prefill'] else 'Disabled'}")
            print(f"  Prefix caching: {'Enabled' if engine_config['enable_prefix_caching'] else 'Disabled'}")
            
            # Estimate batching capacity
            avg_seq_len = engine_config['max_model_len'] // 2  # Rough estimate
            theoretical_batch_size = engine_config['max_num_batched_tokens'] // avg_seq_len
            print(f"  Estimated max batch size: ~{theoretical_batch_size} sequences")
            print(f"  Ready for high-throughput batched inference! 🎯")
            
            # Show model details if accessible
            if hasattr(self.vllm_engine.llm_engine, 'model_executor'):
                model_executor = self.vllm_engine.llm_engine.model_executor
                if hasattr(model_executor, 'driver_worker'):
                    driver_worker = model_executor.driver_worker
                    if hasattr(driver_worker, 'model_runner'):
                        print(f"  Model runner: {type(driver_worker.model_runner).__name__}")
        
        except Exception as e:
            print(f"✗ Failed to initialize vLLM engine: {e}")
            raise
    
    def _init_tokenizer(self):
        """Initialize tokenizer for prompt processing."""
        if self.tokenizer is not None:
            return
            
        print("Loading tokenizer...")
        try:
            # Use base model path for tokenizer if LoRA
            tokenizer_path = self.base_model_path if (
                self.base_model_path and self._is_lora_checkpoint(self.checkpoint_path)
            ) else self.checkpoint_path
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load tokenizer: {e}")
            raise
    
    def _detect_file_format(self, file_path: str, format_hint: str = "auto") -> str:
        """Detect file format based on extension or explicit format."""
        if format_hint != "auto":
            return format_hint
        
        # Auto-detect based on file extension
        path_lower = file_path.lower()
        if path_lower.endswith('.txt'):
            return "text"
        elif path_lower.endswith('.parquet'):
            return "parquet"
        else:
            # Default to parquet for unknown extensions
            print(f"Warning: Unknown file extension for {file_path}, assuming parquet format")
            return "parquet"
    
    def _load_text_file(self, file_path: str) -> pd.DataFrame:
        """Load prompts from a newline-separated text file."""
        print(f"    Loading text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Clean up lines and filter out empty ones
            prompts = []
            for i, line in enumerate(lines):
                line = line.strip()
                if line:  # Skip empty lines
                    prompts.append(line)
            
            # Create DataFrame with prompts (always use "prompt" as column name)
            df = pd.DataFrame({"prompt": prompts})
            print(f"    ✓ Loaded {len(df)} prompts from text file")
            
            return df
            
        except Exception as e:
            print(f"    ✗ Failed to load text file {file_path}: {e}")
            raise
    
    def _load_parquet_file(self, file_path: str) -> pd.DataFrame:
        """Load data from a parquet file."""
        print(f"    Loading parquet file: {file_path}")
        
        try:
            df = pd.read_parquet(file_path)
            print(f"    ✓ Loaded {len(df)} samples from parquet file")
            print(f"    Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"    ✗ Failed to load parquet file {file_path}: {e}")
            raise
    
    def _load_validation_data(self, val_data_paths: List[str], data_format: str = "auto") -> pd.DataFrame:
        """Load and combine validation data from multiple files (parquet or text)."""
        print(f"Loading validation data from {len(val_data_paths)} file(s):")
        for path in val_data_paths:
            detected_format = self._detect_file_format(path, data_format)
            print(f"  - {path} (format: {detected_format})")
        
        try:
            dataframes = []
            total_samples = 0
            
            for i, path in enumerate(val_data_paths):
                print(f"  Loading file {i+1}/{len(val_data_paths)}: {path}")
                
                # Detect format for this file
                file_format = self._detect_file_format(path, data_format)
                
                # Load based on format
                if file_format == "text":
                    df = self._load_text_file(path)
                elif file_format == "parquet":
                    df = self._load_parquet_file(path)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
                
                dataframes.append(df)
                total_samples += len(df)
                
                # Check column consistency for parquet files
                if i == 0:
                    expected_columns = set(df.columns)
                elif file_format == "parquet":  # Only check consistency for parquet files
                    current_columns = set(df.columns)
                    if current_columns != expected_columns:
                        missing = expected_columns - current_columns
                        extra = current_columns - expected_columns
                        print(f"    Warning: Column mismatch detected")
                        if missing:
                            print(f"      Missing columns: {missing}")
                        if extra:
                            print(f"      Extra columns: {extra}")
            
            # Combine all dataframes
            print(f"Combining {len(dataframes)} dataframes...")
            
            if len(dataframes) == 1:
                combined_df = dataframes[0]
            else:
                # For mixed formats, we need to be more careful about concatenation
                combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
            
            print(f"✓ Successfully combined {total_samples} validation samples into {len(combined_df)} rows")
            print(f"  Final columns: {list(combined_df.columns)}")
            
            return combined_df
            
        except Exception as e:
            print(f"✗ Failed to load validation data: {e}")
            raise
    
    def _sample_validation_prompts(self, df: pd.DataFrame, prompt_key: str, num_samples: int) -> List[str]:
        """Sample prompts from validation dataframe."""
        print(f"Sampling {num_samples} prompts from '{prompt_key}' column...")
        
        if prompt_key not in df.columns:
            raise ValueError(f"Prompt key '{prompt_key}' not found in data. Available columns: {list(df.columns)}")
        
        # Sample random rows
        if len(df) > num_samples:
            sampled_df = df.sample(n=num_samples, random_state=42)
        else:
            sampled_df = df
            print(f"  Note: Only {len(df)} samples available, using all")
        
        # Extract prompts
        prompts = sampled_df[prompt_key].tolist()
        
        # Filter out empty prompts
        prompts = [str(p).strip() for p in prompts if pd.notna(p) and str(p).strip()]
        
        print(f"✓ Successfully extracted {len(prompts)} valid prompts")
        if len(prompts) > 0:
            print(f"  Sample prompt: {prompts[0][:100]}...")
        
        return prompts
    
    def _format_prompts_with_chat_template(self, raw_prompts: List[str], system_template_type: str = "interleave") -> List[str]:
        """Apply chat template with system prompt to raw prompts."""
        print(f"Applying chat template with system template type: {system_template_type}")
        
        # Get system prompt
        try:
            system_prompt = get_system_template(system_template_type)
            print(f"Using system prompt: {system_prompt[:100]}...")
        except Exception as e:
            print(f"Warning: Could not load system template '{system_template_type}': {e}")
            print("Using default interleaved template")
            system_prompt = "You are a helpful assistant that thinks compositionally about complex problems. You conduct your reasoning within <think></think> tags, focusing on just one item or component at a time. You then provide that specific item within <answer></answer> tags, including both the item itself and a brief one-sentence summary explaining why this item qualifies or was chosen."
        
        formatted_prompts = []
        
        for raw_prompt in raw_prompts:
            # Create chat messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_prompt}
            ]
            
            # Apply chat template
            try:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )
                else:
                    # Fallback: simple concatenation 
                    formatted_prompt = f"System: {system_prompt}\n\nUser: {raw_prompt}\n\nAssistant:"
                    print("Warning: Tokenizer doesn't support apply_chat_template, using fallback format")
                
                formatted_prompts.append(formatted_prompt)
                
            except Exception as e:
                print(f"Warning: Failed to apply chat template to prompt: {e}")
                # Fallback: simple concatenation
                formatted_prompt = f"System: {system_prompt}\n\nUser: {raw_prompt}\n\nAssistant:"
                formatted_prompts.append(formatted_prompt)
        
        print(f"✓ Successfully formatted {len(formatted_prompts)} prompts with chat template")
        if len(formatted_prompts) > 0:
            print(f"  Sample formatted prompt: {formatted_prompts[0][:200]}...")
        
        return formatted_prompts
    
    def generate_rollouts(self, 
                         val_data_paths: List[str], 
                         output_dir: Optional[str] = None,
                         step: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate validation rollouts using optimized vLLM batching.
        
        Args:
            val_data_paths: List of validation data file paths
            output_dir: Output directory for rollouts
            step: Training step number
            
        Returns:
            Dictionary with rollout results and metadata
        """
        # Check if model loading was skipped
        if self.skip_model_loading:
            raise RuntimeError(
                "Cannot generate rollouts: Model loading was skipped due to LoRA training configuration. "
                "To generate rollouts during LoRA training, either:\n"
                "1. Set skip_model_loading=False when initializing ValidationRolloutGenerator, or\n"
                "2. Use a separate script with the merged LoRA+base model checkpoint."
            )
        
        # Initialize components
        self._init_vllm_engine()
        self._init_tokenizer()
        
        # Double-check that vLLM engine was initialized
        if self.vllm_engine is None:
            raise RuntimeError(
                "vLLM engine failed to initialize. Please check your checkpoint path and configuration."
            )
        
        # Get data format from rollout config
        data_format = self.rollout_config.get('data_format', 'auto')
        
        # Load validation data
        val_df = self._load_validation_data(val_data_paths, data_format)
        
        # Get prompt key from config, with smart defaults
        prompt_key = 'question'  # Default for parquet files
        if self.config and hasattr(self.config.data, 'prompt_key'):
            prompt_key = self.config.data.prompt_key
        else:
            # Auto-detect prompt key based on available columns
            available_columns = list(val_df.columns)
            if 'prompt' in available_columns:
                prompt_key = 'prompt'  # Text files always use "prompt"
            elif 'question' in available_columns:
                prompt_key = 'question'  # Common for parquet files
            else:
                # Use the first column if no standard prompt column found
                prompt_key = available_columns[0]
                print(f"Warning: No standard prompt column found, using '{prompt_key}'")
        
        # Prepare output directory
        if output_dir is None:
            checkpoint_dir = Path(self.checkpoint_path).parent
            output_dir = checkpoint_dir / "val_rollouts"
            
        if step is not None:
            output_dir = Path(output_dir) / f"step_{step}"
        else:
            # Extract step from checkpoint path if available
            checkpoint_name = Path(self.checkpoint_path).name
            if "global_step_" in checkpoint_name:
                try:
                    step = int(checkpoint_name.split("global_step_")[1])
                    output_dir = Path(output_dir) / f"step_{step}"
                except:
                    pass
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample validation prompts
        num_samples = self.rollout_config.get('num_samples', 50)
        raw_prompts = self._sample_validation_prompts(val_df, prompt_key, num_samples)
        
        if not raw_prompts:
            raise ValueError("No validation prompts found")
        
        # Get system template type from config or use default
        system_template_type = "interleave"  # Default to interleaved reasoning
        if self.config and hasattr(self.config, 'system_template_type'):
            system_template_type = self.config.system_template_type
        elif self.rollout_config and 'system_template_type' in self.rollout_config:
            system_template_type = self.rollout_config['system_template_type']
        
        # Format prompts with chat template and system prompt
        val_prompts = self._format_prompts_with_chat_template(raw_prompts, system_template_type)
        
        # Get trajectory parameters
        num_trajectories = self.rollout_config.get('num_trajectories_per_prompt', 1)
        trajectory_temperature = self.rollout_config.get('trajectory_temperature', self.rollout_config.get('temperature', 0.7))
        
        print(f"🚀 OPTIMIZED vLLM BATCHING ENABLED")
        print(f"Generating rollouts for {len(val_prompts)} prompts with {num_trajectories} trajectories each...")
        print(f"Total generations: {len(val_prompts) * num_trajectories}")
        if num_trajectories > 1:
            print(f"Trajectory sampling temperature: {trajectory_temperature}")
        
        # OPTIMIZATION 1: Use vLLM's native n-sampling instead of sequential generation
        if num_trajectories > 1:
            print(f"✓ Using vLLM native n-sampling for {num_trajectories} trajectories per prompt")
            
            # Prepare sampling parameters with native trajectory sampling
            sampling_params = SamplingParams(
                temperature=trajectory_temperature,
                top_p=self.rollout_config.get('top_p', 0.9),
                max_tokens=self.rollout_config.get('max_tokens', 2048),
                stop=self.rollout_config.get('stop_tokens', None),
                n=num_trajectories,  # Generate all trajectories at once per prompt
                seed=None if trajectory_temperature > 0 else 42  # Deterministic if temp=0
            )
            
            # OPTIMIZATION 2: Full batch processing - all prompts at once
            print(f"✓ Processing all {len(val_prompts)} prompts in single vLLM batch")
            start_time = time.time()
            
            # Generate all responses in one batch call
            batch_outputs = self.vllm_engine.generate(val_prompts, sampling_params)
            
            batch_time = time.time() - start_time
            total_trajectories = len(batch_outputs) * num_trajectories
            print(f"✓ Generated {total_trajectories} trajectories in {batch_time:.2f}s ({total_trajectories/batch_time:.1f} traj/s)")
            
            # Process batch outputs into individual trajectory records
            all_outputs = []
            all_prompts_expanded = []
            all_raw_prompts_expanded = []
            trajectory_indices = []
            
            for prompt_idx, (raw_prompt, formatted_prompt, output) in enumerate(zip(raw_prompts, val_prompts, batch_outputs)):
                for traj_idx, completion in enumerate(output.outputs):
                    # Create individual output record for each trajectory
                    single_output = type(output)(
                        request_id=f"{output.request_id}_{traj_idx}",
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=[completion],  # Single completion for this trajectory
                        finished=output.finished
                    )
                    
                    all_outputs.append(single_output)
                    all_prompts_expanded.append(formatted_prompt)
                    all_raw_prompts_expanded.append(raw_prompt)
                    trajectory_indices.append((prompt_idx, traj_idx))
            
        else:
            # Single trajectory per prompt - still use full batching
            print(f"✓ Single trajectory mode - processing all {len(val_prompts)} prompts in one batch")
            
            sampling_params = SamplingParams(
                temperature=trajectory_temperature,
                top_p=self.rollout_config.get('top_p', 0.9),
                max_tokens=self.rollout_config.get('max_tokens', 2048),
                stop=self.rollout_config.get('stop_tokens', None),
                n=1,
                seed=None if trajectory_temperature > 0 else 42
            )
            
            start_time = time.time()
            batch_outputs = self.vllm_engine.generate(val_prompts, sampling_params)
            batch_time = time.time() - start_time
            print(f"✓ Generated {len(batch_outputs)} responses in {batch_time:.2f}s ({len(batch_outputs)/batch_time:.1f} resp/s)")
            
            # Simple mapping for single trajectory
            all_outputs = batch_outputs
            all_prompts_expanded = val_prompts
            all_raw_prompts_expanded = raw_prompts
            trajectory_indices = [(i, 0) for i in range(len(batch_outputs))]
        
        print(f"✓ Total trajectories generated: {len(all_outputs)}")
        
        # Update references for consistent processing
        val_prompts = all_prompts_expanded
        raw_prompts = all_raw_prompts_expanded
        outputs = all_outputs
        
        # Process and save rollouts
        rollouts = []
        for i, (raw_prompt, formatted_prompt, output, (prompt_idx, traj_idx)) in enumerate(zip(raw_prompts, val_prompts, outputs, trajectory_indices)):
            generated_text = output.outputs[0].text
            
            rollout_data = {
                'rollout_id': i,
                'prompt_id': prompt_idx,
                'trajectory_id': traj_idx,
                'raw_prompt': raw_prompt,
                'formatted_prompt': formatted_prompt,
                'prompt': formatted_prompt,  # Keep for backwards compatibility
                'generated_response': generated_text,
                'checkpoint_path': self.checkpoint_path,
                'step': step,
                'system_template_type': system_template_type,
                'generation_params': {
                    'temperature': trajectory_temperature,
                    'top_p': self.rollout_config.get('top_p', 0.9),
                    'max_tokens': self.rollout_config.get('max_tokens', 2048),
                    'num_trajectories_per_prompt': num_trajectories,
                    'batching_optimization': 'enabled',
                    'batch_size': len(val_prompts) // num_trajectories if num_trajectories > 1 else len(val_prompts),
                },
                'model_config': self.vllm_config,
            }
            rollouts.append(rollout_data)
        
        # Save rollouts to file
        rollout_file = output_dir / 'rollouts.jsonl'
        with open(rollout_file, 'w', encoding='utf-8') as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved {len(rollouts)} validation rollouts to {rollout_file}")
        
        # Calculate unique prompts
        unique_prompts = len(set(r['prompt_id'] for r in rollouts))
        
        # Save metadata
        metadata = {
            'checkpoint_path': self.checkpoint_path,
            'val_data_paths': val_data_paths,
            'prompt_key': prompt_key,
            'num_rollouts': len(rollouts),
            'num_unique_prompts': unique_prompts,
            'num_trajectories_per_prompt': num_trajectories,
            'total_trajectories': len(rollouts),
            'step': step,
            'system_template_type': system_template_type,
            'vllm_config': self.vllm_config,
            'rollout_config': self.rollout_config,
            'output_dir': str(output_dir),
            'optimization_enabled': True,
            'batching_mode': 'native_n_sampling' if num_trajectories > 1 else 'full_batch',
        }
        
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved metadata to {metadata_file}")
        
        return {
            'rollouts': rollouts,
            'metadata': metadata,
            'output_dir': str(output_dir),
            'rollout_file': str(rollout_file),
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self._cleanup_merged_model()


def main():
    parser = argparse.ArgumentParser(description="Generate validation rollouts from model checkpoints or base models")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Path to model checkpoint directory (or LoRA adapter). If not provided, uses --base_model_path")
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="Path to base model (required for LoRA checkpoints, e.g., Qwen/Qwen3-8B). Used as fallback if --checkpoint_dir not provided")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to training config file (optional)")
    parser.add_argument("--val_data", type=str, default=None,
                       help="Validation data file(s). Single: 'file.parquet' or multiple: \"['file1.parquet','file2.parquet']\" or text files: 'prompts.txt'")
    parser.add_argument("--val_data_format", type=str, default="auto", choices=["auto", "parquet", "text"],
                       help="Format of validation data files (auto-detect from extension if 'auto')")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (defaults to checkpoint_dir/val_rollouts)")
    parser.add_argument("--step", type=int, default=None,
                       help="Step number for output organization")
    parser.add_argument("--skip_model_loading", action="store_true",
                       help="Skip model loading (useful during LoRA training)")
    parser.add_argument("--force_model_loading", action="store_true",
                       help="Force model loading even during LoRA training")
    
    # vLLM configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                       help="Tensor parallel size for vLLM (auto-detect if not specified)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       help="Model dtype for vLLM")
    parser.add_argument("--max_model_len", type=int, default=4096,
                       help="Maximum model sequence length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization ratio (0.9 for optimal batching)")
    parser.add_argument("--enforce_eager", action="store_true",
                       help="Enforce eager execution (disable CUDA graphs - reduces performance)")
    parser.add_argument("--disable_custom_all_reduce", action="store_true",
                       help="Disable custom all-reduce for multi-GPU setups")
    parser.add_argument("--max_num_batched_tokens", type=int, default=None,
                       help="Maximum number of batched tokens (auto-configured for optimal batching)")
    parser.add_argument("--distributed_executor_backend", type=str, default="ray",
                       choices=["ray", "mp"], help="Distributed executor backend")
    parser.add_argument("--enable_chunked_prefill", action="store_true", default=True,
                       help="Enable chunked prefill for memory efficiency (default: True)")
    parser.add_argument("--disable_chunked_prefill", action="store_false", dest="enable_chunked_prefill",
                       help="Disable chunked prefill")
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True,
                       help="Enable prefix caching for throughput optimization (default: True)")
    parser.add_argument("--disable_prefix_caching", action="store_false", dest="enable_prefix_caching",
                       help="Disable prefix caching")
    parser.add_argument("--disable_log_stats", action="store_true", default=True,
                       help="Disable vLLM logging to reduce overhead (default: True)")
    parser.add_argument("--enable_log_stats", action="store_false", dest="disable_log_stats",
                       help="Enable vLLM logging")
    
    # Generation configuration
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of validation samples to generate")
    parser.add_argument("--max_tokens", type=int, default=4096,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Sampling temperature (0.0 for deterministic)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling parameter (1.0 disables top-p)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic sampling (temperature=0, top_p=1.0)")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic sampling (overrides --deterministic)")
    parser.add_argument("--system_template_type", type=str, default="interleave",
                       choices=["tool", "tool_interleaved", "confidence", "default", "think_answer", "interleave", "code"],
                       help="System template type to use (default: interleave)")
    parser.add_argument("--num_trajectories_per_prompt", type=int, default=1,
                       help="Number of trajectories to sample per prompt (default: 1)")
    parser.add_argument("--trajectory_temperature", type=float, default=None,
                       help="Temperature for trajectory sampling (overrides main temperature for multi-trajectory mode)")
    
    args = parser.parse_args()
    
    # Determine which model path to use
    if args.checkpoint_dir is None and args.base_model_path is None:
        raise ValueError("Must provide either --checkpoint_dir or --base_model_path")
    
    # Use base_model_path as fallback if checkpoint_dir is not provided
    if args.checkpoint_dir is None:
        print(f"No checkpoint directory provided, using base model: {args.base_model_path}")
        checkpoint_path = args.base_model_path
        base_model_path = None  # Don't need separate base model when using it directly
    else:
        print(f"Using checkpoint: {args.checkpoint_dir}")
        if args.base_model_path:
            print(f"Base model for LoRA: {args.base_model_path}")
        checkpoint_path = args.checkpoint_dir
        base_model_path = args.base_model_path
    
    # Handle deterministic vs stochastic sampling
    if args.stochastic:
        # Use stochastic sampling with provided or default values
        temperature = args.temperature if args.temperature > 0 else 0.7
        top_p = args.top_p if args.top_p < 1.0 else 0.9
    elif args.deterministic:
        # Force deterministic sampling
        temperature = 0.0
        top_p = 1.0
    else:
        # Use provided values
        temperature = args.temperature
        top_p = args.top_p
    
    # Determine trajectory sampling parameters early
    num_trajectories = args.num_trajectories_per_prompt
    trajectory_temperature = args.trajectory_temperature
    
    # If multiple trajectories requested but no temperature specified, use stochastic sampling
    if num_trajectories > 1 and trajectory_temperature is None:
        trajectory_temperature = 0.7 if temperature == 0.0 else temperature
    elif trajectory_temperature is not None:
        pass  # Use custom trajectory temperature
    else:
        trajectory_temperature = temperature
    
    print(f"Sampling mode: {'Deterministic' if temperature == 0.0 else 'Stochastic'}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"System template type: {args.system_template_type}")
    print(f"Trajectories per prompt: {num_trajectories}")
    if num_trajectories > 1:
        print(f"Trajectory temperature: {trajectory_temperature}")
        print(f"Expected total trajectories: {args.num_samples * num_trajectories}")
    
    # Load config if provided
    config = None
    if args.config:
        try:
            config = OmegaConf.load(args.config)
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Failed to load config: {e}")
    
    # Determine validation data paths
    val_data_input = args.val_data
    if val_data_input is None and config and hasattr(config.data, 'val_files'):
        val_data_input = config.data.val_files
    
    if val_data_input is None:
        raise ValueError("No validation data specified. Use --val_data or provide config with data.val_files")
    
    # Parse validation data paths
    if isinstance(val_data_input, str):
        # Check if it's a list format
        if val_data_input.startswith('[') and val_data_input.endswith(']'):
            # Remove brackets and split by comma
            inner_content = val_data_input[1:-1].strip()
            if inner_content:
                # Split by comma and clean up each path
                val_data_paths = [path.strip() for path in inner_content.split(',')]
            else:
                val_data_paths = []
        else:
            # Single file path
            val_data_paths = [val_data_input]
    elif isinstance(val_data_input, (list, tuple)):
        # From config file
        val_data_paths = list(val_data_input)
    else:
        val_data_paths = [str(val_data_input)]
    
    print(f"Validation data files ({len(val_data_paths)}):")
    for i, path in enumerate(val_data_paths):
        if args.val_data_format == "auto":
            # Show auto-detected format
            if path.lower().endswith('.txt'):
                format_info = "text (auto-detected)"
            elif path.lower().endswith('.parquet'):
                format_info = "parquet (auto-detected)"
            else:
                format_info = "parquet (default)"
        else:
            format_info = args.val_data_format
        print(f"  {i+1}: {path} ({format_info})")
    
    if args.val_data_format == "text" or any(path.lower().endswith('.txt') for path in val_data_paths):
        print(f"Text files: Each line becomes a prompt (stored in 'prompt' column)")
    
    # Build vLLM config with auto-detection support and batching optimizations
    vllm_config = {
        'dtype': args.dtype,
        'max_model_len': args.max_model_len,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'enforce_eager': args.enforce_eager,
        'disable_custom_all_reduce': args.disable_custom_all_reduce,
        'distributed_executor_backend': args.distributed_executor_backend,
        'enable_chunked_prefill': args.enable_chunked_prefill,
        'enable_prefix_caching': args.enable_prefix_caching,
        'disable_log_stats': args.disable_log_stats,
    }
    
    # Only set tensor_parallel_size if explicitly provided
    if args.tensor_parallel_size is not None:
        vllm_config['tensor_parallel_size'] = args.tensor_parallel_size
        
    # Only set max_num_batched_tokens if provided (otherwise auto-configured)
    if args.max_num_batched_tokens is not None:
        vllm_config['max_num_batched_tokens'] = args.max_num_batched_tokens
    
    # Override with config values if available
    if config and hasattr(config.trainer, 'vllm_config'):
        vllm_config.update(config.trainer.vllm_config)
    
    print(f"\n🚀 vLLM BATCHING OPTIMIZATIONS:")
    print(f"  GPU memory utilization: {vllm_config['gpu_memory_utilization']}")
    print(f"  CUDA graphs: {'Disabled (--enforce_eager)' if vllm_config['enforce_eager'] else 'Enabled'}")
    print(f"  Chunked prefill: {'Enabled' if vllm_config['enable_chunked_prefill'] else 'Disabled'}")
    print(f"  Prefix caching: {'Enabled' if vllm_config['enable_prefix_caching'] else 'Disabled'}")
    print(f"  Log stats: {'Disabled' if vllm_config['disable_log_stats'] else 'Enabled'}")
    if args.max_num_batched_tokens:
        print(f"  Max batched tokens: {args.max_num_batched_tokens}")
    else:
        print(f"  Max batched tokens: Auto-configured based on model size")
    
    # Print trajectory configuration messages
    if num_trajectories > 1 and args.trajectory_temperature is None:
        print(f"Multiple trajectories requested: using temperature {trajectory_temperature} for diversity")
    elif args.trajectory_temperature is not None:
        print(f"Using custom trajectory temperature: {trajectory_temperature}")
    
    # Build rollout config with determined sampling parameters
    rollout_config = {
        'num_samples': args.num_samples,
        'max_tokens': args.max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'stop_tokens': None,
        'system_template_type': args.system_template_type,
        'num_trajectories_per_prompt': num_trajectories,
        'trajectory_temperature': trajectory_temperature,
        'data_format': args.val_data_format,
    }
    
    # Override with config values if available (but preserve deterministic settings)
    if config and hasattr(config.trainer, 'rollout_config'):
        config_rollout = dict(config.trainer.rollout_config)
        # Don't override temperature/top_p if we're using deterministic mode
        if args.deterministic and not args.stochastic:
            config_rollout.pop('temperature', None)
            config_rollout.pop('top_p', None)
        rollout_config.update(config_rollout)
    
    # Determine skip_model_loading setting
    skip_model_loading = None
    if args.force_model_loading:
        skip_model_loading = False
    elif args.skip_model_loading:
        skip_model_loading = True
    # If neither flag is set, let the generator auto-detect based on LoRA config
    
    # Initialize generator
    generator = ValidationRolloutGenerator(
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
        config=config,
        vllm_config=vllm_config,
        rollout_config=rollout_config,
        skip_model_loading=skip_model_loading
    )
    
    # Generate rollouts
    try:
        results = generator.generate_rollouts(
            val_data_paths=val_data_paths,
            output_dir=args.output_dir,
            step=args.step
        )
        
        print(f"\n{'='*60}")
        print("ROLLOUT GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Model path: {checkpoint_path}")
        if base_model_path:
            print(f"Base model: {base_model_path}")
        print(f"System template: {results['metadata']['system_template_type']}")
        print(f"Unique prompts: {results['metadata']['num_unique_prompts']}")
        print(f"Trajectories per prompt: {results['metadata']['num_trajectories_per_prompt']}")
        print(f"Total trajectories: {results['metadata']['total_trajectories']}")
        print(f"Output directory: {results['output_dir']}")
        print(f"Rollout file: {results['rollout_file']}")
        
        # Show trajectory distribution
        if results['metadata']['num_trajectories_per_prompt'] > 1:
            print(f"\nTrajectory organization:")
            print(f"  Each prompt has {results['metadata']['num_trajectories_per_prompt']} different trajectories")
            print(f"  Use 'prompt_id' to group trajectories by prompt")
            print(f"  Use 'trajectory_id' to identify different trajectories for the same prompt")
        
    except Exception as e:
        print(f"Error generating rollouts: {e}")
        raise e
    finally:
        # Explicit cleanup
        generator._cleanup_merged_model()


if __name__ == "__main__":
    main() 