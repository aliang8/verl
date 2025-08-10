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
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

import torch
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

from verl.utils.templates import get_system_template

logger = logging.getLogger(__name__)

# Static vLLM configuration
VLLM_CONFIG = {
    'dtype': 'bfloat16',
    'max_model_len': 4096,
    'gpu_memory_utilization': 0.9,
    'enforce_eager': False,
    'disable_custom_all_reduce': False,
    'max_num_batched_tokens': 65536,
    'distributed_executor_backend': 'ray',
    'enable_chunked_prefill': True,
    'enable_prefix_caching': True,
    'disable_log_stats': True,
    'trust_remote_code': True,
}

# Static generation configuration
GENERATION_CONFIG = {
    'num_samples': 10,
    'max_tokens': 4096,
    'temperature': 0.7,
    'top_p': 0.9,
    'stop_tokens': None,
    'system_template_type': 'plan_first',
    'num_trajectories_per_prompt': 1,
    'data_format': 'auto',
}


class ValidationRolloutGenerator:
    """Generate validation rollouts from model checkpoints using vLLM."""
    
    def __init__(self, 
                 checkpoint_path: str,
                 config: Optional[DictConfig] = None,
                 rollout_config: Optional[Dict] = None):
        """
        Initialize the rollout generator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Training configuration (optional)
            rollout_config: Rollout generation configuration
        """
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.rollout_config = rollout_config or {}
        
        # Initialize components (lazy loading)
        self.vllm_engine = None
        self.tokenizer = None
        
        # Auto-detect tensor parallel size
        available_gpus = torch.cuda.device_count()
        self.vllm_config = VLLM_CONFIG.copy()
        self.vllm_config['tensor_parallel_size'] = min(available_gpus, 4)
        
        print(f"Initialized ValidationRolloutGenerator:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Tensor parallel size: {self.vllm_config['tensor_parallel_size']}")
        print(f"  Available GPUs: {torch.cuda.device_count()}")
    
    def _is_huggingface_model_id(self, model_path: str) -> bool:
        """Check if the model path is a Hugging Face model identifier."""
        return (
            "/" in model_path and
            not Path(model_path).exists() and
            not model_path.startswith("./") and
            not model_path.startswith("../") and
            not model_path.startswith("/")
        )
    
    def _init_vllm_engine(self):
        """Initialize vLLM engine with checkpoint."""
        if self.vllm_engine is not None:
            return
        
        print("Initializing vLLM engine...")
        
        try:
            # Add model path to config
            engine_config = self.vllm_config.copy()
            engine_config['model'] = self.checkpoint_path
            
            print(f"vLLM config: {engine_config}")
            
            # Initialize vLLM engine
            self.vllm_engine = LLM(**engine_config)
            print("✓ vLLM engine initialized successfully")
            
        except Exception as e:
            print(f"✗ Failed to initialize vLLM engine: {e}")
            raise
    
    def _init_tokenizer(self):
        """Initialize tokenizer for prompt processing."""
        if self.tokenizer is not None:
            return
            
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint_path,
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
        
        path_lower = file_path.lower()
        if path_lower.endswith('.txt'):
            return "text"
        elif path_lower.endswith('.parquet'):
            return "parquet"
        else:
            print(f"Warning: Unknown file extension for {file_path}, assuming parquet format")
            return "parquet"
    
    def _load_text_file(self, file_path: str) -> pd.DataFrame:
        """Load prompts from a newline-separated text file."""
        print(f"    Loading text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            prompts = []
            for line in lines:
                line = line.strip()
                if line:
                    prompts.append(line)
            
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
        """Load and combine validation data from multiple files."""
        print(f"Loading validation data from {len(val_data_paths)} file(s):")
        
        try:
            dataframes = []
            total_samples = 0
            
            for i, path in enumerate(val_data_paths):
                print(f"  Loading file {i+1}/{len(val_data_paths)}: {path}")
                
                file_format = self._detect_file_format(path, data_format)
                
                if file_format == "text":
                    df = self._load_text_file(path)
                elif file_format == "parquet":
                    df = self._load_parquet_file(path)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
                
                dataframes.append(df)
                total_samples += len(df)
            
            # Combine all dataframes
            if len(dataframes) == 1:
                combined_df = dataframes[0]
            else:
                combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
            
            print(f"✓ Successfully combined {total_samples} validation samples")
            return combined_df
            
        except Exception as e:
            print(f"✗ Failed to load validation data: {e}")
            raise
    
    def _sample_validation_prompts(self, df: pd.DataFrame, prompt_key: str, num_samples: int) -> List[str]:
        """Sample prompts from validation dataframe."""
        print(f"Sampling {num_samples} prompts from '{prompt_key}' column...")
        
        if prompt_key not in df.columns:
            raise ValueError(f"Prompt key '{prompt_key}' not found in data. Available columns: {list(df.columns)}")
        
        if len(df) > num_samples:
            sampled_df = df.sample(n=num_samples, random_state=42)
        else:
            sampled_df = df
            print(f"  Note: Only {len(df)} samples available, using all")
        
        prompts = sampled_df[prompt_key].tolist()
        # Remove ADDITIONAL_INSTRUCTION from prompts if present
        ADDITIONAL_INSTRUCTION = """First, outline the solution in a markdown format.\nThen, write the code to implement the solution.\nFinally, generate unit tests to test the code. Format the unit tests as a python function with a docstring. Use this exact format:\n```python\nimport unittest\nfrom task_func import task_func\n\nclass Test(unittest.TestCase):\n    def test_case_1(self):\n        # Test case 1 description\n        result = task_func(...)\n        self.assertEqual(result, expected_value)\n```"""
        prompts = [prompt.replace(ADDITIONAL_INSTRUCTION, "").strip() for prompt in prompts]
            
        prompts = [str(p).strip() for p in prompts if pd.notna(p) and str(p).strip()]
        print(f"✓ Successfully extracted {len(prompts)} valid prompts")
        return prompts
    
    def _format_prompts_with_chat_template(self, raw_prompts: List[str], system_template_type: str = "interleave") -> List[str]:
        """Apply chat template with system prompt to raw prompts."""
        print(f"Applying chat template with system template type: {system_template_type}")
        
        try:
            system_prompt = get_system_template(system_template_type)
            print(f"Using system prompt: {system_prompt[:100]}...")
        except Exception as e:
            print(f"Warning: Could not load system template '{system_template_type}': {e}")
            system_prompt = "You are a helpful assistant that thinks compositionally about complex problems. You conduct your reasoning within <think></think> tags, focusing on just one item or component at a time. You then provide that specific item within <answer></answer> tags, including both the item itself and a brief one-sentence summary explaining why this item qualifies or was chosen."
        
        formatted_prompts = []
        
        for raw_prompt in raw_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_prompt}
            ]
            
            try:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )
                else:
                    formatted_prompt = f"System: {system_prompt}\n\nUser: {raw_prompt}\n\nAssistant:"
                
                formatted_prompts.append(formatted_prompt)
                
            except Exception as e:
                print(f"Warning: Failed to apply chat template to prompt: {e}")
                formatted_prompt = f"System: {system_prompt}\n\nUser: {raw_prompt}\n\nAssistant:"
                formatted_prompts.append(formatted_prompt)
        
        print(f"✓ Successfully formatted {len(formatted_prompts)} prompts")
        return formatted_prompts
    
    def generate_rollouts(self, 
                         val_data_paths: List[str], 
                         output_dir: Optional[str] = None,
                         step: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate validation rollouts using vLLM.
        
        Args:
            val_data_paths: List of validation data file paths
            output_dir: Output directory for rollouts
            step: Training step number
            
        Returns:
            Dictionary with rollout results and metadata
        """
        # Initialize components
        self._init_vllm_engine()
        self._init_tokenizer()
        
        # Load validation data
        data_format = self.rollout_config.get('data_format', GENERATION_CONFIG['data_format'])
        val_df = self._load_validation_data(val_data_paths, data_format)
        
        # Get prompt key
        prompt_key = 'question'  # Default for parquet files
        if self.config and hasattr(self.config.data, 'prompt_key'):
            prompt_key = self.config.data.prompt_key
        else:
            available_columns = list(val_df.columns)
            if 'prompt' in available_columns:
                prompt_key = 'prompt'
            elif 'question' in available_columns:
                prompt_key = 'question'
            else:
                prompt_key = available_columns[0]
                print(f"Warning: No standard prompt column found, using '{prompt_key}'")
        
        # Prepare output directory
        if output_dir is None:
            checkpoint_dir = Path(self.checkpoint_path).parent
            output_dir = checkpoint_dir / "val_rollouts"
            
        if step is not None:
            output_dir = Path(output_dir) / f"step_{step}"
        else:
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
        num_samples = self.rollout_config.get('num_samples', GENERATION_CONFIG['num_samples'])
        raw_prompts = self._sample_validation_prompts(val_df, prompt_key, num_samples)
        
        if not raw_prompts:
            raise ValueError("No validation prompts found")
        
        # Get system template type
        system_template_type = self.rollout_config.get('system_template_type', GENERATION_CONFIG['system_template_type'])
        if self.config and hasattr(self.config, 'system_template_type'):
            system_template_type = self.config.system_template_type
        
        # Format prompts with chat template
        val_prompts = self._format_prompts_with_chat_template(raw_prompts, system_template_type)
        
        # Get generation parameters
        num_trajectories = self.rollout_config.get('num_trajectories_per_prompt', GENERATION_CONFIG['num_trajectories_per_prompt'])
        temperature = self.rollout_config.get('temperature', GENERATION_CONFIG['temperature'])
        max_tokens = self.rollout_config.get('max_tokens', GENERATION_CONFIG['max_tokens'])
        top_p = self.rollout_config.get('top_p', GENERATION_CONFIG['top_p'])
        
        print(f"Generating rollouts for {len(val_prompts)} prompts with {num_trajectories} trajectories each...")
        
        # Generate responses
        if num_trajectories > 1:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=self.rollout_config.get('stop_tokens', GENERATION_CONFIG['stop_tokens']),
                n=num_trajectories,
                seed=None if temperature > 0 else 42
            )
            
            start_time = time.time()
            batch_outputs = self.vllm_engine.generate(val_prompts, sampling_params)
            batch_time = time.time() - start_time
            
            # Process batch outputs
            all_outputs = []
            all_prompts_expanded = []
            all_raw_prompts_expanded = []
            trajectory_indices = []
            
            for prompt_idx, (raw_prompt, formatted_prompt, output) in enumerate(zip(raw_prompts, val_prompts, batch_outputs)):
                for traj_idx, completion in enumerate(output.outputs):
                    single_output = type(output)(
                        request_id=f"{output.request_id}_{traj_idx}",
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=[completion],
                        finished=output.finished
                    )
                    
                    all_outputs.append(single_output)
                    all_prompts_expanded.append(formatted_prompt)
                    all_raw_prompts_expanded.append(raw_prompt)
                    trajectory_indices.append((prompt_idx, traj_idx))
            
        else:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=self.rollout_config.get('stop_tokens', GENERATION_CONFIG['stop_tokens']),
                n=1,
                seed=None if temperature > 0 else 42
            )
            
            start_time = time.time()
            batch_outputs = self.vllm_engine.generate(val_prompts, sampling_params)
            batch_time = time.time() - start_time
            
            all_outputs = batch_outputs
            all_prompts_expanded = val_prompts
            all_raw_prompts_expanded = raw_prompts
            trajectory_indices = [(i, 0) for i in range(len(batch_outputs))]
        
        print(f"✓ Generated {len(all_outputs)} trajectories in {batch_time:.2f}s")
        
        # Process and save rollouts
        rollouts = []
        for i, (raw_prompt, formatted_prompt, output, (prompt_idx, traj_idx)) in enumerate(zip(all_raw_prompts_expanded, all_prompts_expanded, all_outputs, trajectory_indices)):
            generated_text = output.outputs[0].text
            
            rollout_data = {
                'rollout_id': i,
                'prompt_id': prompt_idx,
                'trajectory_id': traj_idx,
                'raw_prompt': raw_prompt,
                'formatted_prompt': formatted_prompt,
                'prompt': formatted_prompt,
                'generated_response': generated_text,
                'checkpoint_path': self.checkpoint_path,
                'step': step,
                'system_template_type': system_template_type,
                'generation_params': {
                    'temperature': temperature,
                    'top_p': top_p,
                    'max_tokens': max_tokens,
                    'num_trajectories_per_prompt': num_trajectories,
                },
            }
            rollouts.append(rollout_data)
        
        # Determine data source from validation data paths
        data_source = "unknown"
        if val_data_paths:
            # Extract data source from the first file path
            first_file = Path(val_data_paths[0]).stem
            if "_" in first_file:
                data_source = first_file.split("_")[0]
            else:
                data_source = first_file
        
        # Save rollouts to file with data source in filename
        rollout_file = output_dir / f'{data_source}_rollouts.jsonl'
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
            'data_source': data_source,
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


def main():
    parser = argparse.ArgumentParser(description="Generate validation rollouts from model checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Path to model checkpoint directory")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to training config file (optional)")
    parser.add_argument("--val_data", type=str, default=None,
                       help="Validation data file(s). Single: 'file.parquet' or multiple: \"['file1.parquet','file2.parquet']\"")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (defaults to checkpoint_dir/val_rollouts)")
    parser.add_argument("--step", type=int, default=None,
                       help="Step number for output organization")
    
    args = parser.parse_args()
    
    print(f"Using checkpoint: {args.checkpoint_dir}")
    print(f"Generation config: {GENERATION_CONFIG}")
    
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
        if val_data_input.startswith('[') and val_data_input.endswith(']'):
            inner_content = val_data_input[1:-1].strip()
            if inner_content:
                val_data_paths = [path.strip() for path in inner_content.split(',')]
            else:
                val_data_paths = []
        else:
            val_data_paths = [val_data_input]
    elif isinstance(val_data_input, (list, tuple)):
        val_data_paths = list(val_data_input)
    else:
        val_data_paths = [str(val_data_input)]
    
    print(f"Validation data files ({len(val_data_paths)}):")
    for i, path in enumerate(val_data_paths):
        print(f"  {i+1}: {path}")
    
    # Use global generation config
    rollout_config = GENERATION_CONFIG.copy()
    
    # Override with config values if available
    if config and hasattr(config.trainer, 'rollout_config'):
        config_rollout = dict(config.trainer.rollout_config)
        rollout_config.update(config_rollout)
    
    # Initialize generator
    generator = ValidationRolloutGenerator(
        checkpoint_path=args.checkpoint_dir,
        config=config,
        rollout_config=rollout_config
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
        print(f"Model path: {args.checkpoint_dir}")
        print(f"System template: {results['metadata']['system_template_type']}")
        print(f"Unique prompts: {results['metadata']['num_unique_prompts']}")
        print(f"Trajectories per prompt: {results['metadata']['num_trajectories_per_prompt']}")
        print(f"Total trajectories: {results['metadata']['total_trajectories']}")
        print(f"Output directory: {results['output_dir']}")
        print(f"Rollout file: {results['rollout_file']}")
        
    except Exception as e:
        print(f"Error generating rollouts: {e}")
        raise e


if __name__ == "__main__":
    main() 