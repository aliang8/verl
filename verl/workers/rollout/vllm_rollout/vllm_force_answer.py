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
"""
vLLM Rollout with Interrupted Thinking

This rollout forces the model to generate an answer by explicitly adding </think>
and continuing generation when it reaches max response length, ensuring the model
completes its response with both thinking and answer components.
"""

import logging
import os
from typing import List, Dict, Any

import numpy as np
import torch

from .vllm_rollout_spmd import vLLMRollout
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from tensordict import TensorDict
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove the left padding in the prompt token_id."""
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMForceAnswerRollout(vLLMRollout):
    """
    vLLM Rollout that forces the model to generate an answer.
    
    This rollout:
    1. Generates up to max response length
    2. If </think> is not present, explicitly adds it
    3. Continues generation for additional tokens to complete the answer
    4. Returns the complete response with both thinking and answer parts
    """
    
    def __init__(self, model_path: str, config, tokenizer, model_hf_config, **kwargs):
        """Initialize vLLM Force Answer Rollout.
        
        Args:
            model_path: Path to the model
            config: Configuration dictionary
            tokenizer: The tokenizer
            model_hf_config: HuggingFace model configuration
            **kwargs: Additional arguments
        """
        print(f"Initializing vLLMForceAnswerRollout!!!")
        
        # Call parent constructor to initialize vLLM engine and other components
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        
        # Configuration for interrupt thinking
        self.additional_answer_tokens = config.get("additional_answer_tokens", 1024)
        self.force_thinking_end = config.get("force_thinking_end", True)
        
        print(f"Interrupt thinking config: additional_answer_tokens={self.additional_answer_tokens}, force_thinking_end={self.force_thinking_end}")

    def _check_thinking_complete(self, response_text: str) -> bool:
        """Check if thinking is complete (contains </think>)."""
        return "</think>" in response_text

    def _force_thinking_end(self, response_text: str) -> str:
        """Force thinking to end by adding </think> if not present."""
        if not self._check_thinking_complete(response_text):
            # Add </think> to force the model to stop thinking
            response_text += "</think>"
            logger.info("Forced thinking to end by adding </think>")
        return response_text

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences with interrupted thinking capability."""
        # rebuild vllm cache engine
        if (
            hasattr(self, 'config') and hasattr(self.config, 'free_cache_engine') 
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not working properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        if not do_sample:
            reg_kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            reg_kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            reg_kwargs = {}

        kwargs = {**reg_kwargs, **kwargs}

        lora_requests = None
        if hasattr(self, 'lora_kwargs') and self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # First generation: up to max response length
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # Collect first generation responses
            first_responses = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    first_responses.append(response_ids)

        # Check if thinking is complete and force completion if needed
        completed_responses = []
        
        num_forced_thinking_end = 0
        total_tokens_generated = []  # Track total tokens for each prompt
        
        # First pass: decide which need continuation and prepare batched inputs
        to_continue_indices = []
        batched_continuation_inputs = []
        batched_forced_ids = []
        batched_lora_requests = []

        for i, response_ids in enumerate(first_responses):
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Initialize token count for this prompt
            prompt_total_tokens = len(response_ids)
            total_tokens_generated.append(prompt_total_tokens)
            
            # Check if thinking is complete
            if self._check_thinking_complete(response_text):
                # Thinking is already complete, no need for additional generation
                completed_responses.append(response_ids)
                logger.debug(f"Response {i}: Thinking already complete ({prompt_total_tokens} tokens)")
            else:
                if self.force_thinking_end:
                    # Prepare forced </think> and enqueue for batched continuation
                    forced_response = self._force_thinking_end(response_text)
                    forced_ids = self.tokenizer.encode(forced_response, add_special_tokens=False)
                    continuation_input = {"prompt_token_ids": vllm_inputs[i]["prompt_token_ids"] + forced_ids}
                    
                    to_continue_indices.append(i)
                    batched_continuation_inputs.append(continuation_input)
                    batched_forced_ids.append(forced_ids)
                    if lora_requests:
                        batched_lora_requests.append(lora_requests[i])
                    
                    # Placeholder; will be replaced after batched generation
                    completed_responses.append(None)
                else:
                    # Don't force thinking end, just use original response
                    completed_responses.append(response_ids)
        
        # Check if any responses reached max length without generating a plan (no <answer> tags)
        # This can happen if the model hits max length during initial generation
        max_length_without_plan_indices = []
        for i, response_ids in enumerate(first_responses):
            if response_ids is not None:  # Skip responses that are already being continued
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # Check if response has <answer> tags (indicating a plan was generated)
                has_answer_tags = "<answer>" in response_text and "</answer>" in response_text
                
                # If no answer tags and response is long, it likely hit max length without generating a plan
                if not has_answer_tags and len(response_ids) >= self.sampling_params.max_tokens:
                    max_length_without_plan_indices.append(i)
                    print(f"Response {i}: Reached max length without generating plan, will force answer completion")
        
        # Add max length without plan responses to continuation list
        if max_length_without_plan_indices:
            print(f"Found {len(max_length_without_plan_indices)} responses that reached max length without plan generation")
            
            for i in max_length_without_plan_indices:
                if i not in to_continue_indices:  # Avoid duplicates
                    response_text = self.tokenizer.decode(first_responses[i], skip_special_tokens=True)
                    
                    # Force thinking end and prepare for continuation
                    forced_response = self._force_thinking_end(response_text)
                    forced_ids = self.tokenizer.encode(forced_response, add_special_tokens=False)
                    continuation_input = {"prompt_token_ids": vllm_inputs[i]["prompt_token_ids"] + forced_ids}
                    
                    to_continue_indices.append(i)
                    batched_continuation_inputs.append(continuation_input)
                    batched_forced_ids.append(forced_ids)
                    if lora_requests:
                        batched_lora_requests.append(lora_requests[i])
                    
                    # Replace the original response with placeholder
                    completed_responses[i] = None
        
        # Second pass: run a single batched continuation generation if needed
        if to_continue_indices:
            with self.update_sampling_params(max_tokens=self.additional_answer_tokens, stop=None):
                continuation_outputs = self.inference_engine.generate(
                    prompts=batched_continuation_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=batched_lora_requests if batched_lora_requests else None,
                    use_tqdm=False,
                )
            
            # Merge batched results back
            out_idx = 0
            for j, cont_idx in enumerate(to_continue_indices):
                # continuation_outputs aligns with batched_continuation_inputs order
                output = continuation_outputs[out_idx]
                continuation_ids = output.outputs[0].token_ids
                final_response = batched_forced_ids[j] + continuation_ids
                
                # Update total tokens for this prompt
                total_tokens_generated[cont_idx] += len(continuation_ids)
                
                # Replace placeholder for this index
                completed_responses[cont_idx] = final_response
                num_forced_thinking_end += 1
                out_idx += 1
        
        print(f"number of forced thinking end prompts: {num_forced_thinking_end}")

        # Pad responses to consistent length
        max_response_length = max(len(response) for response in completed_responses)
        padded_responses = []
        
        for response in completed_responses:
            if len(response) < max_response_length:
                # Pad with pad_token_id
                padded_response = response + [self.pad_token_id] * (max_response_length - len(response))
            else:
                padded_response = response
            padded_responses.append(padded_response)

        # Convert to tensor
        response = torch.tensor(padded_responses, device=prompts.batch["input_ids"].device)
        
        # Handle multiple samples if needed
        if self.sampling_params.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
            batch_size = batch_size * self.sampling_params.n

        # Concatenate input and response
        seq = torch.cat([idx, response], dim=-1)

        # Update position IDs
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # Create attention mask for response
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Create dummy log probs (will be recomputed by actor)
        rollout_log_probs = torch.full((batch_size, response_length), -1.0, dtype=torch.float32, device=idx.device)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # Add total tokens generated information to non_tensor_batch
        non_tensor_batch["total_tokens_generated"] = np.array(total_tokens_generated, dtype=np.int64)
        
        # Print summary of total tokens generated
        total_tokens_all_prompts = sum(total_tokens_generated)
        print(f"\n📊 Total Tokens Generated Summary (Force Answer):")
        print(f"  Total tokens across all prompts: {total_tokens_all_prompts:,}")
        print(f"  Average tokens per prompt: {total_tokens_all_prompts / batch_size:.1f}")
        for i, tokens in enumerate(total_tokens_generated):
            print(f"  Prompt {i}: {tokens:,} tokens")
        if num_forced_thinking_end > 0:
            print(f"  Prompts with forced completion: {num_forced_thinking_end}")
        if max_length_without_plan_indices:
            print(f"  Prompts that reached max length without plan: {len(max_length_without_plan_indices)} (indices: {max_length_without_plan_indices})")

        # free vllm cache engine
        if (
            hasattr(self, 'config') and hasattr(self.config, 'free_cache_engine') 
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch) 