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
vLLM Autorater Rollout that implements best-of-n generation with automatic rating.
This rollout generates N responses, parses <answer></answer> chunks, uses an autorater
to select the best answer, and continues generation with the best answer.
"""

import re
import logging
import os
from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np

from .vllm_rollout_spmd import vLLMRollout, _pre_process_inputs, _repeat_interleave
from verl import DataProto
from verl.utils.torch_functional import pad_2d_list_to_length, get_response_mask
from tensordict import TensorDict
from vllm import SamplingParams
from verl.utils.autorater_client import call_autorater_service

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMAutoraterRollout(vLLMRollout):
    """
    vLLM Autorater Rollout that implements best-of-n generation with automatic rating.
    
    This rollout:
    1. Generates N responses for each prompt
    2. Stops generation when </answer> token is encountered
    3. Parses the first <answer></answer> chunk from each response
    4. Uses an autorater to select the best answer
    5. Continues generation with the best answer
    6. Repeats this process for each </answer> token encountered
    """
    
    def __init__(self, model_path: str, config, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        
        # Autorater configuration
        self.n_candidates = config.get("n_candidates", 4)  # Number of candidates to generate
        self.autorater_service_url = config.get("autorater_service_url", "http://localhost:8000")
        self.answer_stop_token = "</answer>"
        self.answer_start_token = "<answer>"
        
        # Get token IDs for answer tokens
        self.answer_stop_token_ids = self.tokenizer.encode(self.answer_stop_token)
        self.answer_start_token_ids = self.tokenizer.encode(self.answer_start_token)
        
        logger.info(f"Initialized vLLMAutoraterRollout with n_candidates={self.n_candidates}")
    
    
    def extract_answer_chunks(self, text: str) -> List[str]:
        """Extract all <answer></answer> chunks from text."""
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]
    
    def extract_first_answer(self, text: str) -> Optional[str]:
        """Extract the first <answer></answer> chunk from text."""
        chunks = self.extract_answer_chunks(text)
        return chunks[0] if chunks else None
    
    def call_autorater_for_candidates(self, question: str, candidates: List[str]) -> Tuple[int, float]:
        """Use the autorater service to select the best candidate."""
        try:
            # Prepare autorater payload
            autorater_payload = {
                "prompts": [question] * len(candidates),
                "responses": candidates,
                "gt_answers": [""] * len(candidates),  # Empty ground truth for candidate selection
                "template_types": ["standard"] * len(candidates),
            }
            
            # Call autorater service
            autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(
                self.autorater_service_url, autorater_payload, batch_size=len(candidates)
            )
            
            # Find the best candidate (highest score)
            best_idx = 0
            best_score = 0.0
            
            for i, score in enumerate(autorater_decisions):
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            logger.debug(f"Autorater selected candidate {best_idx + 1} with score {best_score}")
            return best_idx, best_score
            
        except Exception as e:
            logger.error(f"Error calling autorater service: {e}")
            # Fallback: return first candidate with default score
            return 0, 5.0
    
    def generate_best_of_n(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using best-of-n with autorater.
        
        This method:
        1. Generates N candidate responses for each prompt
        2. Extracts the first <answer></answer> chunk from each response
        3. Uses an autorater to select the best answer
        4. Continues generation with the best answer
        5. Repeats for each </answer> token encountered
        """
        # Rebuild vllm cache engine
        if (vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine):
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

        # Get original prompts for autorater
        original_prompts = []
        for i in range(batch_size):
            prompt_text = self.tokenizer.decode(non_tensor_batch["raw_prompt_ids"][i])
            original_prompts.append(prompt_text)

        # Initialize generation state
        curr_inputs = [ids.copy() for ids in non_tensor_batch["raw_prompt_ids"]]
        init_inputs = [ids.copy() for ids in curr_inputs]
        
        # Track generation state for each sample
        active_indices = list(range(batch_size))
        curr_max_tokens = [self.config.response_length] * batch_size
        
        # Multi-turn generation with autorater
        max_turns = self.config.get("max_turns", 5)
        
        for turn in range(max_turns):
            if not active_indices:
                break
                
            logger.info(f"Turn {turn + 1}: Processing {len(active_indices)} active samples")
            
            # Prepare all inputs for parallel candidate generation
            all_vllm_inputs = []
            input_to_sample_idx = []
            
            for sample_idx in active_indices:
                for candidate_idx in range(self.n_candidates):
                    all_vllm_inputs.append({"prompt_token_ids": curr_inputs[sample_idx]})
                    input_to_sample_idx.append(sample_idx)
            
            # Generate all candidates in parallel
            with self.update_sampling_params(
                max_tokens=min(512, max(curr_max_tokens)),
                temperature=0.7,
                top_p=0.9,
                stop_token_ids=[self.answer_stop_token_ids[-1]] if self.answer_stop_token_ids else None
            ):
                outputs = self.inference_engine.generate(
                    prompts=all_vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )
            
            # Organize candidates by sample
            all_candidates = []
            candidate_indices = []
            
            for sample_idx in active_indices:
                candidates_for_sample = []
                for i, output in enumerate(outputs):
                    if input_to_sample_idx[i] == sample_idx:
                        candidate_ids = output.outputs[0].token_ids
                        candidate_text = self.tokenizer.decode(candidate_ids)
                        candidates_for_sample.append(candidate_text)
                
                all_candidates.append(candidates_for_sample)
                candidate_indices.append(sample_idx)
            
            # Use autorater to select best answers
            new_active_indices = []
            
            for i, sample_idx in enumerate(candidate_indices):
                candidates = all_candidates[i]
                question = original_prompts[sample_idx]
                
                # Extract first answer from each candidate
                first_answers = []
                valid_candidates = []
                
                for candidate in candidates:
                    first_answer = self.extract_first_answer(candidate)
                    if first_answer:
                        first_answers.append(first_answer)
                        valid_candidates.append(candidate)
                    else:
                        # If no answer found, use the full candidate
                        first_answers.append(candidate)
                        valid_candidates.append(candidate)
                
                if not first_answers:
                    # No valid answers found, skip this sample
                    continue
                
                # Use autorater to select best answer
                try:
                    best_idx, score = self.call_autorater_for_candidates(question, first_answers)
                    
                    # Ensure best_idx is within bounds
                    best_idx = min(best_idx, len(valid_candidates) - 1)
                    
                    # Add the best candidate to the current input
                    best_candidate = valid_candidates[best_idx]
                    best_candidate_ids = self.tokenizer.encode(best_candidate)
                    
                    curr_inputs[sample_idx].extend(best_candidate_ids)
                    
                    # Check if we should continue generation
                    if self.answer_stop_token in best_candidate:
                        # Found </answer>, check if we should continue
                        if len(curr_inputs[sample_idx]) - len(init_inputs[sample_idx]) < self.config.response_length:
                            # Continue generation
                            new_active_indices.append(sample_idx)
                            curr_max_tokens[sample_idx] = self.config.response_length - (len(curr_inputs[sample_idx]) - len(init_inputs[sample_idx]))
                        else:
                            # Reached max length
                            pass
                    else:
                        # No </answer> found, continue generation
                        new_active_indices.append(sample_idx)
                        curr_max_tokens[sample_idx] = self.config.response_length - (len(curr_inputs[sample_idx]) - len(init_inputs[sample_idx]))
                    
                    logger.debug(f"Sample {sample_idx}: Selected candidate {best_idx + 1} with score {score}")
                    
                except Exception as e:
                    logger.error(f"Error in autorater for sample {sample_idx}: {e}")
                    # Fallback: use first candidate
                    if valid_candidates:
                        best_candidate_ids = self.tokenizer.encode(valid_candidates[0])
                        curr_inputs[sample_idx].extend(best_candidate_ids)
                        new_active_indices.append(sample_idx)
            
            active_indices = new_active_indices
            
            # Check if any samples have reached max length
            final_active_indices = []
            for sample_idx in active_indices:
                if len(curr_inputs[sample_idx]) - len(init_inputs[sample_idx]) >= self.config.response_length:
                    # Truncate to response length
                    curr_inputs[sample_idx] = init_inputs[sample_idx] + \
                        curr_inputs[sample_idx][len(init_inputs[sample_idx]):len(init_inputs[sample_idx])+self.config.response_length]
                else:
                    final_active_indices.append(sample_idx)
            
            active_indices = final_active_indices
        
        # Collect final responses
        response_list = []
        for i in range(batch_size):
            input_len = len(init_inputs[i])
            response_ids = curr_inputs[i][input_len:]
            response_list.append(response_ids)
        
        # Pad responses to uniform length
        response = pad_2d_list_to_length(response_list, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        
        # Concatenate input and response
        seq = torch.cat([idx, response], dim=-1)
        
        # Update position IDs and attention mask
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # Create dummy log probs (will be recomputed by actor)
        rollout_log_probs = torch.zeros(batch_size, response_length, dtype=torch.float32, device=idx.device)
        
        # Create final batch
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "rollout_log_probs": rollout_log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        
        # Free vllm cache engine
        if (vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine):
            self.inference_engine.free_cache_engine()
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch) 