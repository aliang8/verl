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
import os
from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np

from .vllm_rollout_spmd import vLLMRollout, _pre_process_inputs
from verl import DataProto
from verl.utils.torch_functional import pad_2d_list_to_length, get_response_mask
from tensordict import TensorDict
from vllm import SamplingParams
from verl.utils.autorater_client import call_autorater_service
from verl.single_controller.base.decorator import Dispatch, register
from verl.third_party.vllm import vllm_version
from verl.workers.autorater.autorater_utils import parse_plan_evaluation_response


class vLLMAutoraterRollout(vLLMRollout):
    """
    vLLM Autorater Rollout that implements best-of-n generation with automatic rating.
    
    This rollout:
    1. Generates N candidate responses for each prompt
    2. Extracts the first <answer></answer> chunk from each response
    3. Uses an autorater to select the best answer
    4. Continues generation with the best answer
    5. Repeats for each </answer> token encountered
    """
    
    def __init__(self, model_path: str, config, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        
        # Autorater configuration
        self.n_candidates = config.get("n_candidates", 5)  # Number of candidates to generate
        self.autorater_service_url = config.get("autorater_service_url", "http://10.128.0.30:81")
        self.answer_stop_token = "</answer>"
        self.answer_start_token = "<answer>"
        
        # Get token IDs for answer tokens
        self.answer_stop_token_ids = self.tokenizer.encode(self.answer_stop_token)
        self.answer_start_token_ids = self.tokenizer.encode(self.answer_start_token)
        
        print(f"Initialized vLLMAutoraterRollout with n_candidates={self.n_candidates}")
        print(f"Answer stop token: '{self.answer_stop_token}' -> token IDs: {self.answer_stop_token_ids}")
        print(f"Answer start token: '{self.answer_start_token}' -> token IDs: {self.answer_start_token_ids}")
    
    
    def extract_answer_chunks(self, text: str) -> List[str]:
        """Extract all <answer></answer> chunks from text."""
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]
    
    def extract_last_answer(self, text: str) -> Optional[str]:
        """Extract the last <answer></answer> chunk from text."""
        chunks = self.extract_answer_chunks(text)
        return chunks[-1] if chunks else None
    
    def evaluate_candidate_plan(self, question: str, candidates: List[str], prompt_idx: int = 0, meta_info: Dict = None) -> Tuple[int, float]:
        """Use the autorater service to select the best candidate plan."""
        try:
            # Use explicit task if available, otherwise use original question
            evaluation_question = question
            if meta_info and "explicit_tasks" in meta_info:
                explicit_tasks = meta_info["explicit_tasks"]
                if prompt_idx < len(explicit_tasks) and explicit_tasks[prompt_idx]:
                    evaluation_question = explicit_tasks[prompt_idx]
                    print(f"Using explicit task for evaluation: {evaluation_question[:100]}...")
                else:
                    print(f"Using original question for evaluation: {question[:100]}...")
            else:
                print(f"Using original question for evaluation: {question[:100]}...")
            
            # Prepare autorater payload for plan evaluation
            autorater_payload = {
                "prompts": [evaluation_question],  # Use explicit task if available
                "responses": [candidates],  # The list of plans to evaluate
                "gt_answers": [""],  # Empty ground truth for plan evaluation
                "template_types": ["plan_evaluation"],
            }
            
            # Call autorater service
            autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(
                self.autorater_service_url, autorater_payload, batch_size=1
            )
            
            # Parse the response to get the selected plan number
            if autorater_raw_responses and len(autorater_raw_responses) > 0:
                raw_response = autorater_raw_responses[0]
                # For plan evaluation, the autorater service returns the plan number directly
                # as the decision (not as a raw response that needs parsing)
                if autorater_decisions and len(autorater_decisions) > 0:
                    selected_plan = int(autorater_decisions[0])
                else:
                    # Fallback to parsing the raw response
                    selected_plan = parse_plan_evaluation_response(raw_response, len(candidates))
                
                # Convert to 0-based index and return with a default score
                best_idx = selected_plan - 1  # Convert from 1-based to 0-based
                best_score = 1.0  # Default score for plan evaluation
                
                print(f"Plan evaluator selected candidate {selected_plan} (index {best_idx})")
                return best_idx, best_score
            else:
                print("No response from autorater service, using first candidate")
                return 0, 1.0
            
        except Exception as e:
            print(f"Error calling autorater service for plan evaluation: {e}")
            # Fallback: return first candidate with default score
            return 0, 1.0
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using best-of-n with autorater.
        
        This method:
        1. Generates N candidate responses for each prompt using vLLM's n parameter
        2. Extracts the last <answer></answer> chunk from each response
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

        meta_info = prompts.meta_info

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not working properly.")

        # Get original prompts for autorater
        original_prompts = []
        for i in range(batch_size):
            prompt_text = self.tokenizer.decode(non_tensor_batch["raw_prompt_ids"][i])
            original_prompts.append(prompt_text)

        # Initialize generation state - one input per original prompt
        curr_inputs = []
        init_inputs = []
        active_indices = []
        curr_max_tokens = []
        
        # One input per original prompt (vLLM will generate n_candidates for each)
        for sample_idx in range(batch_size):
            base_input = non_tensor_batch["raw_prompt_ids"][sample_idx].copy()
            curr_inputs.append(base_input.copy())
            init_inputs.append(base_input.copy())
            active_indices.append(sample_idx)
            curr_max_tokens.append(self.config.response_length)
        
        print(f"Initialized {batch_size} prompts, will generate {self.n_candidates} candidates per prompt")
        
        # Multi-turn generation with autorater
        max_turns = self.config.get("max_turns", 2)

        print(f"Multi-turn generation: max_turns={max_turns}")

        for turn in range(max_turns):
            if not active_indices:
                break
                
            print(f"Turn {turn + 1}: Processing {len(active_indices)} active prompts")
            print(f"  Expected answer count for this turn: {turn + 1}")

            # Generate one step for each active prompt with n_candidates
            all_candidates = []
            prompt_indices = []
            
            for prompt_idx in active_indices:
                # Generate n_candidates for this prompt
                prompt_input = {"prompt_token_ids": curr_inputs[prompt_idx]}
                
                with self.update_sampling_params(
                    max_tokens=curr_max_tokens[prompt_idx],
                    temperature=0.8,
                    top_p=0.9,
                    n=self.n_candidates,  # Generate n_candidates per prompt
                    stop=[self.answer_stop_token] if self.answer_stop_token else None,
                    detokenize=True if self.answer_stop_token else None,
                    seed=prompt_idx  # Different seed for each prompt
                ):
                    prompt_outputs = self.inference_engine.generate(
                        prompts=[prompt_input],
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                    
                    if prompt_outputs:
                        # Extract all n_candidates for this prompt
                        prompt_candidates = []
                        for output in prompt_outputs[0].outputs:
                            candidate_ids = output.token_ids
                            candidate_text = self.tokenizer.decode(candidate_ids)
                            prompt_candidates.append(candidate_text)
                        
                        all_candidates.append(prompt_candidates)
                    else:
                        # Handle case where no output is generated
                        all_candidates.append([""] * self.n_candidates)
                
                prompt_indices.append(prompt_idx)
            
            print(f"Generated {len(active_indices)} prompt sets, each with {self.n_candidates} candidates")
            
            # Process each prompt's candidates and use autorater to select best answers
            new_active_indices = []
            
            for i, prompt_idx in enumerate(prompt_indices):
                candidates = all_candidates[i]
                question = original_prompts[prompt_idx]
                
                print(f"Prompt {prompt_idx}: Processing {len(candidates)} candidates")
                
                # Check if any candidate has reached </answer> and extract answers
                candidates_with_answers = []
                candidates_without_answers = []
                
                for candidate_idx, current_turn_generation in enumerate(candidates):
                    # Combine previous turns' generation with current turn's generation
                    input_len = len(init_inputs[prompt_idx])
                    full_generation_ids = curr_inputs[prompt_idx][input_len:]
                    
                    # Add current turn's generation
                    if current_turn_generation:
                        current_turn_ids = self.tokenizer.encode(current_turn_generation)
                        full_generation_ids = full_generation_ids + current_turn_ids
                    
                    # Decode the full combined generation
                    full_generation_text = self.tokenizer.decode(full_generation_ids)
                    
                    # Check if this candidate has reached </answer>
                    if self.answer_stop_token in full_generation_text:
                        # Extract all answers and check if we have the right number for this turn
                        all_answers = self.extract_answer_chunks(full_generation_text)
                        expected_answer_count = turn + 1  # Turn 0 should have 1 answer, Turn 1 should have 2 answers, etc.
                        
                        if len(all_answers) == expected_answer_count:
                            # Take the last answer since it's the most recent
                            last_answer = all_answers[-1]
                            candidates_with_answers.append({
                                'candidate_idx': candidate_idx,
                                'answer': last_answer,
                                'full_generation_text': full_generation_text,
                                'full_generation_ids': full_generation_ids,
                                'current_turn_generation': current_turn_generation
                            })
                            print(f"  Candidate {candidate_idx + 1}: Found {len(all_answers)} answers (expected {expected_answer_count}), using last answer")
                        else:
                            print(f"  Candidate {candidate_idx + 1}: Found {len(all_answers)} answers but expected {expected_answer_count}, skipping")
                    else:
                        candidates_without_answers.append({
                            'candidate_idx': candidate_idx,
                            'current_turn_generation': current_turn_generation
                        })
                        print(f"  Candidate {candidate_idx + 1}: No </answer> reached yet")
                
                # Evaluate and select the best candidate if any have reached </answer>
                if len(candidates_with_answers) > 0:
                    print(f"Prompt {prompt_idx}: {len(candidates_with_answers)}/{len(candidates)} candidates reached </answer>, evaluating and selecting best one")
                    
                    # Use autorater to select the best answer from candidates that have reached </answer>
                    try:
                        answers = [c['answer'] for c in candidates_with_answers]
                        best_idx, score = self.evaluate_candidate_plan(question, answers, prompt_idx, meta_info)
                        
                        # Ensure best_idx is within bounds
                        best_idx = min(best_idx, len(candidates_with_answers) - 1)
                        best_candidate = candidates_with_answers[best_idx]
                        
                        print(f"Prompt {prompt_idx}: Autorater selected candidate {best_candidate['candidate_idx'] + 1} with score {score}")
                        
                        # Copy the context from the best candidate to continue generation
                        best_full_generation_ids = best_candidate['full_generation_ids']
                        
                        # Update the current input to use the best candidate's generation
                        curr_inputs[prompt_idx] = init_inputs[prompt_idx].copy() + best_full_generation_ids
                        
                        # Check if we should continue generation for this prompt
                        current_length = len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx])
                        if current_length < self.config.response_length:
                            new_active_indices.append(prompt_idx)
                            curr_max_tokens[prompt_idx] = self.config.response_length - current_length
                        
                        print(f"Prompt {prompt_idx}: Using best candidate's context (length: {len(best_full_generation_ids)} tokens)")
                        
                    except Exception as e:
                        print(f"Error in autorater for prompt {prompt_idx}: {e}")
                        # Fallback: continue from where the longest candidate left off
                        longest_candidate = max(candidates, key=lambda x: len(x) if x else 0)
                        if longest_candidate:
                            candidate_ids = self.tokenizer.encode(longest_candidate)
                            curr_inputs[prompt_idx].extend(candidate_ids)
                            
                            current_length = len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx])
                            if current_length < self.config.response_length:
                                new_active_indices.append(prompt_idx)
                                curr_max_tokens[prompt_idx] = self.config.response_length - current_length
                
                else:
                    # No candidates have reached </answer> yet, continue from where the longest candidate left off
                    print(f"Prompt {prompt_idx}: No candidates reached </answer> yet, continuing from longest candidate")
                    
                    longest_candidate = max(candidates, key=lambda x: len(x) if x else 0)
                    if longest_candidate:
                        candidate_ids = self.tokenizer.encode(longest_candidate)
                        curr_inputs[prompt_idx].extend(candidate_ids)
                        
                        current_length = len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx])
                        if current_length < self.config.response_length:
                            new_active_indices.append(prompt_idx)
                            curr_max_tokens[prompt_idx] = self.config.response_length - current_length
            
                import ipdb; ipdb.set_trace()
            active_indices = new_active_indices
            
            # Check if any prompts have reached max length
            final_active_indices = []
            for prompt_idx in active_indices:
                if len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx]) >= self.config.response_length:
                    # Truncate to response length
                    curr_inputs[prompt_idx] = init_inputs[prompt_idx] + \
                        curr_inputs[prompt_idx][len(init_inputs[prompt_idx]):len(init_inputs[prompt_idx])+self.config.response_length]
                else:
                    final_active_indices.append(prompt_idx)
            
            active_indices = final_active_indices
        
        # Collect final responses - use the accumulated generation from multi-turn process
        # During each turn, we've already selected the best candidate and continued from there
        # So the final curr_inputs[i] contains the best trajectory we've built up
        response_list = []
        for i in range(batch_size):
            # Get the final accumulated generation for this prompt
            input_len = len(init_inputs[i])
            response_ids = curr_inputs[i][input_len:]
            
            # Use the accumulated generation directly - no need to regenerate
            response_list.append(response_ids)
            print(f"Prompt {i}: Using accumulated generation ({len(response_ids)} tokens)")
        
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