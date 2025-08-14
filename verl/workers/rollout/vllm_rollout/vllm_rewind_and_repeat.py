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
vLLM Rewind and Repeat Rollout that implements iterative plan generation with autorater feedback.

This rollout:
1. Generates 1 candidate response per turn
2. Uses autorater to evaluate if the plan is good given the intent
3. If autorater likes the plan, continues to next turn
4. If autorater rejects the plan, rewinds and regenerates with augmented prompt
5. Repeats until autorater approves or max iterations reached
"""

import re
import os
from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np

from .vllm_autorater_rollout import vLLMAutoraterRollout, _pre_process_inputs
from verl import DataProto
from verl.utils.torch_functional import pad_2d_list_to_length, get_response_mask
from tensordict import TensorDict
from verl.utils.autorater_client import call_autorater_service
from verl.single_controller.base.decorator import Dispatch, register
from verl.third_party.vllm import vllm_version


class vLLMRewindAndRepeatRollout(vLLMAutoraterRollout):
    """
    vLLM Rewind and Repeat Rollout that implements iterative plan generation with autorater feedback.
    
    This rollout:
    1. Generates 1 candidate response per turn (n_candidates=1)
    2. Uses autorater to evaluate if the plan is good given the intent
    3. If autorater likes the plan, continues to next turn
    4. If autorater rejects the plan, rewinds and regenerates with augmented prompt
    5. Repeats until autorater approves or max iterations reached
    """
    
    def __init__(self, model_path: str, config, tokenizer, model_hf_config, **kwargs):
        # Force n_candidates to 1 for this strategy
        config["n_candidates"] = 1
        
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        
        # Rewind and repeat specific configuration
        self.max_rewind_attempts = config.get("max_rewind_attempts", 3)  # Max regeneration attempts per turn
        self.plan_evaluation_threshold = config.get("plan_evaluation_threshold", 0.5)  # Threshold for plan approval
        self.rewind_prompt_template = config.get("rewind_prompt_template", "default")
        
        # Override autorater service URL if specified
        if "plan_evaluation_service_url" in config:
            self.plan_evaluation_service_url = config["plan_evaluation_service_url"]
        else:
            self.plan_evaluation_service_url = self.autorater_service_url
        
        print(f"Initialized vLLMRewindAndRepeatRollout with max_rewind_attempts={self.max_rewind_attempts}")
        print(f"Plan evaluation threshold: {self.plan_evaluation_threshold}")
        print(f"Plan evaluation service: {self.plan_evaluation_service_url}")
    
    def evaluate_plan_quality(self, question: str, plan: str, prompt_idx: int = 0, meta_info: Dict = None) -> Tuple[bool, float]:
        """
        Use the autorater service to evaluate if a plan is good given the intent.
        
        Args:
            question: The original question/prompt
            plan: The generated plan to evaluate
            prompt_idx: Index of the current prompt
            meta_info: Additional metadata
            
        Returns:
            Tuple of (is_approved, confidence_score)
        """
        import ipdb; ipdb.set_trace()
        try:
            # Use explicit task if available, otherwise use original question
            evaluation_question = question
            if meta_info and "explicit_tasks" in meta_info:
                explicit_tasks = meta_info["explicit_tasks"]
                if prompt_idx < len(explicit_tasks) and explicit_tasks[prompt_idx]:
                    evaluation_question = explicit_tasks[prompt_idx]
                    print(f"Using explicit task for plan evaluation: {evaluation_question[:100]}...")
                else:
                    print(f"Using original question for plan evaluation: {question[:100]}...")
            else:
                print(f"Using original question for plan evaluation: {question[:100]}...")
            
            # Prepare autorater payload for plan quality evaluation
            autorater_payload = {
                "prompts": [evaluation_question],
                "responses": [plan],
                "gt_answers": [""],  # Empty ground truth for plan evaluation
                "template_types": ["plan_quality_evaluation"],  # Use the new template type
            }
            
            # Call autorater service
            autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(
                self.plan_evaluation_service_url, autorater_payload, batch_size=1
            )
            
            # Parse the response to get the plan approval decision
            if autorater_decisions and len(autorater_decisions) > 0:
                # The autorater service should return the parsed decision directly
                decision = autorater_decisions[0]
                
                # Handle different response formats
                if isinstance(decision, bool):
                    is_approved = decision
                    confidence_score = 1.0
                elif isinstance(decision, str):
                    # Convert string decision to boolean
                    is_approved = decision.upper() == "TRUE"
                    confidence_score = 1.0 if is_approved else 0.0
                else:
                    # Default to rejection for unknown response types
                    is_approved = False
                    confidence_score = 0.0
                
                print(f"Plan evaluator decision: {decision} (approved: {is_approved})")
                return is_approved, confidence_score
            else:
                print("Warning: No decision from plan evaluator, defaulting to rejection")
                return False, 0.0
                
        except Exception as e:
            print(f"Error calling autorater service for plan evaluation: {e}")
            # Fallback: reject plan with low confidence
            return False, 0.0
    
    def create_rewind_prompt(self, original_question: str, rejected_plans: List[str], turn: int) -> str:
        """
        Create an augmented prompt for regeneration after plan rejection.
        
        Args:
            original_question: The original question/prompt
            rejected_plans: List of previously rejected plans (extracted from <answer> tags)
            turn: Current turn number
            
        Returns:
            Augmented prompt for regeneration
        """
        if self.rewind_prompt_template == "default":
            # Default rewind prompt template
            rewind_prompt = f"""The following plan was rejected as not meeting the requirements. Please generate a better plan.

Original Question: {original_question}

Rejected Plan (Turn {turn}):
{chr(10).join(f"- {plan}" for plan in rejected_plans)}

Please provide a new, improved plan that addresses the original question more effectively. Output your response in <answer> tags.

<answer>"""
            
        elif self.rewind_prompt_template == "constructive":
            # Constructive feedback template
            rewind_prompt = f"""Your previous plan didn't meet the requirements. Let me help you improve it.

Original Question: {original_question}

Previous Attempt (Turn {turn}):
{chr(10).join(f"- {plan}" for plan in rejected_plans)}

Guidance for improvement:
- Consider the core intent more carefully
- Ensure your plan directly addresses the question
- Be more specific and actionable
- Avoid assumptions not supported by the question
- Focus on the most important aspects first

Please generate a revised plan in <answer> tags:

<answer>"""
            
        else:
            # Custom template - fallback to default
            rewind_prompt = f"""Please regenerate your plan for: {original_question}

Previous rejected attempts:
{chr(10).join(f"- {plan}" for plan in rejected_plans)}

Generate a better plan in <answer> tags:

<answer>"""
        
        return rewind_prompt
    
    def validate_response_has_answers(self, response: str) -> bool:
        """
        Validate that a generated response contains <answer> tags.
        
        Args:
            response: The generated response text
            
        Returns:
            True if response contains answer tags, False otherwise
        """
        return self.answer_start_token in response and self.answer_stop_token in response
    
    def generate_plan_with_rewind(self, prompt_idx: int, question: str, curr_inputs: List, 
                                  init_inputs: List, curr_max_tokens: List, turn: int, 
                                  meta_info: Dict = None) -> Tuple[str, List[str], int]:
        """
        Generate a plan with rewind capability if the autorater rejects it.
        
        Args:
            prompt_idx: Index of the current prompt
            question: The original question text
            curr_inputs: Current input tokens for each prompt
            init_inputs: Initial input tokens for each prompt
            curr_max_tokens: Maximum tokens to generate for each prompt
            turn: Current turn number
            meta_info: Additional metadata
            
        Returns:
            Tuple of (approved_plan, rejected_plans, total_attempts)
        """
        rejected_plans = []
        total_attempts = 0
        
        for attempt in range(self.max_rewind_attempts + 1):  # +1 for initial attempt
            total_attempts += 1
            
            if attempt == 0:
                # First attempt: use original prompt
                current_prompt = question
                print(f"  Attempt {attempt + 1}: Using original prompt")
            else:
                # Rewind attempt: use augmented prompt
                current_prompt = self.create_rewind_prompt(question, rejected_plans, turn)
                print(f"  Attempt {attempt + 1}: Using rewind prompt with {len(rejected_plans)} rejected plans")
            
            # Generate single candidate for this attempt
            candidates = self.generate_text_with_model(
                prompt=current_prompt,
                num_outputs=1,  # Always 1 for this strategy
                max_new_tokens=curr_max_tokens[prompt_idx],
                temperature=1.2,
                top_p=0.9,
                seed=prompt_idx + attempt  # Different seed for each attempt
            )
            
            if not candidates or not candidates[0].strip():
                print(f"    ✗ No generation produced, retrying...")
                continue
            
            generated_response = candidates[0].strip()
            print(f"    Generated response ({len(generated_response)} chars): {generated_response[:100]}...")
            
            # Validate that the response contains answer tags
            if not self.validate_response_has_answers(generated_response):
                print(f"    ✗ Response missing <answer> tags, retrying...")
                continue
            
            # Extract the plan from the <answer> tags
            extracted_plan = self.extract_last_answer(generated_response)
            
            if not extracted_plan:
                print(f"    ✗ No <answer> tags found in response, retrying...")
                continue
            
            print(f"    Extracted plan ({len(extracted_plan)} chars): {extracted_plan[:100]}...")
            
            # Evaluate the extracted plan
            is_approved, confidence = self.evaluate_plan_quality(question, extracted_plan, prompt_idx, meta_info)
            
            if is_approved:
                print(f"    ✅ Plan approved with confidence {confidence:.3f}")
                return extracted_plan, rejected_plans, total_attempts
            else:
                print(f"    ❌ Plan rejected with confidence {confidence:.3f}")
                rejected_plans.append(extracted_plan)
                
                if attempt < self.max_rewind_attempts:
                    print(f"    🔄 Rewinding and regenerating...")
                else:
                    print(f"    ⚠️  Max rewind attempts reached, using last rejected plan")
        
        # If we reach here, all attempts were rejected
        # Return the last rejected plan as a fallback
        if rejected_plans:
            print(f"  ⚠️  All {total_attempts} attempts rejected, using last plan as fallback")
            return rejected_plans[-1], rejected_plans[:-1], total_attempts
        else:
            print(f"  ❌ No plans generated, returning empty string")
            return "", [], total_attempts
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using rewind and repeat strategy.
        
        This method:
        1. Generates 1 candidate response per turn
        2. Uses autorater to evaluate plan quality
        3. If rejected, rewinds and regenerates with augmented prompt
        4. Continues until plan is approved or max attempts reached
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
        
        # One input per original prompt
        for sample_idx in range(batch_size):
            base_input = non_tensor_batch["raw_prompt_ids"][sample_idx].copy()
            curr_inputs.append(base_input.copy())
            init_inputs.append(base_input.copy())
            active_indices.append(sample_idx)
            curr_max_tokens.append(self.config.response_length)
        
        print(f"Initialized {batch_size} prompts for rewind and repeat strategy")
        
        # Initialize generation history for each prompt
        generation_history = []
        for i in range(batch_size):
            generation_history.append({
                'prompt_idx': i,
                'original_prompt': original_prompts[i],
                'turns': [],
                'final_response': '',
                'total_regeneration_attempts': 0
            })
        
        # Multi-turn generation with rewind capability
        max_turns = self.config.get("max_turns", 2)
        print(f"Multi-turn generation: max_turns={max_turns}")
        
        for turn in range(max_turns):
            if not active_indices:
                break
                
            print(f"Turn {turn + 1}: Processing {len(active_indices)} active prompts")
            
            # Process each prompt with rewind capability
            new_active_indices = []
            
            for i, prompt_idx in enumerate(active_indices):
                print("="*100)
                print(f"Processing prompt {prompt_idx}, turn {turn + 1}")
                print("="*100)
                
                question = original_prompts[prompt_idx]
                
                # Generate plan with rewind capability
                approved_plan, rejected_plans, total_attempts = self.generate_plan_with_rewind(
                    prompt_idx, question, curr_inputs, init_inputs, curr_max_tokens, turn, meta_info
                )
                
                # Update generation history
                generation_history[prompt_idx]['total_regeneration_attempts'] += total_attempts
                
                turn_data = {
                    'turn_number': turn,
                    'approved_plan': approved_plan,
                    'rejected_plans': rejected_plans,
                    'total_attempts': total_attempts,
                    'plan_approved': len(rejected_plans) < total_attempts
                }
                generation_history[prompt_idx]['turns'].append(turn_data)
                
                print(f"Prompt {prompt_idx}: Turn {turn + 1} completed with {total_attempts} attempts")
                print(f"  Approved plan length: {len(approved_plan)} chars")
                print(f"  Rejected plans: {len(rejected_plans)}")
                
                # Add the approved plan to the current input
                if approved_plan:
                    plan_ids = self.tokenizer.encode(approved_plan)
                    curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx].copy(), plan_ids])
                    
                    # Check if we should continue generation for this prompt
                    current_length = len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx])
                    if current_length < self.config.response_length:
                        new_active_indices.append(prompt_idx)
                        curr_max_tokens[prompt_idx] = self.config.response_length - current_length
                        print(f"  ✓ Continuing to next turn (current length: {current_length})")
                    else:
                        print(f"  ✓ Reached max length, stopping generation")
                else:
                    print(f"  ❌ No plan generated, stopping generation")
            
            active_indices = new_active_indices
            
            # Check if any prompts have reached max length
            final_active_indices = []
            for prompt_idx in active_indices:
                if len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx]) >= self.config.response_length:
                    # Truncate to response length
                    curr_inputs[prompt_idx] = np.concatenate([
                        init_inputs[prompt_idx],
                        curr_inputs[prompt_idx][len(init_inputs[prompt_idx]):len(init_inputs[prompt_idx])+self.config.response_length]
                    ])
                else:
                    final_active_indices.append(prompt_idx)
            
            active_indices = final_active_indices
        
        # Collect final responses and update generation history
        response_list = []
        
        for i in range(batch_size):
            # Get the final accumulated generation for this prompt
            input_len = len(init_inputs[i])
            response_ids = curr_inputs[i][input_len:]
            
            # Use the accumulated generation directly
            response_list.append(response_ids)
            print(f"Prompt {i}: Final generation ({len(response_ids)} tokens)")
            
            # Update the final response in generation history
            generation_history[i]['final_response'] = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
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
        
        # Add generation history to non_tensor_batch
        non_tensor_batch["generation_history"] = np.array(generation_history, dtype=object)
        
        # Free vllm cache engine
        if (vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine):
            self.inference_engine.free_cache_engine()
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch) 