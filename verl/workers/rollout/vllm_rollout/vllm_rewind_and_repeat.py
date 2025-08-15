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
from verl.utils.templates import format_system_message


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
                return is_approved, confidence_score, autorater_explanations[0]
            else:
                print("Warning: No decision from plan evaluator, defaulting to rejection")
                return False, 0.0, ""
                
        except Exception as e:
            print(f"Error calling autorater service for plan evaluation: {e}")
            # Fallback: reject plan with low confidence
            return False, 0.0, ""
    
    def create_rewind_prompt(self, original_question: str, rejected_plans: List[str], turn: int, explicit_task: str = None) -> str:
        """
        Create an augmented prompt for regeneration after plan rejection.
        
        Args:
            original_question: The original question/prompt
            rejected_plans: List of previously rejected plans (extracted from <answer> tags)
            turn: Current turn number
            explicit_task: Optional explicit task description for clearer guidance
            
        Returns:
            Augmented prompt for regeneration
        """
        # Use explicit task if available, otherwise use original question
        evaluation_question = original_question
        if explicit_task:
            print(f"Using explicit task for rewind prompt: {explicit_task}")
            evaluation_question = explicit_task
        
        if self.rewind_prompt_template == "default":
            # Default rewind prompt template
            rewind_prompt = f"""The following plan was rejected as not meeting the requirements. Please generate a better plan.

Question: {evaluation_question}

Rejected Plan (Turn {turn}):
{chr(10).join(f"- {plan}" for plan in rejected_plans)}

Please provide a new, improved plan that addresses the question more effectively."""
            
        else:
            raise ValueError(f"Invalid rewind prompt template: {self.rewind_prompt_template}")
        
        messages = [
            format_system_message("plan_first"),
            {"role": "user", "content": rewind_prompt}
        ]

        rewind_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            
            # Initialize new active indices for next turn
            new_active_indices = []
            
            if turn == 0:
                # First turn: Generate and evaluate initial plans
                print(f"  Turn {turn + 1}: Generating and evaluating initial plans...")
                
                # Batch 1: Generate initial plans for all active prompts
                print(f"  Batch 1: Generating initial plans for {len(active_indices)} prompts...")
                
                # Prepare batch inputs for initial generation
                batch_prompts = []
                batch_max_tokens = []
                batch_seeds = []
                
                for prompt_idx in active_indices:
                    question = original_prompts[prompt_idx]
                    batch_prompts.append(question)
                    batch_max_tokens.append(curr_max_tokens[prompt_idx])
                    batch_seeds.append(prompt_idx)
                
                # Generate initial plans in batch
                initial_candidates = self.generate_text_with_model_batch(
                    prompts=batch_prompts,
                    num_outputs=1,  # Always 1 for this strategy
                    max_new_tokens_list=batch_max_tokens,
                    temperature=1.2,
                    top_p=0.9,
                    seeds=batch_seeds
                )
                
                print(f"  Generated {len(initial_candidates)} initial plans")
                
                # Process initial plans and identify which need rewind
                plans_to_evaluate = []
                prompts_needing_rewind = []
                prompt_indices_needing_rewind = []
                
                for i, prompt_idx in enumerate(active_indices):
                    candidates = initial_candidates[i]
                    if not candidates or not candidates[0].strip():
                        prompts_needing_rewind.append(original_prompts[prompt_idx])
                        prompt_indices_needing_rewind.append(prompt_idx)
                        continue
                    
                    generated_response = candidates[0].strip()
                    
                    # Validate that the response contains answer tags
                    if not self.validate_response_has_answers(generated_response):
                        prompts_needing_rewind.append(original_prompts[prompt_idx])
                        prompt_indices_needing_rewind.append(prompt_idx)
                        continue
                    
                    # Extract the plan from the <answer> tags
                    extracted_plan = self.extract_last_answer(generated_response)
                    
                    if not extracted_plan:
                        prompts_needing_rewind.append(original_prompts[prompt_idx])
                        prompt_indices_needing_rewind.append(prompt_idx)
                        continue
                    
                    # Add to evaluation list
                    plans_to_evaluate.append({
                        'prompt_idx': prompt_idx,
                        'plan': extracted_plan,
                        'response': generated_response
                    })
                
                # Print summary of initial generation results
                print(f"  Initial generation results:")
                print(f"    ✓ Plans ready for evaluation: {len(plans_to_evaluate)} (indices: {[item['prompt_idx'] for item in plans_to_evaluate]})")
                print(f"    ⚠️  Plans needing rewind: {len(prompts_needing_rewind)} (indices: {prompt_indices_needing_rewind})")
                
                # Batch 2: Evaluate all initial plans with autorater
                if plans_to_evaluate:
                    print(f"  Batch 2: Evaluating {len(plans_to_evaluate)} initial plans...")
                    
                    # Prepare batch for autorater evaluation
                    eval_questions = []
                    eval_plans = []
                    
                    for item in plans_to_evaluate:
                        prompt_idx = item['prompt_idx']
                        plan = item['plan']
                        
                        # Use explicit task if available, otherwise use original question
                        evaluation_question = original_prompts[prompt_idx]
                        if meta_info and "explicit_tasks" in meta_info:
                            explicit_tasks = meta_info["explicit_tasks"]
                            if prompt_idx < len(explicit_tasks) and explicit_tasks[prompt_idx]:
                                evaluation_question = explicit_tasks[prompt_idx]
                                print(f"    Using explicit task for evaluation of prompt {prompt_idx}: {evaluation_question[:100]}...")
                            else:
                                print(f"    Using original question for evaluation of prompt {prompt_idx}: {evaluation_question[:100]}...")
                        else:
                            print(f"    Using original question for evaluation of prompt {prompt_idx}: {evaluation_question[:100]}...")
                        
                        eval_questions.append(evaluation_question)
                        eval_plans.append(plan)
                    
                    # Call autorater service in batch
                    autorater_payload = {
                        "prompts": eval_questions,
                        "responses": eval_plans,
                        "gt_answers": [""] * len(eval_plans),
                        "template_types": ["plan_quality_evaluation"] * len(eval_plans),
                    }
                    
                    autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(
                        self.plan_evaluation_service_url, autorater_payload, batch_size=len(eval_plans)
                    )
                    
                    # Process evaluation results
                    approved_indices = []
                    rejected_indices = []
                    
                    for i, item in enumerate(plans_to_evaluate):
                        prompt_idx = item['prompt_idx']
                        plan = item['plan']
                        response = item['response']
                        
                        if i < len(autorater_decisions):
                            decision = autorater_decisions[i]
                            
                            # Handle different response formats
                            if isinstance(decision, bool):
                                is_approved = decision
                            elif isinstance(decision, str):
                                is_approved = decision.upper() == "TRUE"
                            else:
                                is_approved = False
                            
                            if is_approved:
                                approved_indices.append(prompt_idx)
                                
                                # Add the full approved response (including thinking parts) to the current input
                                full_response_ids = self.tokenizer.encode(response)
                                curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx].copy(), full_response_ids])
                                
                                # Update generation history
                                generation_history[prompt_idx]['total_regeneration_attempts'] += 1
                                turn_data = {
                                    'turn_number': turn,
                                    'approved_plan': plan,
                                    'rejected_plans': [],
                                    'total_attempts': 1,
                                    'plan_approved': True
                                }
                                generation_history[prompt_idx]['turns'].append(turn_data)
                                
                                # Check if we should continue generation for this prompt
                                current_length = len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx])
                                if current_length < self.config.response_length:
                                    new_active_indices.append(prompt_idx)
                                    curr_max_tokens[prompt_idx] = self.config.response_length - current_length
                                else:
                                    print(f"      ✓ Reached max length, stopping generation")
                            else:
                                rejected_indices.append(prompt_idx)
                                prompts_needing_rewind.append(original_prompts[prompt_idx])
                                prompt_indices_needing_rewind.append(prompt_idx)
                        else:
                            rejected_indices.append(prompt_idx)
                            prompts_needing_rewind.append(original_prompts[prompt_idx])
                            prompt_indices_needing_rewind.append(prompt_idx)
                    
                    # Print summary of evaluation results
                    print(f"  Evaluation results:")
                    print(f"    ✅ Plans approved: {len(approved_indices)} (indices: {approved_indices})")
                    print(f"    ❌ Plans rejected: {len(rejected_indices)} (indices: {rejected_indices})")
                
                # Batch 3: Handle rewind attempts for rejected plans
                if prompts_needing_rewind:
                    print(f"  Batch 3: Processing {len(prompts_needing_rewind)} prompts needing rewind...")
                    
                    # Track rewind attempts for each prompt
                    rewind_attempts = {idx: 1 for idx in prompt_indices_needing_rewind}
                    max_rewind_attempts = self.max_rewind_attempts
                    
                    while prompts_needing_rewind and max(rewind_attempts.values()) <= max_rewind_attempts:
                        current_rewind_count = max(rewind_attempts.values())
                        print(f"    Rewind iteration {current_rewind_count}: Processing {len(prompts_needing_rewind)} prompts")
                        
                        # Prepare rewind prompts for this batch
                        rewind_prompts = []
                        rewind_max_tokens = []
                        rewind_seeds = []
                        rewind_prompt_indices = []
                        
                        for i, prompt_idx in enumerate(prompt_indices_needing_rewind):
                            if rewind_attempts[prompt_idx] > max_rewind_attempts:
                                continue
                            
                            # Get rejected plans for this prompt (simplified - just track count for now)
                            rejected_count = rewind_attempts[prompt_idx] - 1
                            
                            # Create rewind prompt with explicit task if available
                            explicit_task = None
                            if "explicit_tasks" in meta_info and prompt_idx < len(meta_info["explicit_tasks"]):
                                explicit_task = meta_info["explicit_tasks"][prompt_idx]
                            
                            rewind_prompt = self.create_rewind_prompt(
                                meta_info["original_prompt"][prompt_idx], 
                                [f"Attempt {rewind_attempts[prompt_idx]}"], 
                                turn,
                                explicit_task
                            )
                            
                            rewind_prompts.append(rewind_prompt)
                            rewind_max_tokens.append(curr_max_tokens[prompt_idx])
                            rewind_seeds.append(prompt_idx + rewind_attempts[prompt_idx])
                            rewind_prompt_indices.append(prompt_idx)
                        
                        if not rewind_prompts:
                            break
                        
                        # Generate rewind plans in batch
                        rewind_candidates = self.generate_text_with_model_batch(
                            prompts=rewind_prompts,
                            num_outputs=1,
                            max_new_tokens_list=rewind_max_tokens,
                            temperature=1.2,
                            top_p=0.9,
                            seeds=rewind_seeds
                        )
                        
                        # Process rewind results
                        still_needing_rewind = []
                        still_needing_rewind_indices = []
                        approved_in_this_round = []
                        
                        for i, prompt_idx in enumerate(rewind_prompt_indices):
                            candidates = rewind_candidates[i]
                            if not candidates or not candidates[0].strip():
                                still_needing_rewind.append(original_prompts[prompt_idx])
                                still_needing_rewind_indices.append(prompt_idx)
                                rewind_attempts[prompt_idx] += 1
                                continue
                            
                            generated_response = candidates[0].strip()
                            
                            # Validate and extract plan
                            if not self.validate_response_has_answers(generated_response):
                                still_needing_rewind.append(original_prompts[prompt_idx])
                                still_needing_rewind_indices.append(prompt_idx)
                                rewind_attempts[prompt_idx] += 1
                                continue
                            
                            extracted_plan = self.extract_last_answer(generated_response)
                            if not extracted_plan:
                                still_needing_rewind.append(original_prompts[prompt_idx])
                                still_needing_rewind_indices.append(prompt_idx)
                                rewind_attempts[prompt_idx] += 1
                                continue
                            
                            # Evaluate the rewind plan
                            is_approved, confidence, explanation = self.evaluate_plan_quality(
                                meta_info["original_prompt"][prompt_idx], 
                                extracted_plan, 
                                prompt_idx, 
                                meta_info
                            )
                            
                            if is_approved:
                                approved_in_this_round.append(prompt_idx)
                                
                                # Add the full approved response (including thinking parts) to the current input
                                full_response_ids = self.tokenizer.encode(generated_response)
                                curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx].copy(), full_response_ids])
                                
                                # Update generation history
                                generation_history[prompt_idx]['total_regeneration_attempts'] += rewind_attempts[prompt_idx]
                                turn_data = {
                                    'turn_number': turn,
                                    'approved_plan': extracted_plan,
                                    'rejected_plans': [f"Attempt {j+1}" for j in range(rewind_attempts[prompt_idx] - 1)],
                                    'total_attempts': rewind_attempts[prompt_idx],
                                    'plan_approved': True
                                }
                                generation_history[prompt_idx]['turns'].append(turn_data)
                                
                                # Check if we should continue generation for this prompt
                                current_length = len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx])
                                if current_length < self.config.response_length:
                                    new_active_indices.append(prompt_idx)
                                    curr_max_tokens[prompt_idx] = self.config.response_length - current_length
                                else:
                                    print(f"        ✓ Reached max length, stopping generation")
                            else:
                                still_needing_rewind.append(original_prompts[prompt_idx])
                                still_needing_rewind_indices.append(prompt_idx)
                                rewind_attempts[prompt_idx] += 1
                        
                        # Print summary of this rewind round
                        print(f"      Rewind round {current_rewind_count} results:")
                        print(f"        ✅ Plans approved: {len(approved_in_this_round)} (indices: {approved_in_this_round})")
                        print(f"        🔄 Still needing rewind: {len(still_needing_rewind)} (indices: {still_needing_rewind_indices})")
                        
                        # Update for next iteration
                        prompts_needing_rewind = still_needing_rewind
                        prompt_indices_needing_rewind = still_needing_rewind_indices
                    
                    # Handle prompts that exceeded max rewind attempts
                    exceeded_max_attempts = []
                    for prompt_idx in prompt_indices_needing_rewind:
                        if rewind_attempts[prompt_idx] > max_rewind_attempts:
                            exceeded_max_attempts.append(prompt_idx)
                            
                            # Update generation history
                            generation_history[prompt_idx]['total_regeneration_attempts'] += max_rewind_attempts
                            turn_data = {
                                'turn_number': turn,
                                'approved_plan': "",
                                'rejected_plans': [f"Attempt {j+1}" for j in range(max_rewind_attempts)],
                                'total_attempts': max_rewind_attempts,
                                'plan_approved': False
                            }
                            generation_history[prompt_idx]['turns'].append(turn_data)
                    
                    if exceeded_max_attempts:
                        print(f"    ⚠️  Max rewind attempts exceeded for {len(exceeded_max_attempts)} prompts (indices: {exceeded_max_attempts})")
            else:
                # Subsequent turns: Continue generation with already-approved plans
                print(f"  Turn {turn + 1}: Continuing generation with approved plans...")
                
                # Prepare batch inputs for continuation generation
                batch_prompts = []
                batch_max_tokens = []
                batch_seeds = []
                batch_prompt_indices = []
                
                for prompt_idx in active_indices:
                    # Get the full accumulated response from the previous turn (including original prompt)
                    full_response = self.tokenizer.decode(curr_inputs[prompt_idx], skip_special_tokens=False)
                    
                    # Remove the im_end token for continued generation
                    full_response = full_response.replace(self.tokenizer.special_tokens_map["eos_token"], "")

                    batch_prompts.append(full_response)
                    batch_max_tokens.append(curr_max_tokens[prompt_idx])
                    batch_seeds.append(prompt_idx + turn)
                    batch_prompt_indices.append(prompt_idx)
                
                # Generate continuations in batch
                continuation_candidates = self.generate_text_with_model_batch(
                    prompts=batch_prompts,
                    num_outputs=1,
                    max_new_tokens_list=batch_max_tokens,
                    temperature=1.2,
                    top_p=0.9,
                    seeds=batch_seeds
                )
                
                # Process continuation results
                successful_continuations = []
                failed_continuations = []
                
                for i, prompt_idx in enumerate(batch_prompt_indices):
                    candidates = continuation_candidates[i]
                    if candidates and candidates[0].strip():
                        continuation = candidates[0].strip()
                        
                        # Add continuation to current input
                        continuation_ids = self.tokenizer.encode(continuation)
                        curr_inputs[prompt_idx] = np.concatenate([curr_inputs[prompt_idx], continuation_ids])
                        
                        # Update generation history
                        turn_data = {
                            'turn_number': turn,
                            'approved_plan': continuation,
                            'rejected_plans': [],
                            'total_attempts': 1,
                            'plan_approved': True  # No evaluation needed on subsequent turns
                        }
                        generation_history[prompt_idx]['turns'].append(turn_data)
                        
                        # Check if we should continue generation for this prompt
                        current_length = len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx])
                        if current_length < self.config.response_length:
                            new_active_indices.append(prompt_idx)
                            curr_max_tokens[prompt_idx] = self.config.response_length - current_length
                            successful_continuations.append(prompt_idx)
                        else:
                            print(f"      ✓ Reached max length, stopping generation")
                    else:
                        failed_continuations.append(prompt_idx)
                        print(f"    Prompt {prompt_idx}: ❌ No continuation generated, stopping generation")
                
                # Print summary of continuation results
                print(f"  Continuation generation results:")
                print(f"    ✅ Successful continuations: {len(successful_continuations)} (indices: {successful_continuations})")
                if failed_continuations:
                    print(f"    ❌ Failed continuations: {len(failed_continuations)} (indices: {failed_continuations})")
            
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