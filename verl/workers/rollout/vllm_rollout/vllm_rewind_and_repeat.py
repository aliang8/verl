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

from .vllm_best_of_n import vLLMBestOfN, _pre_process_inputs
from verl import DataProto
from verl.utils.torch_functional import pad_2d_list_to_length, get_response_mask
from tensordict import TensorDict
from verl.utils.autorater_client import call_autorater_service
from verl.single_controller.base.decorator import Dispatch, register
from verl.third_party.vllm import vllm_version
from verl.utils.templates import format_system_message


class vLLMRewindAndRepeatRollout(vLLMBestOfN):
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
        self.max_rewind_attempts = config.get("max_rewind_attempts", 2)  # Max regeneration attempts per turn
        self.plan_evaluation_threshold = config.get("plan_evaluation_threshold", 0.5)  # Threshold for plan approval
        self.rewind_prompt_template = config.get("rewind_prompt_template", "default")
        
        # Force answer completion configuration
        self.force_answer_completion = config.get("force_answer_completion", True)
        self.additional_answer_tokens = config.get("additional_answer_tokens", 512)

        # Override autorater service URL if specified
        if "plan_evaluation_service_url" in config:
            self.plan_evaluation_service_url = config["plan_evaluation_service_url"]
        else:
            self.plan_evaluation_service_url = self.autorater_service_url

        print(f"Initialized vLLMRewindAndRepeatRollout with max_rewind_attempts={self.max_rewind_attempts}")
        print(f"Plan evaluation threshold: {self.plan_evaluation_threshold}")
        print(f"Plan evaluation service: {self.plan_evaluation_service_url}")
        print(f"Force answer completion: {self.force_answer_completion}")
        print(f"Additional answer tokens: {self.additional_answer_tokens}")

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
                    # print(f"Using explicit task for plan evaluation: {evaluation_question[:100]}...")
                else:
                    pass
                    # print(f"Using original question for plan evaluation: {question[:100]}...")
            # else:
                # print(f"Using original question for plan evaluation: {question[:100]}...")

            # Prepare autorater payload for plan quality evaluation
            autorater_payload = {
                "prompts": [evaluation_question],
                "responses": [plan],
                "gt_answers": [""],  # Empty ground truth for plan evaluation
                "template_types": ["plan_quality_evaluation"],  # Use the new template type
            }

            # Call autorater service
            autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(self.plan_evaluation_service_url, autorater_payload, batch_size=1)

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

                # print(f"Plan evaluator decision: {decision} (approved: {is_approved})")
                return is_approved, confidence_score, autorater_explanations[0]
            else:
                # print("Warning: No decision from plan evaluator, defaulting to rejection")
                return False, 0.0, ""

        except Exception as e:
            print(f"Error calling autorater service for plan evaluation: {e}")
            # Fallback: reject plan with low confidence
            return False, 0.0, ""

    def create_rewind_prompt(self, original_question: str, rejected_plans: List[str], turn: int, explicit_task: str = None, hit_max_length: bool = False) -> str:
        """
        Create an augmented prompt for regeneration after plan rejection.

        Args:
            original_question: The original question/prompt
            rejected_plans: List of previously rejected plans (extracted from <answer> tags)
            turn: Current turn number
            explicit_task: Optional explicit task description for clearer guidance
            hit_max_length: Whether the prompt hit max length without generating a plan

        Returns:
            Augmented prompt for regeneration
        """
        # Use explicit task if available, otherwise use original question
        evaluation_question = original_question
        if explicit_task:
            print(f"Using explicit task for rewind prompt: {explicit_task}")
            evaluation_question = explicit_task

        if self.rewind_prompt_template == "default":
            if hit_max_length:
                if rejected_plans:
                    rewind_prompt = f"""The previous generation hit the maximum length limit without producing a complete plan. Please generate a concise plan that addresses the question effectively within the length constraints.

Question: {evaluation_question}

Previous incomplete attempts:
{chr(10).join(f"- {plan}" for plan in rejected_plans)}

Please provide a new, concise plan that fits within the max length and addresses the question effectively."""
                else:
                    rewind_prompt = f"""The previous generation hit the maximum length limit without producing a complete plan. Please generate a concise plan that addresses the question effectively within the length constraints.

Question: {evaluation_question}

Please provide a concise plan that fits within the max length and addresses the question effectively."""
            else:
                if rejected_plans:
                    rewind_prompt = f"""The following plan was rejected as not meeting the requirements. Please generate a better plan.

Question: {evaluation_question}

Rejected Plan (Turn {turn}):
{chr(10).join(f"- {plan}" for plan in rejected_plans)}

Please provide a new, improved plan that addresses the question more effectively."""
                else:
                    rewind_prompt = f"""The previous attempt did not produce a valid plan. Please generate a new plan.

Question: {evaluation_question}

Please provide a new plan that addresses the question effectively."""

        else:
            raise ValueError(f"Invalid rewind prompt template: {self.rewind_prompt_template}")

        messages = [format_system_message("plan_first"), {"role": "user", "content": rewind_prompt}]

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
        1. Turn 1: Generate plan, evaluate with autorater, rewind if rejected until max attempts
        2. Turn 2: Generate answer based on approved plan
        3. Force completion only if:
           - Turn 1 hits max length without generating plan, OR
           - Turn 2 doesn't have 2 complete <answer></answer> blocks
        """
        # Rebuild vllm cache engine
        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
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
                "prompt_idx": i, 
                "original_prompt": original_prompts[i], 
                "turns": [], 
                "final_response": "", 
                "total_regeneration_attempts": 0,
                "total_tokens_generated": 0  # Track total tokens across all attempts
            })
        
        # Track rejected plans for each prompt
        rejected_plans_per_prompt = {i: [] for i in range(batch_size)}
        
        # For prompts that hit max length without generating a plan, we'll add them to rewind pool
        # but they won't have any rejected plans initially

        # Fixed 2-turn generation
        max_turns = 2
        print(f"Fixed 2-turn generation: plan generation + answer generation")

        # TURN 1: Plan Generation with Rewind on Rejection
        print(f"\n🔄 TURN 1: Plan Generation with Rewind Capability")
        
        # Generate initial plans for all prompts
        print(f"  Generating initial plans for {len(active_indices)} prompts...")
        
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
            temperature=0.6,
            top_p=0.9,
            seeds=batch_seeds,
        )

        print(f"  Generated {len(initial_candidates)} initial plans")
        
        # Track tokens generated in initial generation
        for i, prompt_idx in enumerate(active_indices):
            if initial_candidates[i] and initial_candidates[i][0].strip():
                response_text = initial_candidates[i][0].strip()
                response_tokens = self.tokenizer.encode(response_text)
                generation_history[prompt_idx]["total_tokens_generated"] += len(response_tokens)

        # Process initial plans and identify which need rewind
        plans_to_evaluate = []
        prompts_needing_rewind = []
        prompt_indices_needing_rewind = []
        max_length_without_plan_indices = []  # Track responses that hit max length without plan

        for i, prompt_idx in enumerate(active_indices):
            candidates = initial_candidates[i]
            if not candidates or not candidates[0].strip():
                prompts_needing_rewind.append(original_prompts[prompt_idx])
                prompt_indices_needing_rewind.append(prompt_idx)
                continue

            generated_response = candidates[0].strip()

            # Validate that the response contains answer tags (plan)
            if not self.validate_response_has_answers(generated_response):
                # Check if this response hit max length without generating a plan
                response_tokens = self.tokenizer.encode(generated_response)
                if len(response_tokens) >= curr_max_tokens[prompt_idx]:
                    # Hit max length without plan - add to rewind pool instead of forcing completion
                    max_length_without_plan_indices.append(prompt_idx)
                    prompts_needing_rewind.append(original_prompts[prompt_idx])
                    prompt_indices_needing_rewind.append(prompt_idx)
                    print(f"      Prompt {prompt_idx}: Hit max length without generating plan, adding to rewind pool")
                else:
                    # Didn't hit max length but still no answer tags - needs rewind
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
            plans_to_evaluate.append({"prompt_idx": prompt_idx, "plan": extracted_plan, "response": generated_response})

        # Print summary of initial generation results
        print(f"  Initial generation results:")
        print(f"    ✓ Plans ready for evaluation: {len(plans_to_evaluate)} (indices: {[item['prompt_idx'] for item in plans_to_evaluate]})")
        print(f"    ⚠️  Plans needing rewind: {len(prompts_needing_rewind)} (indices: {prompt_indices_needing_rewind})")
        if max_length_without_plan_indices:
            print(f"    🔄 Max length without plan (added to rewind): {len(max_length_without_plan_indices)} (indices: {max_length_without_plan_indices})")

        # Evaluate all initial plans with autorater
        if plans_to_evaluate:
            print(f"  Evaluating {len(plans_to_evaluate)} initial plans...")

            # Prepare batch for autorater evaluation
            eval_questions = []
            eval_plans = []

            for item in plans_to_evaluate:
                prompt_idx = item["prompt_idx"]
                plan = item["plan"]

                # Use explicit task if available, otherwise use original question
                evaluation_question = original_prompts[prompt_idx]
                if meta_info and "explicit_tasks" in meta_info:
                    explicit_tasks = meta_info["explicit_tasks"]
                    if prompt_idx < len(explicit_tasks) and explicit_tasks[prompt_idx]:
                        evaluation_question = explicit_tasks[prompt_idx]

                eval_questions.append(evaluation_question)
                eval_plans.append(plan)

            # Call autorater service in batch
            autorater_payload = {
                "prompts": eval_questions,
                "responses": eval_plans,
                "gt_answers": [""] * len(eval_plans),
                "template_types": ["plan_quality_evaluation"] * len(eval_plans),
            }

            autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(self.plan_evaluation_service_url, autorater_payload, batch_size=len(eval_plans))

            # Process evaluation results
            approved_indices = []
            rejected_indices = []

            for i, item in enumerate(plans_to_evaluate):
                prompt_idx = item["prompt_idx"]
                plan = item["plan"]
                response = item["response"]

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
                        generation_history[prompt_idx]["total_regeneration_attempts"] += 1
                        turn_data = {"turn_number": 1, "approved_plan": plan, "rejected_plans": [], "total_attempts": 1, "plan_approved": True}
                        generation_history[prompt_idx]["turns"].append(turn_data)
                    else:
                        rejected_indices.append(prompt_idx)
                        # Store the rejected plan
                        rejected_plans_per_prompt[prompt_idx].append(plan)
                        prompts_needing_rewind.append(original_prompts[prompt_idx])
                        prompt_indices_needing_rewind.append(prompt_idx)
                else:
                    rejected_indices.append(prompt_idx)
                    # Store the rejected plan (even if extraction failed)
                    rejected_plans_per_prompt[prompt_idx].append("Failed to extract plan")
                    prompts_needing_rewind.append(original_prompts[prompt_idx])
                    prompt_indices_needing_rewind.append(prompt_idx)

            # Print summary of evaluation results
            print(f"  Evaluation results:")
            print(f"    ✅ Plans approved: {len(approved_indices)} (indices: {approved_indices})")
            print(f"    ❌ Plans rejected: {len(rejected_indices)} (indices: {rejected_indices})")

        # Handle rewind attempts for rejected plans
        if prompts_needing_rewind:
            print(f"  Processing {len(prompts_needing_rewind)} prompts needing rewind...")
            if max_length_without_plan_indices:
                print(f"    Note: {len(max_length_without_plan_indices)} prompts hit max length without generating plan and will be rewound")

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

                    # Get rejected plans for this prompt
                    rejected_plans = rejected_plans_per_prompt[prompt_idx]

                    # Create rewind prompt with explicit task if available
                    explicit_task = None
                    if "explicit_tasks" in meta_info and prompt_idx < len(meta_info["explicit_tasks"]):
                        explicit_task = meta_info["explicit_tasks"][prompt_idx]

                    # Check if this prompt hit max length without generating a plan
                    hit_max_length = prompt_idx in max_length_without_plan_indices
                    
                    rewind_prompt = self.create_rewind_prompt(
                        meta_info["original_prompt"][prompt_idx], 
                        rejected_plans, 
                        1, 
                        explicit_task,
                        hit_max_length
                    )

                    rewind_prompts.append(rewind_prompt)
                    rewind_max_tokens.append(curr_max_tokens[prompt_idx])
                    rewind_seeds.append(prompt_idx + rewind_attempts[prompt_idx])
                    rewind_prompt_indices.append(prompt_idx)

                if not rewind_prompts:
                    break

                # Generate rewind plans in batch
                rewind_candidates = self.generate_text_with_model_batch(prompts=rewind_prompts, num_outputs=1, max_new_tokens_list=rewind_max_tokens, temperature=0.6, top_p=0.9, seeds=rewind_seeds)

                # Track tokens generated in rewind attempts
                for i, prompt_idx in enumerate(rewind_prompt_indices):
                    if rewind_candidates[i] and rewind_candidates[i][0].strip():
                        response_text = rewind_candidates[i][0].strip()
                        response_tokens = self.tokenizer.encode(response_text)
                        generation_history[prompt_idx]["total_tokens_generated"] += len(response_tokens)

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
                    is_approved, confidence, explanation = self.evaluate_plan_quality(meta_info["original_prompt"][prompt_idx], extracted_plan, prompt_idx, meta_info)

                    if is_approved:
                        approved_in_this_round.append(prompt_idx)

                        # Add the full approved response (including thinking parts) to the current input
                        full_response_ids = self.tokenizer.encode(generated_response)
                        curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx].copy(), full_response_ids])

                        # Update generation history
                        generation_history[prompt_idx]["total_regeneration_attempts"] += rewind_attempts[prompt_idx]
                        turn_data = {"turn_number": 1, "approved_plan": extracted_plan, "rejected_plans": rejected_plans_per_prompt[prompt_idx].copy(), "total_attempts": rewind_attempts[prompt_idx], "plan_approved": True}
                        generation_history[prompt_idx]["turns"].append(turn_data)

                        print(f"        ✓ Prompt {prompt_idx}: Plan approved after {rewind_attempts[prompt_idx]} rewind attempts")
                    else:
                        # Store the rejected plan
                        rejected_plans_per_prompt[prompt_idx].append(extracted_plan)
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

            # Handle prompts that exceeded max rewind attempts - continue with most recent rejected plan
            exceeded_max_attempts = []
            for prompt_idx in prompt_indices_needing_rewind:
                if rewind_attempts[prompt_idx] > max_rewind_attempts:
                    exceeded_max_attempts.append(prompt_idx)

                    # Get the most recent rejected plan to continue generation
                    rejected_plans = rejected_plans_per_prompt[prompt_idx]
                    if rejected_plans:
                        most_recent_plan = rejected_plans[-1]
                        print(f"      Prompt {prompt_idx}: Using most recent rejected plan for continuation: {most_recent_plan[:100]}...")
                        
                        # Create a response with the rejected plan (wrapped in <answer> tags)
                        rejected_response = f"<answer>{most_recent_plan}</answer><think>"
                        rejected_response_ids = self.tokenizer.encode(rejected_response)
                        
                        # Add the rejected plan response to current input
                        curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx].copy(), rejected_response_ids])
                        
                        # Update generation history
                        generation_history[prompt_idx]["total_regeneration_attempts"] += max_rewind_attempts
                        turn_data = {"turn_number": 1, "approved_plan": most_recent_plan, "rejected_plans": rejected_plans.copy(), "total_attempts": max_rewind_attempts, "plan_approved": False, "continued_with_rejected": True}
                        generation_history[prompt_idx]["turns"].append(turn_data)
                        
                        print(f"        ✓ Continuing generation with rejected plan")
                    else:
                        # No rejected plans available, update history but don't continue
                        generation_history[prompt_idx]["total_regeneration_attempts"] += max_rewind_attempts
                        turn_data = {"turn_number": 1, "approved_plan": "", "rejected_plans": [], "total_attempts": max_rewind_attempts, "plan_approved": False, "continued_with_rejected": False}
                        generation_history[prompt_idx]["turns"].append(turn_data)
                        print(f"        ❌ No rejected plans available, stopping generation")

            if exceeded_max_attempts:
                print(f"    ⚠️  Max rewind attempts exceeded for {len(exceeded_max_attempts)} prompts (indices: {exceeded_max_attempts})")
                print(f"      Continuing generation with most recent rejected plans for these prompts")

        # TURN 2: Answer Generation
        print(f"\n🔄 TURN 2: Answer Generation")
        
        # Get all prompts that have completed turn 1 (either approved plan or continued with rejected plan)
        active_for_turn2 = []
        for i in range(batch_size):
            if len(generation_history[i]["turns"]) > 0:  # Has completed turn 1
                active_for_turn2.append(i)
        
        print(f"  Continuing with {len(active_for_turn2)} prompts that completed turn 1")
        
        if active_for_turn2:
            # Prepare batch inputs for answer generation
            batch_prompts = []
            batch_max_tokens = []
            batch_seeds = []
            batch_prompt_indices = []

            for prompt_idx in active_for_turn2:
                # Get the full accumulated response from turn 1 (including original prompt)
                full_response = self.tokenizer.decode(curr_inputs[prompt_idx], skip_special_tokens=False)

                # Remove the im_end token for continued generation and replace with a <think> tag
                eos_token = self.tokenizer.special_tokens_map["eos_token"]

                # check if the last token is eos_token
                if full_response.endswith(eos_token):
                    full_response = full_response[:-len(eos_token)] + "<think>"
                else:
                    if full_response.endswith("</think>"):
                        pass
                    else:
                        full_response = full_response + "<think>"

                batch_prompts.append(full_response)
                batch_max_tokens.append(curr_max_tokens[prompt_idx])
                batch_seeds.append(prompt_idx + 100)  # Different seed for turn 2
                batch_prompt_indices.append(prompt_idx)

            # Generate answers in batch
            answer_candidates = self.generate_text_with_model_batch(prompts=batch_prompts, num_outputs=1, max_new_tokens_list=batch_max_tokens, temperature=0.6, top_p=0.9, seeds=batch_seeds)

            # Track tokens generated in answer generation
            for i, prompt_idx in enumerate(batch_prompt_indices):
                if answer_candidates[i] and answer_candidates[i][0].strip():
                    response_text = answer_candidates[i][0].strip()
                    response_tokens = self.tokenizer.encode(response_text)
                    generation_history[prompt_idx]["total_tokens_generated"] += len(response_tokens)

            # Process answer generation results
            for i, prompt_idx in enumerate(batch_prompt_indices):
                candidates = answer_candidates[i]
                if candidates and candidates[0].strip():
                    answer = candidates[0].strip()

                    # answer should start with <think>
                    if not answer.startswith("<think>"):
                        answer = "<think>" + answer

                    # Add answer to current input
                    answer_ids = self.tokenizer.encode(answer)
                    curr_inputs[prompt_idx] = np.concatenate([curr_inputs[prompt_idx], answer_ids])

                    # Update generation history
                    turn_data = {
                        "turn_number": 2,
                        "approved_plan": answer,
                        "rejected_plans": [],
                        "total_attempts": 1,
                        "plan_approved": True,  # No evaluation needed on turn 2
                    }
                    generation_history[prompt_idx]["turns"].append(turn_data)

                    print(f"    ✓ Prompt {prompt_idx}: Answer generation completed")
                else:
                    print(f"    ❌ Prompt {prompt_idx}: No answer generated")

        # import ipdb; ipdb.set_trace()
        # Collect final responses and update generation history
        response_list = []

        for i in range(batch_size):
            # Get the final accumulated generation for this prompt
            input_len = len(init_inputs[i])
            response_ids = curr_inputs[i][input_len:]

            # Use the accumulated generation directly
            response_list.append(response_ids)

            # Update the final response in generation history
            generation_history[i]["final_response"] = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Force completion if needed
        if self.force_answer_completion:
            print(f"\n🔍 Checking for incomplete answers and forcing completion...")
            
            # Check which responses need answer completion
            to_force_answer_indices = []
            batched_force_answer_inputs = []
            batched_force_answer_responses = []
            
            # Check for incomplete answers after turn 2
            for i, response_ids in enumerate(response_list):
                if response_ids is None:
                    continue
                    
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # Count <answer></answer> blocks
                answer_blocks = response_text.count("<answer>")
                answer_end_blocks = response_text.count("</answer>")
                
                # Check if we have at least 2 complete answer blocks
                if answer_blocks < 2 or answer_end_blocks < 2:
                    print(f"  Prompt {i}: Incomplete answer after turn 2 (found {answer_blocks} <answer> and {answer_end_blocks} </answer>), forcing completion")
                    
                    # Force answer completion by adding </think><answer> to force the model to answer
                    if not response_text.endswith("</answer>"):
                        if not response_text.endswith("</think>"):
                            response_text += "</think>"
                        if not response_text.endswith("<answer>"):
                            response_text += "<answer>"
                    
                        # Encode the forced response
                        forced_response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
                        
                        # Prepare continuation input
                        continuation_input = np.concatenate([init_inputs[i].copy(), forced_response_ids])
                        
                        to_force_answer_indices.append(i)
                        batched_force_answer_inputs.append(continuation_input)
                        batched_force_answer_responses.append(forced_response_ids)
                        
                        # Placeholder; will be replaced after forced generation
                        response_list[i] = None
                else:
                    print(f"  Prompt {i}: Complete answer after turn 2 (found {answer_blocks} <answer> and {answer_end_blocks} </answer>)")
            
            # Run forced answer completion if needed
            if to_force_answer_indices:
                print(f"  Forcing answer completion for {len(to_force_answer_indices)} prompts...")
                
                # Generate additional tokens to complete the answer
                with self.update_sampling_params(max_tokens=self.additional_answer_tokens, stop=None):
                    force_answer_outputs = self.inference_engine.generate(
                        prompts=[{"prompt_token_ids": input_ids.tolist()} for input_ids in batched_force_answer_inputs],
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )
                
                # Merge forced answer results back
                for j, force_idx in enumerate(to_force_answer_indices):
                    output = force_answer_outputs[j]
                    continuation_ids = output.outputs[0].token_ids
                    
                    # Combine forced response with continuation
                    final_response = batched_force_answer_responses[j] + continuation_ids
                    
                    # Update response list
                    response_list[force_idx] = final_response
                    
                    # Update generation history with forced completion tokens
                    generation_history[force_idx]["total_tokens_generated"] += len(continuation_ids)
                    
                    # Update the final response in generation history to reflect the complete response
                    generation_history[force_idx]["final_response"] = self.tokenizer.decode(final_response, skip_special_tokens=True)
                    
                    print(f"    Prompt {force_idx}: Added {len(continuation_ids)} forced completion tokens")
                    print(f"      Total tokens for prompt {force_idx}: {generation_history[force_idx]['total_tokens_generated']:,} (including forced completion)")
        else:
            print(f"\n⚠️  Force answer completion disabled, skipping answer completion check")
            
            # When force answer completion is disabled, ensure we have the final responses
            for i, response_ids in enumerate(response_list):
                if response_ids is not None:
                    # Update the final response in generation history
                    generation_history[i]["final_response"] = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Convert responses back to tensors
        final_response_list = []
        for i, response_ids in enumerate(response_list):
            if response_ids is None:
                # This shouldn't happen, but handle gracefully
                print(f"Warning: Response {i} is None, using empty response")
                final_response_list.append(torch.tensor([], dtype=torch.long, device=idx.device))
            else:
                final_response_list.append(torch.tensor(response_ids, dtype=torch.long, device=idx.device))

        # Pad responses to uniform length
        response = pad_2d_list_to_length(final_response_list, self.pad_token_id, max_length=self.config.response_length).to(idx.device)

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
        
        # Print summary of total tokens generated
        total_tokens_all_prompts = sum(gh["total_tokens_generated"] for gh in generation_history)
        tokens_per_prompt = [gh["total_tokens_generated"] for gh in generation_history]
        print(f"\n📊 Total Tokens Generated Summary:")
        print(f"Tokens per prompt: {tokens_per_prompt}")
        print(f"  Total tokens across all prompts: {total_tokens_all_prompts:,}")
        print(f"  Average tokens per prompt: {total_tokens_all_prompts / batch_size:.1f}")
        
        # Print detailed rewind statistics
        prompts_needing_rewind = []
        rewind_attempts_breakdown = {}
        
        for i, gh in enumerate(generation_history):
            if gh['total_regeneration_attempts'] > 1:  # More than just initial generation
                prompts_needing_rewind.append(i)
                rewind_count = gh['total_regeneration_attempts'] - 1  # Subtract initial generation
                if rewind_count not in rewind_attempts_breakdown:
                    rewind_attempts_breakdown[rewind_count] = []
                rewind_attempts_breakdown[rewind_count].append(i)
        
        if prompts_needing_rewind:
            print(f"\n🔄 Rewind Statistics:")
            print(f"  Total prompts needing rewind: {len(prompts_needing_rewind)} (indices: {prompts_needing_rewind})")
            
            # Print breakdown by number of rewind attempts
            for rewind_count in sorted(rewind_attempts_breakdown.keys()):
                prompt_indices = rewind_attempts_breakdown[rewind_count]
                print(f"  Prompts with {rewind_count} rewind attempt(s): {len(prompt_indices)} (indices: {prompt_indices})")
            
            # Print detailed breakdown per prompt
            print(f"\n📊 Detailed Rewind Breakdown per Prompt:")
            for i, gh in enumerate(generation_history):
                if gh['total_regeneration_attempts'] > 1:
                    rewind_count = gh['total_regeneration_attempts'] - 1
                    print(f"    Prompt {i}: {rewind_count} rewind attempt(s)")
                    
                    # Show details about each turn
                    for turn_idx, turn in enumerate(gh['turns']):
                        if turn.get('plan_approved', False):
                            status = "✅ APPROVED"
                        elif turn.get('continued_with_rejected', False):
                            status = "🔄 CONTINUED WITH REJECTED"
                        else:
                            status = "❌ REJECTED"
                        
                        print(f"      Turn {turn_idx + 1}: {status} ({turn.get('total_attempts', 1)} attempt(s))")
        else:
            print(f"\n🔄 Rewind Statistics:")
            print(f"  No prompts needed rewind - all plans were approved on first attempt! 🎉")


        # Validate final response structure
        # print(f"\n🔍 Validating final response structure...")
        # from helpers import parse_interleaved_components
        
        # responses_with_issues = []
        # for i, response_ids in enumerate(response_list):
        #     if response_ids is not None:
        #         response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        #         interleaved_components = parse_interleaved_components(response_text)
                
        #         # Check if we have the expected structure: 2 think blocks and 2 answer blocks
        #         think_components = [comp for comp in interleaved_components if comp['type'] == 'think']
        #         answer_components = [comp for comp in interleaved_components if comp['type'] == 'answer']
                
        #         if len(think_components) < 2 or len(answer_components) < 2:
        #             responses_with_issues.append({
        #                 'prompt_idx': i,
        #                 'think_count': len(think_components),
        #                 'answer_count': len(answer_components),
        #                 'response_text': response_text[:200] + "..." if len(response_text) > 200 else response_text
        #             })
        #             print(f"  ❌ Prompt {i}: Expected 2 think + 2 answer blocks, got {len(think_components)} think + {len(answer_components)} answer")
        #         else:
        #             print(f"  ✅ Prompt {i}: Correct structure ({len(think_components)} think + {len(answer_components)} answer blocks)")


        # # Free vllm cache engine
        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
