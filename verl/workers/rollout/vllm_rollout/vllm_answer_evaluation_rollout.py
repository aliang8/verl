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
vLLM Answer Evaluation Rollout that implements answer generation with autorater feedback.

This rollout:
1. Generates response until max tokens is reached
2. Extracts the answer portion after </think> tags
3. Uses autorater to evaluate if the answer is correct given the explicit task
4. If answer is correct, generation is complete
5. If answer is incorrect and tokens remain, reprompts with rejected answer and tries again
6. Repeats until answer is approved or max tokens reached
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


class vLLMAnswerEvaluationRollout(vLLMBestOfN):
    """
    vLLM Answer Evaluation Rollout that implements answer generation with autorater feedback.

    This rollout:
    1. Generates response until max tokens is reached
    2. Extracts the answer portion after </think> tags
    3. Uses autorater to evaluate if the answer is correct given the explicit task
    4. If answer is correct, generation is complete
    5. If answer is incorrect and tokens remain, reprompts with rejected answer and tries again
    6. Repeats until answer is approved or max tokens reached
    """

    def __init__(self, model_path: str, config, tokenizer, model_hf_config, **kwargs):
        # Force n_candidates to 1 for this strategy
        config["n_candidates"] = 1

        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)

        # Answer evaluation specific configuration
        self.max_regeneration_attempts = config.get("max_regeneration_attempts", 3)  # Max regeneration attempts
        self.answer_evaluation_threshold = config.get("answer_evaluation_threshold", 0.5)  # Threshold for answer approval
        self.reprompt_template = config.get("reprompt_template", "default")

        # Override autorater service URL if specified
        if "answer_evaluation_service_url" in config:
            self.answer_evaluation_service_url = config["answer_evaluation_service_url"]
        else:
            self.answer_evaluation_service_url = self.autorater_service_url

        print(f"Initialized vLLMAnswerEvaluationRollout with max_regeneration_attempts={self.max_regeneration_attempts}")
        print(f"Answer evaluation threshold: {self.answer_evaluation_threshold}")
        print(f"Answer evaluation service: {self.answer_evaluation_service_url}")

    def extract_answer_after_think(self, response: str) -> str:
        """
        Extract the answer portion that comes after the last </think> tag.

        Args:
            response: The generated response text

        Returns:
            The answer portion after </think>, or empty string if no </think> found
        """
        if not response or not isinstance(response, str):
            return ""

        # Find the last occurrence of </think>
        think_end_pattern = r"</think>"
        think_end_matches = list(re.finditer(think_end_pattern, response, re.IGNORECASE))

        if not think_end_matches:
            # No </think> found, return the entire response as answer
            return response.strip()

        # Get the last </think> position
        last_think_end = think_end_matches[-1].end()

        # Extract everything after the last </think>
        answer = response[last_think_end:].strip()

        return answer

    def evaluate_answer_correctness(self, explicit_task: str, answer: str, prompt_idx: int = 0, meta_info: Dict = None) -> Tuple[bool, float, str]:
        """
        Use the autorater service to evaluate if an answer is correct given the explicit task.

        Args:
            explicit_task: The explicit task description
            answer: The generated answer to evaluate
            prompt_idx: Index of the current prompt
            meta_info: Additional metadata

        Returns:
            Tuple of (is_correct, confidence_score, explanation)
        """
        try:
            print(f"Evaluating answer correctness for prompt {prompt_idx}")
            print(f"Explicit task: {explicit_task[:100]}...")
            print(f"Answer: {answer[:100]}...")

            # Prepare autorater payload for answer correctness evaluation
            # Use coding_answer_correctness template for coding tasks
            template_type = "coding_answer_correctness"

            autorater_payload = {
                "prompts": [explicit_task],
                "responses": [answer],
                "gt_answers": [""],  # Empty ground truth for answer evaluation
                "template_types": [template_type],  # Use specialized template for coding tasks
            }

            # Call autorater service
            autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(self.answer_evaluation_service_url, autorater_payload, batch_size=1)

            # Parse the response to get the answer correctness decision
            if autorater_decisions and len(autorater_decisions) > 0:
                # The autorater service should return the parsed decision directly
                decision = autorater_decisions[0]

                # Handle different response formats
                if isinstance(decision, bool):
                    is_correct = decision
                    confidence_score = 1.0
                elif isinstance(decision, str):
                    # Convert string decision to boolean
                    is_correct = decision.upper() == "TRUE"
                    confidence_score = 1.0 if is_correct else 0.0
                else:
                    # Default to incorrect for unknown response types
                    is_correct = False
                    confidence_score = 0.0

                explanation = autorater_explanations[0] if autorater_explanations else ""
                print(f"Answer evaluator decision: {decision} (correct: {is_correct})")
                return is_correct, confidence_score, explanation
            else:
                print("Warning: No decision from answer evaluator, defaulting to incorrect")
                return False, 0.0, ""

        except Exception as e:
            print(f"Error calling autorater service for answer evaluation: {e}")
            # Fallback: mark answer as incorrect with low confidence
            return False, 0.0, ""

    def create_reprompt_with_rejected_answer(self, original_question: str, rejected_answer: str, attempt: int) -> str:
        """
        Create a reprompt for regeneration after answer rejection.

        Args:
            original_question: The original question/prompt
            rejected_answer: The previously rejected answer
            attempt: Current attempt number

        Returns:
            Reprompt for regeneration
        """
        if self.reprompt_template == "default":
            # Default reprompt template
            reprompt = f"""The following answer was evaluated as incorrect. Please generate a better answer.

Original Question: {original_question}

Rejected Answer (Attempt {attempt}):
{rejected_answer}

Please provide a new, improved answer that correctly addresses the question."""

        else:
            raise ValueError(f"Invalid reprompt template: {self.reprompt_template}")

        messages = [format_system_message("default"), {"role": "user", "content": reprompt}]

        reprompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return reprompt

    def validate_response_has_think_tags(self, response: str) -> bool:
        """
        Validate that a generated response contains <think> tags.

        Args:
            response: The generated response text

        Returns:
            True if response contains think tags, False otherwise
        """
        return "<think>" in response and "</think>" in response

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using answer evaluation strategy.

        This method:
        1. Generates response until max tokens is reached
        2. Extracts answer after </think> tags
        3. Evaluates answer correctness with autorater
        4. If incorrect and tokens remain, reprompts and tries again
        5. Continues until answer is correct or max attempts reached
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

        print(f"Initialized {batch_size} prompts for answer evaluation strategy")

        # Initialize generation history for each prompt
        generation_history = []
        for i in range(batch_size):
            generation_history.append({"prompt_idx": i, "original_prompt": original_prompts[i], "attempts": [], "final_response": "", "total_regeneration_attempts": 0, "final_answer_approved": False, "total_tokens_generated": 0})

        # Main generation loop with answer evaluation
        while active_indices:
            print(f"\n🔄 Processing {len(active_indices)} active prompts...")

            # Prepare batch inputs for generation
            batch_prompts = []
            batch_max_tokens = []
            batch_seeds = []
            batch_prompt_indices = []

            for prompt_idx in active_indices:
                # Get the current accumulated response (including original prompt)
                full_response = self.tokenizer.decode(curr_inputs[prompt_idx], skip_special_tokens=False)

                # Remove the im_end token for continued generation
                full_response = full_response.replace(self.tokenizer.special_tokens_map["eos_token"], "")

                batch_prompts.append(full_response)
                batch_max_tokens.append(curr_max_tokens[prompt_idx])
                batch_seeds.append(prompt_idx + generation_history[prompt_idx]["total_regeneration_attempts"])
                batch_prompt_indices.append(prompt_idx)

            # Generate responses in batch
            print(f"  Generating responses for {len(batch_prompts)} prompts...")
            generation_candidates = self.generate_text_with_model_batch(
                prompts=batch_prompts,
                num_outputs=1,  # Always 1 for this strategy
                max_new_tokens_list=batch_max_tokens,
                temperature=1.2,
                top_p=0.9,
                seeds=batch_seeds,
            )

            # Process generation results and evaluate answers
            new_active_indices = []

            for i, prompt_idx in enumerate(batch_prompt_indices):
                candidates = generation_candidates[i]
                if not candidates or not candidates[0].strip():
                    print(f"    Prompt {prompt_idx}: ❌ No response generated, stopping generation")
                    continue

                generated_response = candidates[0].strip()

                # Validate that the response contains think tags
                if not self.validate_response_has_think_tags(generated_response):
                    print(f"    Prompt {prompt_idx}: ⚠️  Response missing <think> tags, stopping generation")
                    continue

                # Extract the answer portion after </think>
                extracted_answer = self.extract_answer_after_think(generated_response)

                if not extracted_answer:
                    print(f"    Prompt {prompt_idx}: ⚠️  No answer found after </think>, stopping generation")
                    continue

                # Get explicit task for evaluation
                explicit_task = original_prompts[prompt_idx]
                if meta_info and "explicit_tasks" in meta_info:
                    explicit_tasks = meta_info["explicit_tasks"]
                    if prompt_idx < len(explicit_tasks) and explicit_tasks[prompt_idx]:
                        explicit_task = explicit_tasks[prompt_idx]

                # Evaluate answer correctness
                is_correct, confidence, explanation = self.evaluate_answer_correctness(explicit_task, extracted_answer, prompt_idx, meta_info)

                # Update generation history
                attempt_data = {
                    "attempt_number": generation_history[prompt_idx]["total_regeneration_attempts"] + 1,
                    "full_response": generated_response,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "explanation": explanation,
                    "tokens_generated": len(self.tokenizer.encode(generated_response)),
                }
                generation_history[prompt_idx]["attempts"].append(attempt_data)
                generation_history[prompt_idx]["total_regeneration_attempts"] += 1
                generation_history[prompt_idx]["total_tokens_generated"] += attempt_data["tokens_generated"]

                if is_correct:
                    # Answer is correct, mark as complete
                    generation_history[prompt_idx]["final_response"] = generated_response
                    generation_history[prompt_idx]["final_answer_approved"] = True

                    # Add the full response to current input
                    full_response_ids = self.tokenizer.encode(generated_response)
                    curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx].copy(), full_response_ids])

                else:
                    # Answer is incorrect, check if we can try again
                    if generation_history[prompt_idx]["total_regeneration_attempts"] < self.max_regeneration_attempts:
                        # Create reprompt with rejected answer
                        reprompt = self.create_reprompt_with_rejected_answer(original_prompts[prompt_idx], extracted_answer, generation_history[prompt_idx]["total_regeneration_attempts"])

                        # Restart from scratch with the reprompt (don't accumulate previous response)
                        reprompt_ids = self.tokenizer.encode(reprompt)
                        curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx].copy(), reprompt_ids])
                        curr_max_tokens[prompt_idx] = self.config.response_length  # Reset to full response length

                        # Keep this prompt active for next iteration
                        new_active_indices.append(prompt_idx)

                    else:
                        # Max attempts reached
                        generation_history[prompt_idx]["final_response"] = generated_response
                        generation_history[prompt_idx]["final_answer_approved"] = False

                        # Add the final response to current input
                        full_response_ids = self.tokenizer.encode(generated_response)
                        curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx].copy(), full_response_ids])

            # Update active indices for next iteration
            active_indices = new_active_indices

            # Print summary of this iteration
            approved_indices = [i for i in range(batch_size) if generation_history[i]["final_answer_approved"]]
            rejected_indices = [i for i in range(batch_size) if not generation_history[i]["final_answer_approved"] and generation_history[i]["final_response"]]
            still_active_indices = [i for i in active_indices]

            print(f"  Iteration Summary:")
            if approved_indices:
                print(f"    ✅ Approved answers: {approved_indices}")
            if rejected_indices:
                print(f"    ❌ Rejected answers: {rejected_indices}")
            if still_active_indices:
                print(f"    🔄 Still active: {still_active_indices}")

            # Check if any prompts have reached max length
            final_active_indices = []
            for prompt_idx in active_indices:
                if len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx]) >= self.config.response_length:
                    # Truncate to response length
                    curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx], curr_inputs[prompt_idx][len(init_inputs[prompt_idx]) : len(init_inputs[prompt_idx]) + self.config.response_length]])
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

            # Update the final response in generation history if not already set
            if not generation_history[i]["final_response"]:
                generation_history[i]["final_response"] = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Update the final response in generation history if not already set
            if not generation_history[i]["final_response"]:
                generation_history[i]["final_response"] = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Print final summary
        approved_indices = [i for i in range(batch_size) if generation_history[i]["final_answer_approved"]]
        rejected_indices = [i for i in range(batch_size) if not generation_history[i]["final_answer_approved"]]

        # Calculate total tokens across all attempts
        total_tokens_all_attempts = sum(generation_history[i]["total_tokens_generated"] for i in range(batch_size))

        print(f"\n🎯 Final Generation Summary:")
        print(f"  ✅ Total approved answers: {len(approved_indices)} (indices: {approved_indices})")
        print(f"  ❌ Total rejected answers: {len(rejected_indices)} (indices: {rejected_indices})")
        print(f"  🔢 Total tokens generated across all attempts: {total_tokens_all_attempts:,}")

        for i in range(batch_size):
            print(f"  Prompt {i}: {len(response_ids)} tokens, {generation_history[i]['total_regeneration_attempts']} attempts, {generation_history[i]['total_tokens_generated']:,} total tokens, {'✅ Approved' if generation_history[i]['final_answer_approved'] else '❌ Rejected'}")

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
        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
