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
to select the best answer, and continues generation w78
ith the best answer.
"""

from email.charset import add_alias
import re
import os
from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
        self.n_candidates = config.get("n_candidates", 1)  # Number of candidates to generate
        self.autorater_service_url = config.get("autorater_service_url", "http://10.128.0.30:81")
        self.answer_stop_token = "</answer>"
        self.answer_start_token = "<answer>"
        
        # Similarity filtering configuration
        self.similarity_threshold = config.get("similarity_threshold", 0.85)  # Threshold for filtering similar plans
        self.use_similarity_filtering = config.get("use_similarity_filtering", True)
        
        # Iterative reprompting configuration
        self.enable_iterative_reprompting = config.get("enable_iterative_reprompting", True)
        self.max_iterative_iterations = config.get("max_iterative_iterations", 3)
        
        # Initialize sentence transformer for similarity filtering
        self.sentence_transformer = None
        if self.use_similarity_filtering:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"✓ Sentence transformer initialized for similarity filtering (threshold: {self.similarity_threshold})")
            except ImportError:
                print("⚠️  Sentence transformers not available, similarity filtering disabled")
                self.use_similarity_filtering = False
        
        # Get token IDs for answer tokens
        self.answer_stop_token_ids = self.tokenizer.encode(self.answer_stop_token)
        self.answer_start_token_ids = self.tokenizer.encode(self.answer_start_token)
        
        print(f"Initialized vLLMAutoraterRollout with n_candidates={self.n_candidates}")
        print(f"Answer stop token: '{self.answer_stop_token}' -> token IDs: {self.answer_stop_token_ids}")
        print(f"Answer start token: '{self.answer_start_token}' -> token IDs: {self.answer_start_token_ids}")
        print(f"Iterative reprompting: enabled={self.enable_iterative_reprompting}, max_iterations={self.max_iterative_iterations}")
    
    
    def extract_answer_chunks(self, text: str) -> List[str]:
        """Extract all <answer></answer> chunks from text."""
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]
    
    def extract_last_answer(self, text: str) -> Optional[str]:
        """Extract the last <answer></answer> chunk from text."""
        chunks = self.extract_answer_chunks(text)
        return chunks[-1] if chunks else None
    
    def evaluate_response(self, question: str, candidates: List[str], prompt_idx: int = 0, meta_info: Dict = None) -> Tuple[int, float]:
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
        except Exception as e:
            print(f"Error calling autorater service for plan evaluation: {e}")
            # Fallback: return first candidate with default score
            return 0, 1.0
    
    def filter_similar_candidates(self, candidates: List[Dict], threshold: float = None) -> Tuple[List[Dict], List[int]]:
        """
        Filter out candidates that are too similar based on sentence embeddings of their answers.
        
        Args:
            candidates: List of candidate dictionaries with 'answer' field
            threshold: Similarity threshold (uses self.similarity_threshold if None)
            
        Returns:
            Tuple of (filtered_candidates, kept_indices)
        """
        if not self.use_similarity_filtering or not self.sentence_transformer or len(candidates) <= 1:
            return candidates, list(range(len(candidates)))
        
        if threshold is None:
            threshold = self.similarity_threshold
        
        # Extract answers from candidates for similarity comparison
        answers = [candidate.get('answer', '') for candidate in candidates]
        
        print(f"  Filtering {len(candidates)} candidates with similarity threshold: {threshold}")
        
        try:
            # Generate embeddings for all answers
            embeddings = self.sentence_transformer.encode(answers, convert_to_tensor=True)
            
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings.cpu().numpy())
            
            # Filter candidates based on similarity of their answers
            kept_indices = [0]  # Always keep the first candidate
            filtered_candidates = [candidates[0]]
            
            for i in range(1, len(candidates)):
                # Check similarity with all previously kept candidates
                max_similarity = max(similarity_matrix[i][j] for j in kept_indices)
                
                if max_similarity < threshold:
                    # This candidate is sufficiently different, keep it
                    kept_indices.append(i)
                    filtered_candidates.append(candidates[i])
                #     print(f"    Kept candidate {i+1} (max similarity: {max_similarity:.3f})")
                # else:
                #     print(f"    Filtered out candidate {i+1} (max similarity: {max_similarity:.3f} >= {threshold})")
            
            print(f"  Kept {len(filtered_candidates)}/{len(candidates)} candidates after similarity filtering")
            return filtered_candidates, kept_indices
            
        except Exception as e:
            print(f"  Error in similarity filtering: {e}, returning all candidates")
            return candidates, list(range(len(candidates)))
    
    def process_candidates_for_turn(self, candidates: List[str], prompt_idx: int, question: str, 
                                   init_inputs: List, curr_inputs: List, turn: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Process candidates for a specific turn, extracting answers and checking completion.
        
        Args:
            candidates: List of candidate generations for this turn
            prompt_idx: Index of the current prompt
            question: The original question text
            init_inputs: Initial input tokens for each prompt
            curr_inputs: Current accumulated input tokens for each prompt
            turn: Current turn number
            
        Returns:
            Tuple of (candidates_with_answers, candidates_without_answers)
        """
        candidates_with_answers = []
        candidates_without_answers = []
        
        for candidate_idx, current_turn_generation in enumerate(candidates):
            # The current input already contains all previous turns' generation
            # We just need to add the current turn's generation to it
            input_len = len(init_inputs[prompt_idx])
            
            # Create a copy of the current accumulated input
            full_generation_ids = curr_inputs[prompt_idx][input_len:].copy()
            
            # Add current turn's generation
            if current_turn_generation:
                current_turn_ids = self.tokenizer.encode(current_turn_generation)
                # Handle numpy array concatenation properly
                if len(full_generation_ids) == 0:
                    full_generation_ids = np.array(current_turn_ids)
                else:
                    full_generation_ids = np.concatenate([full_generation_ids, current_turn_ids])
            
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
                        'full_generation_ids': full_generation_ids,  # This now contains the complete generation
                        'current_turn_generation': current_turn_generation
                    })
                #     print(f"  Candidate {candidate_idx + 1}: Found {len(all_answers)} answers (expected {expected_answer_count}), using last answer")
                #     print(f"  Candidate {candidate_idx + 1}: Full generation length: {len(full_generation_ids)} tokens")
                #     print(f"  Candidate {candidate_idx + 1}: Contains <think>: {'<think>' in full_generation_text}")
                #     print(f"  Candidate {candidate_idx + 1}: Contains <answer>: {'<answer>' in full_generation_text}")
                #     print(f"  Candidate {candidate_idx + 1}: Full text preview: {full_generation_text[:200]}...")
                # else:
                #     print(f"  Candidate {candidate_idx + 1}: Found {len(all_answers)} answers but expected {expected_answer_count}, skipping")
            else:
                candidates_without_answers.append({
                    'candidate_idx': candidate_idx,
                    'current_turn_generation': current_turn_generation
                })
                print(f"  Candidate {candidate_idx + 1}: No </answer> reached yet")
        
        return candidates_with_answers, candidates_without_answers
    
    def generate_text_with_model(self, prompt: str, num_outputs: int = 1, max_new_tokens: int = 256, 
                                temperature: float = 0.8, top_p: float = 0.9, seed: int = None) -> List[str]:
        """
        Generic function to generate text using the model.
        
        Args:
            prompt: The input prompt text
            num_outputs: Number of outputs to generate (n parameter)
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            seed: Random seed for generation
            
        Returns:
            List of generated text strings
        """
        try:
            # Encode the prompt
            prompt_input = {"prompt_token_ids": self.tokenizer.encode(prompt)}
            
            # Use provided seed or generate one from prompt
            if seed is None:
                seed = hash(prompt) % 10000
            
            with self.update_sampling_params(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=num_outputs,
                stop=[self.answer_stop_token] if self.answer_stop_token else None,
                detokenize=True if self.answer_stop_token else None,
                seed=seed
            ):
                outputs = self.inference_engine.generate(
                    prompts=[prompt_input],
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )
                
                if outputs and outputs[0].outputs:
                    generated_texts = []
                    for output in outputs[0].outputs:
                        text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                        if text.strip():
                            generated_texts.append(text.strip())
                    
                    return generated_texts
                else:
                    print(f"  Warning: No output generated for prompt")
                    return [""] * num_outputs
                    
        except Exception as e:
            print(f"  Error generating text: {e}")
            return [""] * num_outputs
    
    def generate_text_with_model_batch(self, prompts: List[str], num_outputs: int = 1, max_new_tokens_list: List[int] = None, 
                                     temperature: float = 0.8, top_p: float = 0.9, seeds: List[int] = None) -> List[List[str]]:
        """
        Batch version of text generation for multiple prompts simultaneously.
        
        Args:
            prompts: List of input prompt texts
            num_outputs: Number of outputs to generate per prompt (n parameter)
            max_new_tokens_list: List of maximum tokens to generate for each prompt
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            seeds: List of random seeds for generation (one per prompt)
            
        Returns:
            List of lists of generated text strings (outer list = prompts, inner list = candidates)
        """
        if not prompts:
            return []
        
        # Use default max_new_tokens if not provided
        if max_new_tokens_list is None:
            max_new_tokens_list = [256] * len(prompts)
        
        # Use default seeds if not provided
        if seeds is None:
            seeds = [hash(prompt) % 10000 for prompt in prompts]
        
        try:
            # Encode all prompts
            prompt_inputs = [{"prompt_token_ids": self.tokenizer.encode(prompt)} for prompt in prompts]
            
            # Use the maximum max_new_tokens for batch processing (vLLM requirement)
            max_tokens = max(max_new_tokens_list)
            
            with self.update_sampling_params(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=num_outputs,
                stop=[self.answer_stop_token] if self.answer_stop_token else None,
                detokenize=True if self.answer_stop_token else None,
                seed=seeds[0] if seeds else None  # Use first seed for batch
            ):
                outputs = self.inference_engine.generate(
                    prompts=prompt_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )
                
                if outputs:
                    all_generated_texts = []
                    for i, output in enumerate(outputs):
                        if output.outputs:
                            prompt_candidates = []
                            for candidate_output in output.outputs:
                                text = self.tokenizer.decode(candidate_output.token_ids, skip_special_tokens=True)
                                if text.strip():
                                    prompt_candidates.append(text.strip())
                            
                            # Ensure we have the right number of candidates
                            while len(prompt_candidates) < num_outputs:
                                prompt_candidates.append("")
                            
                            all_generated_texts.append(prompt_candidates[:num_outputs])
                        else:
                            # No outputs for this prompt
                            all_generated_texts.append([""] * num_outputs)
                    
                    return all_generated_texts
                else:
                    print(f"  Warning: No outputs generated for batch")
                    return [[""] * num_outputs for _ in prompts]
                    
        except Exception as e:
            print(f"  Error generating text in batch: {e}")
            return [[""] * num_outputs for _ in prompts]
    
    def generate_candidates_for_prompt(self, prompt_idx: int, curr_inputs: List, curr_max_tokens: List) -> List[str]:
        """
        Generate candidates for a specific prompt using vLLM.
        
        Args:
            prompt_idx: Index of the current prompt
            curr_inputs: Current input tokens for each prompt
            curr_max_tokens: Maximum tokens to generate for each prompt
            
        Returns:
            List of candidate generations
        """
        # Decode the current input to get the prompt text
        prompt_text = self.tokenizer.decode(curr_inputs[prompt_idx])
        
        # Generate candidates using the generic function
        candidates = self.generate_text_with_model(
            prompt=prompt_text,
            num_outputs=self.n_candidates,
            max_new_tokens=curr_max_tokens[prompt_idx],
            temperature=1.2,
            top_p=0.9,
            seed=prompt_idx
        )
        
        return candidates
    
    def generate_candidates_for_all_prompts_batch(self, active_indices: List[int], curr_inputs: List, curr_max_tokens: List) -> List[List[str]]:
        """
        Generate candidates for all prompts simultaneously using vLLM batch processing.
        
        Args:
            active_indices: List of active prompt indices
            curr_inputs: Current input tokens for each prompt
            curr_max_tokens: Maximum tokens to generate for each prompt
            
        Returns:
            List of candidate lists for each prompt
        """
        if not active_indices:
            return []
        
        # Prepare batch inputs
        batch_prompts = []
        batch_max_tokens = []
        batch_seeds = []
        
        for prompt_idx in active_indices:
            # Decode the current input to get the prompt text
            prompt_text = self.tokenizer.decode(curr_inputs[prompt_idx])
            batch_prompts.append(prompt_text)
            batch_max_tokens.append(curr_max_tokens[prompt_idx])
            batch_seeds.append(prompt_idx)  # Use prompt_idx as seed for reproducibility
        
        # Generate all candidates simultaneously using batch processing
        all_candidates = self.generate_text_with_model_batch(
            prompts=batch_prompts,
            num_outputs=self.n_candidates,
            max_new_tokens_list=batch_max_tokens,
            temperature=1.2,
            top_p=0.9,
            seeds=batch_seeds
        )
        
        return all_candidates
    
    def generate_diverse_answers_batch(self, prompt: str, existing_answers: List[str], num_to_generate: int, max_new_tokens: int = 256) -> List[str]:
        """
        Generate multiple new answers that are different from existing answers.
        
        Args:
            prompt: The original prompt
            existing_answers: List of existing answers to avoid
            num_to_generate: Number of new answers to generate
            max_new_tokens: Maximum tokens for each new answer
            
        Returns:
            List of new diverse full responses (including think sections)
        """
        if num_to_generate <= 0:
            return []
        
        if not existing_answers:
            # If no existing answers, generate normal answers
            return self.generate_text_with_model(
                prompt=prompt,
                num_outputs=num_to_generate,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9
            )

        prompt = prompt.replace("<|im_end|>\n<|im_start|>assistant", "")
        # choose 3 random existing answers
        import random
        negative_examples = random.sample(existing_answers, min(5, len(existing_answers)))

        # Construct the diverse generation prompt
        diverse_prompt = f"""{prompt}\
Generate a different answer that is DIFFERENT from the existing approaches shown below.
Existing approaches to avoid:
{chr(10).join(negative_examples)}<im_end>\n<im_start|>assistant\n"""
        
        # Generate diverse answers using the generic function
        new_responses = self.generate_text_with_model(
            prompt=diverse_prompt,
            num_outputs=10,
            max_new_tokens=1024,
            temperature=1.2,
            top_p=0.9
        )

        # Return the full responses without extracting answers
        # The answer extraction will happen later when creating synthetic candidates
        return new_responses

    def create_synthetic_candidate(self, answer: str, candidate_idx: int, candidates: List, 
                                   prompt_idx: int, curr_inputs: List, init_inputs: List, 
                                   full_response: str = None) -> Dict:
        """
        Create a synthetic candidate from a generated answer.
        
        Args:
            answer: The generated answer text
            candidate_idx: Index of the new candidate
            candidates: Original candidates list for reference
            prompt_idx: Index of the current prompt
            curr_inputs: Current accumulated input tokens for each prompt
            init_inputs: Initial input tokens for each prompt
            full_response: The complete generated response (including think sections)
            
        Returns:
            Synthetic candidate dictionary with complete generation context
        """
        if full_response:
            # Use the full response directly to preserve think sections
            full_generation_ids = self.tokenizer.encode(full_response)
            full_generation_text = full_response
        else:
            # Fallback: get the current accumulated generation context and add the answer
            input_len = len(init_inputs[prompt_idx])
            current_context_ids = curr_inputs[prompt_idx][input_len:].copy()
            
            # Ensure we have integer token IDs
            if hasattr(current_context_ids, 'cpu'):
                current_context_ids = current_context_ids.cpu().numpy()
            if hasattr(current_context_ids, 'dtype') and current_context_ids.dtype != np.int64:
                current_context_ids = current_context_ids.astype(np.int64)
            
            # Add the new answer to the context
            answer_ids = self.tokenizer.encode(f"<answer>{answer}</answer>")
            answer_ids = np.array(answer_ids, dtype=np.int64)
            
            # Handle empty context case
            if len(current_context_ids) == 0:
                full_generation_ids = answer_ids
            else:
                full_generation_ids = np.concatenate([current_context_ids, answer_ids])
            
            # Decode to get the full text
            full_generation_text = self.tokenizer.decode(full_generation_ids)
        
        return {
            'candidate_idx': candidate_idx,
            'answer': answer,
            'full_generation_text': full_generation_text,
            'full_generation_ids': full_generation_ids,
            'current_turn_generation': f"<answer>{answer}</answer>",
            'is_synthetic': True  # Mark as synthetically generated
        }
    
    def continue_from_candidate(self, candidate: str, prompt_idx: int, curr_inputs: List, 
                               init_inputs: List, curr_max_tokens: List, new_active_indices: List) -> None:
        """
        Continue generation from a specific candidate.
        
        Args:
            candidate: The candidate text to continue from
            prompt_idx: Index of the current prompt
            curr_inputs: Current input tokens for each prompt
            init_inputs: Initial input tokens for each prompt
            curr_max_tokens: Maximum tokens to generate for each prompt
            new_active_indices: List to append active prompt indices to
        """
        if candidate:
            candidate_ids = self.tokenizer.encode(candidate)
            curr_inputs[prompt_idx].extend(candidate_ids)
            
            current_length = len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx])
            if current_length < self.config.response_length:
                new_active_indices.append(prompt_idx)
                curr_max_tokens[prompt_idx] = self.config.response_length - current_length
    
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
        
        # Initialize generation history for each prompt
        generation_history = []
        for i in range(batch_size):
            generation_history.append({
                'prompt_idx': i,
                'original_prompt': original_prompts[i],
                'turns': [],
                'final_response': ''
            })
        
        # Multi-turn generation with autorater
        max_turns = self.config.get("max_turns", 2)

        print(f"Multi-turn generation: max_turns={max_turns}")
        
        for turn in range(max_turns):
            if not active_indices:
                break
                
            print(f"Turn {turn + 1}: Processing {len(active_indices)} active prompts")
            print(f"  Expected answer count for this turn: {turn + 1}")

            # Generate candidates for all prompts simultaneously using batch processing
            print(f"Generating {self.n_candidates} candidates for {len(active_indices)} prompts simultaneously...")
            
            all_candidates = self.generate_candidates_for_all_prompts_batch(active_indices, curr_inputs, curr_max_tokens)
            prompt_indices = active_indices  # Keep original order
            
            print(f"Generated {len(active_indices)} prompt sets, each with {self.n_candidates} candidates")
            
            # Process each prompt's candidates and use autorater to select best answers
            new_active_indices = []
            
            for i, prompt_idx in enumerate(prompt_indices):
                print("="*100)
                print(f"Processing prompt {prompt_idx}, turn {turn + 1}")
                print("="*100)
                
                candidates = all_candidates[i]
                question = original_prompts[prompt_idx]
                
                print(f"Prompt {prompt_idx}: Processing {len(candidates)} candidates")
                
                # Check if any candidate has reached </answer> and extract answers
                candidates_with_answers, candidates_without_answers = self.process_candidates_for_turn(candidates, prompt_idx, question, init_inputs, curr_inputs, turn)

                # Evaluate and select the best candidate if any have reached </answer>
                if len(candidates_with_answers) > 0:
                    print(f"Prompt {prompt_idx}: {len(candidates_with_answers)}/{len(candidates)} candidates reached </answer>, evaluating and selecting best one")
                    
                    # Apply similarity filtering to original candidates first
                    if self.use_similarity_filtering and len(candidates_with_answers) > 1:
                        print(f"  Applying similarity filtering to {len(candidates_with_answers)} original candidates...")
                        filtered_original_candidates, kept_indices = self.filter_similar_candidates(candidates_with_answers)
                        
                        if len(filtered_original_candidates) < len(candidates_with_answers):
                            print(f"  Reduced from {len(candidates_with_answers)} to {len(filtered_original_candidates)} diverse original candidates")
                    else:
                        filtered_original_candidates = candidates_with_answers

                    candidates_with_answers = filtered_original_candidates
                    
                    # Iterative answer generation: generate additional diverse answers if needed
                    # Only do this on the first turn (turn 0) to establish the initial diverse set
                    if turn == 0 and self.enable_iterative_reprompting:
                        target_diverse_candidates = min(self.n_candidates, 10)  # Cap at 10 to avoid infinite loops
                        
                        if len(candidates_with_answers) < target_diverse_candidates:
                            iteration = 0
                            while len(candidates_with_answers) < target_diverse_candidates:
                                num_needed = target_diverse_candidates - len(candidates_with_answers)
                                print(f"  Need {num_needed} more diverse candidates, generating batch...")
                                
                                # Extract existing answers for diversity comparison
                                existing_answers = [c['answer'] for c in candidates_with_answers]
                                
                                # Generate all needed answers in one batch
                                new_responses = self.generate_diverse_answers_batch(
                                    prompt=question,
                                    existing_answers=existing_answers,
                                    num_to_generate=num_needed,
                                    max_new_tokens=curr_max_tokens[prompt_idx]
                                )

                                print(f"  Generated {len(new_responses)} new responses")
                                
                                if new_responses:
                                    added_candidates = 0
                                    # Create synthetic candidates for the new responses and add them to candidates_with_answers
                                    for new_response in new_responses:
                                        # Extract the answer from the full response for the candidate structure
                                        extracted_answer = self.extract_last_answer(new_response)
                                        if extracted_answer:
                                            new_candidate = self.create_synthetic_candidate(extracted_answer, len(candidates_with_answers), candidates, prompt_idx, curr_inputs, init_inputs, new_response)
                                            candidates_with_answers.append(new_candidate)
                                            candidates.append(new_candidate)
                                            added_candidates += 1
                                    print(f"    ✓ Added {added_candidates} new candidates to the pool")
                                else:
                                    print(f"    ✗ Failed to generate new responses")

                                iteration += 1

                                # Apply similarity filtering to all candidates (including synthetic ones) after iterative generation
                                if self.use_similarity_filtering and len(candidates_with_answers) > 1:
                                    print(f"  Applying similarity filtering to {len(candidates_with_answers)} total candidates (original + synthetic)...")
                                    candidates_with_answers, kept_indices = self.filter_similar_candidates(candidates_with_answers)

                                if iteration > self.max_iterative_iterations:
                                    break
                        
                        print(f"  Final candidate count after iterative generation: {len(candidates_with_answers)}")
                    elif turn == 0 and not self.enable_iterative_reprompting:
                        print(f"  Turn {turn}: Iterative reprompting disabled, using {len(candidates_with_answers)} original candidates")
                    else:
                        print(f"  Turn {turn}: Using existing candidates without regeneration")
                    
                    # Use autorater to select the best answer from filtered candidates
                    # Extract answers from filtered candidates for autorater evaluation
                    filtered_answers = [c['answer'] for c in candidates_with_answers]
                    best_idx, score = self.evaluate_response(question, filtered_answers, prompt_idx, meta_info)
                    
                    # Get the best candidate from candidates_with_answers
                    # Now they are perfectly aligned since we filtered the candidates themselves
                    best_idx = min(best_idx, len(candidates_with_answers) - 1)
                    best_candidate = candidates_with_answers[best_idx]
                    
                    # Check if the selected candidate came from iterative generation
                    is_iterative_candidate = best_candidate.get('is_synthetic', False)
                    candidate_source = "iterative generation" if is_iterative_candidate else "original candidates"
                    
                    print(f"Prompt {prompt_idx}: Autorater selected candidate {best_candidate['candidate_idx'] + 1} with score {score}")
                    print(f"Prompt {prompt_idx}: Selected candidate source: {candidate_source}")
                    
                    # Store turn data in generation history
                    turn_data = {
                        'turn_number': turn,
                        'candidates_generated': len(candidates),
                        'candidates_with_answers': candidates_with_answers,
                        'selected_candidate_idx': best_idx,
                        'selected_answer': best_candidate['answer'],
                        'autorater_score': score,
                    }
                    generation_history[prompt_idx]['turns'].append(turn_data)
                    print(f"Prompt {prompt_idx}: Stored turn {turn + 1} data with {len(candidates_with_answers)} filtered answers")
                    
                    # Copy the context from the best candidate to continue generation
                    best_full_generation_ids = best_candidate['full_generation_ids']
                    
                    # Ubate the current input to use the best candidate's generation
                    curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx].copy(), best_full_generation_ids])
                    
                    # Check if we should continue generation for this prompt
                    current_length = len(curr_inputs[prompt_idx]) - len(init_inputs[prompt_idx])
                    if current_length < self.config.response_length:
                        new_active_indices.append(prompt_idx)
                        curr_max_tokens[prompt_idx] = self.config.response_length - current_length
                    
                    print(f"Prompt {prompt_idx}: Using best candidate's context (length: {len(best_full_generation_ids)} tokens)")
                else:
                    print(f"Prompt {prompt_idx}: No candidates reached </answer>, using first candidate")
                    best_idx = 0
                    best_candidate = candidates[0]
                    # best_full_generation_ids = best_candidate['full_generation_ids']
                    best_candidate_id = self.tokenizer.encode(best_candidate)
                    curr_inputs[prompt_idx] = np.concatenate([init_inputs[prompt_idx].copy(), best_candidate_id])
                    new_active_indices.append(prompt_idx)
                    curr_max_tokens[prompt_idx] = self.config.response_length - current_length
            
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
            
            # Use the accumulated generation directly - no need to regenerate
            response_list.append(response_ids)
            print(f"Prompt {i}: Using accumulated generation ({len(response_ids)} tokens)")
            
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
        # Convert to numpy array to be compatible with DataProto
        non_tensor_batch["generation_history"] = np.array(generation_history, dtype=object)
        
        # Free vllm cache engine
        if (vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine):
            self.inference_engine.free_cache_engine()
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch) 