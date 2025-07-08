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

import logging
import os
import re
from typing import Any, Dict, List, Tuple, Optional

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl.workers.autorater.autorater_utils import extract_solution, format_code_outline_prompt
from verl.utils.autorater_client import call_autorater_service

# Import SandboxSession for code execution
try:
    from llm_sandbox import SandboxSession
except ImportError:
    SandboxSession = None
    logging.warning("llm_sandbox not available, code execution will be disabled")

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CodeEvaluator:
    """
    Handles code evaluation and interleaved reasoning functionality using sandbox execution.
    Supports both standard code evaluation and multi-part interleaved reasoning.
    Includes unit test execution capabilities using SandboxSession.
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        autorater_service_url: Optional[str] = None,
        template_type: Optional[str] = None,
    ):
        """
        Initializes the CodeEvaluator.

        Args:
            config: Configuration object for the code evaluator.
            tokenizer: The tokenizer instance to use for decoding.
            autorater_service_url: The base URL of the AutoRater service.
            template_type: The template type used for generation (e.g., "interleave").
        """
        self.config = config
        self.tokenizer = tokenizer
        self.autorater_base_url = autorater_service_url.rstrip("/") if autorater_service_url else None
        self.template_type = template_type

        # Interleaved reasoning configuration
        self.enable_interleaved_reasoning = self.config.get("enable_interleaved_reasoning", False)
        self.interleaved_reward_weights = self.config.get("interleaved_reward_weights", {
            "description": 1.0,
            "code": 1.0,
            "unit_tests": 1.0
        })

        # Initialize SandboxSession for code execution
        self.sandbox_session = None
        self._init_sandbox()

    def _init_sandbox(self):
        """Initialize SandboxSession with proper error handling."""
        if SandboxSession is None:
            logger.warning("llm_sandbox not available, code execution will be disabled")
            return
        
        self.sandbox_session = SandboxSession(lang="python")
        
        # Try to open the session - some versions require explicit open()
        if hasattr(self.sandbox_session, 'open'):
            try:
                self.sandbox_session.open()
                logger.info("SandboxSession initialized and opened successfully")
            except Exception as e:
                # Handle Docker connection errors specifically
                if "Connection aborted" in str(e) or "No such file or directory" in str(e):
                    logger.error(f"Docker/sandbox service not available: {e}")
                    logger.info("Code execution will fall back to heuristic evaluation")
                else:
                    logger.warning(f"SandboxSession created but failed to open explicitly: {e}")
                    logger.info("Will try to rely on automatic session management")
                self.sandbox_session = None
                return
        else:
            raise Exception("SandboxSession not available")
    

    def run_unit_tests(
        self,
        predicted_answers: List[str],
        reward_model_info: List[Dict[str, Any]],
    ) -> Tuple[List[float], List[int], List[int], List[str], List[str], List[str]]:
        """
        Execute unit tests in SandboxSession and aggregate scores and outputs.
        
        Args:
            predicted_answers: List of predicted code answers
            reward_model_info: List of reward model info containing unit tests
            
        Returns:
            Tuple of (code_scores, tests_passed, total_tests, stdout_list, stderr_list, error_list)
        """
        batch_size = len(predicted_answers)

        # Ensure sandbox session exists
        if self.sandbox_session is None:
            logger.info("SandboxSession not available, attempting to initialize on demand")
            self._init_sandbox()

        if self.sandbox_session is None:
            logger.warning("No SandboxSession available, returning zero scores")
            raise Exception("No SandboxSession available")

        sess = self.sandbox_session

        code_scores: List[float] = [0.0] * batch_size
        tests_passed: List[int] = [0] * batch_size
        total_tests: List[int] = [0] * batch_size
        stdout_list: List[str] = [""] * batch_size
        stderr_list: List[str] = [""] * batch_size
        error_list: List[str] = [""] * batch_size

        for idx, rm_info in enumerate(reward_model_info):
            tests_raw: List[str] = []
            libs: Optional[List[str]] = None

            if isinstance(rm_info, dict):
                # collect libs for this sample
                raw_libs = rm_info.get("libs")
                if isinstance(raw_libs, list):
                    libs = raw_libs
                elif raw_libs is not None:
                    libs = [str(raw_libs)]

                # gather tests definitions
                if isinstance(rm_info.get("unit_tests"), list):
                    tests_raw = rm_info["unit_tests"]
                elif isinstance(rm_info.get("tests"), list):
                    tests_raw = rm_info["tests"]
                else:
                    tc = rm_info.get("unit_tests") or rm_info.get("tests")
                    if tc:
                        tests_raw = [tc]

            if not tests_raw:
                continue

            code_match = re.search(r"```[\w]*\n(.*?)```", predicted_answers[idx], re.DOTALL)
            pred_code_block = code_match.group(1) if code_match else predicted_answers[idx]

            passes = 0
            split_tests: List[str] = []

            for snippet in tests_raw:
                # Simple approach: split on 'def' and treat each as a test case
                normalized_snippet = snippet.replace('\\n', '\n')
                
                # Extract methods
                test_methods = re.findall(r'def\s+(\w+)\s*\([^)]*\)\s*:(.*?)(?=\n\s*def|\Z)', normalized_snippet, re.DOTALL)
                
                # Extract setUp method if present
                setup_method = ""
                test_only_methods = []
                
                for method_name, method_body in test_methods:
                    if method_name == "setUp":
                        setup_method = f"    def setUp(self):{method_body}"
                    elif method_name.startswith("test_"):
                        test_only_methods.append((method_name, method_body))
                
                for method_name, method_body in test_only_methods:
                    # Create complete test with user code + just the test method
                    complete_test = f"""import unittest
import pandas as pd
import numpy as np

{pred_code_block}

class TestCases(unittest.TestCase):
{setup_method}
    def {method_name}(self):{method_body}

if __name__ == '__main__':
    unittest.main()
"""
                    split_tests.append(complete_test)
                
                # If no test methods found, treat whole snippet as one test
                if not test_only_methods:
                    # For plain assert statements, combine with user code
                    complete_fallback = f"""{pred_code_block}

{normalized_snippet}
"""
                    split_tests.append(complete_fallback)

            for test_snippet in split_tests:
                exec_code = test_snippet
                try:
                    res = sess.run(exec_code, libraries=libs)
                    if res.exit_code == 0:
                        passes += 1
                    stdout_list[idx] += res.stdout + "\n"
                    stderr_list[idx] += res.stderr + "\n"
                except Exception as exec_e:
                    error_list[idx] += str(exec_e) + "\n"

            total_tests[idx] = len(split_tests)
            tests_passed[idx] = passes
            code_scores[idx] = float(passes) * 0.2

        return code_scores, tests_passed, total_tests, stdout_list, stderr_list, error_list

    def run_unit_tests_combined(
        self,
        predicted_answers: List[str],
        reward_model_info: List[Dict[str, Any]],
    ) -> Tuple[List[float], List[int], List[int], List[str], List[str], List[str]]:
        """
        Execute unit tests by combining user code with unit test blocks and parsing unittest output.
        This is a cleaner approach that uses unit test blocks directly without extraction.
        
        Args:
            predicted_answers: List of predicted code answers
            reward_model_info: List of reward model info containing unit tests
            
        Returns:
            Tuple of (code_scores, tests_passed, total_tests, stdout_list, stderr_list, error_list)
        """
        batch_size = len(predicted_answers)

        # Ensure sandbox session exists
        if self.sandbox_session is None:
            logger.info("SandboxSession not available, attempting to initialize on demand")
            self._init_sandbox()

        if self.sandbox_session is None:
            logger.warning("No SandboxSession available, returning zero scores")
            raise Exception("No SandboxSession available")

        sess = self.sandbox_session

        code_scores: List[float] = [0.0] * batch_size
        tests_passed: List[int] = [0] * batch_size
        total_tests: List[int] = [0] * batch_size
        stdout_list: List[str] = [""] * batch_size
        stderr_list: List[str] = [""] * batch_size
        error_list: List[str] = [""] * batch_size

        for idx, rm_info in enumerate(reward_model_info):
            tests_raw: List[str] = []
            libs: Optional[List[str]] = None

            if isinstance(rm_info, dict):
                # collect libs for this sample
                raw_libs = rm_info.get("libs")
                if isinstance(raw_libs, list):
                    libs = raw_libs
                elif raw_libs is not None:
                    libs = [str(raw_libs)]

                # gather tests definitions
                if isinstance(rm_info.get("unit_tests"), list):
                    tests_raw = rm_info["unit_tests"]
                elif isinstance(rm_info.get("tests"), list):
                    tests_raw = rm_info["tests"]
                else:
                    tc = rm_info.get("unit_tests") or rm_info.get("tests")
                    if tc:
                        tests_raw = [tc]

            if not tests_raw:
                continue

            code_match = re.search(r"```[\w]*\n(.*?)```", predicted_answers[idx], re.DOTALL)
            pred_code_block = code_match.group(1) if code_match else predicted_answers[idx]

            # Combine user code with all unit test blocks directly
            all_test_blocks = []
            for snippet in tests_raw:
                normalized_snippet = snippet.replace('\\n', '\n')
                all_test_blocks.append(normalized_snippet)
            
            # Create combined test file with user code + all test blocks
            combined_test = f"""import unittest
import pandas as pd
import numpy as np

{pred_code_block}

{chr(10).join(all_test_blocks)}

if __name__ == '__main__':
    unittest.main(verbosity=2)
"""

            try:
                res = sess.run(combined_test, libraries=libs)
                stdout_list[idx] = res.stdout
                stderr_list[idx] = res.stderr
                
                # Parse unittest output to count tests
                output = res.stdout + res.stderr
                
                # Look for patterns like "Ran X tests in Y.YYYs"
                ran_match = re.search(r'Ran (\d+) tests? in', output)
                if ran_match:
                    total_tests[idx] = int(ran_match.group(1))
                else:
                    # Fallback: count test methods in the test blocks
                    test_count = 0
                    for block in all_test_blocks:
                        test_count += len(re.findall(r'def\s+test_\w+', block))
                    total_tests[idx] = test_count
                
                # Count failures and errors
                failures = 0
                errors = 0
                
                # Look for "FAILED (failures=X, errors=Y)" or "FAILED (failures=X)" or "FAILED (errors=Y)"
                failed_match = re.search(r'FAILED \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?\)', output)
                if failed_match:
                    if failed_match.group(1):
                        failures = int(failed_match.group(1))
                    if failed_match.group(2):
                        errors = int(failed_match.group(2))
                
                # Calculate passed tests
                tests_passed[idx] = total_tests[idx] - failures - errors
                
                # If exit code is 0, all tests passed
                if res.exit_code == 0:
                    tests_passed[idx] = total_tests[idx]
                
                # Calculate score (0.2 points per passing test, matching original function)
                code_scores[idx] = float(tests_passed[idx]) * 0.2
                
            except Exception as exec_e:
                error_list[idx] = str(exec_e)
                # Fallback: count test methods in the test blocks
                test_count = 0
                for block in all_test_blocks:
                    test_count += len(re.findall(r'def\s+test_\w+', block))
                total_tests[idx] = test_count
                tests_passed[idx] = 0
                code_scores[idx] = 0.0

        return code_scores, tests_passed, total_tests, stdout_list, stderr_list, error_list

    def evaluate_code(
        self,
        decoded_pred_answers: List[str],
        original_prompts: List[str],
        ground_truth_infos: List[Dict[str, Any]],
        batch_size: int,
    ) -> Tuple[List[float], List[int], List[str], List[str]]:
        """
        Evaluate code responses using either standard or interleaved reasoning.

        Args:
            decoded_pred_answers: List of decoded response strings
            original_prompts: List of original prompts
            ground_truth_infos: List of ground truth information dictionaries
            batch_size: Number of samples in the batch

        Returns:
            Tuple of (scores, decisions, explanations, raw_responses)
        """
        # Check if we're doing interleaved reasoning
        is_interleaved = (
            self.enable_interleaved_reasoning or 
            (self.template_type and "interleave" in self.template_type.lower())
        )
        
        if is_interleaved:
            logger.info("Using interleaved reasoning evaluation")
            return self._evaluate_interleaved_reasoning(
                decoded_pred_answers, original_prompts, ground_truth_infos, batch_size
            )
        else:
            logger.info("Using standard code evaluation")
            return self._evaluate_code(
                decoded_pred_answers, ground_truth_infos, batch_size
            )

    def extract_code_snippet(self, predicted_answer: str) -> str:
        """Extract code snippet from predicted answer."""
        if not predicted_answer.strip():
            return ""
        
        # Method 1: Try to extract code between triple backticks
        code_block_pattern = r'```(?:python)?\s*(.*?)```'
        code_matches = re.findall(code_block_pattern, predicted_answer, re.DOTALL)
        
        if code_matches:
            # Return the first code block found
            code_snippet = code_matches[0].strip()
            logger.debug(f"Extracted code from ``` block: {len(code_snippet)} characters")
            return code_snippet
        
        # Method 2: Try to extract from "def task_func" onwards
        task_func_pattern = r'(def task_func.*?)(?=\n\n|\n(?:def |class |import |from |#|$)|\Z)'
        task_func_match = re.search(task_func_pattern, predicted_answer, re.DOTALL)
        
        if task_func_match:
            code_snippet = task_func_match.group(1).strip()
            logger.debug(f"Extracted code from def task_func: {len(code_snippet)} characters")
            return code_snippet
        
        # Method 3: Look for any function definition as fallback
        function_pattern = r'(def \w+.*?)(?=\n\n|\n(?:def |class |import |from |#|$)|\Z)'
        function_matches = re.findall(function_pattern, predicted_answer, re.DOTALL)
        
        if function_matches:
            # Return the first function found
            code_snippet = function_matches[0].strip()
            logger.debug(f"Extracted code from function def: {len(code_snippet)} characters")
            return code_snippet
        
        logger.debug(f"No code snippet found in predicted answer")
        return None 

    def _evaluate_code(
        self,
        decoded_pred_answers: List[str],
        ground_truth_infos: List[Dict[str, Any]],
        batch_size: int,
    ) -> Tuple[List[float], List[int], List[str], List[str]]:
        """
        Evaluate standard code responses using sandbox execution.
        
        Args:
            decoded_pred_answers: List of decoded response strings
            ground_truth_infos: List of ground truth information dictionaries
            batch_size: Number of samples in the batch
            
        Returns:
            Tuple of (scores, decisions, explanations, raw_responses)
        """
        # Check if any unit tests are present
        def _has_tests(info: Dict[str, Any]):
            return bool(
                isinstance(info, dict)
                and (
                    ("unit_tests" in info and info["unit_tests"])
                    or ("tests" in info and info["tests"])
                )
            )

        use_code_evaluator = any(_has_tests(info) for info in ground_truth_infos)

        if use_code_evaluator:
            # Extract code snippets from predicted answers first
            logger.info("Extracting code snippets from predicted answers")
            extracted_code_answers = []
            failed_extraction_indices = []
            
            for i, pred_answer in enumerate(decoded_pred_answers):
                extracted_code = self.extract_code_snippet(pred_answer)
                if extracted_code is None:
                    failed_extraction_indices.append(i)
                    extracted_code_answers.append("")  # Use empty string for failed extractions
                else:
                    extracted_code_answers.append(extracted_code)
            
            if failed_extraction_indices:
                logger.warning(f"Failed to extract code from {len(failed_extraction_indices)} samples: {failed_extraction_indices}")
            
            # Use local unit test execution
            logger.info("Using local unit test execution for code evaluation")
            (
                code_scores,
                code_tests_passed,
                code_total_tests,
                code_stdout,
                code_stderr,
                code_error,
            ) = self.run_unit_tests_combined(extracted_code_answers, ground_truth_infos)

            # Create decisions and normalized scores based on test results
            decisions = []
            explanations = []
            normalized_scores = []
            
            for i in range(batch_size):
                if i in failed_extraction_indices:
                    # Default values for failed code extraction
                    normalized_scores.append(0.0)
                    decisions.append(0)
                    explanations.append("failed code extraction")
                else:
                    # Normalize score as passed_tests / total_tests
                    if code_total_tests[i] > 0:
                        normalized_score = code_tests_passed[i] / code_total_tests[i]
                    else:
                        normalized_score = 0.0
                    
                    normalized_scores.append(normalized_score)
                    decisions.append(1 if normalized_score > 0 else 0)
                    explanations.append(f"passed {code_tests_passed[i]}/{code_total_tests[i]} tests")
            
            raw_responses = [""] * batch_size
            
            return normalized_scores, decisions, explanations, raw_responses
        else:
            raise ValueError("No unit tests available, using simple heuristic evaluation")

    def _evaluate_interleaved_reasoning(
        self,
        decoded_pred_answers: List[str],
        original_prompts: List[str],
        ground_truth_infos: List[Dict[str, Any]],
        batch_size: int,
    ) -> Tuple[List[float], List[int], List[str], List[str]]:
        """
        Evaluate interleaved reasoning responses with multiple <answer> tags.
        
        Expected structure:
        1. First <answer>: Description/explanation of the solution approach
        2. Second <answer>: Code implementation 
        3. Third <answer>: Self-generated unit tests 
        
        Args:
            decoded_pred_answers: List of decoded response strings
            original_prompts: List of original prompts
            ground_truth_infos: List of ground truth information dictionaries
            batch_size: Number of samples in the batch
            
        Returns:
            Tuple of (scores, decisions, explanations, raw_responses)
        """
        total_scores = []
        decisions = []
        explanations = []
        raw_responses = []
        
        # --- Batch-evaluate all descriptions first ---
        description_scores_map = self._evaluate_all_descriptions(
            decoded_pred_answers, original_prompts
        )

        for i, (pred_answer, gt_info) in enumerate(zip(decoded_pred_answers, ground_truth_infos)):
            # Extract all answers from the interleaved response using extract_all=True
            all_answers = extract_solution(pred_answer, extract_all=True)
            
            # The format is guaranteed by the caller (RewardManager), which filters
            # for responses with >= 3 answers. We can assert this.
            assert all_answers is not None, "all_answers should not be None"
            if isinstance(all_answers, list):
                answer_parts = [part.strip() for part in all_answers]
            else:
                # This path should ideally not be taken if extract_solution is consistent
                answer_parts = [part.strip() for part in str(all_answers).split(",")]

            assert len(answer_parts) >= 3, f"Expected >=3 answer parts, but got {len(answer_parts)}"

            # Initialize component scores
            description_score = description_scores_map.get(i, 0.0)
            code_score = 0.0
            unit_test_score = 0.0
            component_explanations = []
            
            # Evaluate first answer (description/explanation)
            description_score = description_scores_map.get(i, 0.0)
            component_explanations.append(f"Description Score: {description_score:.2f}")

            # Evaluate second answer (code implementation)
            code_text = answer_parts[1]
            # Extract clean code snippet from the code answer
            extracted_code = self.extract_code_snippet(code_text)
            
            # Check if code extraction failed
            if extracted_code is None:
                code_score = 0.0
                component_explanations.append("Code: failed code extraction")
            # Check if we have unit tests available for code evaluation
            elif isinstance(gt_info, dict) and (gt_info.get("unit_tests") or gt_info.get("tests")):
                (
                    code_scores_list,
                    code_tests_passed,
                    code_total_tests,
                    _, _, _,
                ) = self.run_unit_tests_combined([extracted_code], [{
                    "ground_truth": "code",
                    "unit_tests": gt_info.get("unit_tests") or gt_info.get("tests"),
                    "libs": gt_info.get("libs", [])
                }])
                
                passed = code_tests_passed[0] if code_tests_passed else 0
                total = code_total_tests[0] if code_total_tests else 0
                
                # Normalize score as passed_tests / total_tests
                if total > 0:
                    code_score = passed / total
                else:
                    code_score = 0.0
                    
                component_explanations.append(f"Code: passed {passed}/{total} tests")
            else:
                code_score = 0.0
                component_explanations.append("Code: No gt unit tests provided")

            # Evaluate third answer (self-generated unit tests) - optional
            unit_test_text = answer_parts[2]
            # Check if unit tests look reasonable
            if ("assert" in unit_test_text or "test" in unit_test_text.lower() or 
                "unittest" in unit_test_text.lower() or "def test_" in unit_test_text):
                unit_test_score = 1.0
                component_explanations.append("Unit Tests: Self-generated tests provided")
            else:
                unit_test_score = 0.0
                component_explanations.append("Unit Tests: No unit tests found")
                
            # Combine scores with weights
            weights = self.interleaved_reward_weights
            total_score = (
                description_score * weights.get("description", 1.0)
                + code_score * weights.get("code", 1.0)
                + unit_test_score * weights.get("unit_tests", 1.0)
            )
            
            total_scores.append(total_score)
            decisions.append(1 if total_score > 1.0 else 0)  # Threshold for success
            explanations.append(" | ".join(component_explanations))
            raw_responses.append(f"Interleaved evaluation: {len(answer_parts)} answers found")
            
        return total_scores, decisions, explanations, raw_responses

    def _evaluate_all_descriptions(
        self, decoded_pred_answers: List[str], original_prompts: List[str]
    ) -> Dict[int, float]:
        """
        Evaluate all description parts of interleaved answers in a single batch.
        """
        if not self.autorater_base_url:
            logger.warning("AutoRater service URL not configured in CodeEvaluator; skipping description evaluation.")
            return {}

        descriptions_to_eval: List[Tuple[int, str, str]] = []
        for i, pred_answer in enumerate(decoded_pred_answers):
            first_answer = extract_solution(pred_answer, extract_all=False)
            if first_answer and isinstance(first_answer, str):
                descriptions_to_eval.append((i, original_prompts[i], first_answer))

        if not descriptions_to_eval:
            return {}

        # Prepare payload for the AutoRater service
        batch_indices, batch_prompts, batch_responses = zip(*descriptions_to_eval)
        
        tokenized_prompts = self.tokenizer(list(batch_prompts), add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids.tolist()
        tokenized_responses = self.tokenizer(list(batch_responses), add_special_tokens=False, padding=True, truncation=True, return_tensors="pt").input_ids.tolist()

        # We need to construct a valid-looking payload even if some parts are dummy
        batch_size = len(batch_indices)
        payload = {
            "prompts": tokenized_prompts,
            "responses": tokenized_responses,
            # Dummy values for fields that are not used by outline evaluation but required by schema
            "attention_mask": [[1] * len(r) for r in tokenized_responses],
            "position_ids": [list(range(len(r))) for r in tokenized_responses],
            "reward_model_info": [
                {"template": "outline", "ground_truth": ""} for _ in range(batch_size)
            ],
        }

        try:
            logger.info(f"Calling AutoRater to evaluate {batch_size} description outlines.")
            scores, decisions, _, _ = call_autorater_service(
                self.autorater_base_url,
                payload,
                batch_size=batch_size,
                endpoint="/evaluate_autorater", # Use the main endpoint
            )
            
            # The decision is what matters: 1 for TRUE, 0 for FALSE. Score is shaped, so use decision.
            final_scores = [1.0 if d == 1 else 0.0 for d in decisions]
            return dict(zip(batch_indices, final_scores))

        except Exception as e:
            logger.error(f"Failed to evaluate descriptions via AutoRater: {e}")
            return {}

    def close(self):
        """Clean up resources."""
        if self.sandbox_session is not None:
            try:
                if hasattr(self.sandbox_session, "close"):
                    self.sandbox_session.close()
                    logger.info("SandboxSession closed successfully")
                elif hasattr(self.sandbox_session, "__exit__"):
                    self.sandbox_session.__exit__(None, None, None)
                    logger.info("SandboxSession cleaned up via __exit__")
                else:
                    logger.info("SandboxSession cleanup - no explicit close method available")
            except Exception as e:
                logger.warning(f"Error closing SandboxSession: {e}")
            finally:
                self.sandbox_session = None

    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.close() 