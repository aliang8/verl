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

from verl.workers.autorater.autorater_utils import extract_solution

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
        template_type: Optional[str] = None,
    ):
        """
        Initializes the CodeEvaluator.

        Args:
            config: Configuration object for the code evaluator.
            tokenizer: The tokenizer instance to use for decoding.
            template_type: The template type used for generation (e.g., "interleave").
        """
        self.config = config
        self.tokenizer = tokenizer
        self.template_type = template_type

        # Interleaved reasoning configuration
        self.enable_interleaved_reasoning = self.config.get("enable_interleaved_reasoning", False)
        self.interleaved_reward_weights = self.config.get("interleaved_reward_weights", {
            "description": 1.0,
            "code": 2.0,
            "unit_tests": 1.5
        })

        # Initialize SandboxSession for code execution
        self.sandbox_session = None
        self._init_sandbox()

    def _init_sandbox(self):
        """Initialize SandboxSession with proper error handling."""
        if SandboxSession is None:
            logger.warning("llm_sandbox not available, code execution will be disabled")
            return
        
        try:
            self.sandbox_session = SandboxSession(lang="python")
            
            # Try to open the session - some versions require explicit open()
            if hasattr(self.sandbox_session, 'open'):
                try:
                    self.sandbox_session.open()
                    logger.info("SandboxSession initialized and opened successfully")
                except Exception as e:
                    logger.warning(f"SandboxSession created but failed to open explicitly: {e}")
                    logger.info("Will rely on automatic session management")
            else:
                logger.info("SandboxSession initialized (no explicit open() method)")
                
        except Exception as e:
            logger.error(f"Failed to initialize SandboxSession: {e}")
            self.sandbox_session = None

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
            code_scores[idx] = float(passes) * 0.3

        return code_scores, tests_passed, total_tests, stdout_list, stderr_list, error_list

    def evaluate_code(
        self,
        decoded_pred_answers: List[str],
        ground_truth_infos: List[Dict[str, Any]],
        batch_size: int,
    ) -> Tuple[List[float], List[int], List[str], List[str]]:
        """
        Evaluate code responses using either standard or interleaved reasoning.

        Args:
            decoded_pred_answers: List of decoded response strings
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
                decoded_pred_answers, ground_truth_infos, batch_size
            )
        else:
            logger.info("Using standard code evaluation")
            return self._evaluate_standard_code(
                decoded_pred_answers, ground_truth_infos, batch_size
            )



    def _evaluate_standard_code(
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
            # Use local unit test execution
            logger.info("Using local unit test execution for code evaluation")
            (
                code_scores,
                code_tests_passed,
                code_total_tests,
                code_stdout,
                code_stderr,
                code_error,
            ) = self.run_unit_tests(decoded_pred_answers, ground_truth_infos)

            # Create decisions based on code scores
            decisions = [1 if score > 0 else 0 for score in code_scores]
            explanations = [f"passed {p}/{t} tests" for p, t in zip(code_tests_passed, code_total_tests)]
            raw_responses = [""] * batch_size
            
            return code_scores, decisions, explanations, raw_responses
        else:
            # Simple heuristic evaluation when no unit tests available
            logger.info("No unit tests available, using simple heuristic evaluation")
            scores = []
            decisions = []
            explanations = []
            
            for pred_ans in decoded_pred_answers:
                # Extract solution from <answer> tags if present
                extracted = extract_solution(pred_ans)
                code_text = extracted if extracted else pred_ans
                
                # Simple heuristic: check if code contains function definition
                if "def " in code_text or "function" in code_text.lower():
                    score = 1.0
                    decision = 1
                    explanation = "Function definition found"
                else:
                    score = 0.0
                    decision = 0
                    explanation = "No function definition found"
                    
                scores.append(score)
                decisions.append(decision)
                explanations.append(explanation)
            
            raw_responses = ["Heuristic evaluation"] * batch_size
            return scores, decisions, explanations, raw_responses

    def _evaluate_interleaved_reasoning(
        self,
        decoded_pred_answers: List[str],
        ground_truth_infos: List[Dict[str, Any]],
        batch_size: int,
    ) -> Tuple[List[float], List[int], List[str], List[str]]:
        """
        Evaluate interleaved reasoning responses with multiple <answer> tags.
        
        Expected structure:
        1. First <answer>: Description/explanation of the solution approach
        2. Second <answer>: Code implementation 
        3. Third <answer>: Self-generated unit tests (optional)
        
        Args:
            decoded_pred_answers: List of decoded response strings
            ground_truth_infos: List of ground truth information dictionaries
            batch_size: Number of samples in the batch
            
        Returns:
            Tuple of (scores, decisions, explanations, raw_responses)
        """
        total_scores = []
        decisions = []
        explanations = []
        raw_responses = []
        
        for i, (pred_answer, gt_info) in enumerate(zip(decoded_pred_answers, ground_truth_infos)):
            # Extract all answers from the interleaved response using extract_all=True
            all_answers = extract_solution(pred_answer, extract_all=True)
            
            if all_answers is None:
                # No <answer> tags found
                total_scores.append(-2.0)
                decisions.append(0)
                explanations.append("No <answer> tags found in interleaved response")
                raw_responses.append("Parse failure")
                continue
                
            # Split the comma-separated answers
            answer_parts = [part.strip() for part in all_answers.split(",")]
            
            # Initialize component scores
            description_score = 0.0
            code_score = 0.0
            unit_test_score = 0.0
            component_explanations = []
            
            # Evaluate first answer (description/explanation)
            if len(answer_parts) >= 1:
                description_text = answer_parts[0]
                # Simple heuristic: check if description is substantial and explains approach
                if len(description_text.split()) >= 10:  # At least 10 words
                    description_score = 1.0
                    component_explanations.append("Description: Good explanation provided")
                else:
                    description_score = 0.5
                    component_explanations.append("Description: Brief explanation provided")
            else:
                component_explanations.append("Description: Missing")
                
            # Evaluate second answer (code implementation)
            if len(answer_parts) >= 2:
                code_text = answer_parts[1]
                
                # Check if we have unit tests available for code evaluation
                if isinstance(gt_info, dict) and (gt_info.get("unit_tests") or gt_info.get("tests")):
                    # Use local unit test execution
                    try:
                        (
                            code_scores,
                            code_tests_passed,
                            code_total_tests,
                            code_stdout,
                            code_stderr,
                            code_error,
                        ) = self.run_unit_tests([code_text], [{
                            "ground_truth": "code",
                            "unit_tests": gt_info.get("unit_tests") or gt_info.get("tests"),
                            "libs": gt_info.get("libs", [])
                        }])
                        
                        code_score = code_scores[0] if code_scores else 0.0
                        passed = code_tests_passed[0] if code_tests_passed else 0
                        total = code_total_tests[0] if code_total_tests else 0
                        component_explanations.append(f"Code: passed {passed}/{total} tests")
                        
                    except Exception as e:
                        logger.warning(f"Code evaluation failed: {e}")
                        # Fallback: simple heuristic evaluation
                        if "def " in code_text or "function" in code_text.lower():
                            code_score = 1.0
                            component_explanations.append("Code: Function definition found")
                        else:
                            code_score = 0.0
                            component_explanations.append("Code: No function definition found")
                else:
                    # Simple heuristic evaluation when no unit tests available
                    if "def " in code_text or "function" in code_text.lower():
                        code_score = 1.0
                        component_explanations.append("Code: Function definition found")
                    else:
                        code_score = 0.0
                        component_explanations.append("Code: No function definition found")
            else:
                component_explanations.append("Code: Missing")
                
            # Evaluate third answer (self-generated unit tests) - optional
            if len(answer_parts) >= 3:
                unit_test_text = answer_parts[2]
                
                # Check if unit tests look reasonable
                if ("assert" in unit_test_text or "test" in unit_test_text.lower() or 
                    "unittest" in unit_test_text.lower() or "def test_" in unit_test_text):
                    unit_test_score = 1.0
                    component_explanations.append("Unit Tests: Self-generated tests provided")
                else:
                    unit_test_score = 0.5
                    component_explanations.append("Unit Tests: Attempted but incomplete")
            else:
                component_explanations.append("Unit Tests: Not provided")
                
            # Combine scores with weights
            weights = self.interleaved_reward_weights
            total_score = (
                description_score * weights.get("description", 1.0) +
                code_score * weights.get("code", 2.0) +
                unit_test_score * weights.get("unit_tests", 1.5)
            )
            
            total_scores.append(total_score)
            decisions.append(1 if total_score > 2.0 else 0)  # Threshold for success
            explanations.append(" | ".join(component_explanations))
            raw_responses.append(f"Interleaved evaluation: {len(answer_parts)} answers found")
            
        return total_scores, decisions, explanations, raw_responses

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