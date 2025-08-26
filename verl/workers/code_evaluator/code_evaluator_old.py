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
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import collections
import threading
import queue
from contextlib import contextmanager
from omegaconf import DictConfig
from transformers import AutoTokenizer
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

from verl.workers.autorater.autorater_utils import (
    extract_solution,
)
from verl.utils.autorater_client import call_autorater_service
from verl.utils.debug.performance import _timer  # Add timing support

# Import SandboxSession for code execution
try:
    from llm_sandbox import SandboxSession
except ImportError:
    SandboxSession = None
    logging.warning("llm_sandbox not available, code execution will be disabled")

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ErrorTracker:
    """Tracks different types of errors that occur during code evaluation"""

    def __init__(self):
        # Track errors by prompt ID and error type
        self.error_logs: Dict[str, Dict[str, Any]] = {}
        self.error_counts: Dict[str, int] = collections.defaultdict(int)
        self.batch_counter = 0
        self.epoch_start_time = None
        self.current_epoch = None

    def start_epoch(self, epoch: int):
        """Mark the start of a new epoch"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.error_logs.clear()
        self.error_counts.clear()
        self.batch_counter = 0

    def log_error(
        self,
        prompt_id: str,
        error_type: str,
        error_details: Dict[str, Any],
        prompt_text: str = "",
        response_text: str = "",
    ):
        """Log an error for a specific prompt"""
        if prompt_id not in self.error_logs:
            self.error_logs[prompt_id] = {
                "prompt_text": prompt_text[:500],  # Truncate for storage
                "response_text": response_text[:1000],  # Truncate for storage
                "errors": [],
                "success": False,
                "batch_id": self.batch_counter,
            }

        self.error_logs[prompt_id]["errors"].append(
            {
                "error_type": error_type,
                "error_details": error_details,
                "timestamp": time.time(),
            }
        )

        self.error_counts[error_type] += 1

    def log_success(
        self,
        prompt_id: str,
        success_details: Dict[str, Any],
        prompt_text: str = "",
        response_text: str = "",
    ):
        """Log a successful evaluation for a specific prompt"""
        if prompt_id not in self.error_logs:
            self.error_logs[prompt_id] = {
                "prompt_text": prompt_text[:500],
                "response_text": response_text[:1000],
                "errors": [],
                "success": True,
                "batch_id": self.batch_counter,
                "success_details": success_details,
            }
        else:
            self.error_logs[prompt_id]["success"] = True
            self.error_logs[prompt_id]["success_details"] = success_details

    def increment_batch(self):
        """Increment the batch counter"""
        self.batch_counter += 1

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors"""
        total_prompts = len(self.error_logs)
        successful_prompts = sum(1 for log in self.error_logs.values() if log["success"])
        failed_prompts = total_prompts - successful_prompts

        return {
            "total_prompts": total_prompts,
            "successful_prompts": successful_prompts,
            "failed_prompts": failed_prompts,
            "success_rate": (successful_prompts / total_prompts if total_prompts > 0 else 0.0),
            "error_counts": dict(self.error_counts),
            "epoch_duration": (time.time() - self.epoch_start_time if self.epoch_start_time else 0),
            "epoch": self.current_epoch,
        }

    def get_wandb_metrics(self) -> Dict[str, Any]:
        """Get metrics formatted for wandb logging"""
        summary = self.get_error_summary()

        # Create wandb-friendly metrics
        wandb_metrics = {
            "code_eval/total_prompts": summary["total_prompts"],
            "code_eval/successful_prompts": summary["successful_prompts"],
            "code_eval/failed_prompts": summary["failed_prompts"],
            "code_eval/success_rate": summary["success_rate"],
            "code_eval/epoch_duration": summary["epoch_duration"],
        }

        # Add individual error type counts
        for error_type, count in summary["error_counts"].items():
            wandb_metrics[f"code_eval/errors/{error_type}"] = count

        return wandb_metrics

    def save_metadata(self, output_dir: str, epoch: int):
        """Save detailed metadata to files"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed error logs
        detailed_file = os.path.join(output_dir, f"epoch_{epoch}_detailed_logs_{timestamp}.json")
        with open(detailed_file, "w") as f:
            json.dump(self.error_logs, f, indent=2)

        # Save error summary
        summary_file = os.path.join(output_dir, f"epoch_{epoch}_error_summary_{timestamp}.json")
        summary = self.get_error_summary()
        summary["epoch"] = epoch
        summary["timestamp"] = timestamp
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save CSV for easy analysis
        csv_file = os.path.join(output_dir, f"epoch_{epoch}_prompt_results_{timestamp}.csv")
        try:
            import pandas as pd

            rows = []
            for prompt_id, log in self.error_logs.items():
                row = {
                    "prompt_id": prompt_id,
                    "success": log["success"],
                    "batch_id": log["batch_id"],
                    "num_errors": len(log["errors"]),
                    "error_types": ",".join([e["error_type"] for e in log["errors"]]),
                    "prompt_text": log["prompt_text"],
                    "response_text": log["response_text"],
                }
                if log["success"] and "success_details" in log:
                    row.update({f"success_{k}": v for k, v in log["success_details"].items()})
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(csv_file, index=False)
        except ImportError:
            logger.warning("pandas not available, skipping CSV export")

        logger.info(f"Saved epoch {epoch} metadata to {output_dir}")
        print(f"Saved epoch {epoch} error tracking metadata:")
        print(f"  - Detailed logs: {detailed_file}")
        print(f"  - Summary: {summary_file}")
        if os.path.exists(csv_file):
            print(f"  - CSV: {csv_file}")


class SafeResourceManagedExecutor:
    """Combined robust code executor with resource management that reuses sessions"""

    def __init__(self, max_concurrent=3):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.execution_queue = queue.Queue()

    @contextmanager
    def acquire_resources(self):
        """Acquire execution resources"""
        self.semaphore.acquire()
        try:
            yield
        finally:
            self.semaphore.release()

    def execute_safely(self, code: str, session=None, libraries=None, **kwargs):
        """Execute code safely with comprehensive error handling and resource management"""
        try:
            # Pre-execution validation
            if not code or not code.strip():
                return {
                    "error": "Empty code provided",
                    "error_type": "empty_code",
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "Empty code provided",
                }

            with self.acquire_resources():
                # Use provided session or fallback behavior
                if session is None:
                    logger.warning("No session provided, this may cause issues")
                    return {
                        "error": "No sandbox session available",
                        "error_type": "no_session",
                        "exit_code": 1,
                        "stdout": "",
                        "stderr": "No sandbox session available",
                    }

                # Security check if available
                if hasattr(session, "is_safe"):
                    try:
                        is_safe, violations = session.is_safe(code)
                        if not is_safe:
                            return {
                                "error": "Security violation",
                                "error_type": "security_violation",
                                "violations": [(v.description if hasattr(v, "description") else str(v)) for v in violations],
                                "exit_code": 1,
                                "stdout": "",
                                "stderr": "Security violation detected",
                            }
                    except Exception as e:
                        logger.warning(f"Security check failed: {e}, proceeding with execution")

                # Execute using the reused session
                result = session.run(code, libraries=libraries)

                # Post-execution validation
                if result.exit_code != 0:
                    error_type = "execution_failed"
                    # Categorize error types based on stderr content
                    stderr_lower = result.stderr.lower()
                    if "memoryerror" in stderr_lower or "memory" in stderr_lower:
                        error_type = "memory_error"
                    elif "timeout" in stderr_lower:
                        error_type = "timeout_error"
                    elif "syntaxerror" in stderr_lower:
                        error_type = "syntax_error"
                    elif "importerror" in stderr_lower or "modulenotfounderror" in stderr_lower:
                        error_type = "import_error"
                    elif "assertionerror" in stderr_lower:
                        error_type = "assertion_error"

                    return {
                        "error": "Execution failed",
                        "error_type": error_type,
                        "stderr": result.stderr,
                        "stdout": result.stdout,
                        "exit_code": result.exit_code,
                    }

                return {
                    "success": True,
                    "output": result.stdout,
                    "stderr": result.stderr,
                    "stdout": result.stdout,
                    "exit_code": result.exit_code,
                }

        except TimeoutError:
            return {
                "error": "Execution timeout",
                "error_type": "timeout_error",
                "exit_code": 124,
                "stdout": "",
                "stderr": "Execution timeout",
            }
        except MemoryError:
            return {
                "error": "Memory limit exceeded",
                "error_type": "memory_error",
                "exit_code": 125,
                "stdout": "",
                "stderr": "Memory limit exceeded",
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "error_type": "unexpected_error",
                "exit_code": 126,
                "stdout": "",
                "stderr": str(e),
            }


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
        template_type: Optional[str] = "default",
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

        self.interleaved_reward_weights = self.config.get(
            "interleaved_reward_weights",
            {"description": 0.5, "code": 1.0, "unit_tests": 0.5},
        )
        # Parallelism config
        self.num_workers = self.config.get("num_workers", 10)

        # Initialize robust execution components
        self.sandbox_session = None
        self.safe_executor = SafeResourceManagedExecutor(max_concurrent=10)

        # Initialize error tracking
        self.error_tracker = ErrorTracker()

        self.is_interleaved = bool(self.template_type and "interleave" in self.template_type.lower())

        self._init_sandbox()

    def start_epoch(self, epoch: int):
        """Start tracking for a new epoch"""
        self.error_tracker.start_epoch(epoch)

    def save_epoch_metadata(self, output_dir: str, epoch: int):
        """Save metadata for the completed epoch"""
        self.error_tracker.save_metadata(output_dir, epoch)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get current error summary"""
        return self.error_tracker.get_error_summary()

    def get_wandb_metrics(self) -> Dict[str, Any]:
        """Get metrics formatted for wandb logging"""
        return self.error_tracker.get_wandb_metrics()

    def _init_sandbox(self):
        """Initialize SandboxSession with proper error handling."""
        if SandboxSession is None:
            logger.warning("llm_sandbox not available, code execution will be disabled")
            return

        self.sandbox_session = SandboxSession(
            lang="python",
            execution_timeout=10,
            verbose=False,
            runtime_configs={"cpu_count": 50, "mem_limit": "4096m"},
        )

        try:
            self.sandbox_session.open()
            logger.info("SandboxSession initialized and opened successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SandboxSession: {e}")
            print(f"Failed to initialize SandboxSession: {e}")
            self.sandbox_session = None
            raise e

    def run_unit_tests_combined(
        self,
        predicted_answers: List[str],
        reward_model_info: List[Dict[str, Any]],
        prompt_ids: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
        timing_raw: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[float], List[int], List[int], List[str], List[str], List[str]]:
        """
        Execute unit tests by combining user code with unit test blocks and parsing unittest output.
        This is a cleaner approach that uses unit test blocks directly without extraction.

        Args:
            predicted_answers: List of predicted code answers
            reward_model_info: List of reward model info containing unit tests
            prompt_ids: List of prompt IDs for error tracking
            prompts: List of prompts for error tracking
            timing_raw: Dictionary to store timing information

        Returns:
            Tuple of (code_scores, tests_passed, total_tests, stdout_list, stderr_list, error_list)
        """
        if timing_raw is None:
            timing_raw = {}

        batch_size = len(predicted_answers)

        # Ensure sandbox session exists
        if self.sandbox_session is None:
            logger.info("SandboxSession not available, attempting to initialize on demand")
            with _timer("init_sandbox", timing_raw):
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

        def run_single(idx, rm_info):
            prompt_id = prompt_ids[idx] if prompt_ids and idx < len(prompt_ids) else f"batch_{self.error_tracker.batch_counter}_idx_{idx}"
            prompt_text = prompts[idx] if prompts and idx < len(prompts) else ""
            response_text = predicted_answers[idx] if idx < len(predicted_answers) else ""
            tests_raw: List[str] = []
            libs: Optional[List[str]] = None
            import ast

            if isinstance(rm_info, dict):
                raw_libs = rm_info.get("libs")
                if isinstance(raw_libs, list):
                    libs = raw_libs
                elif isinstance(raw_libs, str):
                    try:
                        libs = ast.literal_eval(raw_libs)
                    except Exception:
                        libs = [str(raw_libs)]
                elif raw_libs is not None:
                    libs = [str(raw_libs)]
                if isinstance(rm_info.get("unit_tests"), list):
                    tests_raw = rm_info["unit_tests"]
                elif isinstance(rm_info.get("tests"), list):
                    tests_raw = rm_info["tests"]
                else:
                    tc = rm_info.get("unit_tests") or rm_info.get("tests")
                    if tc:
                        tests_raw = [tc]
            if not tests_raw:
                self.error_tracker.log_error(
                    prompt_id,
                    "no_unit_tests",
                    {"message": "No unit tests provided"},
                    prompt_text,
                    response_text,
                )
                return (idx, 0.0, 0, 0, "", "", "No unit tests provided")
            code_match = re.search(r"```[\w]*\n(.*?)```", predicted_answers[idx], re.DOTALL)
            pred_code_block = code_match.group(1) if code_match else predicted_answers[idx]
            all_test_blocks = []
            for snippet in tests_raw:
                normalized_snippet = snippet.replace("\\n", "\n")
                all_test_blocks.append(normalized_snippet)
            combined_test = f"""import unittest\nimport pandas as pd\nimport numpy as np\n\n{pred_code_block}\n\n{chr(10).join(all_test_blocks)}\n\nif __name__ == '__main__':\n    unittest.main(verbosity=2)\n"""
            try:
                result = self.safe_executor.execute_safely(combined_test, session=sess, libraries=libs)
                is_unittest_failure = "Ran " in result.get("stderr", "") or "FAILED" in result.get("stderr", "") or "PASSED" in result.get("stderr", "")
                if result.get("success") or is_unittest_failure:
                    stdout = result.get("stdout", "")
                    stderr = result.get("stderr", "")
                    output = stdout + stderr
                    ran_match = re.search(r"Ran (\d+) tests? in", output)
                    if ran_match:
                        total = int(ran_match.group(1))
                    else:
                        test_count = 0
                        for block in all_test_blocks:
                            test_count += len(re.findall(r"def\\s+test_\\w+", block))
                        total = test_count
                    failures = 0
                    errors = 0
                    failed_match = re.search(
                        r"FAILED \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?\)",
                        output,
                    )
                    if failed_match:
                        if failed_match.group(1):
                            failures = int(failed_match.group(1))
                        if failed_match.group(2):
                            errors = int(failed_match.group(2))
                    passed = total - failures - errors
                    if result["exit_code"] == 0:
                        passed = total
                    code_score = float(passed) * 0.2
                    self.error_tracker.log_success(
                        prompt_id,
                        {
                            "tests_passed": passed,
                            "total_tests": total,
                            "code_score": code_score,
                            "success_rate": passed / total if total > 0 else 0.0,
                            "libraries": libs or [],
                        },
                        prompt_text,
                        response_text,
                    )
                    return (idx, code_score, passed, total, stdout, stderr, "")
                else:
                    stdout = result.get("stdout", "")
                    stderr = result.get("stderr", result.get("error", ""))
                    error = result.get("error", "Unknown error")
                    self.error_tracker.log_error(
                        prompt_id,
                        result.get("error_type", "execution_error"),
                        {
                            "error_message": result.get("error", "Unknown error"),
                            "exit_code": result.get("exit_code", -1),
                            "stdout": result.get("stdout", ""),
                            "stderr": result.get("stderr", ""),
                            "libraries": libs or [],
                        },
                        prompt_text,
                        response_text,
                    )
                    return (idx, 0.0, 0, 0, stdout, stderr, error)
            except Exception as exec_e:
                error = str(exec_e)
                self.error_tracker.log_error(
                    prompt_id,
                    "unexpected_exception",
                    {
                        "exception_type": type(exec_e).__name__,
                        "exception_message": str(exec_e),
                        "libraries": libs or [],
                    },
                    prompt_text,
                    response_text,
                )
                test_count = 0
                for block in all_test_blocks:
                    test_count += len(re.findall(r"def\s+test_\w+", block))
                return (idx, 0.0, 0, test_count, "", "", error)

        # Parallel execution
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_single, idx, rm_info) for idx, rm_info in enumerate(reward_model_info)]
            for future in as_completed(futures):
                idx, code_score, passed, total, stdout, stderr, error = future.result()
                code_scores[idx] = code_score
                tests_passed[idx] = passed
                total_tests[idx] = total
                stdout_list[idx] = stdout
                stderr_list[idx] = stderr
                error_list[idx] = error

        # Increment batch counter for error tracking
        self.error_tracker.increment_batch()

        return (
            code_scores,
            tests_passed,
            total_tests,
            stdout_list,
            stderr_list,
            error_list,
        )

    def evaluate_code(
        self,
        decoded_pred_answers: List[str],
        original_prompts: List[str],
        ground_truth_infos: List[Dict[str, Any]],
        batch_size: int,
        batch_indices: List[int],
        timing_raw: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[float], List[int], List[str], List[str], Dict[str, List[float]]]:
        """
        Evaluate code responses using either standard or interleaved reasoning.

        Args:
            decoded_pred_answers: List of decoded response strings
            original_prompts: List of original prompts
            ground_truth_infos: List of ground truth information dictionaries
            batch_size: Number of samples in the batch
            timing_raw: Dictionary to store timing information

        Returns:
            Tuple of (scores, decisions, explanations, raw_responses, component_rewards)
        """
        if timing_raw is None:
            timing_raw = {}

        if self.is_interleaved:
            print("Using interleaved reasoning evaluation")
            with _timer("interleaved_reasoning_evaluation", timing_raw):
                return self._evaluate_interleaved_reasoning(
                    decoded_pred_answers,
                    original_prompts,
                    ground_truth_infos,
                    batch_size,
                    batch_indices,
                    timing_raw,
                )
        else:
            print("Using standard code evaluation")
            with _timer("standard_code_evaluation", timing_raw):
                return self._evaluate_code(
                    decoded_pred_answers,
                    ground_truth_infos,
                    batch_size,
                    batch_indices,
                    timing_raw,
                )

    def extract_code_snippet(self, predicted_answer: str) -> str:
        """Extract code snippet from predicted answer."""
        if not predicted_answer.strip():
            return ""

        # Method 1: Try to extract code between triple backticks
        code_block_pattern = r"```(?:python)?\s*(.*?)```"
        code_matches = re.findall(code_block_pattern, predicted_answer, re.DOTALL)

        if code_matches:
            # Return the first code block found
            code_snippet = code_matches[0].strip()
            logger.debug(f"Extracted code from ``` block: {len(code_snippet)} characters")
            return code_snippet

        # Method 2: Try to extract from "def task_func" onwards
        task_func_pattern = r"(def task_func.*?)(?=\n\n|\n(?:def |class |import |from |#|$)|\Z)"
        task_func_match = re.search(task_func_pattern, predicted_answer, re.DOTALL)

        if task_func_match:
            code_snippet = task_func_match.group(1).strip()
            logger.debug(f"Extracted code from def task_func: {len(code_snippet)} characters")
            return code_snippet

        # Method 3: Look for any function definition as fallback
        function_pattern = r"(def \w+.*?)(?=\n\n|\n(?:def |class |import |from |#|$)|\Z)"
        function_matches = re.findall(function_pattern, predicted_answer, re.DOTALL)

        if function_matches:
            # Return the first function found
            code_snippet = function_matches[0].strip()
            logger.debug(f"Extracted code from function def: {len(code_snippet)} characters")
            return code_snippet

        logger.debug(f"No code snippet found in predicted answer")
        return ""

    def _check_generated_unit_tests(self, unit_test_text: str) -> Tuple[float, str]:
        """
        Check if generated unit tests look reasonable.

        Args:
            unit_test_text: The generated unit test text

        Returns:
            Tuple of (score, explanation)
        """
        if not unit_test_text or not unit_test_text.strip():
            return 0.0, "Unit Tests: No unit tests found"

        # Check if unit tests look reasonable (same logic as in interleaved reasoning)
        if "assert" in unit_test_text or "test" in unit_test_text.lower() or "unittest" in unit_test_text.lower() or "def test_" in unit_test_text:
            return 1.0, "Unit Tests: Self-generated tests provided"
        else:
            return 0.0, "Unit Tests: No unit tests found"

    def _evaluate_code(
        self,
        decoded_pred_answers: List[str],
        ground_truth_infos: List[Dict[str, Any]],
        batch_size: int,
        batch_indices: List[int],
        timing_raw: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[float], List[int], List[str], List[str], Dict[str, List[float]]]:
        """
        Evaluate standard code responses using sandbox execution.

        Args:
            decoded_pred_answers: List of decoded response strings
            ground_truth_infos: List of ground truth information dictionaries
            batch_size: Number of samples in the batch
            timing_raw: Dictionary to store timing information

        Returns:
            Tuple of (scores, decisions, explanations, raw_responses, component_rewards)
        """
        if timing_raw is None:
            timing_raw = {}

        # Check if any unit tests are present
        def _has_tests(info: Dict[str, Any]):
            return bool(isinstance(info, dict) and (("unit_tests" in info and info["unit_tests"]) or ("tests" in info and info["tests"])))

        use_code_evaluator = any(_has_tests(info) for info in ground_truth_infos)

        if use_code_evaluator:
            # Extract code snippets from predicted answers first
            logger.info("Extracting code snippets from predicted answers")
            extracted_code_answers = []
            failed_extraction_indices = []

            for i, pred_answer in enumerate(decoded_pred_answers):
                extracted_code = self.extract_code_snippet(pred_answer)
                if not extracted_code.strip():
                    failed_extraction_indices.append(i)
                    extracted_code_answers.append("")  # Use empty string for failed extractions
                else:
                    extracted_code_answers.append(extracted_code)

            if failed_extraction_indices:
                logger.warning(f"Failed to extract code from {len(failed_extraction_indices)} samples: {failed_extraction_indices}")

            # Use local unit test execution
            logger.info("Using local unit test execution for code evaluation")
            with _timer("run_unit_tests", timing_raw):
                (
                    code_scores,
                    code_tests_passed,
                    code_total_tests,
                    code_stdout,
                    code_stderr,
                    code_error,
                ) = self.run_unit_tests_combined(
                    extracted_code_answers,
                    ground_truth_infos,
                    prompt_ids=[f"prompt_{i}" for i in batch_indices],
                    prompts=decoded_pred_answers,
                    timing_raw=timing_raw,
                )

            # Create decisions and normalized scores based on test results
            decisions = []
            explanations = []
            total_scores = []
            code_scores = []
            pass_at_1 = []
            unit_test_scores = []

            for i in range(batch_size):
                if i in failed_extraction_indices:
                    # Default values for failed code extraction
                    total_scores.append(0.0)
                    decisions.append(0)
                    explanations.append("failed code extraction")
                    pass_at_1.append(0)
                    unit_test_scores.append(0.0)
                    code_scores.append(0.0)
                else:
                    # Normalize score as passed_tests / total_tests
                    if code_total_tests[i] > 0:
                        normalized_score = code_tests_passed[i] / code_total_tests[i]
                    else:
                        normalized_score = 0.0

                    # Check for generated unit tests in the original response
                    unit_test_score, unit_test_explanation = self._check_generated_unit_tests(decoded_pred_answers[i])

                    weights = self.interleaved_reward_weights
                    total_score = normalized_score * weights["code"] + unit_test_score * weights["unit_tests"]
                    code_scores.append(normalized_score)
                    total_scores.append(total_score)
                    decisions.append(1 if normalized_score > 0 else 0)
                    explanations.append(f"passed {code_tests_passed[i]}/{code_total_tests[i]} tests | {unit_test_explanation} \\nstdout: {code_stdout[i]} \\nstderr: {code_stderr[i]} \\nerror: {code_error[i]}")
                    pass_at_1.append(1 if normalized_score == 1.0 else 0)
                    unit_test_scores.append(unit_test_score)

                raw_responses = [""] * batch_size

            return (
                total_scores,
                decisions,
                explanations,
                raw_responses,
                {
                    "pass@1": pass_at_1,
                    "code_scores": code_scores,
                    "unit_test_scores": unit_test_scores,
                    "total_scores": total_scores,
                },
            )
        else:
            raise ValueError("No unit tests available, using simple heuristic evaluation")

    def _evaluate_interleaved_reasoning(
        self,
        decoded_pred_answers: List[str],
        original_prompts: List[str],
        ground_truth_infos: List[Dict[str, Any]],
        batch_size: int,
        batch_indices: List[int],
        timing_raw: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[float], List[int], List[str], List[str], Dict[str, List[float]]]:
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
            batch_indices: List of batch indices
            timing_raw: Dictionary to store timing information

        Returns:
            Tuple of (scores, decisions, explanations, raw_responses)
        """
        if timing_raw is None:
            timing_raw = {}

        total_scores = [0.0] * batch_size
        decisions = [0] * batch_size
        explanations = [""] * batch_size
        raw_responses = [""] * batch_size

        # --- Batch-evaluate all descriptions first ---
        with _timer("evaluate_descriptions", timing_raw):
            description_scores_map = self._evaluate_all_descriptions(decoded_pred_answers, original_prompts, timing_raw)

        component_rewards = collections.defaultdict(list)

        # --- Collect all code/unit-test pairs for batch execution ---
        all_extracted_code = []
        all_unit_test_info = []
        code_eval_indices = []  # Indices where we have code/unit-tests to evaluate
        for i, (pred_answer, gt_info) in enumerate(zip(decoded_pred_answers, ground_truth_infos)):
            all_answers = extract_solution(pred_answer, template_type=self.template_type)
            if isinstance(all_answers, list):
                answer_parts = [part.strip() for part in all_answers]
            else:
                answer_parts = [part.strip() for part in str(all_answers).split(",")]
            # Defensive: ensure at least 3 parts
            if len(answer_parts) < 3:
                all_extracted_code.append("")
                all_unit_test_info.append({})
                continue

            code_text = answer_parts[1]
            extracted_code = self.extract_code_snippet(code_text)
            all_extracted_code.append(extracted_code)
            if isinstance(gt_info, dict) and (gt_info.get("unit_tests") or gt_info.get("tests")):
                all_unit_test_info.append(
                    {
                        "ground_truth": "code",
                        "unit_tests": gt_info.get("unit_tests") or gt_info.get("tests"),
                        "libs": gt_info.get("libs", []),
                    }
                )
                code_eval_indices.append(i)
            else:
                all_unit_test_info.append({})
                # raise ValueError(f"No unit tests found for sample {i}, batch_indx: {batch_indices[i]}")

        # --- Run all code/unit-tests in parallel (single batch call) ---
        code_scores_list = [0.0] * batch_size
        code_tests_passed = [0] * batch_size
        code_total_tests = [0] * batch_size
        print(f"Running unit tests for {len(code_eval_indices)} samples")
        if any(all_unit_test_info[i] for i in code_eval_indices):
            (batch_code_scores, batch_tests_passed, batch_total_tests, _, _, _) = self.run_unit_tests_combined(
                all_extracted_code,
                all_unit_test_info,
                prompt_ids=[f"interleaved_prompt_{i}" for i in batch_indices],
                prompts=decoded_pred_answers,
                timing_raw=timing_raw,
            )
            for idx in code_eval_indices:
                code_scores_list[idx] = batch_code_scores[idx]
                code_tests_passed[idx] = batch_tests_passed[idx]
                code_total_tests[idx] = batch_total_tests[idx]
        print(f"Done running unit tests")

        # --- Main evaluation loop (now just uses batch results) ---
        with _timer("evaluate_interleaved_components", timing_raw):
            for i, (pred_answer, gt_info) in enumerate(zip(decoded_pred_answers, ground_truth_infos)):
                all_answers = extract_solution(pred_answer, template_type=self.template_type)
                if isinstance(all_answers, list):
                    answer_parts = [part.strip() for part in all_answers]
                else:
                    answer_parts = [part.strip() for part in str(all_answers).split(",")]
                if len(answer_parts) < 3:
                    explanations[i] = "Not enough answer parts"
                    continue
                # Initialize component scores
                description_score = description_scores_map.get(i, 0.0)
                code_score = 0.0
                unit_test_score = 0.0
                component_explanations = []
                # Evaluate first answer (description/explanation)
                description_score = description_scores_map.get(i, 0.0)
                component_explanations.append(f"Description Score: {description_score:.2f}")
                # Evaluate second answer (code implementation)
                if code_total_tests[i] > 0:
                    code_score = code_tests_passed[i] / code_total_tests[i]
                    component_explanations.append(f"Code: passed {code_tests_passed[i]}/{code_total_tests[i]} tests")
                else:
                    component_explanations.append("Code: No gt unit tests provided or failed extraction")
                # Evaluate third answer (self-generated unit tests) - optional
                unit_test_text = answer_parts[2]
                unit_test_score, unit_test_explanation = self._check_generated_unit_tests(unit_test_text)
                component_explanations.append(unit_test_explanation)
                # Combine scores with weights
                weights = self.interleaved_reward_weights
                total_score = description_score * weights["description"] + code_score * weights["code"] + unit_test_score * weights["unit_tests"]
                total_scores[i] = total_score
                decisions[i] = 1 if total_score > 1.0 else 0  # Threshold for success
                explanations[i] = " | ".join(component_explanations)
                raw_responses[i] = f"Interleaved evaluation: {len(answer_parts)} answers found"
                component_rewards["description_scores"].append(description_score)
                component_rewards["code_scores"].append(code_score)
                component_rewards["unit_test_scores"].append(unit_test_score)
                component_rewards["pass@1"].append(1 if code_score == 1.0 else 0)
        return total_scores, decisions, explanations, raw_responses, component_rewards

    def _evaluate_all_descriptions(
        self,
        decoded_pred_answers: List[str],
        original_prompts: List[str],
        timing_raw: Optional[Dict[str, float]] = None,
    ) -> Dict[int, float]:
        """
        Evaluate all description parts of interleaved answers in a single batch.
        """
        if timing_raw is None:
            timing_raw = {}

        if not self.autorater_base_url:
            logger.warning("AutoRater service URL not configured in CodeEvaluator; skipping description evaluation.")
            return {}

        descriptions_to_eval: List[Tuple[int, str, str]] = []

        for i, pred_answer in enumerate(decoded_pred_answers):
            answers = extract_solution(pred_answer, template_type=self.template_type)
            if isinstance(answers, list):
                first_answer = answers[0]
            else:
                first_answer = answers
            if first_answer and isinstance(first_answer, str):
                descriptions_to_eval.append((i, original_prompts[i], first_answer))

        if not descriptions_to_eval:
            return {}

        # Prepare payload for the AutoRater service
        batch_indices, batch_prompts, batch_responses = zip(*descriptions_to_eval)

        with _timer("tokenize_descriptions", timing_raw):
            tokenized_prompts = self.tokenizer(
                list(batch_prompts),
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).input_ids.tolist()
            tokenized_responses = self.tokenizer(
                list(batch_responses),
                add_special_tokens=False,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).input_ids.tolist()

        # We need to construct a valid-looking payload even if some parts are dummy
        batch_size = len(batch_indices)
        payload = {
            "prompts": tokenized_prompts,
            "responses": tokenized_responses,
            # Dummy values for fields that are not used by outline evaluation but required by schema
            "attention_mask": [[1] * len(r) for r in tokenized_responses],
            "position_ids": [list(range(len(r))) for r in tokenized_responses],
            "reward_model_info": [{"template": "outline", "ground_truth": ""} for _ in range(batch_size)],
        }

        logger.info(f"Calling AutoRater to evaluate {batch_size} description outlines.")
        scores, decisions, _, _ = call_autorater_service(
            self.autorater_base_url,
            payload,
            batch_size=batch_size,
            endpoint="/evaluate_autorater",  # Use the main endpoint
        )

        # The decision is what matters: 1 for TRUE, 0 for FALSE. Score is shaped, so use decision.
        final_scores = [1.0 if d == 1 else 0.0 for d in decisions]
        return dict(zip(batch_indices, final_scores))

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
