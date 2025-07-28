import threading
import queue
from contextlib import contextmanager
import logging
from verl.workers.code_evaluator.llm_sandbox import SafeResourceManagedExecutor
from omegaconf import DictConfig
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple, Union
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from verl.utils.autorater_client import call_autorater_service

logger = logging.getLogger(__name__)

class CodeEvaluator:
    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.executor = SafeResourceManagedExecutor(max_concurrent=config.max_concurrent)
    
    def evaluate_code_outlines(self, code_outlines: List[str], prompts: List[str]) -> List[int]:
        tokenized_prompts = self.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids.tolist()

        tokenized_outlines = self.tokenizer(
            code_outlines,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids.tolist()

        autorater_payload = {
            "prompts": tokenized_prompts,
            "responses": tokenized_outlines,
            "attention_mask": [[1] * len(r) for r in tokenized_outlines],
            "position_ids": [list(range(len(r))) for r in tokenized_outlines],
            "reward_model_info": [{"template": "outline", "ground_truth": ""} for _ in range(len(code_outlines))],
        }

        decisions, explanations, raw_responses = call_autorater_service(
            self.config.autorater_service_url,
            autorater_payload,
            batch_size=len(code_outlines),
            endpoint="/evaluate_autorater",
        )

        return decisions

    def evaluate_unit_tests(self, unit_tests: List[str]) -> List[int]:
        # for now, we just check if unit tests exist and follow the correct format 
        unit_test_rewards = []
        for unit_test in unit_tests:
            # check if using unit_test format 
            if "unittest.TestCase" in unit_test and re.search(r"def test_", unit_test):
                unit_test_rewards.append(1.0)
            else:
                unit_test_rewards.append(0.0)

        return unit_test_rewards
        
    def _test_code_snippets(self, code_snippets: List[str], unit_tests: str, required_libs: str) -> Dict[str, Any]:
        if len(code_snippets) > len(unit_tests):
            # repeat the unit tests for each code snippet 
            unit_tests = [unit_tests[0]] * len(code_snippets)
        elif len(code_snippets) < len(unit_tests) and len(code_snippets) > 0:
            # repeat the code snippets for each unit test 
            code_snippets = [code_snippets[0]] * len(unit_tests)
        
        results = []
        for code_snippet, unit_test, libs in zip(code_snippets, unit_tests, required_libs):
            code_snippet = code_snippet.replace("\\n", "\n")
            combined_test = f"""import unittest\nimport pandas as pd\nimport numpy as np\n\n{code_snippet}\n\n{unit_test}\n\nif __name__ == '__main__':\n    unittest.main(verbosity=2)\n"""

            # convert required_libs to list
            if isinstance(libs, str):
                if libs == "":
                    libs = []
                else:
                    libs = ast.literal_eval(libs)
            elif isinstance(libs, list):
                libs = [ast.literal_eval(lib) for lib in libs]
            
            result = self.executor.execute_safely(combined_test, libraries=libs)
            results.append(result)
        return results

    def _evaluate_code_helper(self, code_snippets: List[str], rm_infos: List[Dict[str, Any]], batch_indices: List[int]) -> Tuple[List[float], List[Dict[str, Any]]]:
        code_snippets = [self._extract_code_snippets(snippet) for snippet in code_snippets]

        # First extract the ground truth unit tests 
        unit_tests = []
        for info in rm_infos:
            if "unit_tests" in info:
                unit_tests.append(info["unit_tests"])
            elif "tests" in info:
                unit_tests.append(info["tests"])
            else:
                raise ValueError(f"No unit tests found in ground truth info: {info}")
        
        # run the code snippets 
        sandbox_results = []

        if self.config.execute_sequential:
            for i, code_snippet in enumerate(code_snippets):
                import ipdb; ipdb.set_trace()
                code_snippet = code_snippets[i]
                unit_tests = unit_tests[i]
                required_libs = rm_infos[i].get("libs", [])
                result = self._test_code_snippets(code_snippet, unit_tests, required_libs)
                sandbox_results.append(result)
        else:
            # run the code snippets in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
                futures = [
                    executor.submit(self._test_code_snippets, code_snippets[i], unit_tests[i], rm_infos[i].get("libs", []))
                    for i, code_snippet in enumerate(code_snippets)
                ]
                for future in as_completed(futures):
                    sandbox_results.append(future.result())
                
        # print(sandbox_results)
        # parse the results to compute success rate 
        unit_test_pass_rate = []
        
        for result in sandbox_results:
            if result is None:
                unit_test_pass_rate.append(0)
                continue
            
            single_pass_rate = []
            for r in result:
                ran_successfully = "PASSED" in r["stderr"] or "FAILED" in r["stderr"] or "Ran" in r["stderr"]
                if not ran_successfully:
                    unit_test_pass_rate.append(0)
                else:
                    stdout = r.get("stdout", "")
                    stderr = r.get("stderr", "")
                    output = stdout + stderr
                    ran_match = re.search(r"Ran (\d+) tests? in", output)
                    total_unit_tests = int(ran_match.group(1)) if ran_match else 0

                    failed_match = re.search(r"FAILED \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?\)", output)
                    
                    failures = 0
                    errors = 0
                    if failed_match:
                        if failed_match.group(1):
                            failures = int(failed_match.group(1))
                        if failed_match.group(2):
                            errors = int(failed_match.group(2))

                    passed = total_unit_tests - failures - errors 
                    passed = max(passed, 0)
                    single_pass_rate.append(passed / total_unit_tests if total_unit_tests > 0 else 0.0)
            
            if len(single_pass_rate) > 0:
                unit_test_pass_rate.append(sum(single_pass_rate) / len(single_pass_rate))
            else:
                unit_test_pass_rate.append(0)

        return unit_test_pass_rate, sandbox_results

    def _extract_code_snippets(self, answer: Union[str, List[str]]) -> List[str]:
        """Extract a list of code snippets from predicted answer."""
        code_block_pattern = r"```python\s*(.*?)\s*```"
        
        if isinstance(answer, str):
            # Try to extract code between triple backticks and has def or class 
            code_matches = re.findall(code_block_pattern, answer, re.DOTALL)

            if code_matches:
                return code_matches

            return [answer]
            
        elif isinstance(answer, list):
            code_matches = []
            for a in answer:
                matches = re.findall(code_block_pattern, a, re.DOTALL)
                if matches:
                    code_matches.extend(matches)
                else:
                    code_matches.append(a)
            return code_matches
        
        return [answer] 

    def evaluate_code(self, answers: List[str], prompts: List[str], rm_infos: List[Dict[str, Any]], batch_indices: List[int]) -> Dict[str, List[float]]:
        unit_test_pass_rate, sandbox_results = self._evaluate_code_helper(answers, rm_infos, batch_indices)
        pass_1 = [1 if r == 1.0 else 0 for r in unit_test_pass_rate]

        code_rewards = {
            "unit_test_pass_rate": unit_test_pass_rate,
            "pass@1": pass_1,
        }

        return code_rewards

    def evaluate_interleaved_outline_code_test(self, answers: List[List[str]], prompts: List[str], rm_infos: List[Dict[str, Any]], batch_indices: List[int]) -> Dict[str, List[float]]:
        # answers are already extracted and should be a list of lists of answers

        # these ones have 3 answers,
        # the first answer is the code outline,
        # the second answer is the code,
        # the third answer is the unit tests
        code_outlines = [answers[i][0] for i in range(len(answers))]
        code_snippets = [answers[i][1] for i in range(len(answers))]
        unit_tests = [answers[i][2] for i in range(len(answers))]

        # run the code snippets 
        unit_test_pass_rate, sandbox_results = self._evaluate_code_helper(
            code_snippets, 
            [rm_infos[i] for i in range(len(answers))], 
            [batch_indices[i] for i in range(len(answers))]
        )

        pass_1 = [1 if r == 1.0 else 0 for r in unit_test_pass_rate]

        # evaluate the outlines with llm autorater
        code_outline_helpfulness = self.evaluate_code_outlines(
            code_outlines, 
            prompts
        )

        # evaluate the generated unit tests  
        unit_test_rewards = self.evaluate_unit_tests(unit_tests)

        code_rewards = {
            "unit_test_pass_rate": unit_test_pass_rate,
            "code_outline_helpfulness": code_outline_helpfulness,
            "unit_test_rewards": unit_test_rewards,
            "pass@1": pass_1,
        }

        return code_rewards