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
Preprocess the MATH-500 dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    """Extract the final answer from the solution string."""
    # For MATH-500, the answer is typically in a \\boxed{} format
    # First try to find \\boxed{} pattern
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    match = re.search(boxed_pattern, solution_str)
    if match:
        return match.group(1)
    
    # If no boxed pattern, try to find the answer after common patterns
    answer_patterns = [
        r'final answer is[:\s]*([^\n\.]+)',
        r'answer is[:\s]*([^\n\.]+)',
        r'the answer[:\s]*([^\n\.]+)',
        r'therefore[:\s]*([^\n\.]+)',
        r'thus[:\s]*([^\n\.]+)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, solution_str, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, return the original solution
    return solution_str.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math500")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "HuggingFaceH4/MATH-500"

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source)

    # MATH-500 only has a test split
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem_raw = example.pop("problem")

            question = problem_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": problem_raw,
                    "answer": answer_raw,
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Print an example of the processed data
    print(f"\nExample of processed MATH-500 data:")
    print(f"Dataset size: {len(test_dataset)}")
    if len(test_dataset) > 0:
        example = test_dataset[0]
        print(f"Question preview: {example['prompt'][0]['content']}...")
        print(f"Ground truth: {example['reward_model']['ground_truth']}")
        print(f"Data source: {example['data_source']}")
        print(f"Ability: {example['ability']}")
        print()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create a dummy train dataset for consistency
    train_dataset = test_dataset
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir) 