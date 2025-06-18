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
Preprocess the Knights and Knaves dataset to parquet format
"""

import argparse
import os

import datasets
from datasets import concatenate_datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_text):
    """Extract the ground truth solution from the solution_text field."""
    # The solution_text already contains the formatted conclusion
    # Example: "(1) Zoey is a knave (2) Oliver is a knight"
    return solution_text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/knights_and_knaves")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--subsets", nargs='+', default=["2ppl"], 
                        choices=["2ppl", "3ppl", "4ppl", "5ppl", "6ppl", "7ppl", "8ppl"],
                        help="Which subsets to process (number of people in puzzles). Can specify multiple subsets.")

    args = parser.parse_args()

    data_source = "K-and-K/knights-and-knaves"

    print(f"Loading the {data_source} dataset (subsets: {args.subsets}) from huggingface...", flush=True)

    # Load datasets for all specified subsets
    train_datasets = []
    test_datasets = []
    
    for subset in args.subsets:
        print(f"Loading subset: {subset}")
        train_dataset = datasets.load_dataset(data_source, "train", split=subset)
        test_dataset = datasets.load_dataset(data_source, "test", split=subset)
        
        print(f"  Train dataset size for {subset}: {len(train_dataset)}")
        print(f"  Test dataset size for {subset}: {len(test_dataset)}")
        
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    # Combine all train and test datasets
    if len(train_datasets) > 1:
        combined_train_dataset = concatenate_datasets(train_datasets)
        combined_test_dataset = concatenate_datasets(test_datasets)
    else:
        combined_train_dataset = train_datasets[0]
        combined_test_dataset = test_datasets[0]
    
    print(f"\nCombined train dataset size: {len(combined_train_dataset)}")
    print(f"Combined test dataset size: {len(combined_test_dataset)}")

    instruction_following = 'You must infer the identity of each character. At the end of your answer, you must clearly state the identity of each character by following the format:\n\nCONCLUSION:\n(1) ...\n(2) ...\n(3) ...'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            quiz_raw = example.pop("quiz")

            question = quiz_raw + "\n\n" + instruction_following

            solution_text_raw = example.pop("solution_text")
            solution = extract_solution(solution_text_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "logical_reasoning",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": quiz_raw,
                    "answer": solution_text_raw,
                    # "quiz": quiz_raw,
                    # "names": example.get("names", []),
                    # "solution_details": example.get("solution", []),
                    # "subsets": args.subsets,
                },
            }
            return data

        return process_fn

    train_dataset = combined_train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = combined_test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Print an example of the processed data
    print(f"\nExample of processed Knights and Knaves data:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    if len(test_dataset) > 0:
        example = test_dataset[0]
        print(f"Question preview: {example['prompt'][0]['content'][:300]}...")
        print(f"Ground truth: {example['reward_model']['ground_truth']}")
        print(f"Data source: {example['data_source']}")
        print(f"Ability: {example['ability']}")
        # print(f"Subsets: {example['extra_info']['subsets']}")
        print()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir) 