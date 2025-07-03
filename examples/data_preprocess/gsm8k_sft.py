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
Preprocess the GSM8k dataset to simple question/answer format for SFT
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k_sft")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--add_instruction", action="store_true", 
                       help="Add step-by-step instruction to questions")

    args = parser.parse_args()

    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_text = 'Let\'s think step by step and output the final answer in \\boxed{answer here}.'

    def process_fn(example):
        question = example["question"]
        answer = example["answer"]
        
        # Optionally add instruction to question
        if args.add_instruction:
            question = question + " " + instruction_text
        
        return {
            "question": question,
            "answer": answer
        }

    train_dataset = train_dataset.map(function=process_fn)
    test_dataset = test_dataset.map(function=process_fn)

    # Print examples of the processed data
    print(f"\nExample of processed GSM8K SFT data:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    if len(train_dataset) > 0:
        example = train_dataset[0]
        print(f"\nTrain example:")
        print(f"Question: {example['question'][:200]}...")
        print(f"Answer: {example['answer'][:200]}...")
    
    if len(test_dataset) > 0:
        example = test_dataset[0]
        print(f"\nTest example:")
        print(f"Question: {example['question'][:200]}...")
        print(f"Answer: {example['answer'][:200]}...")
    
    print()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"Saved datasets to {local_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied to HDFS: {hdfs_dir}") 