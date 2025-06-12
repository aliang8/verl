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
Preprocess the MuSiQue dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_final_answer(answer_text):
    """
    Extract the final answer from MuSiQue's answer text.
    MuSiQue answers are typically straightforward text, but we clean them up.
    """
    if not answer_text:
        return ""
    
    # Clean up the answer text
    answer = str(answer_text).strip()
    
    # Remove common prefixes/suffixes if they exist
    answer = re.sub(r'^(the answer is:?\s*)', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^(answer:?\s*)', '', answer, flags=re.IGNORECASE)
    
    # Remove trailing punctuation if it's just a period
    answer = re.sub(r'\.$', '', answer)
    
    return answer.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/musique")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "dgslibisey/MuSiQue"

    # Load the dataset - MuSiQue has train and validation splits
    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]  # MuSiQue uses "validation" as test split

    instruction_following = 'Please read the following context carefully and answer the question. Think step by step and provide your final answer.'

    # Add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_final_answer(answer_raw)
            
            # Get additional fields that might be useful
            extra_info = {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            }
            
            # Add other available fields from the dataset
            if "id" in example:
                extra_info["dataset_id"] = example.pop("id")
            if "paragraphs" in example:
                extra_info["paragraphs"] = example.pop("paragraphs")
            if "question_decomposition" in example:
                extra_info["question_decomposition"] = example.pop("question_decomposition")
            if "answer_aliases" in example:
                extra_info["answer_aliases"] = example.pop("answer_aliases")
            if "answerable" in example:
                extra_info["answerable"] = example.pop("answerable")
            
            # Add any remaining fields to extra_info
            extra_info.update(example)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "multi_hop_reasoning",  # MuSiQue is specifically designed for multi-hop reasoning
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": extra_info,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Print an example of the processed data
    print(f"\nExample of processed MuSiQue data:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    if len(test_dataset) > 0:
        example = test_dataset[0]
        print(f"Question: {example['prompt'][0]['content']}")
        print(f"Ground truth: {example['reward_model']['ground_truth']}")
        print(f"Data source: {example['data_source']}")
        print(f"Ability: {example['ability']}")
        
        # Show some extra info if available
        if "dataset_id" in example['extra_info']:
            print(f"Dataset ID: {example['extra_info']['dataset_id']}")
        if "answerable" in example['extra_info']:
            print(f"Answerable: {example['extra_info']['answerable']}")
        print()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir) 