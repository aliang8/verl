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


def filter_by_hops(dataset, hop_types):
    """
    Filter dataset to include only questions with specified hop counts.
    Args:
        dataset: The dataset to filter
        hop_types: List of hop types to include (e.g., ["3hop", "4hop"])
    """
    def should_include(example):
        if "id" not in example:
            return False
        
        dataset_id = example["id"]
        for hop_type in hop_types:
            if dataset_id.startswith(f"{hop_type}"):
                return True
        return False
    
    return dataset.filter(should_include)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/musique")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--hop_types", nargs='+', default=["3hop", "4hop"],
                        choices=["2hop", "3hop", "4hop"],
                        help="Which hop types to include (e.g., 3hop 4hop)")

    args = parser.parse_args()

    data_source = "dgslibisey/MuSiQue"

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    # Load the dataset - MuSiQue has train and validation splits
    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]  # MuSiQue uses "validation" as test split
    
    print(f"Original train dataset size: {len(train_dataset)}")
    print(f"Original test dataset size: {len(test_dataset)}")

    # Filter datasets by hop types
    print(f"Filtering for hop types: {args.hop_types}")
    train_dataset = filter_by_hops(train_dataset, args.hop_types)
    test_dataset = filter_by_hops(test_dataset, args.hop_types)
    
    print(f"Filtered train dataset size: {len(train_dataset)}")
    print(f"Filtered test dataset size: {len(test_dataset)}")

    # Add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw

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
                dataset_id = example.pop("id")
                extra_info["dataset_id"] = dataset_id
                # Extract hop type from id
                for hop_type in ["2hop", "3hop", "4hop"]:
                    if dataset_id.startswith(f"{hop_type}_"):
                        extra_info["hop_type"] = hop_type
                        break
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
        if "hop_type" in example['extra_info']:
            print(f"Hop type: {example['extra_info']['hop_type']}")
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