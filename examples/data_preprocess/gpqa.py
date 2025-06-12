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
Preprocess the GPQA dataset to parquet format
"""

import argparse
import os
import random

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_correct_answer_letter(example):
    """Extract the correct answer letter (A, B, C, or D) from the GPQA example."""
    # Create a list of all choices including the correct answer
    choices = [
        example["Incorrect Answer 1"], 
        example["Incorrect Answer 2"], 
        example["Incorrect Answer 3"],
        example["Correct Answer"]
    ]
    
    # Randomly shuffle the incorrect answers
    incorrect_answers = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
    random.shuffle(incorrect_answers)
    
    # Insert the correct answer at a random position
    gold_index = random.randint(0, 3)
    final_choices = incorrect_answers.copy()
    final_choices.insert(gold_index, example["Correct Answer"])
    
    # Return the letter corresponding to the correct answer position
    gold_choice = "ABCD"[gold_index]
    return final_choices, gold_choice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gpqa")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--subset", default="gpqa_diamond", choices=["gpqa_diamond", "gpqa_main", "gpqa_extended"], 
                        help="Which GPQA subset to process")

    args = parser.parse_args()

    data_source = "Idavidrein/gpqa"

    print(f"Loading the {data_source} dataset (subset: {args.subset}) from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, args.subset)

    # GPQA only has a train split (which is actually test data)
    test_dataset = dataset["train"]

    # Template for GPQA multiple choice questions
    GPQA_QUERY_TEMPLATE = 'Answer the following multiple choice question. The last line of your response should be of the following format: \'Answer: $LETTER\' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("Question")
            
            # Extract choices and determine correct answer position
            choices, correct_letter = extract_correct_answer_letter(example)
            
            # Format the question with choices
            question = GPQA_QUERY_TEMPLATE.format(
                Question=question_raw,
                A=choices[0],
                B=choices[1], 
                C=choices[2],
                D=choices[3]
            )
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "science",
                "reward_model": {"style": "rule", "ground_truth": correct_letter},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "answer": example.get("Correct Answer", ""),
                    # "choices": choices,
                    # "subset": args.subset,
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Print an example of the processed data
    print(f"\nExample of processed GPQA data:")
    print(f"Dataset size: {len(test_dataset)}")
    if len(test_dataset) > 0:
        example = test_dataset[0]
        print(f"Question preview: {example['prompt'][0]['content']}...")
        print(f"Ground truth: {example['reward_model']['ground_truth']}")
        print(f"Data source: {example['data_source']}")
        print(f"Ability: {example['ability']}")
        print(f"Correct answer text: {example['extra_info']['answer']}")
        print()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create a dummy train dataset for consistency
    # train_dataset = test_dataset.select(range(0))  # Empty dataset
    train_dataset = test_dataset

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir) 