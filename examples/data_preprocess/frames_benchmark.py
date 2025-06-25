#!/usr/bin/env python3

"""
Preprocessing script for Google Frames Benchmark dataset.
This script converts the raw Frames dataset into the format expected by VERL.

The Google Frames Benchmark is a multi-hop reasoning dataset with complex questions
that require information synthesis from multiple Wikipedia sources.

Dataset format:
- Input: "Prompt" column containing complex reasoning questions
- Output: "Answer" column containing ground truth answers
- Additional metadata: reasoning_types, wikipedia_links, etc.

Example:
    Input: "If my future wife has the same first name as the 15th first lady..."
    Output: "Jane Ballou"
"""

import re
import argparse
import datasets
from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    """
    Extract the final answer from the solution string.
    For Frames benchmark, the answer is typically straightforward.
    """
    # Clean up the solution string
    solution = solution_str.strip()
    
    # For Frames benchmark, answers are usually direct
    # Remove any trailing punctuation
    solution = re.sub(r'[.!?]+$', '', solution)
    
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/frames_benchmark")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    data_source = "google/frames-benchmark"

    # load dataset from huggingface
    dataset = datasets.load_dataset(data_source, split="test")  # Frames only has test split
    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))

    instruction_following = 'Let\'s think step by step and output the final answer in \\boxed{answer here}.'

    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract the prompt and answer from the Frames dataset
            prompt_raw = example.pop("Prompt")
            answer_raw = example.pop("Answer")
            
            # Add instruction following to the prompt
            prompt = prompt_raw + " " + instruction_following
            
            # Extract the clean solution
            final_solution = extract_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": final_solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": prompt_raw,
                },
            }
            return data

        return process_fn

    test_dataset = dataset.map(function=make_map_fn("test"), with_indices=True)

    # Print an example of the processed data
    print(f"\nExample of processed Frames Benchmark data:")
    print(f"Test dataset size: {len(test_dataset)}")
    if len(test_dataset) > 0:
        example = test_dataset[0]
        print(f"Question preview: {example['prompt'][0]['content'][:100]}...")
        print(f"Ground truth: {example['reward_model']['ground_truth']}")
        print(f"Data source: {example['data_source']}")
        print(f"Ability: {example['ability']}")
        print()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # create the directory
    makedirs(local_dir, exist_ok=True)

    # dump to parquet (following GSM8K format)
    test_dataset.to_parquet(f"{local_dir}/test.parquet")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
    
    print(f"Processed {len(test_dataset)} examples and saved to {local_dir}/test.parquet")


if __name__ == "__main__":
    main() 