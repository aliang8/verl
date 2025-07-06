#!/usr/bin/env python3
"""
Preprocess the BigCodeBench dataset to simple question/answer format for SFT
"""

import argparse
import os

import datasets  # type: ignore

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/bigcodebench_sft")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--dataset_name", default="bigcode/bigcodebench", help="HF dataset name")
    parser.add_argument("--target_split", default="v0.1.4", help="Dataset split/version to preprocess (default: v0.1.4)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of validation set size (default: 0.2 for 20%)")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed for shuffling before split")
    parser.add_argument("--add_instruction", action="store_true", 
                       help="Add step-by-step instruction to questions")
    parser.add_argument("--include_solution", action="store_true",
                       help="Include canonical solution in the answer")

    args = parser.parse_args()

    data_source = args.dataset_name

    print(f"Loading {data_source} from HuggingFace…", flush=True)

    # Load only the requested split (e.g., v0.1.4)
    dataset_dict = datasets.load_dataset(data_source)
    if args.target_split not in dataset_dict:
        raise ValueError(f"Split '{args.target_split}' not found in dataset. Available splits: {list(dataset_dict.keys())}")

    raw_ds = dataset_dict[args.target_split]
    print(f"Processing split '{args.target_split}' with {len(raw_ds)} examples…", flush=True)

    instruction_text = 'Please implement the function step by step and provide the complete solution in a code block.'

    def process_fn(example):
        question = example.get("instruct_prompt", "")
        
        # Optionally add instruction to question
        if args.add_instruction:
            question = question + " " + instruction_text
        
        # Use canonical solution as answer if available and requested
        if args.include_solution and example.get("canonical_solution"):
            answer = f"```python\n{example['canonical_solution']}\n```"
        else:
            # For SFT without canonical solution, we might want to use a placeholder
            # or skip examples without solutions
            answer = example.get("canonical_solution", "")
            if not answer:
                return None  # Skip examples without solutions
            answer = f"```python\n{answer}\n```"
        
        return {
            "question": question,
            "answer": answer
        }

    # Process and filter out None results
    processed_ds = raw_ds.map(function=process_fn)
    processed_ds = processed_ds.filter(lambda x: x is not None)

    # Shuffle & train/val split
    processed_ds = processed_ds.shuffle(seed=args.shuffle_seed, keep_in_memory=True)
    split_dict = processed_ds.train_test_split(test_size=args.val_ratio, seed=args.shuffle_seed)
    train_ds, val_ds = split_dict["train"], split_dict["test"]

    # Print examples of the processed data
    print(f"\nExample of processed BigCodeBench SFT data:")
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    
    if len(train_ds) > 0:
        example = train_ds[0]
        print(f"\nTrain example:")
        print(f"Question: {example['question'][:200]}...")
        print(f"Answer: {example['answer'][:200]}...")
    
    if len(val_ds) > 0:
        example = val_ds[0]
        print(f"\nVal example:")
        print(f"Question: {example['question'][:200]}...")
        print(f"Answer: {example['answer'][:200]}...")
    
    print()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    train_ds.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_ds.to_parquet(os.path.join(local_dir, "val.parquet"))

    print(f"Saved datasets to {local_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied to HDFS: {hdfs_dir}") 