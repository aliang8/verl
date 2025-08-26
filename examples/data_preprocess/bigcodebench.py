#!/usr/bin/env python3
"""
Preprocess the BigCodeBench dataset to parquet format compatible with VERL.

Each example is converted to a schema similar to GSM8K/MATH500 preprocessors:

{
  "data_source": <dataset name>,
  "prompt": [{"role": "user", "content": <instruct_prompt>}],
  "ability": "code",
  "reward_model": {
     "style": "code",
     "unit_tests": <list[str]>,
     "libs": <list[str] | None>
  },
  "extra_info": {...}
}

Ground-truth functional correctness is determined by unit tests, so we do not
include a scalar ground_truth answer.
"""

import argparse
import os
from typing import Any, Dict, List, Optional

import datasets  # type: ignore

from verl.utils.hdfs_io import copy, makedirs


def process_example(example: Dict[str, Any], idx: int, split: str, data_source: str) -> Dict[str, Any]:
    """Transform a single BigCodeBench example into VERL json-able dict."""
    instruct_prompt: str = example.get("instruct_prompt", "")
    unit_tests: List[str] = example.get("test", [])
    libs: Optional[List[str]] = example.get("libs")
    canonical_solution: Optional[str] = example.get("canonical_solution")

    # Basic prompt list (single user turn)
    prompt = [{"role": "user", "content": instruct_prompt}]

    reward_model: Dict[str, Any] = {
        "style": "code",
        "unit_tests": unit_tests,
    }
    if libs:
        reward_model["libs"] = libs
    if canonical_solution is not None:
        reward_model["canonical_solution"] = canonical_solution

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "code",
        "reward_model": reward_model,
        "extra_info": {
            "split": split,
            "index": idx,
            "task_id": example.get("task_id"),
            "question": instruct_prompt,
            "answer": canonical_solution,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/bigcodebench", help="Local output directory")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy data to")
    parser.add_argument("--dataset_name", default="bigcode/bigcodebench", help="HF dataset name")
    parser.add_argument("--target_split", default="v0.1.4", help="Dataset split/version to preprocess (default: v0.1.4)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of validation set size (default: 0.2 for 20%)")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed for shuffling before split")
    args = parser.parse_args()

    data_source = args.dataset_name

    print(f"Loading {data_source} from HuggingFace…", flush=True)

    # Load only the requested split (e.g., v0.1.4)
    dataset_dict = datasets.load_dataset(data_source)
    if args.target_split not in dataset_dict:
        raise ValueError(f"Split '{args.target_split}' not found in dataset. Available splits: {list(dataset_dict.keys())}")

    raw_ds = dataset_dict[args.target_split]
    print(f"Processing split '{args.target_split}' with {len(raw_ds)} examples…", flush=True)

    # Map to VERL schema
    processed_ds = raw_ds.map(lambda ex, idx: process_example(ex, idx, args.target_split, data_source), with_indices=True)

    # Shuffle & train/val split
    processed_ds = processed_ds.shuffle(seed=args.shuffle_seed, keep_in_memory=True)
    split_dict = processed_ds.train_test_split(test_size=args.val_ratio, seed=args.shuffle_seed)
    train_ds, val_ds = split_dict["train"], split_dict["test"]

    # Preview one example
    if len(train_ds) > 0:
        print("Example processed item:")
        print(train_ds[0])

    # Save to parquet
    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, "train.parquet")
    val_path = os.path.join(args.local_dir, "val.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    print(f"Saved {len(train_ds)} training examples to {train_path}")
    print(f"Saved {len(val_ds)} validation examples to {val_path}")

    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print(f"Copied data to HDFS directory {args.hdfs_dir}")
