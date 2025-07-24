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
Generalized preprocessing script for Knights and Knaves and SimpleQA datasets.
Outputs a structured format with reward_model and data_source fields.

python3 verl/examples/data_preprocess/create_parquets.py --dataset_name mbpp --local_dir data --combine_n 2 --prompt_combine_mode and
python3 verl/examples/data_preprocess/create_parquets.py --dataset_name simpleqa --local_dir data --combine_n 2 --prompt_combine_mode and
"""

import argparse
import os
import datasets
from datasets import concatenate_datasets
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from generate_concat_interleaved import test_list_to_unittest

LLM_MODEL = None
LLM_TOKENIZER = None
LLM_MODEL_NAME = None
LLM_DEVICE_MAP = None

def process_knights_and_knaves(local_dir, subsets):
    data_source = "K-and-K/knights-and-knaves"
    instruction_following = 'You must infer the identity of each character. At the end of your answer, you must clearly state the identity of each character by following the format:\n\nCONCLUSION:\n(1) ...\n(2) ...\n(3) ...'

    train_datasets = []
    test_datasets = []
    for subset in subsets:
        print(f"Loading subset: {subset}")
        train_dataset = datasets.load_dataset(data_source, "train", split=subset)
        test_dataset = datasets.load_dataset(data_source, "test", split=subset)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    if len(train_datasets) > 1:
        combined_train_dataset = concatenate_datasets(train_datasets)
        combined_test_dataset = concatenate_datasets(test_datasets)
    else:
        combined_train_dataset = train_datasets[0]
        combined_test_dataset = test_datasets[0]

    def extract_solution(solution_text):
        return solution_text.strip()

    def make_map_fn(split):
        def process_fn(example, idx):
            quiz_raw = example.pop("quiz")
            question = quiz_raw + "\n\n" + instruction_following
            solution_text_raw = example.pop("solution_text")
            solution = extract_solution(solution_text_raw)
            data = {
                "data_source": "k&k",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution_text_raw,
                    "question": quiz_raw,
                },
            }
            return data
        return process_fn

    train_dataset = combined_train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = combined_test_dataset.map(function=make_map_fn("test"), with_indices=True)

    return train_dataset, test_dataset

def process_simpleqa(local_dir):
    data_source = "SimpleQA"
    print("Loading SimpleQA from HuggingFace...")
    ds = datasets.load_dataset("basicv8vc/SimpleQA")
    test_dataset = ds["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example["problem"]
            answer = example["answer"]
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "reward_model": {"ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question,
                    "answer": answer,
                },
            }
            return data
        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return test_dataset

def process_mbpp(local_dir):
    data_source = "mbpp"
    print("Loading MBPP from HuggingFace...")
    ds = datasets.load_dataset("mbpp")
    test_dataset = ds["test"]
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example["text"]
            answer = example["code"]
            # Generate unit tests with function renaming
            unit_tests = test_list_to_unittest(example["test_list"], func_name="task_func")
            answer = re.sub(r'def\s+\w+\(', 'def task_func(', answer)
            answer = answer.replace('solution(', 'task_func(')
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "reward_model": {"ground_truth": answer, "unit_tests": unit_tests},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": prompt,
                    "answer": answer,
                },
            }
            return data
        return process_fn
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return test_dataset

def process_mbpp_combined(local_dir, n=2, sep=" ", answer_sep="\n\n", prompt_combine_mode="space", llm_model_name="Qwen/Qwen3-8B", llm_device_map="auto"):
    data_source = f"mbpp_combined_{prompt_combine_mode}_{n}"
    print("Loading MBPP from HuggingFace...")
    ds = datasets.load_dataset("mbpp")
    test_dataset = ds["test"]
    def combine_dataset(dataset, split):
        combined = []
        for i in range(0, len(dataset), n):
            group = dataset[i:i+n]
            if len(group) < n:
                continue
            # For MBPP, use 'text' as prompt, 'code' as answer
            group_dict = {
                "prompt": group["text"],
                "code": group["code"],
                "test_list": group["test_list"],
            }
            # Combine unit tests for all problems in the group
            combined_unit_tests = '\n\n'.join([test_list_to_unittest(tests, func_name="task_func") for tests in group_dict["test_list"]])
            # Rename function in code
            group_dict["code"] = [re.sub(r'def\s+\w+\(', 'def task_func(', c).replace('solution(', 'task_func(') for c in group_dict["code"]]
            combined_ex = combine_examples(
                group_dict,
                prompt_key="prompt",
                answer_key="code",
                sep=sep,
                answer_sep=answer_sep,
                prompt_combine_mode=prompt_combine_mode,
                llm_model_name=llm_model_name,
                llm_device_map=llm_device_map
            )
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": combined_ex["prompt"],
                    }
                ],
                "reward_model": {"ground_truth": combined_ex["answer"], "unit_tests": combined_unit_tests},
                "extra_info": {
                    "split": split,
                    "index": i // n,
                    "questions": group_dict["prompt"],
                    "answers": group_dict["code"],
                },
            }
            combined.append(data)
        return combined
    test_combined = combine_dataset(test_dataset, "test")
    return test_combined

def combine_examples(examples, prompt_key="question", answer_key="answer", sep=" ", answer_sep=", ", prompt_combine_mode="space", llm_model_name=None, llm_device_map=None):
    """
    Combine a list of examples into a single prompt and a single answer.
    - prompt_combine_mode: 'space', 'and', or 'llm'
    """
    prompts = examples[prompt_key]
    # Remove trailing punctuation from all but the last prompt if combining with 'and' or 'space', and lowercase the second and later prompts
    if prompt_combine_mode in ("and", "space") and len(prompts) > 1:
        cleaned_prompts = []
        for i, p in enumerate(prompts):
            cleaned = p.strip()
            if i < len(prompts) - 1:
                cleaned = re.sub(r'[\.,!?;:]+$', '', cleaned)
            if i > 0:
                cleaned = cleaned.lower()
            cleaned_prompts.append(cleaned)
        prompts = cleaned_prompts
    if prompt_combine_mode == "and":
        combined_prompt = " and ".join(prompts)
    elif prompt_combine_mode == "llm":
        combined_prompt = merge_prompts_with_llm(prompts, model_name=llm_model_name, device_map=llm_device_map)
    else:
        combined_prompt = sep.join(prompts)
    combined_answer = answer_sep.join(examples[answer_key])
    return {"prompt": combined_prompt, "answer": combined_answer}


def merge_prompts_with_llm(prompts, model_name=None, device_map=None):
    """
    Use Qwen LLM to merge prompts into a single, natural question.
    """
    global LLM_MODEL, LLM_TOKENIZER, LLM_MODEL_NAME, LLM_DEVICE_MAP
    if LLM_MODEL is None or LLM_TOKENIZER is None or model_name != LLM_MODEL_NAME or device_map != LLM_DEVICE_MAP:
        print(f"Loading LLM for prompt merging: {model_name} (device_map={device_map})")
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True
        )
        LLM_MODEL_NAME = model_name
        LLM_DEVICE_MAP = device_map
    # Compose a system prompt
    system_prompt = "You are an expert at combining multiple short questions into a single, natural, clear question."
    user_prompt = "Combine the following questions into a single, natural question that asks for all the information together.\n\n" + "\n".join(f"- {p}" for p in prompts)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = LLM_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = LLM_TOKENIZER([text], return_tensors="pt").to(LLM_MODEL.device)
    with torch.no_grad():
        output = LLM_MODEL.generate(
            **model_inputs,
            max_new_tokens=128,
            temperature=0.2,
            top_p=0.7,
            do_sample=True,
            pad_token_id=LLM_TOKENIZER.eos_token_id
        )
    output_ids = output[0][len(model_inputs.input_ids[0]):].tolist()
    response = LLM_TOKENIZER.decode(output_ids, skip_special_tokens=True).strip()
    return response


def process_simpleqa_combined(local_dir, n=2, sep=" ", answer_sep=", ", prompt_combine_mode="space", llm_model_name="Qwen/Qwen3-8B", llm_device_map="auto"):
    """
    Loads simpleqa, combines every n examples into one, and saves to parquet.
    prompt_combine_mode: 'space', 'and', or 'llm'
    """
    data_source = f"simpleqa_combined_{prompt_combine_mode}_{n}"
    print("Loading SimpleQA from HuggingFace...")
    ds = datasets.load_dataset("basicv8vc/SimpleQA")
    test_dataset = ds["test"]
    print(f"Loaded {len(test_dataset)} examples from SimpleQA")

    def combine_dataset(dataset, split):
        combined = []
        for i in range(0, len(dataset), n):
            group = dataset[i:i+n]

            combined_ex = combine_examples(
                group,
                prompt_key="problem",
                answer_key="answer",
                sep=sep,
                answer_sep=answer_sep,
                prompt_combine_mode=prompt_combine_mode,
                llm_model_name=llm_model_name,
                llm_device_map=llm_device_map
            )
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": combined_ex["prompt"],
                    }
                ],
                "reward_model": {"ground_truth": combined_ex["answer"]},
                "extra_info": {
                    "split": split,
                    "index": i // n,
                    "questions": group["problem"],
                    "answers": group["answer"],
                },
            }
            combined.append(data)
        return combined

    test_combined = combine_dataset(test_dataset, "test")
    return test_combined

def save_to_parquet(train_dataset, test_dataset, local_dir, prefix):
    os.makedirs(local_dir, exist_ok=True)
    if train_dataset is not None:
        train_path = os.path.join(local_dir, f"{prefix}_train.parquet")
        pd.DataFrame(train_dataset).to_parquet(train_path)
        print(f"Saved train to {train_path}")

    if test_dataset is not None:
        test_path = os.path.join(local_dir, f"{prefix}_test.parquet")
        pd.DataFrame(test_dataset).to_parquet(test_path)
        print(f"Saved test to {test_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=["knights_and_knaves", "simpleqa", "mbpp"], help="Dataset to preprocess")
    parser.add_argument("--local_dir", type=str, default="./data_structured", help="Output directory")
    parser.add_argument("--subsets", nargs='+', default=["2ppl"], help="Knights and Knaves: which subsets to process")
    parser.add_argument("--combine_n", type=int, default=2, help="For simpleqa_combined: how many prompts to combine")
    parser.add_argument("--prompt_combine_mode", type=str, default="space", choices=["space", "and", "llm"], help="How to combine prompts: 'space', 'and', or 'llm'")
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen3-8B", help="LLM model name for prompt merging (if --prompt_combine_mode=llm)")
    parser.add_argument("--llm_device_map", type=str, default="auto", help="Device map for LLM (if --prompt_combine_mode=llm)")
    args = parser.parse_args()

    if args.dataset_name == "knights_and_knaves":
        train_dataset, test_dataset = process_knights_and_knaves(args.local_dir, args.subsets)
        prefix = "knights_and_knaves"
        save_to_parquet(train_dataset, test_dataset, args.local_dir, prefix)
        print(f"\nExample from train:")
        print(train_dataset[0])
        print(f"\nExample from test:")
        print(test_dataset[0])
    elif args.dataset_name == "simpleqa":
        if args.combine_n > 1:
            test_dataset = process_simpleqa_combined(args.local_dir, n=args.combine_n, prompt_combine_mode=args.prompt_combine_mode, llm_model_name=args.llm_model_name, llm_device_map=args.llm_device_map)
            prefix = f"simpleqa_{args.prompt_combine_mode}_{args.combine_n}"
        else:
            test_dataset = process_simpleqa(args.local_dir)
            prefix = "simpleqa"
        save_to_parquet(None, test_dataset, args.local_dir, prefix)
        print(f"\nExample from test:")
        print(test_dataset[0])
    elif args.dataset_name == "mbpp":
        if args.combine_n > 1:
            test_dataset = process_mbpp_combined(args.local_dir, n=args.combine_n, prompt_combine_mode=args.prompt_combine_mode, llm_model_name=args.llm_model_name, llm_device_map=args.llm_device_map)
            prefix = f"mbpp_{args.prompt_combine_mode}_{args.combine_n}"
        else:
            test_dataset = process_mbpp(args.local_dir)
            prefix = "mbpp"
        save_to_parquet(None, test_dataset, args.local_dir, prefix)
        print(f"\nExample from test:")
        print(test_dataset[0])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

if __name__ == "__main__":
    main() 