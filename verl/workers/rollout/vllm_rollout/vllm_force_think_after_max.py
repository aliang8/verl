#!/usr/bin/env python3
"""
vLLM rollout that forces </think> only when the initial generation hits the
max response length without closing the thinking block, then continues
for a fixed number of tokens to allow the answer to be produced.
"""

import logging
import os
from typing import List

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from vllm.lora.request import LoRARequest
from verl.utils.torch_functional import get_response_mask

try:
    # Prefer the SPMD rollout base if available (as in force answer file)
    from .vllm_rollout_spmd import vLLMRollout
except ImportError:
    # Fallback to the standard rollout base
    from .vllm_rollout import vLLMRollout


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove the left padding in the prompt token_id."""
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMForceThinkAfterMaxRollout(vLLMRollout):
    """
    vLLM rollout that:
    1) Generates up to max response length (regular vLLM generation)
    2) If the response reached max length AND does not contain </think>,
       force-append </think> and continue generation for `additional_answer_tokens`.
    3) Otherwise, returns the first generation as-is.
    """

    def __init__(self, model_path: str, config, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.additional_answer_tokens = int(config.get("additional_answer_tokens", 4096))

    def _has_think_closed(self, text: str) -> bool:
        return "</think>" in text

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if getattr(self.config, "free_cache_engine", False):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([
                _pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)
            ], dtype=object)

        vllm_inputs = [{"prompt_token_ids": raw} for raw in non_tensor_batch.pop("raw_prompt_ids")]
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        reg_kwargs = {}
        if not do_sample:
            reg_kwargs = {"best_of": 1, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "temperature": 0, "n": 1}
        elif is_validate:
            reg_kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        kwargs = {**reg_kwargs, **kwargs}

        lora_requests = None
        if getattr(self, "lora_kwargs", None):
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # First generation: up to configured response length
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

        first_responses: List[List[int]] = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response_ids = output.outputs[sample_id].token_ids
                first_responses.append(response_ids)

        # Decide which prompts need forced </think> + continuation
        to_continue_indices: List[int] = []
        forced_prefix_ids: List[List[int]] = []
        total_tokens_generated: List[int] = []
        completed_responses: List[List[int]] = [None] * len(first_responses)

        max_tokens = int(getattr(self.sampling_params, "max_tokens", self.config.response_length))

        for i, response_ids in enumerate(first_responses):
            text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            total_tokens_generated.append(len(response_ids))
            hit_max = len(response_ids) >= max_tokens
            has_closed = self._has_think_closed(text)

            if hit_max and not has_closed:
                # Force append </think> and enqueue for continuation
                forced_text = text + "</think>"
                forced_ids = self.tokenizer.encode(forced_text, add_special_tokens=False)
                vllm_input = {"prompt_token_ids": vllm_inputs[i]["prompt_token_ids"] + forced_ids}

                to_continue_indices.append(i)
                forced_prefix_ids.append(forced_ids)
                completed_responses[i] = None  # placeholder
                vllm_inputs[i] = vllm_input  # reuse array for second pass order
            else:
                # Keep as-is
                completed_responses[i] = response_ids

        # Single batched continuation for those needing more tokens
        if to_continue_indices:
            with self.update_sampling_params(max_tokens=self.additional_answer_tokens, stop=None):
                continuation_outputs = self.inference_engine.generate(
                    prompts=[vllm_inputs[i] for i in to_continue_indices],
                    sampling_params=self.sampling_params,
                    lora_request=[lora_requests[i] for i in to_continue_indices] if lora_requests else None,
                    use_tqdm=False,
                )

            out_idx = 0
            for j, cont_idx in enumerate(to_continue_indices):
                output = continuation_outputs[out_idx]
                continuation_ids = output.outputs[0].token_ids
                final_ids = forced_prefix_ids[j] + continuation_ids
                total_tokens_generated[cont_idx] += len(continuation_ids)
                completed_responses[cont_idx] = final_ids
                out_idx += 1

        # Prepare tensor outputs
        max_resp_len = max(len(r) for r in completed_responses)
        padded_responses = []
        for r in completed_responses:
            if len(r) < max_resp_len:
                r = r + [self.pad_token_id] * (max_resp_len - len(r))
            padded_responses.append(r)

        response = torch.tensor(padded_responses, device=prompts.batch["input_ids"].device)

        if self.sampling_params.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
            batch_size = batch_size * self.sampling_params.n

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        rollout_log_probs = torch.full((batch_size, response_length), -1.0, dtype=torch.float32, device=idx.device)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "rollout_log_probs": rollout_log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # Attach total token stats
        non_tensor_batch["total_tokens_generated"] = np.array(total_tokens_generated, dtype=np.int64)

        # Optional summary
        try:
            total_tokens_all = int(sum(total_tokens_generated))
            print(f"\n📊 Total Tokens Generated Summary (Force Think After Max):")
            print(f"  Total tokens across all prompts: {total_tokens_all:,}")
            print(f"  Average tokens per prompt: {total_tokens_all / len(total_tokens_generated):.1f}")
        except Exception:
            pass

        if getattr(self.config, "free_cache_engine", False):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch) 