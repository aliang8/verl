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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import ast 
import json
import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

# Additional imports for tool and MCP functionality
import re
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from verl.utils.torch_functional import pad_sequence_to_length
import asyncio
import time
from contextlib import AsyncExitStack

# Conditional MCP imports (may not be available in all environments)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)


class vLLMRolloutWithTool(vLLMRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        self.gen_str = "\n<|im_start|>assistant\n<think>"
        self.gen_ids = self.tokenizer.encode(self.gen_str)
    
    def format_tool_call(self, tool_call_str: str):
        """Convert JSON function call description to Python executable code string."""
        try:
            call_json = json.loads(tool_call_str)
            func_name = call_json['name']
            arguments = call_json.get('arguments', {})
            
            args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
            return f"{func_name}({args_str})"
        except Exception as e:
            return f"Parse tool call failed: {e}"

    def validate_tool_calls(self, output_str):
        start_tags = re.findall(r'<tool_call>', output_str)
        end_tags = re.findall(r'</tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_tool_calls(self, output_str):
        if not self.validate_tool_calls(output_str):
            return []

        try:
            pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []
    
    def batch_execute(self, env_list: List[str], tool_calls_list: List[List[str]]):
        def exe_tool_call(env, call):
            url = f'{self.config.sandbox_url}/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("Parse tool call failed"):
                return call_str
            
            try:
                data = {
                    'env': env,
                    'call': call_str
                }                
                response = requests.post(url, json=data, timeout=10)
                if response.status_code != 200:
                    return f"error: {response.status_code}"
                response = response.json()
                ret_str = ''
                if response['result']:
                    ret_str += f'result: \n{response["result"]}\n'
                if response['output']:
                    ret_str += f'output: \n{response["output"]}\n'
                if response['error']:
                    ret_str += f'error: \n{response["error"]}\n'
                return ret_str.strip()
            except requests.exceptions.Timeout:
                return "error: execution timed out"
            except Exception as e:
                return str(e)

        # flatten all tasks
        all_tasks = []
        task_indices = []
        for env_idx, (env, tool_calls) in enumerate(zip(env_list, tool_calls_list)):
            for call_idx, tool_call in enumerate(tool_calls):
                all_tasks.append((env, tool_call))
                task_indices.append((env_idx, call_idx))

        # parallel execute all tasks
        all_results = [None] * len(all_tasks)
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {executor.submit(exe_tool_call, env, call): i 
                            for i, (env, call) in enumerate(all_tasks)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                all_results[index] = future.result()

        # reorganize results to original structure
        results_list = [[None for _ in range(len(tool_calls_list[i]))] for i, _ in enumerate(env_list)]
        for (env_idx, call_idx), result in zip(task_indices, all_results):
            results_list[env_idx][call_idx] = result

        return results_list

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:        
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = ori_input_ids.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        with self.update_sampling_params(**kwargs):
            # prepare n copies for each input
            curr_inputs = []
            for input_ids in idx_list:
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(input_ids.copy())
            init_inputs = [ids.copy() for ids in curr_inputs]

            # if there are envs, prepare n copies for each env
            env_list = None
            if 'env' in prompts.non_tensor_batch:
                env_list = []
                for env in prompts.non_tensor_batch['env']:
                    for _ in range(self.sampling_params.n):
                        env_list.append(env)

            # track the status of each input
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))

            # collect the result mask of each rollout, 1 for non-result, 0 for tool call result or pad
            result_mask_list = [[] for _ in range(len(curr_inputs))]

            # generate until all inputs are completed
            for step in range(self.config.max_turns):
                if len(active_indices) == 0:
                    break

                # only process the active inputs
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]
                
                with self.update_sampling_params(
                    n=1, 
                    max_tokens=min(512, max(active_max_tokens)),
                    stop_token_ids=[151644],
                    top_p=0.99,
                ):  # 512 at most, and add <|im_start|> as stop for corner case
                    vllm_inputs = [{
                        'prompt_token_ids': raw_prompt_ids
                    } for raw_prompt_ids in active_inputs]
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )

                # collect all tool calls
                tool_calls_list: List[List[str]] = []
                call_indices: List[int] = []

                # process each output
                new_active_indices = []
                for i, idx in enumerate(active_indices):
                    output_ids = outputs[i].outputs[0].token_ids
                    finish_reason = outputs[i].outputs[0].finish_reason
                    stop_reason = outputs[i].outputs[0].stop_reason

                    if finish_reason == 'stop' and (stop_reason == None or stop_reason == self.tokenizer.pad_token_id):
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)

                        output_str = self.tokenizer.decode(output_ids)
                        tool_calls: List[str] = self.extract_tool_calls(output_str)
                        if tool_calls:
                            tool_calls_list.append(tool_calls)
                            call_indices.append(idx)
                            new_active_indices.append(idx)
                        else:
                            pass # no tool calls
                    elif finish_reason == 'length':
                        # output over max tokens
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and stop_reason == 151644: # 151644 is the id of <|im_start|>, is a illigal stop, we stop here
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    else:
                        raise ValueError(f"unknown stop reason. finish_reason: {finish_reason}, stop_reason: {stop_reason}")

                # batch process tool calls
                if tool_calls_list:
                    # Only tp_rank 0 executes the tools
                    if self.tp_rank == 0:
                        active_env_list = [env_list[i] for i in call_indices]
                        tool_responses_list = self.batch_execute(active_env_list, tool_calls_list)
                        
                        # Prepare data for broadcasting
                        broadcast_data = {
                            'tool_calls_list': tool_calls_list,
                            'call_indices': call_indices,
                            'tool_responses_list': tool_responses_list
                        }
                    else:
                        broadcast_data = None
                    
                    broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
                    
                    # All ranks process the broadcasted data
                    if broadcast_data is not None:
                        tool_calls_list = broadcast_data['tool_calls_list']
                        call_indices = broadcast_data['call_indices']
                        tool_responses_list = broadcast_data['tool_responses_list']

                        for idx, tool_calls, tool_responses in zip(call_indices, tool_calls_list, tool_responses_list):
                            tool_response_str = ''
                            for call, response in zip(tool_calls, tool_responses):
                                tool_response_str += f"<tool_response>{call}\n{response}\n</tool_response>\n"
                            tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"
                            output_ids = self.tokenizer.encode(tool_response_str)
                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [0] * len(output_ids)

                            curr_inputs[idx] += self.gen_ids
                            result_mask_list[idx] += [0] * len(self.gen_ids)

                # Update active indices and check length constraints
                length_checked_active_indices = []
                for idx in active_indices:
                    if len(curr_inputs[idx]) - len(init_inputs[idx]) >= self.config.response_length:
                        # Truncate to response length
                        curr_inputs[idx] = init_inputs[idx] + \
                            curr_inputs[idx][len(init_inputs[idx]):len(init_inputs[idx])+self.config.response_length]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                    else:
                        curr_max_tokens[idx] = self.config.response_length - len(curr_inputs[idx]) + len(init_inputs[idx])
                        if idx in new_active_indices:
                            length_checked_active_indices.append(idx)
                
                active_indices = length_checked_active_indices

            output_ids_list = []
            # collect the all rollouts
            for i, input_ids in enumerate(idx_list):
                for j in range(self.sampling_params.n):
                    idx = i * self.sampling_params.n + j
                    input_len = len(input_ids)
                    output_ids_list.append(curr_inputs[idx][input_len:])

        response_attention_mask_list = []
        response_list = []
        result_mask_list_padded = []
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            # to tensor 
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            result_mask = torch.tensor(result_mask, device=ori_input_ids.device)
            # response attention mask, 1 for valid, 0 for invalid
            response_attention_mask = torch.ones_like(response, dtype=torch.int64)
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
            response_attention_mask_list.append(response_attention_mask)
            # response, pad to response_length
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_list.append(response)
            # result mask, 1 for non-result, 0 for result or pad
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            result_mask_list_padded.append(result_mask)
        response_attention_mask = torch.stack(response_attention_mask_list, dim=0)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)

        if self.config.n > 1 and do_sample:
            ori_input_ids = ori_input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([ori_input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                
        # concat attenion_mask for input and response
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # result mask: result part is 0, other part is 1
        loss_mask = result_mask * response_attention_mask
        
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict({
            "prompts": ori_input_ids,
            "responses": response,
            "input_ids": seq,  # here input_ids become the whole sentences
            "rollout_log_probs": result_mask,  # we will recompute old log prob with actor
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)


class vLLMRolloutWithMCP(vLLMRollout):
    """
    vLLM Rollout with MCP (Model Context Protocol) integration.
    
    This class extends vLLMRollout to support batch calls to MCP servers
    during the generation process, enabling retrieval-augmented generation
    and other external tool integrations.
    """
    
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        
        if not MCP_AVAILABLE:
            raise ImportError("MCP (Model Context Protocol) is not available. Please install the mcp package.")
        
        # MCP server configuration
        self.mcp_server_params = StdioServerParameters(
            command=config.get("mcp_command", "python3"),
            args=config.get("mcp_args", ["-m", "wikipedia_mcp"])
        )
        
        # MCP tool configuration
        self.mcp_batch_size = config.get("mcp_batch_size", 8)
        self.mcp_timeout = config.get("mcp_timeout", 10.0)
        
        # MCP mode: "search_summary" or "direct_article"
        self.mcp_mode = config.get("mcp_mode", "direct_article")
        self.max_article_tokens = config.get("max_article_tokens", 2048)
        
        # Generation prompts
        self.gen_str = "\n<|im_start|>assistant\n"
        self.gen_ids = self.tokenizer.encode(self.gen_str)

    async def create_mcp_session_and_query(self, query: str, limit: int = 1) -> Dict[str, Any]:
        """Create a new MCP session and execute a multi-step Wikipedia query."""
        exit_stack = AsyncExitStack()
        start_time = time.time()
        
        logger.debug(f"[{time.strftime('%H:%M:%S')}] Starting MCP query for: {query}")
        
        # Create new server connection
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(self.mcp_server_params)
        )
        stdio, write = stdio_transport
        session = await exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        
        await session.initialize()
        
        if self.mcp_mode == "direct_article":
            # Mode 1: Direct article retrieval
            # First search to get the actual article title
            search_args = {"query": query, "limit": 1}
            search_result = await session.call_tool("search_wikipedia", search_args)
            
            # Extract the top result title
            search_data = search_result.content[0]
            article_title = json.loads(search_data.text)["results"][0]["title"]
            
            # Now get the full article content
            article_args = {"title": article_title}
            article_result = await session.call_tool("get_article", article_args)
            
            article_data = article_result.content[0]
            article_content = json.loads(article_data.text)["text"]
            article_tokens = self.tokenizer.encode(article_content)
            
            # if len(article_tokens) > self.max_article_tokens:
            #     truncated_tokens = article_tokens[:self.max_article_tokens]
            #     truncated_text = self.tokenizer.decode(truncated_tokens)
            # else:
            truncated_text = article_content
            
            combined_result = {
                "mode": "direct_article",
                "search_query": query,
                "article_title": article_title,
                "article_content": truncated_text,
                "token_count": min(len(article_tokens), self.max_article_tokens),
                "was_truncated": len(article_tokens) > self.max_article_tokens
            }
            
        else:
            # Mode 2: Search + Summary + Key Facts (default)
            # Step 1: Search Wikipedia
            search_args = {"query": query, "limit": limit}
            search_result = await session.call_tool("search_wikipedia", search_args)
            
            # Extract the top result title
            search_data = search_result.content[0]
            article_title = json.loads(search_data.text)["results"][0]["title"]
            
            # Step 2: Get summary
            summary_args = {"title": article_title}
            summary_result = await session.call_tool("get_summary", summary_args)
            
            summary_data = summary_result.content[0] 
            summary = json.loads(summary_data.text)["summary"]
            
            # Step 3: Extract key facts
            key_facts_args = {"title": article_title, "count": 5}
            key_facts_result = await session.call_tool("extract_key_facts", key_facts_args)
            
            key_facts_data = key_facts_result.content[0]
            # this is a list
            key_facts = json.loads(key_facts_data.text)["facts"]
            
            # Combine all results
            combined_result = {
                "mode": "search_summary",
                "search_result": search_result.content,
                "article_title": article_title,
                "summary": summary,
                "key_facts": key_facts
            }
        
        end_time = time.time()
        duration = end_time - start_time
        logger.debug(f"[{time.strftime('%H:%M:%S')}] Completed MCP query for: {query} (took {duration:.2f}s)")
        
        await exit_stack.aclose()
        
        return {
            "query": query,
            "success": True,
            "result": combined_result,
            "error": None,
            "duration": duration,
        }

    async def batch_mcp_query(self, queries: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Execute multiple MCP queries with true parallelism."""
        if not queries:
            return []
        
        logger.info(f"[{time.strftime('%H:%M:%S')}] Executing {len(queries)} MCP queries in parallel...")
        start_total = time.time()
        
        # Create tasks for parallel execution
        tasks = [self.create_mcp_session_and_query(query, limit) for query in queries]
        
        # Execute with timeout
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=self.mcp_timeout
        )
        
        end_total = time.time()
        total_duration = end_total - start_total
        logger.info(f"[{time.strftime('%H:%M:%S')}] All MCP queries completed in {total_duration:.2f}s")
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "query": queries[i],
                    "success": False,
                    "result": None,
                    "error": str(result),
                    "duration": 0,
                })
            else:
                processed_results.append(result)
        
        return processed_results

    def extract_mcp_queries(self, output_str: str) -> List[str]:
        """Extract MCP queries from model output using regex patterns."""
        patterns = [
            r'<search>([^<]+)</search>',
        ]
        
        queries = []
        for pattern in patterns:
            matches = re.findall(pattern, output_str, re.IGNORECASE)
            queries.extend([match.strip() for match in matches])
        
        return list(set(queries))  # Remove duplicates

    def format_mcp_response(self, query: str, mcp_result: Dict[str, Any]) -> str:
        """Format MCP response for inclusion in the generation."""
        if mcp_result["success"]:
            result_data = mcp_result["result"]
            mode = result_data.get("mode", "search_summary")
            
            if mode == "direct_article":
                # Format direct article response
                search_query = result_data.get('search_query', query)
                article_title = result_data.get('article_title', 'Unknown')
                article_content = result_data.get('article_content', 'No content available')
                token_count = result_data.get('token_count', 0)
                was_truncated = result_data.get('was_truncated', False)
                
                formatted_content = f"Search: {search_query}\nArticle: {article_title}\n\n"
                formatted_content += f"Content:\n{article_content}"
                
            else:
                # Format search+summary response
                formatted_content = f"Article: {result_data.get('article_title', 'Unknown')}\n\n"
                
                # Add summary
                summary = result_data.get('summary', 'No summary available')
                if isinstance(summary, dict):
                    summary = summary.get('summary', str(summary))
                formatted_content += f"Summary: {summary}\n\n"
                
                # Add key facts
                key_facts = result_data.get('key_facts', 'No key facts available')
                if isinstance(key_facts, dict):
                    facts_list = key_facts.get('facts', [])
                    if facts_list:
                        formatted_content += "Key Facts:\n"
                        for i, fact in enumerate(facts_list[:5], 1):
                            formatted_content += f"{i}. {fact}\n"
                    else:
                        formatted_content += f"Key Facts: {str(key_facts)}\n"
                elif isinstance(key_facts, list):
                    formatted_content += "Key Facts:\n"
                    for i, fact in enumerate(key_facts[:5], 1):
                        formatted_content += f"{i}. {fact}\n"
                else:
                    formatted_content += f"Key Facts: {str(key_facts)}\n"
            
            # Truncate if too long
            if len(formatted_content) > 3000:
                formatted_content = formatted_content[:3000] + "..."
            
            return f"<result>\n{formatted_content}</result>\n"
        else:
            return f"<error>{mcp_result['error']}</error>\n"

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences with MCP integration."""
        # Rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = ori_input_ids.size(0)

        # Parse input IDs
        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        # Set up sampling parameters
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1
            }
        elif is_validate:
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,
            }

        kwargs.update(
            {"stop": ["</search>"], "detokenize": True}
        )

        with self.update_sampling_params(**kwargs):
            # Prepare inputs for generation
            curr_inputs = []
            for input_ids in idx_list:
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(input_ids.copy())
            init_inputs = [ids.copy() for ids in curr_inputs]

            # Track generation state
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))
            result_mask_list = [[] for _ in range(len(curr_inputs))]

            # Multi-turn generation with MCP integration
            for step in range(self.config.get("max_turns", 3)):
                if len(active_indices) == 0:
                    break

                # Generate for active inputs
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]
                
                with self.update_sampling_params(
                    n=1,
                    max_tokens=min(512, max(active_max_tokens)),
                    top_p=0.95,
                    **kwargs
                ):
                    vllm_inputs = [{'prompt_token_ids': raw_prompt_ids} for raw_prompt_ids in active_inputs]
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )

                # Collect MCP queries from outputs
                mcp_queries = []
                query_indices = []
                
                new_active_indices = []
                
                for i, idx in enumerate(active_indices):
                    output_ids = outputs[i].outputs[0].token_ids
                    finish_reason = outputs[i].outputs[0].finish_reason
                    
                    # Add generated tokens
                    curr_inputs[idx] += output_ids
                    result_mask_list[idx] += [1] * len(output_ids)
                    
                    # Extract MCP queries from the output
                    output_str = self.tokenizer.decode(output_ids)
                    queries = self.extract_mcp_queries(output_str)
                    
                    if queries and finish_reason != 'length':
                        mcp_queries.extend(queries)
                        query_indices.extend([idx] * len(queries))
                        new_active_indices.append(idx)

                # Execute MCP queries in batch (only on rank 0)
                if mcp_queries:
                    if self.tp_rank == 0:
                        # Run MCP queries asynchronously
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        mcp_results = loop.run_until_complete(
                            self.batch_mcp_query(mcp_queries, limit=1)
                        )
                        loop.close()
                        
                        broadcast_data = {
                            'mcp_queries': mcp_queries,
                            'query_indices': query_indices,
                            'mcp_results': mcp_results
                        }
                    else:
                        broadcast_data = None
                    
                    # Broadcast results to all ranks
                    broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
                    
                    if broadcast_data is not None:
                        # Process MCP results
                        query_to_result = {}
                        for query, result in zip(broadcast_data['mcp_queries'], broadcast_data['mcp_results']):
                            if query not in query_to_result:  # Avoid duplicates
                                query_to_result[query] = result
                        
                        # Add MCP responses to generation
                        for idx in set(broadcast_data['query_indices']):
                            if idx in new_active_indices:
                                mcp_response_str = ""
                                for query, result in query_to_result.items():
                                    mcp_response_str += self.format_mcp_response(query, result)
                                
                                # Add MCP responses and continue generation prompt
                                mcp_response_str += self.gen_str
                                mcp_response_ids = self.tokenizer.encode(mcp_response_str)
                                
                                curr_inputs[idx] += mcp_response_ids
                                result_mask_list[idx] += [0] * len(mcp_response_ids)

                # Update active indices and check length constraints
                length_checked_active_indices = []
                for idx in active_indices:
                    if len(curr_inputs[idx]) - len(init_inputs[idx]) >= self.config.response_length:
                        # Truncate to response length
                        curr_inputs[idx] = init_inputs[idx] + \
                            curr_inputs[idx][len(init_inputs[idx]):len(init_inputs[idx])+self.config.response_length]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                    else:
                        curr_max_tokens[idx] = self.config.response_length - len(curr_inputs[idx]) + len(init_inputs[idx])
                        if idx in new_active_indices:
                            length_checked_active_indices.append(idx)
                
                active_indices = length_checked_active_indices

            # Collect final outputs
            output_ids_list = []
            for i, input_ids in enumerate(idx_list):
                for j in range(self.sampling_params.n):
                    idx = i * self.sampling_params.n + j
                    input_len = len(input_ids)
                    output_ids_list.append(curr_inputs[idx][input_len:])

        # Process outputs into tensors
        response_attention_mask_list = []
        response_list = []
        result_mask_list_padded = []
        
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            # Convert to tensors
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            result_mask = torch.tensor(result_mask, device=ori_input_ids.device)
            
            # Create attention mask
            response_attention_mask = torch.ones_like(response, dtype=torch.int64)
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
            response_attention_mask_list.append(response_attention_mask)
            
            # Pad response
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_list.append(response)
            
            # Pad result mask
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            result_mask_list_padded.append(result_mask)
        
        response_attention_mask = torch.stack(response_attention_mask_list, dim=0)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)

        # Handle multiple samples
        if self.config.n > 1 and do_sample:
            ori_input_ids = ori_input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n

        # Concatenate input and response
        seq = torch.cat([ori_input_ids, response], dim=-1)

        # Update position IDs
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # Update attention mask
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Create loss mask (0 for MCP results, 1 for model-generated content)
        loss_mask = result_mask * response_attention_mask

        # Create final batch
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }, batch_size=batch_size)

        # Free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)