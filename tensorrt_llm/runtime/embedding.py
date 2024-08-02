import copy
import math
from functools import reduce, wraps
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import tensorrt as trt

# isort: off
import torch
import tensorrt as trt
# isort: on
from cuda import cudart

from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime.generation import CUASSERT, _Runtime, ModelConfig, RuntimeTensor, _prepare_attention_mask

from .._ipc_utils import set_peer_access
from .._utils import (preview_trt_version, str_dtype_to_torch,
                      torch_to_numpy, trt_dtype_to_torch)
from ..logger import logger
from ..lora_manager import LoraManager
from ..mapping import Mapping
from ..quantization import QuantMode
from .kv_cache_manager import GenerationSequence, KVCacheManager
from .session import _scoped_stream


class EmbeddingSession(object):
    BEAM_WIDTH = 1

    _model_config: ModelConfig
    mapping: Mapping
    runtime: _Runtime
    device: torch.device
    batch_size: int
    buffer_allocated: bool
    debug_mode: bool
    quant_mode: QuantMode
    cuda_graph_mode: bool
    dtype: trt.DataType
    debug_tensors_to_save: None

    def __init__(self,
                 model_config: ModelConfig,
                 engine_buffer,
                 mapping: Mapping,
                 debug_mode=False,
                 debug_tensors_to_save=None,
                 cuda_graph_mode=False,
                 stream: torch.cuda.Stream = None):
        assert isinstance(model_config, ModelConfig)
        self._model_config = model_config
        self.mapping = mapping
        self.runtime = _Runtime(engine_buffer, mapping)
        self.device = torch.device(
            f'cuda:{self.runtime.runtime_rank % mapping.gpus_per_node}')
        torch.cuda.set_device(self.device)
        # dynamic_decoder currently use torch's current stream, so must let TRT enqueue use same stream here
        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)
        self.debug_mode = debug_mode
        self.debug_tensors_to_save = debug_tensors_to_save

        self.cuda_graph_mode = cuda_graph_mode

        self.buffer = None
        self.buffer_allocated = False

        if self.paged_kv_cache:
            logger.warning(
                "The paged KV cache in Python runtime is experimental. For performance and correctness, please, use C++ runtime."
            )

        if self.mapping.has_pp():
            self.nccl_comm = torch.classes.trtllm.NcclCommunicatorOp(
                self.mapping.tp_size, self.mapping.pp_size, self.mapping.rank)

        # if self.mapping.is_last_pp_rank():
        #     self.decoder_logits_dtype = self._tensor_dtype('logits')
        #     if self.decoder_logits_dtype not in [torch.float16, torch.float32]:
        #         logger.warning(
        #             "Logits dtype not supported by decoder. Falling back to float32. You may want to change the logits dtype to float16 in your model definition."
        #         )
        #         self.decoder_logits_dtype = torch.float32

        if model_config.use_context_fmha_for_generation:
            logger.warning(
                "Context FMHA is used for generation. Use it only for testing")

        expected_tensor_names = []
        if self.mapping.is_first_pp_rank():
            expected_tensor_names += ['input_ids']
        else:
            expected_tensor_names += ['hidden_states_input']

        if self.mapping.is_last_pp_rank():
            expected_tensor_names += ['embeddings_output']
            if not model_config.gather_context_logits:
                expected_tensor_names += ['last_token_ids']
        else:
            expected_tensor_names += ['hidden_states_output']

        if model_config.has_position_embedding and self.mapping.is_first_pp_rank(
        ):
            expected_tensor_names += ['position_ids']
        if model_config.has_token_type_embedding and self.mapping.is_first_pp_rank(
        ):
            expected_tensor_names += ['token_type_ids']

        expected_tensor_names += ['cache_indirection']

        if self.paged_kv_cache:
            expected_tensor_names += [f'kv_cache_block_pointers']
            expected_tensor_names += [f'host_kv_cache_block_pointers']
        else:
            expected_tensor_names += [
                f'past_key_value_{i}'
                for i in range(self.first_layer, self.last_layer)
            ]
            expected_tensor_names += [
                f'present_key_value_{i}'
                for i in range(self.first_layer, self.last_layer)
            ]

        if model_config.gpt_attention_plugin:
            expected_tensor_names += [
                'sequence_length', 'context_lengths', 'host_request_types',
                'host_past_key_value_lengths', 'host_sink_token_length'
            ]
            expected_tensor_names += [f'host_max_attention_window_sizes']
            if model_config.remove_input_padding:
                expected_tensor_names.append('host_context_lengths')
        else:
            expected_tensor_names += [
                'attention_mask',
            ]

        if model_config.max_prompt_embedding_table_size > 0:
            expected_tensor_names += [
                'prompt_embedding_table', 'tasks', 'prompt_vocab_size'
            ]

        if model_config.cross_attention:
            if model_config.gpt_attention_plugin:
                expected_tensor_names += [
                    f'cross_present_key_value_{i}'
                    for i in range(self.first_layer, self.last_layer)
                ]
                expected_tensor_names += [
                    f'cross_past_key_value_{i}'
                    for i in range(self.first_layer, self.last_layer)
                ]
            else:
                expected_tensor_names += [
                    'cross_attention_mask',
                ]

            expected_tensor_names += [
                'encoder_output',
                'encoder_input_lengths',
                'encoder_max_input_length',
                'cross_kv_cache_gen',
            ]
            self.skip_cross_qkv = model_config.skip_cross_qkv
            if self.skip_cross_qkv:
                expected_tensor_names += ['cross_qkv_reuse']

        if self.mapping.tp_size > 1 and model_config.use_custom_all_reduce:
            expected_tensor_names += ['all_reduce_workspace']

        self.lora_target_modules = model_config.lora_target_modules

        # In current design, q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time.
        # However, there are some cases that the lora modules only contain one or two of them, so we use zero tensor to fill the missing ones.
        self.missing_qkv_modules = []
        if self.lora_target_modules is not None:
            if any(x in self.lora_target_modules
                   for x in ["attn_q", "attn_k", "attn_v"]):
                for lora_module in ["attn_q", "attn_k", "attn_v"]:
                    if lora_module not in self.lora_target_modules:
                        self.missing_qkv_modules.append(lora_module)
            if any(x in self.lora_target_modules
                   for x in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]):
                for lora_module in [
                        "cross_attn_q", "cross_attn_k", "cross_attn_v"
                ]:
                    if lora_module not in self.lora_target_modules:
                        self.missing_qkv_modules.append(lora_module)

        if model_config.lora_plugin:

            for lora_module in (self.lora_target_modules +
                                self.missing_qkv_modules):
                expected_tensor_names += [
                    f'{lora_module}_lora_ranks_{i}'
                    for i in range(self.first_layer, self.last_layer)
                ]

                expected_tensor_names += [
                    f'{lora_module}_lora_weights_pointers_{i}'
                    for i in range(self.first_layer, self.last_layer)
                ]
            if self.cross_attention and self.remove_input_padding:
                expected_tensor_names += ['host_encoder_input_lengths']

        found_tensor_names = [
            self.runtime.engine.get_tensor_name(i)
            for i in range(self.runtime.engine.num_io_tensors)
        ]
        print(expected_tensor_names)
        print(found_tensor_names)
        if not self.debug_mode and set(expected_tensor_names) != set(
                found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError(
                "Tensor names in engine are not the same as expected, to use this GenerationSession, "
                "you need to use PretrainedModel.prepare_inputs to create TRT Network inputs."
            )
        if self.debug_mode:
            self.debug_tensors = list(
                set(found_tensor_names) - set(expected_tensor_names))

    @property
    def vocab_size(self):
        return self._model_config.vocab_size

    @property
    def num_layers(self):
        assert self._model_config.num_layers % self.mapping.pp_size == 0, \
            f"num_layers {self._model_config.num_layers} must be a multiple of pipeline parallelism size {self.mapping.pp_size}"
        return self._model_config.num_layers // self.mapping.pp_size

    @property
    def first_layer(self):
        return self.num_layers * self.mapping.pp_rank

    @property
    def last_layer(self):
        return self.first_layer + self.num_layers

    @property
    def num_heads(self):
        return self._model_config.num_heads

    @property
    def hidden_size(self):
        return self._model_config.hidden_size

    @property
    def use_gpt_attention_plugin(self):
        return self._model_config.gpt_attention_plugin

    @property
    def use_mamba_conv1d_plugin(self):
        return self._model_config.mamba_conv1d_plugin

    @property
    def paged_kv_cache(self):
        return self._model_config.paged_kv_cache

    @property
    def tokens_per_block(self):
        return self._model_config.tokens_per_block

    @property
    def remove_input_padding(self):
        return self._model_config.remove_input_padding

    @property
    def num_heads_kv(self):
        return self._model_config.num_kv_heads

    @property
    def head_size(self):
        return self.hidden_size // self.num_heads if self._model_config.head_size is None else self._model_config.head_size

    @property
    def max_prompt_embedding_table_size(self):
        return self._model_config.max_prompt_embedding_table_size

    @property
    def quant_mode(self):
        return self._model_config.quant_mode

    @property
    def gather_context_logits(self):
        return self._model_config.gather_context_logits

    @property
    def dtype(self):
        return str_dtype_to_torch(self._model_config.dtype)

    @property
    def use_custom_all_reduce(self):
        return self._model_config.use_custom_all_reduce

    # def cuda_stream_guard(func):
    #     """Sync external stream and set current stream to the one bound to the session. Reset on exit.
    #     """

    #     @wraps(func)
    #     def wrapper(self, *args, **kwargs):
    #         external_stream = torch.cuda.current_stream()
    #         if external_stream != self.stream:
    #             external_stream.synchronize()
    #             torch.cuda.set_stream(self.stream)
    #         ret = func(self, *args, **kwargs)
    #         if external_stream != self.stream:
    #             self.stream.synchronize()
    #             torch.cuda.set_stream(external_stream)
    #         return ret

    #     return wrapper

    @property
    def cross_attention(self):
        return self._model_config.cross_attention

    @property
    def has_position_embedding(self):
        return self._model_config.has_position_embedding

    @property
    def has_token_type_embedding(self):
        return self._model_config.has_token_type_embedding

    @property
    def use_lora_plugin(self):
        return self._model_config.lora_plugin

    @property
    def use_context_fmha_for_generation(self):
        return self._model_config.use_context_fmha_for_generation

    @property
    def paged_state(self):
        return self._model_config.paged_state

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.runtime.engine.get_tensor_dtype(name))
        return dtype

    def setup(self,
              batch_size: int,
              max_context_length: int,
              max_attention_window_size: Optional[int] = None,
              sink_token_length: Optional[int] = None,
              encoder_max_input_length: Optional[int] = None,
              lora_manager: LoraManager = None,
              lora_uids: List[str] = None):
        # Store these params related to buffer size to check against
        # the input shape with the params given in decode()
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.max_seq_length = max_context_length
        self.encoder_max_input_length = encoder_max_input_length
        if max_attention_window_size is None:
            self.max_attention_window_size = self.max_seq_length
            logger.debug(
                "The max_attention_window_size is not set, we will use max_seq_length by default."
            )
            self.host_max_attention_window_sizes = torch.ones(
                (self.num_layers, ),
                dtype=torch.int32) * self.max_attention_window_size

        elif isinstance(max_attention_window_size, int):
            if max_attention_window_size > self.max_seq_length:
                logger.warning(
                    "The value of max_attention_window_size should ideally not exceed max_seq_length. "
                    "Therefore, it has been adjusted to match the value of max_seq_length."
                )
            self.max_attention_window_size = min(max_attention_window_size,
                                                 self.max_seq_length)
            self.host_max_attention_window_sizes = torch.ones(
                (self.num_layers, ),
                dtype=torch.int32) * self.max_attention_window_size

        elif isinstance(max_attention_window_size, torch.Tensor):
            self.max_attention_window_size = int(
                torch.max(max_attention_window_size).item())
            if self.max_attention_window_size > self.max_seq_length:
                logger.warning(
                    "The value of max_attention_window_size should ideally not exceed max_seq_length. "
                    "Therefore, it has been adjusted to match the value of max_seq_length."
                )
            self.max_attention_window_size = min(self.max_attention_window_size,
                                                 self.max_seq_length)
            if max_attention_window_size.shape[0] != self.num_layers:
                logger.error(
                    "max_attention_window_size tensor's size is not equal to num_layers! "
                    "Note that num_layers = num_total_layers // pipeline_parallelism_size."
                )
                assert False
            self.host_max_attention_window_sizes = torch.minimum(
                max_attention_window_size.to(torch.int32),
                torch.IntTensor([self.max_seq_length] * self.num_layers))
        else:
            assert False, "invalid max_attention_window_size!"

        if sink_token_length is None:
            self.sink_token_length = 0
            self.host_sink_token_length = torch.zeros((1, ), dtype=torch.int32)
        elif isinstance(sink_token_length, int):
            self.sink_token_length = sink_token_length
            self.host_sink_token_length = torch.ones(
                (1, ), dtype=torch.int32) * self.sink_token_length
        else:
            assert False, "invalid sink_token_length!"

        self.lora_manager = lora_manager

        self.buffer = {}
        if self.mapping.is_last_pp_rank():
            self.buffer['embeddings_output'] = torch.empty(
                (batch_size, self.hidden_size)  # TODO(ilyaonoff)
                if not self.gather_context_logits else
                (batch_size, max_context_length, self.hidden_size),  # TODO(ilyaonoff)
                dtype=self._tensor_dtype('embeddings_output'),
                device=self.device)

        if self.cross_attention:
            # use shape info to pass max length info in remove padding mode
            self.buffer['encoder_max_input_length'] = torch.empty(
                (encoder_max_input_length, ),
                dtype=self._tensor_dtype('encoder_max_input_length'),
                device=self.device)

        if self.paged_kv_cache:
            bubble_len = 0
            if self.sink_token_length % self.tokens_per_block > 0:
                bubble_len += (self.tokens_per_block -
                               self.sink_token_length % self.tokens_per_block)
            blocks = batch_size * math.ceil(
                (self.max_attention_window_size + bubble_len) /
                self.tokens_per_block)
            cache_shape = (
                blocks,
                2,
                self.num_heads_kv,
                self.tokens_per_block,
                self.head_size,
            )
        else:
            cache_shape = (
                batch_size,
                2,
                self.num_heads_kv,
                self.max_attention_window_size,
                self.head_size,
            )
            if self.cross_attention:
                cross_cache_shape = (
                    batch_size,
                    2,
                    self.num_heads_kv,
                    self.encoder_max_input_length,
                    self.head_size,
                )

        for i in range(self.first_layer, self.last_layer):
            if self.quant_mode.has_kv_cache_quant():
                # Since torch does not support fp8 now, using int8 here.
                kv_cache_type = torch.int8
            else:
                kv_cache_type = self.dtype if self.paged_kv_cache else self._tensor_dtype(
                    f'present_key_value_{i}')
            self.buffer[f'present_key_value_{i}'] = torch.empty(
                cache_shape, dtype=kv_cache_type, device=self.device)
            if self.cross_attention:
                self.buffer[f'cross_present_key_value_{i}'] = torch.empty(
                    cross_cache_shape, dtype=kv_cache_type, device=self.device)

        if self.use_gpt_attention_plugin:
            self.sequence_length_buffer = torch.ones((batch_size, ),
                                                     dtype=torch.int32,
                                                     device=self.device)
        else:
            # without plugin, we need two set of kv cache buffers,
            # one for inputs, and the other for outputs.
            # They will take turns to act as input and output buffers.
            # Not applicable to cross KV buffers as it's constant
            for i in range(self.first_layer, self.last_layer):
                trt_dtype = self.runtime.engine.get_tensor_dtype(
                    f'present_key_value_{i}')
                if trt_dtype == trt.fp8:
                    # PyTorch doesn't support fp8 datatype, use int8 instead of it because int8 datatype size is same with fp8.
                    # TODO: Remove this section when PyTorch support fp8 datatype
                    dtype = torch.int8
                else:
                    dtype = self._tensor_dtype(f'present_key_value_{i}')
                self.buffer[f'1_present_key_value_{i}'] = torch.empty(
                    cache_shape, dtype=dtype, device=self.device)

        if self.use_custom_all_reduce and self.mapping.tp_size > 1:
            set_peer_access(self.mapping)
            self.ipc_buffers, self.all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
                self.mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.mapping.tp_size))

        if self.use_lora_plugin and self.lora_manager is not None:
            assert lora_uids is not None

            for idx in range(self.num_layers):
                layer_idx = idx + self.first_layer

                for lora_module in (self.lora_target_modules +
                                    self.missing_qkv_modules):
                    lora_ranks_ = []
                    lora_ptrs_ = []
                    for batch_idx in range(batch_size):
                        lora_uid = lora_uids[batch_idx]
                        if lora_uid is not None and lora_uid != "-1" and self.lora_manager.uid_to_low_ranks(
                                lora_uid)[layer_idx][lora_module] != 0:
                            lora_ranks_.append(
                                self.lora_manager.uid_to_low_ranks(lora_uid)
                                [layer_idx][lora_module])
                            lora_ptrs_.append(
                                self.lora_manager.lora_weights_pointers_list[
                                    layer_idx][lora_uid][lora_module])
                        else:
                            lora_ranks_.append(0)
                            lora_ptrs_.append([0, 0])

                    self.buffer.update({
                        f'{lora_module}_lora_ranks_{layer_idx}':
                        torch.IntTensor(lora_ranks_)
                    })
                    self.buffer.update({
                        f'{lora_module}_lora_weights_pointers_{layer_idx}':
                        torch.LongTensor(lora_ptrs_)
                    })

        self.buffer_allocated = True

    def _get_context_shape_buffer(
            self,
            input_ids: torch.Tensor,
            context_lengths: torch.Tensor,
            host_context_lengths: torch.Tensor,
            position_ids: torch.Tensor,
            last_token_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            cross_attention_mask: torch.Tensor,
            cache_indirection: torch.Tensor,
            kv_cache_block_pointers: List[torch.Tensor],
            host_kv_cache_block_pointers: List[torch.Tensor],
            hidden_states_input: torch.Tensor = None,
            prompt_embedding_table: torch.Tensor = None,
            tasks: torch.Tensor = None,
            prompt_vocab_size: torch.Tensor = None,
            encoder_output: torch.Tensor = None,
            encoder_input_lengths: torch.Tensor = None) -> List[RuntimeTensor]:
        tensors = {}

        def sym(x, name):
            return RuntimeTensor.from_torch(name, x)

        def add_tensor(x, name):
            return tensors.update({name: sym(x, name)})

        def add_tensor_with_shape(x, name, shape):
            return tensors.update(
                {name: RuntimeTensor.from_torch(name, x, override_shape=shape)})

        if self.use_gpt_attention_plugin:
            add_tensor(context_lengths, 'context_lengths')
        add_tensor(cache_indirection, 'cache_indirection')

        if self.has_position_embedding:
            add_tensor(position_ids, 'position_ids')

        if self.cross_attention:
            # in context phase, need to generate cross kv cache, set to True
            add_tensor(torch.ones(1, dtype=torch.bool, device=self.device),
                       'cross_kv_cache_gen')
            if self.skip_cross_qkv:
                if self.cross_qkv_reuse is None:
                    # see Attention's self.qkv output dim
                    cross_qkv_out_dim = self.num_heads * self.head_size + (
                        2 * self.num_heads_kv * self.head_size)
                    cross_qkv_shape = encoder_output.shape[:-1] + (
                        cross_qkv_out_dim, )
                    cross_qkv_reuse = torch.empty(cross_qkv_shape,
                                                  dtype=encoder_output.dtype,
                                                  device=encoder_output.device)
                    self.cross_qkv_reuse = cross_qkv_reuse
                add_tensor(self.cross_qkv_reuse, 'cross_qkv_reuse')
            add_tensor(encoder_output, 'encoder_output')
            add_tensor(encoder_input_lengths, 'encoder_input_lengths')
            add_tensor(self.buffer['encoder_max_input_length'],
                       'encoder_max_input_length')
            if not self.use_gpt_attention_plugin:
                add_tensor(cross_attention_mask, 'cross_attention_mask')

        if self.mapping.has_pp():
            hidden_size = self.hidden_size * self.mapping.tp_size
            if input_ids.dim() == 2:
                hidden_states_input = hidden_states_input.resize_(
                    input_ids.shape[0], input_ids.shape[1], hidden_size)
            else:
                hidden_states_input = hidden_states_input.resize_(
                    input_ids.shape[0], hidden_size)

        if self.mapping.is_last_pp_rank():
            add_tensor(self.buffer['embeddings_output'], 'embeddings_output')

            if not self.gather_context_logits:
                add_tensor(last_token_ids, 'last_token_ids')
        else:
            add_tensor(hidden_states_input, 'hidden_states_output')

        if self.mapping.is_first_pp_rank():
            add_tensor(input_ids, 'input_ids')
        else:
            add_tensor(hidden_states_input, 'hidden_states_input')

        if prompt_embedding_table is not None:
            add_tensor(prompt_embedding_table, 'prompt_embedding_table')

            if self.remove_input_padding:
                tasks_generation = torch.concat([
                    torch.full([context_lengths[b].item()],
                               tasks[b].item(),
                               dtype=torch.int32)
                    for b in range(context_lengths.size(0))
                ]).cuda()
            else:
                tasks_generation = tasks.unsqueeze(-1)
            add_tensor(tasks_generation, 'tasks')
            add_tensor(prompt_vocab_size, 'prompt_vocab_size')

        if self.paged_kv_cache:
            buffer = kv_cache_block_pointers.contiguous()
            shape = kv_cache_block_pointers.shape
            shape = [shape[0], shape[1] * shape[2], *shape[3:]]
            add_tensor_with_shape(buffer, f'kv_cache_block_pointers', shape)
            add_tensor_with_shape(host_kv_cache_block_pointers,
                                  f'host_kv_cache_block_pointers', shape)

        batch_size = context_lengths.shape[0]
        if not self.paged_kv_cache:
            for idx in range(self.first_layer, self.last_layer):
                if not self.use_gpt_attention_plugin:
                    kv_cache_shape = (batch_size, 2, self.num_heads_kv, 0,
                                      self.head_size)
                    # for empty tensor, TRT does not really use the tensor data, so any dtype is fine
                    kv_cache_buffer = torch.zeros((1, ),
                                                  dtype=torch.float32,
                                                  device=self.device)
                    add_tensor_with_shape(kv_cache_buffer,
                                          f'past_key_value_{idx}',
                                          kv_cache_shape)
                    present = f'present_key_value_{idx}'
                    add_tensor(self.buffer[present], present)

                    if self.cross_attention:
                        cross_kv_cache_shape = (batch_size, 2,
                                                self.num_heads_kv, 0,
                                                self.head_size)
                        # for empty tensor, TRT does not really use the tensor data, so any dtype is fine
                        cross_kv_cache_buffer = torch.zeros((1, ),
                                                            dtype=torch.float32,
                                                            device=self.device)
                        add_tensor_with_shape(cross_kv_cache_buffer,
                                              f'cross_past_key_value_{idx}',
                                              cross_kv_cache_shape)
                        cross_present = f'cross_present_key_value_{idx}'
                        add_tensor(self.buffer[cross_present], cross_present)
                else:
                    key_value_cache = self.buffer[f'present_key_value_{idx}']
                    # when plugin is used, past_ket_value tensor does not need to be empty tensor
                    # because plugin does not care, and does not use this shape.
                    add_tensor(key_value_cache, f'past_key_value_{idx}')
                    add_tensor(key_value_cache, f'present_key_value_{idx}')

                    if self.cross_attention:
                        cross_cache_buffer = self.buffer[
                            f'cross_present_key_value_{idx}']
                        add_tensor(cross_cache_buffer,
                                   f'cross_past_key_value_{idx}')
                        add_tensor(cross_cache_buffer,
                                   f'cross_present_key_value_{idx}')

        if self.use_gpt_attention_plugin:
            # context request
            host_request_types = torch.zeros_like(context_lengths,
                                                  device='cpu').int()
            self.sequence_length_buffer = context_lengths.detach().clone()
            add_tensor_with_shape(self.sequence_length_buffer,
                                  'sequence_length', (batch_size, ))

            # field 0: past_key_value_length, field 1: is_context (deprecated). changed to [0], otherwise affects batch padded input mode
            add_tensor_with_shape(host_context_lengths,
                                  'host_past_key_value_lengths', (batch_size, ))
            add_tensor_with_shape(self.host_sink_token_length,
                                  'host_sink_token_length', (1, ))
            add_tensor(host_request_types, 'host_request_types')
            add_tensor_with_shape(self.host_max_attention_window_sizes,
                                  f'host_max_attention_window_sizes',
                                  (self.num_layers, ))
            if self.remove_input_padding:
                add_tensor(host_context_lengths, 'host_context_lengths')
        else:
            add_tensor(attention_mask, 'attention_mask')

        if self.use_custom_all_reduce and self.mapping.tp_size > 1:
            add_tensor(self.all_reduce_workspace, 'all_reduce_workspace')

        if self.use_lora_plugin:
            for idx in range(self.num_layers):
                for lora_module in (self.lora_target_modules +
                                    self.missing_qkv_modules):
                    layer_idx = idx + self.first_layer
                    lora_ranks = f'{lora_module}_lora_ranks_{layer_idx}'
                    add_tensor(self.buffer[lora_ranks], lora_ranks)
                    lora_weights = f'{lora_module}_lora_weights_pointers_{layer_idx}'
                    add_tensor(self.buffer[lora_weights], lora_weights)
            if self.cross_attention and self.remove_input_padding:
                add_tensor(encoder_input_lengths.to('cpu'),
                           'host_encoder_input_lengths')

        return tensors

    def _prepare_context_inputs(self, batch_size, context_lengths,
                                host_context_lengths, use_gpt_attention_plugin,
                                remove_input_padding, **kwargs):

        last_token_ids = context_lengths.detach().clone()
        if use_gpt_attention_plugin:
            max_context_length = kwargs.pop('max_context_length')
            if remove_input_padding:
                position_ids = torch.concat([
                    torch.arange(0,
                                 host_context_lengths[i],
                                 dtype=torch.int32,
                                 device='cuda') for i in range(batch_size)
                ])
                last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
            else:
                position_ids = torch.tensor(range(max_context_length),
                                            dtype=torch.int32,
                                            device='cuda').reshape(
                                                [1,
                                                 -1]).expand([batch_size, -1])
            ret = {'last_token_ids': last_token_ids}
        else:
            input_ids = kwargs.pop('input_ids')
            pad_id = kwargs.pop('pad_id', None)
            attention_mask = _prepare_attention_mask(input_ids, pad_id)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.int()

            ret = {
                'attention_mask': attention_mask,
                'last_token_ids': last_token_ids
            }

        if self.has_position_embedding:
            ret['position_ids'] = position_ids

        return ret

    def pp_communicate(self, sequence_length):
        if self.mapping.is_last_pp_rank():
            for pg in self.mapping.pp_group:
                if pg == self.mapping.rank:
                    continue
                self.nccl_comm.send(sequence_length, pg)
        else:
            self.nccl_comm.recv(sequence_length, self.mapping.pp_group[-1])

    def handle_per_step(
            self, cache_indirection: torch.Tensor, batch_size: int,
            max_context_length: int, input_ids: torch.Tensor,
            hidden_states: torch.Tensor, pad_id: int,
            kv_cache_block_pointers: list, host_kv_cache_block_pointers: list,
            prompt_embedding_table: torch.Tensor, tasks: torch.Tensor,
            context_lengths: torch.Tensor, host_context_lengths,
            attention_mask: torch.Tensor, cross_attention_mask: torch.Tensor,
            prompt_vocab_size: torch.Tensor,
            next_step_tensors: Dict[str, RuntimeTensor], encoder_output: torch.Tensor,
            encoder_input_lengths: torch.Tensor):
        model_inputs = self._prepare_context_inputs(
            batch_size=batch_size,
            context_lengths=context_lengths,
            host_context_lengths=host_context_lengths,
            use_gpt_attention_plugin=self.use_gpt_attention_plugin,
            remove_input_padding=self.remove_input_padding,
            max_context_length=max_context_length,
            input_ids=input_ids,
            pad_id=pad_id)

        position_ids = model_inputs.get('position_ids', None)
        last_token_ids = model_inputs.get('last_token_ids')
        attention_mask = model_inputs.get('attention_mask', None)

        if self.paged_kv_cache:
            host_kv_cache_block_pointers = self.kv_cache_manager.get_block_pointers(
                1)
            kv_cache_block_pointers = host_kv_cache_block_pointers.to(
                'cuda')

        ctx_tensors = self._get_context_shape_buffer(
            input_ids, context_lengths, host_context_lengths, position_ids,
            last_token_ids, attention_mask, cross_attention_mask,
            cache_indirection, kv_cache_block_pointers,
            host_kv_cache_block_pointers, hidden_states,
            prompt_embedding_table, tasks, prompt_vocab_size,
            encoder_output, encoder_input_lengths)
        context = self.runtime.ctx_context
        self.runtime._set_tensors(context, ctx_tensors)
        if self.debug_mode:
            self.debug_buffer = {
                name: tensor.to_torch()
                for name, tensor in ctx_tensors.items()
            }
        if self.cuda_graph_mode:
            # context mode, clean cuda graph instances
            self.runtime.cuda_graph_instances = [None for _ in range(2)]

        if self.debug_mode:
            self.runtime._check_tensors(context)
        # dynamic_decoder currently use torch's current stream, so must let TRT enqueue use same stream here
        stream = torch.cuda.current_stream().cuda_stream
        instance_idx = 0  # step % 2
        if self.cuda_graph_mode and self.runtime.cuda_graph_instances[
                instance_idx] is not None:
            # launch cuda graph
            CUASSERT(
                cudart.cudaGraphLaunch(
                    self.runtime.cuda_graph_instances[instance_idx], stream))
            ok = True
        else:
            ok = self.runtime._run(context, stream)

        if not ok:
            raise RuntimeError(f"Executing TRT engine failed!")
        if self.debug_mode:
            torch.cuda.synchronize()

        # TODO(ilyaonoff) do we really need this?
        # Initialize sequence_lengths (no paddings) for the generation phase.
        self.sequence_length_buffer = context_lengths.detach().clone()

        # TODO(ilyaonoff) what is it?
        if self.mapping.has_pp():
            self.pp_communicate(self.sequence_length_buffer)

        if self.paged_kv_cache:
            self.kv_cache_manager.step([True] * batch_size)

        if self.debug_mode:
            self.dump_debug_buffers()

            if next_step_tensors is not None:
                self.debug_buffer = {
                    name: tensor.to_torch()
                    for name, tensor in next_step_tensors.items()
                }

        return next_step_tensors, tasks, context_lengths, host_context_lengths, attention_mask, encoder_input_lengths

    def dump_debug_buffers(self) -> None:
        if self.debug_tensors_to_save is not None:
            # restricted written tensors according to filter
            debug_tensor_names = copy.deepcopy(list(self.debug_buffer.keys()))
            for k in debug_tensor_names:
                if all([kk not in k for kk in self.debug_tensors_to_save]):
                    self.debug_buffer.pop(k)

        debug_dir = Path(
            f"tllm_debug/PP_{self.mapping.pp_rank}/TP_{self.mapping.tp_rank}")
        debug_dir.mkdir(parents=True, exist_ok=True)

        for name, t in self.debug_buffer.items():
            # convert tensor name to valid file name
            fname = name.replace("/", ".")
            t = torch_to_numpy(t)
            np.save(debug_dir / f"{fname}.npy", t)

            txt_format = "%d" if t.dtype in [np.int32, np.int8] else '%.18e'
            np.savetxt(
                debug_dir / f"{fname}.txt",
                t.reshape(-1, t.shape[-1]),  # savetxt accepts 2 dims only
                fmt=txt_format)

    def decode_regular(self,
                       batch_size: int,
                       pad_id: int,
                       context_lengths: torch.Tensor,
                       host_context_lengths,
                       max_context_length: int,
                       cache_indirection: torch.Tensor,
                       input_ids: torch.Tensor,
                       hidden_states: torch.Tensor,
                       prompt_embedding_table: torch.Tensor,
                       tasks: torch.Tensor,
                       prompt_vocab_size: torch.Tensor,
                       output_sequence_lengths: bool = False,
                       return_dict: bool = False,
                       encoder_output: torch.Tensor = None,
                       encoder_input_lengths: torch.Tensor = None,
                       cross_attention_mask: torch.Tensor = None,
                       **kwargs):
        kv_cache_block_pointers = []
        host_kv_cache_block_pointers = []
        attention_mask = None

        benchmark_profiler = kwargs.get('benchmark_profiler', None)
        generation_phase_step_count = 0

        def profile_fn(benchmark_profiler_obj, step_count):
            if benchmark_profiler_obj is not None:
                benchmark_profiler_obj.record_cuda_event('last_token')
                benchmark_profiler_obj.record_elapsed_time(
                    'first_token', 'last_token', 'generation_time')
                benchmark_profiler_obj.add_aux_info('generation_step_count',
                                                    step_count)

        next_step_tensors = None
        next_step_tensors, tasks, context_lengths, host_context_lengths, attention_mask, encoder_input_lengths = self.handle_per_step(
            cache_indirection, batch_size, max_context_length, input_ids, hidden_states, pad_id,
            kv_cache_block_pointers, host_kv_cache_block_pointers,
            prompt_embedding_table, tasks, context_lengths,
            host_context_lengths, attention_mask, cross_attention_mask,
            prompt_vocab_size, next_step_tensors, encoder_output,
            encoder_input_lengths)
        if benchmark_profiler is not None:
            benchmark_profiler.record_cuda_event('first_token')

        profile_fn(benchmark_profiler, generation_phase_step_count)

        return {
            'embeddings_output': self.buffer['embeddings_output']
        }

    # def decode_batch(self,
    #                  input_ids: Sequence[torch.Tensor],
    #                  sampling_config: SamplingConfig,
    #                  streaming: bool = False,
    #                  **kwargs):
    #     input_ids, context_lengths = _prepare_input_ids(input_ids)
    #     return self.decode(input_ids,
    #                        context_lengths,
    #                        sampling_config,
    #                        streaming=streaming,
    #                        **kwargs)

    # As dynamic_decoder uses torch's current stream, we must ensure it runs on the same stream that
    # dynamic_decoder was set up with
    # @cuda_stream_guard
    def decode(self,
               input_ids: torch.Tensor,
               context_lengths: torch.Tensor,
               pad_id: int,
               prompt_embedding_table: torch.Tensor = None,
               tasks: torch.Tensor = None,
               prompt_vocab_size: torch.Tensor = None,
               output_sequence_lengths: bool = False,
               return_dict: bool = False,
               encoder_output: torch.Tensor = None,
               encoder_input_lengths: torch.Tensor = None,
               cross_attention_mask: torch.Tensor = None,
               **kwargs):
        batch_size = context_lengths.size(0)
        max_context_length = torch.max(context_lengths).item()
        host_context_lengths = context_lengths.cpu()
        assert batch_size == self.batch_size, \
            "Given batch size is different from the one used in setup()," \
            "rerun the setup function with the new batch size to avoid buffer overflow."
        assert max_context_length <= self.max_context_length, \
            "Given input length is large then the one used in setup()," \
            "rerun the setup function with the new max_context_length to avoid buffer overflow."
        assert self.sink_token_length <= torch.min(context_lengths).item(), \
            "Given sink token length is larger than shortest context length," \
            "rerun the setup function with a smaller sink token length."

        if self.remove_input_padding and input_ids.dim() == 2:
            assert input_ids.shape[
                0] == 1, "Packed 2D input must have shape [1, <sum of input lengths>]"
            input_ids = input_ids.squeeze(0)

        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        cache_indirection = torch.full(
            (
                batch_size,
                self.BEAM_WIDTH,
                self.max_attention_window_size,
            ),
            0,
            dtype=torch.int32,
            device=self.device
        )

        hidden_states = None
        if self.mapping.has_pp():
            max_num_tokens = max(batch_size, batch_size * self.max_seq_length)
            hidden_size = self.hidden_size * self.mapping.tp_size
            hidden_states = torch.zeros((1, max_num_tokens, hidden_size))

        # Init KV cache block manager
        if self.paged_kv_cache:
            bubble_len = 0
            if self.sink_token_length % self.tokens_per_block > 0:
                bubble_len += (self.tokens_per_block -
                               self.sink_token_length % self.tokens_per_block)
            max_blocks_per_seq = math.ceil(
                (self.max_attention_window_size + bubble_len) /
                self.tokens_per_block)
            blocks = batch_size * max_blocks_per_seq
            memory_pools = [
                self.buffer[f'present_key_value_{i}']
                for i in range(self.first_layer, self.last_layer)
            ]
            self.kv_cache_manager = KVCacheManager(
                memory_pools, blocks, self.tokens_per_block, max_blocks_per_seq,
                self.max_attention_window_size, self.sink_token_length,
                self.BEAM_WIDTH, False)

            # Add sequences to the manager
            for bi in range(batch_size):
                generation_sequence = GenerationSequence(seq_idx=bi,
                                                         batch_idx=bi)
                self.kv_cache_manager.add_sequence(generation_sequence,
                                                   max_context_length)

        return self.decode_regular(
            batch_size, pad_id, context_lengths,
            host_context_lengths, max_context_length,
            cache_indirection, input_ids, hidden_states,
            prompt_embedding_table, tasks, prompt_vocab_size, output_sequence_lengths, return_dict,
            encoder_output, encoder_input_lengths, cross_attention_mask, **kwargs)
