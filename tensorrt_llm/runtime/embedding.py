
import copy
import math
import platform
from dataclasses import dataclass, field
from functools import reduce, wraps
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

import numpy as np
import tensorrt as trt

# isort: off
import torch
import tensorrt as trt
# isort: on
from cuda import cudart

from .._ipc_utils import set_peer_access
from .._utils import (pad_vocab_size, str_dtype_to_torch, torch_to_numpy,
                      trt_dtype_to_torch, trt_gte_10)
from ..logger import logger
from ..lora_manager import LoraManager
from ..mapping import Mapping
from ..plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime.generation import CUASSERT, _Runtime, LogitsProcessor, ModelConfig, RuntimeTensor, SamplingConfig, StoppingCriteria, _prepare_attention_mask, _tile_beam_width, _update_cuda_graph_instance

from ..quantization import QuantMode
from .kv_cache_manager import GenerationSequence, KVCacheManager, KVCacheUpdater
from .session import _scoped_stream


class EmbeddingSession(object):

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
        # TODO: in tensorrt_llm/cpp/tensorrt_llm/thop/dynamicDecodeOp.cpp it's T, can be float or half?
        self.embedding_bias_opt = None
        # use one more block in paged kv cache.
        self.use_one_more_block = False

        self.buffer = None
        self.buffer_allocated = False

        self.vocab_size_padded = pad_vocab_size(self.vocab_size,
                                                self.mapping.tp_size)
        if len(model_config.layer_types) == 0:
            self.layer_types = ['attention'] * model_config.num_layers
        else:
            layer_types = model_config.layer_types
            layer_types = layer_types * (model_config.num_layers //
                                         len(layer_types))
            layer_types = layer_types + layer_types[0:(model_config.num_layers %
                                                       len(layer_types))]
            self.layer_types = layer_types
        self.num_attn_layers = \
            self.layer_types[self.first_layer:self.last_layer].count('attention')
        self.has_attn_layers = self.num_attn_layers > 0
        self.has_rnn_layers = 'recurrent' in self.layer_types[
            self.first_layer:self.last_layer]
        self.attn_to_general_idx = {}
        attn_layer_idx = 0
        for i in range(self.first_layer, self.last_layer):
            if self.layer_types[i] == 'attention':
                self.attn_to_general_idx[attn_layer_idx] = i
                attn_layer_idx += 1

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

        if self.use_custom_all_reduce and self.mapping.tp_size > 1:
            set_peer_access(self.mapping)
            self.ipc_buffers, self.all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
                self.mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.mapping.tp_size))

        expected_tensor_names = []
        if self.mapping.is_first_pp_rank():
            expected_tensor_names += ['input_ids']
        else:
            expected_tensor_names += ['hidden_states_input']

        if self.mapping.is_last_pp_rank():
            expected_tensor_names += ['embeddings_output']
            if not model_config.gather_context_logits or self.has_rnn_layers:
                expected_tensor_names += ['last_token_ids']
        else:
            expected_tensor_names += ['hidden_states_output']

        if self.has_attn_layers:
            if model_config.has_position_embedding and self.mapping.is_first_pp_rank(
            ):
                expected_tensor_names += ['position_ids']
            if model_config.has_token_type_embedding and self.mapping.is_first_pp_rank(
            ):
                expected_tensor_names += ['token_type_ids']

            expected_tensor_names += ['cache_indirection']

        if self.paged_kv_cache and self.has_attn_layers:
            expected_tensor_names += [f'kv_cache_block_offsets']
            expected_tensor_names += [f'host_kv_cache_block_offsets']
            expected_tensor_names += [f'host_kv_cache_pool_pointers']
            if self.cross_attention:
                expected_tensor_names += [f'cross_kv_cache_block_offsets']
                expected_tensor_names += [f'host_cross_kv_cache_block_offsets']
                expected_tensor_names += [f'host_cross_kv_cache_pool_pointers']
        else:
            for i in range(self.first_layer, self.last_layer):
                if self.layer_types[i] == 'attention':
                    expected_tensor_names += [
                        f'past_key_value_{i}', f'present_key_value_{i}'
                    ]
            if model_config.cross_attention:
                if model_config.gpt_attention_plugin:
                    for i in range(self.first_layer, self.last_layer):
                        if self.layer_types[i] == 'attention':
                            expected_tensor_names += [
                                f'cross_present_key_value_{i}',
                                f'cross_past_key_value_{i}'
                            ]
                else:
                    expected_tensor_names += [
                        'cross_attention_mask',
                    ]

        if self.paged_state and self.has_rnn_layers:
            for i in range(self.first_layer, self.last_layer):
                if self.layer_types[i] == 'recurrent':
                    expected_tensor_names += [
                        f'conv_state_ptr_{i}', f'rnn_state_ptr_{i}'
                    ]
            expected_tensor_names += ['slot_mapping']
        else:
            for i in range(self.first_layer, self.last_layer):
                if self.layer_types[i] == 'recurrent':
                    expected_tensor_names += [
                        f'past_conv_state_{i}', f'present_conv_state_{i}',
                        f'past_rnn_state_{i}', f'present_rnn_state_{i}'
                    ]

        if model_config.gpt_attention_plugin and self.has_attn_layers:
            expected_tensor_names += [
                'sequence_length', 'context_lengths', 'host_request_types',
                'host_past_key_value_lengths', 'host_sink_token_length'
            ]
            expected_tensor_names += [f'host_max_attention_window_sizes']
            if model_config.remove_input_padding:
                expected_tensor_names.append('host_context_lengths')
        else:
            if self.has_rnn_layers:
                expected_tensor_names += ['host_request_types']
                if model_config.mamba_conv1d_plugin and model_config.remove_input_padding:
                    expected_tensor_names.append('host_context_lengths')
            if self.has_attn_layers:
                expected_tensor_names += ['attention_mask']

        if model_config.max_prompt_embedding_table_size > 0:
            expected_tensor_names += [
                'prompt_embedding_table', 'tasks', 'prompt_vocab_size'
            ]

        if model_config.cross_attention:
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
        self.missing_qkv_modules = LoraManager.get_missing_qkv_modules(
            self.lora_target_modules)
        if model_config.lora_plugin:
            for lora_module in (self.lora_target_modules +
                                self.missing_qkv_modules):
                for i in range(self.first_layer, self.last_layer):
                    expected_tensor_names += [
                        f'{lora_module}_lora_ranks_{i}',
                        f'{lora_module}_lora_weights_pointers_{i}'
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
    def context_mem_size(self) -> int:
        return self.runtime.context_mem_size

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
    def gather_generation_logits(self):
        return self._model_config.gather_generation_logits

    @property
    def dtype(self):
        return str_dtype_to_torch(self._model_config.dtype)

    @property
    def use_custom_all_reduce(self):
        return self._model_config.use_custom_all_reduce

    @property
    def profiler(self):
        return self.runtime.profiler

    @property
    def engine_inspector(self):
        return self.runtime.engine_inspector

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session. Reset on exit.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret

        return wrapper

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
    def paged_state(self):
        return self._model_config.paged_state

    @property
    def conv_kernel(self):
        return self._model_config.conv_kernel

    @property
    def rnn_hidden_size(self):
        return self._model_config.rnn_hidden_size

    @property
    def state_size(self):
        return self._model_config.state_size

    @property
    def state_dtype(self):
        if self._model_config.state_dtype == "":
            return str_dtype_to_torch(self._model_config.dtype)
        return str_dtype_to_torch(self._model_config.state_dtype)

    def _capture_cuda_graph_and_instantiate(self, context, stream, step):
        instance_idx = (step + 1) % 2
        if not self.has_attn_layers:
            # Create two cuda graph once.If cuda graph has already existed, skip it.
            if self.runtime.cuda_graph_instances[instance_idx] is not None:
                return
            # WAR for TRT 9.x
            if not trt_gte_10() and step < 3:
                return
        # capture cuda graph
        CUASSERT(
            cudart.cudaStreamBeginCapture(
                stream,
                cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
        context.execute_async_v3(stream)
        next_graph = CUASSERT(cudart.cudaStreamEndCapture(stream))[0]

        if self.runtime.cuda_graph_instances[instance_idx] is not None:
            self.runtime.cuda_graph_instances[
                instance_idx] = _update_cuda_graph_instance(
                    self.runtime.cuda_graph_instances[instance_idx], next_graph)
        else:
            self.runtime.cuda_graph_instances[instance_idx] = CUASSERT(
                cudart.cudaGraphInstantiate(next_graph, 0))[0]

        # Pre-upload cuda graph to stream
        CUASSERT(
            cudart.cudaGraphUpload(
                self.runtime.cuda_graph_instances[instance_idx], stream))

    def __setup_decoder(self, input_ids: torch.Tensor,
                        sampling_config: SamplingConfig,
                        host_context_lengths: torch.Tensor):
        '''Allocate buffers and setup the post-processing decoder kernel
        '''
        batch_size = host_context_lengths.shape[0]
        scfg = sampling_config  # just to make a shorter name, no other meaning
        if isinstance(scfg.top_k, torch.Tensor):
            assert scfg.top_k.dtype == torch.int32, f"scfg.top_k.dtype ({scfg.top_k.dtype}) must be torch.int32"
            assert scfg.top_k.shape[
                0] == batch_size, f"scfg.top_k.shape[0] ({scfg.top_k.shape[0]}) must equal to batch_size ({batch_size})"
            self.top_k = scfg.top_k
        else:
            self.top_k = torch.full([batch_size], scfg.top_k, dtype=torch.int32)

        if isinstance(scfg.top_p, torch.Tensor):
            assert scfg.top_p.dtype == torch.float32, f"scfg.top_p.dtype ({scfg.top_p.dtype}) must be torch.float32"
            assert scfg.top_p.shape[
                0] == batch_size, f"scfg.top_p.shape[0] ({scfg.top_p.shape[0]}) must equal to batch_size ({batch_size})"
            self.top_p = scfg.top_p
        else:
            self.top_p = torch.full([batch_size],
                                    scfg.top_p,
                                    dtype=torch.float32)

        if isinstance(scfg.temperature, torch.Tensor):
            assert scfg.temperature.dtype == torch.float32, f"scfg.temperature.dtype ({scfg.temperature.dtype}) must be torch.float32"
            assert scfg.temperature.shape[
                0] == batch_size, f"scfg.temperature.shape[0] ({scfg.temperature.shape[0]}) must equal to batch_size ({batch_size})"
            self.temperature = scfg.temperature
        else:
            self.temperature = torch.full([batch_size],
                                          scfg.temperature,
                                          dtype=torch.float32)

        if isinstance(scfg.repetition_penalty, torch.Tensor):
            assert scfg.repetition_penalty.dtype == torch.float32, f"scfg.repetition_penalty.dtype ({scfg.repetition_penalty.dtype}) must be torch.float32"
            assert scfg.repetition_penalty.shape[
                0] == batch_size, f"scfg.repetition_penalty.shape[0] ({scfg.repetition_penalty.shape[0]}) must equal to batch_size ({batch_size})"
            self.repetition_penalty = scfg.repetition_penalty
        elif scfg.repetition_penalty == 1.0:
            self.repetition_penalty = None
        else:
            self.repetition_penalty = torch.full([batch_size],
                                                 scfg.repetition_penalty,
                                                 dtype=torch.float32)

        if isinstance(scfg.length_penalty, torch.Tensor):
            assert scfg.length_penalty.dtype == torch.float32, f"scfg.length_penalty.dtype ({scfg.length_penalty.dtype}) must be torch.float32"
            assert scfg.length_penalty.shape[
                0] == batch_size, f"scfg.length_penalty.shape[0] ({scfg.length_penalty.shape[0]}) must equal to batch_size ({batch_size})"
            self.host_length_penalty = scfg.length_penalty
        else:
            self.host_length_penalty = torch.full([batch_size],
                                                  scfg.length_penalty,
                                                  dtype=torch.float32)
        self.length_penalty = self.host_length_penalty.to(self.device)

        if isinstance(scfg.early_stopping, torch.Tensor):
            assert scfg.early_stopping.dtype == torch.int32, f"scfg.early_stopping.dtype ({scfg.early_stopping.dtype}) must be torch.int32"
            assert scfg.early_stopping.shape[
                0] == batch_size, f"scfg.early_stopping.shape[0] ({scfg.early_stopping.shape[0]}) must equal to batch_size ({batch_size})"
            self.host_early_stopping = scfg.early_stopping
        else:
            self.host_early_stopping = torch.full([batch_size],
                                                  scfg.early_stopping,
                                                  dtype=torch.int32)

        if isinstance(scfg.presence_penalty, torch.Tensor):
            assert scfg.presence_penalty.dtype == torch.float32, f"scfg.presence_penalty.dtype ({scfg.presence_penalty.dtype}) must be torch.float32"
            assert scfg.presence_penalty.shape[
                0] == batch_size, f"scfg.presence_penalty.shape[0] ({scfg.presence_penalty.shape[0]}) must equal to batch_size ({batch_size})"
            self.presence_penalty = scfg.presence_penalty
        elif scfg.presence_penalty == 0.0:
            self.presence_penalty = None
        else:
            self.presence_penalty = torch.full([batch_size],
                                               scfg.presence_penalty,
                                               dtype=torch.float32)

        if isinstance(scfg.frequency_penalty, torch.Tensor):
            assert scfg.frequency_penalty.dtype == torch.float32, f"scfg.frequency_penalty.dtype ({scfg.frequency_penalty.dtype}) must be torch.float32"
            assert scfg.frequency_penalty.shape[
                0] == batch_size, f"scfg.frequency_penalty.shape[0] ({scfg.frequency_penalty.shape[0]}) must equal to batch_size ({batch_size})"
            self.frequency_penalty = scfg.frequency_penalty
        elif scfg.frequency_penalty == 0.0:
            self.frequency_penalty = None
        else:
            self.frequency_penalty = torch.full([batch_size],
                                                scfg.frequency_penalty,
                                                dtype=torch.float32)

        if isinstance(scfg.min_length, torch.Tensor):
            assert scfg.min_length.dtype == torch.int32, f"scfg.min_length.dtype ({scfg.min_length.dtype}) must be torch.int32"
            assert scfg.min_length.shape[
                0] == batch_size, f"scfg.min_length.shape[0] ({scfg.min_length.shape[0]}) must equal to batch_size ({batch_size})"
            self.min_length = scfg.min_length
        else:
            self.min_length = torch.full([batch_size],
                                         scfg.min_length,
                                         dtype=torch.int32)

        if isinstance(scfg.beam_search_diversity_rate, torch.Tensor):
            assert scfg.beam_search_diversity_rate.dtype == torch.float32, f"scfg.beam_search_diversity_rate.dtype ({scfg.beam_search_diversity_rate.dtype}) must be torch.float32"
            assert scfg.beam_search_diversity_rate.shape[
                0] == batch_size, f"scfg.beam_search_diversity_rate.shape[0] ({scfg.beam_search_diversity_rate.shape[0]}) must equal to batch_size ({batch_size})"
            self.beam_search_diversity_rate = scfg.beam_search_diversity_rate
        elif scfg.beam_search_diversity_rate is not None:
            self.beam_search_diversity_rate = torch.full(
                [batch_size],
                scfg.beam_search_diversity_rate,
                dtype=torch.float32)
        else:
            self.beam_search_diversity_rate = None

        if isinstance(scfg.random_seed, torch.Tensor):
            assert scfg.random_seed.dtype == torch.int64, f"scfg.random_seed.dtype ({scfg.random_seed.dtype}) must be torch.int64"
            assert scfg.random_seed.shape[
                0] == batch_size, f"scfg.random_seed.shape[0] ({scfg.random_seed.shape[0]}) must equal to batch_size ({batch_size})"
            self.random_seed = scfg.random_seed
        elif scfg.random_seed is not None:
            self.random_seed = torch.full([batch_size],
                                          scfg.random_seed,
                                          dtype=torch.int64)
        else:
            self.random_seed = None

        # if self.mapping.is_last_pp_rank():
        #     self.dynamic_decoder.setup(
        #         batch_size, scfg.num_beams, self.top_k, self.top_p,
        #         self.temperature, self.repetition_penalty,
        #         self.presence_penalty, self.frequency_penalty, self.min_length,
        #         self.host_length_penalty, self.host_early_stopping,
        #         self.beam_search_diversity_rate, self.random_seed,
        #         self.top_p_decay, self.top_p_min, self.top_p_reset_ids,
        #         self.no_repeat_ngram_size, scfg.output_log_probs,
        #         scfg.num_beams > 1 or scfg.output_cum_log_probs)

        assert scfg.end_id is not None, "end_id cannot be none"
        assert scfg.pad_id is not None, 'pad_id cannot be none'
        self.end_ids = torch.full((batch_size * scfg.num_beams, ),
                                  scfg.end_id,
                                  dtype=torch.int32,
                                  device=self.device)
        max_context_length = host_context_lengths.max()

        # setup output ids buffer
        if input_ids.dim() == 1:
            # input_ids only have one dimension, which means remove_padding is enabled
            split_ids_list = list(
                torch.split(input_ids.unsqueeze(0),
                            host_context_lengths.numpy().tolist(),
                            dim=1))
            padded_input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(split_ids_list,
                                           dtype=torch.int32,
                                           device='cuda'),
                scfg.pad_id).reshape(batch_size, max_context_length)
        else:
            padded_input_ids = input_ids
        if scfg.num_beams > 1:
            tiled_input_ids = _tile_beam_width(padded_input_ids, scfg.num_beams)
            tiled_input_ids = tiled_input_ids.reshape(batch_size,
                                                      scfg.num_beams,
                                                      max_context_length)
            tiled_input_ids.permute(2, 0, 1)  # TODO: delete?
            self.output_ids = torch.cat(
                (tiled_input_ids,
                 torch.full((batch_size, scfg.num_beams,
                             self.max_seq_length - max_context_length),
                            scfg.end_id,
                            dtype=padded_input_ids.dtype,
                            device=padded_input_ids.device)),
                axis=-1)
        else:
            self.output_ids = torch.cat(
                (padded_input_ids,
                 torch.full(
                     (batch_size, self.max_seq_length - max_context_length),
                     scfg.end_id,
                     dtype=padded_input_ids.dtype,
                     device=padded_input_ids.device)),
                axis=-1)

        # Note: we still allocate max_seq_length size of parent ids (not max_attention_window_size).
        self.parent_ids = torch.zeros(
            (batch_size, scfg.num_beams, self.max_seq_length),
            dtype=torch.int32,
            device=self.device)

        if scfg.num_beams > 1:
            self.new_tokens = torch.zeros([batch_size, scfg.num_beams, 1],
                                          dtype=torch.int32,
                                          device=self.device)
        else:
            self.new_tokens = torch.zeros([batch_size, 1],
                                          dtype=torch.int32,
                                          device=self.device)

        if scfg.num_beams > 1 or scfg.output_cum_log_probs:
            self.cum_log_probs = torch.full((batch_size, scfg.num_beams),
                                            -1e20,
                                            dtype=torch.float32,
                                            device=self.device)
            self.cum_log_probs[:, 0] = 0.0
        else:
            self.cum_log_probs = None

        if scfg.output_log_probs:
            self.log_probs = torch.zeros(
                (batch_size, scfg.num_beams, self.max_seq_length),
                dtype=torch.float32,
                device=self.device)
            self.log_probs_tiled = torch.zeros(
                (self.max_seq_length, self._model_config.max_batch_size,
                 scfg.num_beams),
                dtype=torch.float32,
                device=self.device)
        else:
            self.log_probs = None
            self.log_probs_tiled = None

        self.finished = torch.zeros((batch_size, scfg.num_beams),
                                    dtype=torch.uint8,
                                    device=self.device)

        if scfg.use_beam_hyps:
            self.beam_hyps_output_ids_cba = torch.full(
                size=[batch_size, scfg.num_beams * 2, self.max_seq_length],
                fill_value=scfg.end_id,
                dtype=torch.int32,
                device=self.device)
            self.beam_hyps_seq_len_cba = torch.zeros(
                [batch_size, scfg.num_beams * 2],
                dtype=torch.int32,
                device=self.device)
            self.beam_hyps_cum_log_probs_cba = torch.zeros(
                [batch_size, scfg.num_beams * 2],
                dtype=torch.float,
                device=self.device)
            self.beam_hyps_normed_scores_cba = torch.zeros(
                [batch_size, scfg.num_beams * 2],
                dtype=torch.float,
                device=self.device)
            self.beam_hyps_log_probs_cba = torch.zeros(
                [batch_size, scfg.num_beams * 2, self.max_seq_length],
                dtype=torch.float,
                device=self.device)
            self.beam_hyps_min_normed_scores = torch.zeros([batch_size],
                                                           dtype=torch.float,
                                                           device=self.device)
            self.beam_hyps_num_beams = torch.zeros([batch_size],
                                                   dtype=torch.int32,
                                                   device=self.device)
            self.beam_hyps_is_done = torch.zeros([batch_size],
                                                 dtype=torch.bool,
                                                 device=self.device)
        else:
            self.beam_hyps_output_ids_cba = None
            self.beam_hyps_seq_len_cba = None
            self.beam_hyps_cum_log_probs_cba = None
            self.beam_hyps_normed_scores_cba = None
            self.beam_hyps_log_probs_cba = None
            self.beam_hyps_min_normed_scores = None
            self.beam_hyps_num_beams = None
            self.beam_hyps_is_done = None

        self.cross_qkv_reuse = None

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.runtime.engine.get_tensor_dtype(name))
        return dtype

    def setup(self,
              batch_size: int,
              max_context_length: int,
              max_new_tokens: int,
              beam_width: int = 1,
              max_attention_window_size: Optional[int] = None,
              sink_token_length: Optional[int] = None,
              encoder_max_input_length: Optional[int] = None,
              lora_manager: LoraManager = None,
              lora_uids: List[str] = None,
              medusa_choices: List[List[int]] = None):
        # Store these params related to buffer size to check against
        # the input shape with the params given in decode()
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_context_length + max_new_tokens
        self.beam_width = beam_width
        self.encoder_max_input_length = encoder_max_input_length
        if max_attention_window_size is None:
            self.max_attention_window_size = self.max_seq_length
            logger.debug(
                "The max_attention_window_size is not set, we will use max_seq_length by default."
            )
            self.host_max_attention_window_sizes = torch.ones(
                (self.num_attn_layers, ),
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
                (self.num_attn_layers, ),
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
            if max_attention_window_size.shape[0] != self.num_attn_layers:
                logger.error(
                    "max_attention_window_size tensor's size is not equal to num_layers! "
                    "Note that num_layers = num_total_layers // pipeline_parallelism_size."
                )
                assert False
            self.host_max_attention_window_sizes = torch.minimum(
                max_attention_window_size.to(torch.int32),
                torch.IntTensor([self.max_seq_length] * self.num_attn_layers))
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

        self.use_one_more_block = (
            self.paged_kv_cache and beam_width > 1
            and self.max_seq_length > self.max_attention_window_size)
        self.lora_manager = lora_manager

        self.buffer = {}
        if self.mapping.is_last_pp_rank():
            self.buffer['embeddings_output'] = torch.empty(
                (batch_size, 4096) # TODO(ilyaonoff)
                if not self.gather_context_logits else
                (batch_size, max_context_length, self.vocab_size_padded),
                dtype=self._tensor_dtype('embeddings_output'),
                device=self.device)

        if self.cross_attention:
            # use shape info to pass max length info in remove padding mode
            self.buffer['encoder_max_input_length'] = torch.empty(
                (encoder_max_input_length, ),
                dtype=self._tensor_dtype('encoder_max_input_length'),
                device=self.device)

        if self.quant_mode.has_kv_cache_quant():
            # Since torch does not support fp8 now, using int8 here.
            kv_cache_type = torch.int8
        else:
            if self.has_attn_layers:
                first_atten_layer = self.layer_types.index('attention')
                kv_cache_type = self.dtype if self.paged_kv_cache else self._tensor_dtype(
                    f'present_key_value_{first_atten_layer}')
            else:
                kv_cache_type = None

        if self.paged_kv_cache and self.has_attn_layers:
            num_blocks, _ = self._get_num_paged_blocks(
                self.max_attention_window_size, self.sink_token_length,
                self.use_one_more_block)
            cache_shape = (
                num_blocks,
                self.num_attn_layers,
                2,
                self.num_heads_kv,
                self.tokens_per_block,
                self.head_size,
            )
            self.kv_cache_pool = torch.empty(cache_shape,
                                             dtype=kv_cache_type,
                                             device=self.device)
            if self.cross_attention:  # As for now we enable cross paged kv and self paged kv to share the same tokens_per_block
                cross_num_blocks, _ = self._get_num_paged_blocks(
                    self.encoder_max_input_length,
                    sink_token_length=0,
                    use_one_more_block=False)
                cross_cache_shape = (
                    cross_num_blocks,
                    self.num_layers,
                    2,
                    self.num_heads_kv,
                    self.tokens_per_block,
                    self.head_size,
                )
                self.cross_kv_cache_pool = torch.empty(cross_cache_shape,
                                                       dtype=kv_cache_type,
                                                       device=self.device)
        elif self.has_attn_layers:
            cache_shape = (
                batch_size,
                2,
                self.num_heads_kv,
                self.max_attention_window_size,
                self.head_size,
            )
            for i in range(self.first_layer, self.last_layer):
                if self.layer_types[i] == 'attention':
                    self.buffer[f'present_key_value_{i}'] = torch.empty(
                        cache_shape, dtype=kv_cache_type, device=self.device)

            if self.cross_attention:
                cross_cache_shape = (
                    batch_size,
                    2,
                    self.num_heads_kv,
                    self.encoder_max_input_length,
                    self.head_size,
                )
                for i in range(self.first_layer, self.last_layer):
                    if self.layer_types[i] == 'attention':
                        self.buffer[
                            f'cross_present_key_value_{i}'] = torch.empty(
                                cross_cache_shape,
                                dtype=kv_cache_type,
                                device=self.device)

        if self.use_gpt_attention_plugin:
            self.sequence_length_buffer = torch.ones((batch_size, ),
                                                     dtype=torch.int32,
                                                     device=self.device)
        else:
            # Without plugin, we need extra kv cache buffers.
            # Because we don't support inplace update, so we need separate buffer for inputs and outputs.
            # We can do reuse between different layers' inputs and outputs, i.e. current layer's output can
            # reuse previous layer's input memory. But this need one extra buffer as the guard.
            if self.has_attn_layers:  # Not applicable to cross KV buffers as it's constant
                i = self.attn_to_general_idx[0]
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

        if self.use_mamba_conv1d_plugin:
            conv_state_shape = (
                batch_size,
                self.conv_kernel - 1,
                self.rnn_hidden_size,
            )
        else:
            conv_state_shape = (
                batch_size,
                self.rnn_hidden_size,
                self.conv_kernel - 1,
            )

        rnn_state_shape = (
            batch_size,
            self.state_size,
            self.rnn_hidden_size,
        )

        for i in range(self.first_layer, self.last_layer):
            if self.layer_types[i] == 'recurrent':
                dtype = self.dtype
                self.buffer[f'present_conv_state_{i}'] = torch.empty(
                    conv_state_shape, dtype=dtype, device=self.device)
                self.buffer[f'1_present_conv_state_{i}'] = torch.empty(
                    conv_state_shape, dtype=dtype, device=self.device)
                self.buffer[f'present_rnn_state_{i}'] = torch.empty(
                    rnn_state_shape, dtype=self.state_dtype, device=self.device)
                if self.paged_state:
                    conv_state_ptr = torch.tensor(
                        [self.buffer[f'present_conv_state_{i}'].data_ptr()],
                        dtype=torch.int64,
                        device='cpu')
                    rnn_state_ptr = torch.tensor(
                        [self.buffer[f'present_rnn_state_{i}'].data_ptr()],
                        dtype=torch.int64,
                        device='cpu')
                    self.buffer[f'conv_state_ptr_{i}'] = conv_state_ptr
                    self.buffer[f'rnn_state_ptr_{i}'] = rnn_state_ptr

        if self.use_lora_plugin and self.lora_manager is not None:
            lora_uids = lora_uids or ["-1"]
            self.buffer.update(
                self.lora_manager.input_buffers(
                    lora_uids,
                    self.mapping,
                    self._model_config.num_layers,
                ))

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
            kv_cache_block_offsets: torch.Tensor,
            host_kv_cache_block_offsets: torch.Tensor,
            cross_kv_cache_block_offsets: torch.Tensor = None,
            host_cross_kv_cache_block_offsets: torch.Tensor = None,
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

        if self.has_attn_layers:
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

            if not self.gather_context_logits or self.has_rnn_layers:
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

        if self.paged_kv_cache and self.has_attn_layers:
            buffer = kv_cache_block_offsets.contiguous()
            shape = kv_cache_block_offsets.shape
            shape = [shape[0] * shape[1], *shape[2:]]
            add_tensor_with_shape(buffer, f'kv_cache_block_offsets', shape)
            add_tensor_with_shape(host_kv_cache_block_offsets,
                                  f'host_kv_cache_block_offsets', shape)
            pool_pointers = f'host_kv_cache_pool_pointers'
            add_tensor(self.buffer[pool_pointers], pool_pointers)
            if self.cross_attention:
                cross_buffer = cross_kv_cache_block_offsets.contiguous()
                cross_shape = cross_kv_cache_block_offsets.shape
                cross_shape = [
                    cross_shape[0] * cross_shape[1], *cross_shape[2:]
                ]
                add_tensor_with_shape(cross_buffer,
                                      f'cross_kv_cache_block_offsets',
                                      cross_shape)
                add_tensor_with_shape(host_cross_kv_cache_block_offsets,
                                      f'host_cross_kv_cache_block_offsets',
                                      cross_shape)
                cross_pool_pointers = f'host_cross_kv_cache_pool_pointers'
                add_tensor(self.buffer[cross_pool_pointers],
                           cross_pool_pointers)

        batch_size = context_lengths.shape[0]
        if not self.paged_kv_cache:
            for idx in range(self.first_layer, self.last_layer):
                if not self.use_gpt_attention_plugin and self.layer_types[
                        idx] == 'attention':
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
                elif self.layer_types[idx] == 'attention':
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

        for idx in range(self.first_layer, self.last_layer):
            if self.layer_types[idx] != 'recurrent':
                continue
            if self.paged_state:
                add_tensor(self.buffer[f'conv_state_ptr_{idx}'],
                           f'conv_state_ptr_{idx}')
                add_tensor(self.buffer[f'rnn_state_ptr_{idx}'],
                           f'rnn_state_ptr_{idx}')
            else:
                # conv state
                dtype = self._tensor_dtype(f'present_conv_state_{idx}')
                if self.use_mamba_conv1d_plugin:
                    conv_state_shape = (batch_size, self.conv_kernel - 1,
                                        self.rnn_hidden_size)
                else:
                    conv_state_shape = (batch_size, self.rnn_hidden_size,
                                        self.conv_kernel - 1)

                conv_state = torch.zeros(conv_state_shape,
                                         dtype=dtype,
                                         device=self.device)
                add_tensor(conv_state, f'past_conv_state_{idx}')
                present = f'present_conv_state_{idx}'
                add_tensor(self.buffer[present], present)
                # rnn state
                rnn_state = self.buffer[f'present_rnn_state_{idx}']
                add_tensor(rnn_state, f'past_rnn_state_{idx}')
                add_tensor(rnn_state, f'present_rnn_state_{idx}')

        if self.paged_state and self.has_rnn_layers:
            slot_mapping = torch.arange(0,
                                        batch_size,
                                        device='cuda',
                                        dtype=torch.int32)
            add_tensor(slot_mapping, 'slot_mapping')

        if self.use_gpt_attention_plugin and self.has_attn_layers:
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
                                  (self.num_attn_layers, ))
            if self.remove_input_padding:
                add_tensor(host_context_lengths, 'host_context_lengths')
        else:
            if self.has_rnn_layers:
                host_request_types = torch.zeros_like(context_lengths,
                                                      device='cpu').int()
                add_tensor(host_request_types, 'host_request_types')
                if self.use_mamba_conv1d_plugin and self.remove_input_padding:
                    add_tensor(host_context_lengths, 'host_context_lengths')
            if self.has_attn_layers:
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
        if (use_gpt_attention_plugin
                or self.has_rnn_layers) and remove_input_padding:
            last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
        ret = {'last_token_ids': last_token_ids}

        if use_gpt_attention_plugin:
            max_context_length = kwargs.pop('max_context_length')
            if remove_input_padding:
                position_ids = torch.concat([
                    torch.arange(0,
                                 host_context_lengths[i],
                                 dtype=torch.int32,
                                 device='cuda') for i in range(batch_size)
                ])
            else:
                position_ids = torch.tensor(range(max_context_length),
                                            dtype=torch.int32,
                                            device='cuda').reshape(
                                                [1,
                                                 -1]).expand([batch_size, -1])
        else:
            if self.has_attn_layers:
                input_ids = kwargs.pop('input_ids')
                pad_id = kwargs.pop('pad_id', None)
                attention_mask = _prepare_attention_mask(input_ids, pad_id)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.int()
                ret['attention_mask'] = attention_mask

        if self.has_position_embedding and self.has_attn_layers:
            ret['position_ids'] = position_ids

        return ret

    def pp_communicate_new_tokens(self, should_stop, cache_indir,
                                  sequence_length):
        if self.mapping.is_last_pp_rank():
            for pg in self.mapping.pp_group:
                if pg == self.mapping.rank:
                    continue
                should_stop = should_stop.to(self.device)
                self.nccl_comm.send(should_stop, pg)
                self.nccl_comm.send(cache_indir, pg)
                self.nccl_comm.send(sequence_length, pg)
            self.nccl_comm.send(self.new_tokens, self.mapping.pp_group[0])
        else:
            should_stop = torch.zeros(1, dtype=torch.bool, device=self.device)
            self.nccl_comm.recv(should_stop, self.mapping.pp_group[-1])
            self.nccl_comm.recv(cache_indir, self.mapping.pp_group[-1])
            self.nccl_comm.recv(sequence_length, self.mapping.pp_group[-1])
            if self.mapping.is_first_pp_rank():
                self.nccl_comm.recv(self.new_tokens, self.mapping.pp_group[-1])
        return should_stop

    def pp_communicate_final_output_ids(self, final_output_ids, batch_size,
                                        beam_width):
        if self.mapping.is_last_pp_rank():
            self.nccl_comm.send(final_output_ids, self.mapping.pp_group[0])
        elif self.mapping.is_first_pp_rank():
            final_output_ids = torch.zeros(
                (batch_size, beam_width, self.max_seq_length),
                dtype=torch.int32,
                device=self.device)
            self.nccl_comm.recv(final_output_ids, self.mapping.pp_group[-1])
        return final_output_ids

    def update_output_ids_by_offset(self, new_generated_ids, offsets):
        # output_ids [batch_size, padded_input_length]
        # new_generated_ids [batch_size, padded_accepted_length]
        # offsets [batch_size]
        # FIXME: using fused kernel to update the padded output ids.
        batch_size = self.output_ids.shape[0]
        for b in range(batch_size):
            self.output_ids[b, offsets[b]:(
                offsets[b] + self.accept_lengths[b]
            )] = new_generated_ids[b][:self.accept_lengths[b]]

    def handle_per_step(
            self, cache_indirections: list, step: int, batch_size: int,
            max_context_length: int, beam_width: int, input_ids: torch.Tensor,
            hidden_states: torch.Tensor, scfg: SamplingConfig,
            kv_cache_block_offsets: torch.Tensor,
            host_kv_cache_block_offsets: torch.Tensor,
            cross_kv_cache_block_offsets: torch.Tensor,
            host_cross_kv_cache_block_offsets: torch.Tensor,
            prompt_embedding_table: torch.Tensor, tasks: torch.Tensor,
            context_lengths: torch.Tensor, host_context_lengths,
            attention_mask: torch.Tensor, cross_attention_mask: torch.Tensor,
            prompt_vocab_size: torch.Tensor, ite: int,
            sequence_limit_lengths: torch.Tensor,
            sequence_lengths: torch.Tensor,
            next_step_tensors: Dict[str, RuntimeTensor], stop_words_data,
            bad_words_data, encoder_output: torch.Tensor,
            encoder_input_lengths: torch.Tensor,
            stopping_criteria: StoppingCriteria,
            logits_processor: LogitsProcessor, **kwargs):
        if step % 2:
            context = self.runtime.context_0
            this_src_cache_indirection = cache_indirections[1]
            this_tgt_cache_indirection = cache_indirections[0]
            next_src_cache_indirection = cache_indirections[0]
        else:
            context = self.runtime.context_1
            this_src_cache_indirection = cache_indirections[0]
            this_tgt_cache_indirection = cache_indirections[1]
            next_src_cache_indirection = cache_indirections[1]

        if step == 0:
            model_inputs = self._prepare_context_inputs(
                batch_size=batch_size,
                context_lengths=context_lengths,
                host_context_lengths=host_context_lengths,
                use_gpt_attention_plugin=self.use_gpt_attention_plugin,
                remove_input_padding=self.remove_input_padding,
                max_context_length=max_context_length,
                input_ids=input_ids,
                pad_id=scfg.pad_id,
                eos_id=scfg.end_id)

            position_ids = model_inputs.get('position_ids', None)
            last_token_ids = model_inputs.get('last_token_ids')
            attention_mask = model_inputs.get('attention_mask', None)

            if self.paged_kv_cache and self.has_attn_layers:
                host_kv_cache_block_offsets = self.kv_cache_manager.get_block_offsets(
                    beam_width=1)
                kv_cache_block_offsets = host_kv_cache_block_offsets.to('cuda')
                if self.cross_attention:
                    host_cross_kv_cache_block_offsets = self.cross_kv_cache_manager.get_block_offsets(
                        beam_width=1)
                    cross_kv_cache_block_offsets = host_cross_kv_cache_block_offsets.to(
                        'cuda')

            ctx_tensors = self._get_context_shape_buffer(
                input_ids, context_lengths, host_context_lengths, position_ids,
                last_token_ids, attention_mask, cross_attention_mask,
                this_src_cache_indirection, kv_cache_block_offsets,
                host_kv_cache_block_offsets, cross_kv_cache_block_offsets,
                host_cross_kv_cache_block_offsets, hidden_states,
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
        instance_idx = step % 2
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
            raise RuntimeError(f"Executing TRT engine failed step={step}!")

        # TODO: remove this Windows WAR after https://nvbugs/4460474 is fixed.
        if platform.system() == "Windows" or self.debug_mode:
            torch.cuda.synchronize()

        context_logits = None
        generation_logits = None
        # if self.mapping.is_last_pp_rank():
        #     if self.gather_generation_logits:
        #         generation_logits = self.buffer['logits'].detach().clone()

        # Initialize sequence_lengths (no paddings) for the generation phase.
        if step == 0:
            self.sequence_length_buffer = context_lengths.detach().clone()

        should_stop = None
        logits = None
        if self.mapping.is_last_pp_rank():
            should_stop = torch.tensor(True) # TODO(ilyaonoff)

        if self.runtime._is_profiling():
            if not context.report_to_profiler():
                logger.warning("Runtime report to profiler failed.")
            self.runtime._insert_step_to_profiler(step)

        if self.mapping.has_pp():
            should_stop = self.pp_communicate_new_tokens(
                should_stop, this_tgt_cache_indirection,
                self.sequence_length_buffer)

        if self.paged_kv_cache and self.has_attn_layers:
            if (step >= self.max_new_tokens - 1) or (should_stop is not None
                                                     and should_stop.item()):
                # Free all blocks in all sequences.
                # With in-flight batching and while loop we'll free some sequences, when they are done
                self.kv_cache_manager.step([True] * batch_size)
                if self.cross_attention:
                    self.cross_kv_cache_manager.step([True] * batch_size)

        if self.debug_mode:
            self.dump_debug_buffers(step)

            if next_step_tensors is not None:
                self.debug_buffer = {
                    name: tensor.to_torch()
                    for name, tensor in next_step_tensors.items()
                }

        return should_stop, next_step_tensors, tasks, context_lengths, host_context_lengths, attention_mask, context_logits, generation_logits, encoder_input_lengths

    def dump_debug_buffers(self, step: int) -> None:
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
            np.save(debug_dir / f"{fname}-step{step}.npy", t)

            txt_format = "%d" if t.dtype in [np.int32, np.int8] else '%.18e'
            np.savetxt(
                debug_dir / f"{fname}-step{step}.txt",
                t.reshape(-1, t.shape[-1]),  # savetxt accepts 2 dims only
                fmt=txt_format)

    def decode_regular(self,
                       batch_size: int,
                       scfg: SamplingConfig,
                       sequence_lengths: torch.Tensor,
                       context_lengths: torch.Tensor,
                       host_context_lengths,
                       max_context_length: int,
                       beam_width: int,
                       cache_indirections: list,
                       input_ids: torch.Tensor,
                       hidden_states: torch.Tensor,
                       prompt_embedding_table: torch.Tensor,
                       tasks: torch.Tensor,
                       prompt_vocab_size: torch.Tensor,
                       ite: int,
                       sequence_limit_lengths: torch.Tensor,
                       stop_words_data,
                       bad_words_data,
                       output_sequence_lengths: bool = False,
                       return_dict: bool = False,
                       encoder_output: torch.Tensor = None,
                       encoder_input_lengths: torch.Tensor = None,
                       stopping_criteria: StoppingCriteria = None,
                       logits_processor: LogitsProcessor = None,
                       cross_attention_mask: torch.Tensor = None,
                       **kwargs):
        kv_cache_block_offsets = None
        host_kv_cache_block_offsets = None
        cross_kv_cache_block_offsets = None
        host_cross_kv_cache_block_offsets = None
        attention_mask = None
        outputs_context_logits = None
        outputs_generation_logits = []

        def get_outputs_dict(output_ids):
            outputs = {}
            outputs['output_ids'] = output_ids
            if scfg.output_log_probs:
                outputs['log_probs'] = self.log_probs
            if scfg.output_cum_log_probs:
                outputs['cum_log_probs'] = self.cum_log_probs
            if output_sequence_lengths:
                outputs[
                    'sequence_lengths'] = self.sequence_length_buffer.reshape(
                        [batch_size, beam_width])
            if self.gather_context_logits:
                outputs['context_logits'] = outputs_context_logits
            if self.gather_generation_logits:
                outputs['generation_logits'] = outputs_generation_logits
            return outputs

        benchmark_profiler = kwargs.get('benchmark_profiler', None)
        generation_phase_step_count = 0

        if benchmark_profiler is not None and benchmark_profiler.is_recording_perf_profile:
            self.runtime._set_profiler()

        def profile_fn(benchmark_profiler_obj, step_count):
            if benchmark_profiler_obj is not None:
                benchmark_profiler_obj.record_cuda_event('last_token')
                benchmark_profiler_obj.record_elapsed_time(
                    'first_token', 'last_token', 'generation_time')
                benchmark_profiler_obj.add_aux_info('generation_step_count',
                                                    step_count)

        next_step_tensors = None
        step = 0
        should_stop, next_step_tensors, tasks, context_lengths, host_context_lengths, attention_mask, context_logits, generation_logits, encoder_input_lengths = self.handle_per_step(
            cache_indirections, step, batch_size, max_context_length,
            beam_width, input_ids, hidden_states, scfg,
            kv_cache_block_offsets, host_kv_cache_block_offsets,
            cross_kv_cache_block_offsets, host_cross_kv_cache_block_offsets,
            prompt_embedding_table, tasks, context_lengths,
            host_context_lengths, attention_mask, cross_attention_mask,
            prompt_vocab_size, ite, sequence_limit_lengths,
            sequence_lengths, next_step_tensors, stop_words_data,
            bad_words_data, encoder_output, encoder_input_lengths,
            stopping_criteria, logits_processor, **kwargs)

        if benchmark_profiler is not None:
            benchmark_profiler.record_cuda_event('first_token')

        if self.mapping.is_last_pp_rank():
            if step == 0 and self.gather_context_logits:
                outputs_context_logits = context_logits
            if self.gather_generation_logits:
                outputs_generation_logits.append(generation_logits)

        if should_stop is not None and should_stop.item():
            profile_fn(benchmark_profiler, generation_phase_step_count)
            # if self.is_medusa_mode:
            #     # just hack away for now
            #     final_output_ids = self.output_ids.clone().unsqueeze(1)
            #     final_output_ids = final_output_ids[:, :, :self.
            #                                         max_seq_length -
            #                                         self._model_config.
            #                                         max_medusa_tokens]
            # else:
            #     final_output_ids = self.finalize_decoder(
            #         context_lengths, batch_size, beam_width, scfg)

            # if self.mapping.is_first_pp_rank():
            #     if return_dict:
            #         return get_outputs_dict(final_output_ids)
            #     else:
            #         return final_output_ids
            # elif self.mapping.is_last_pp_rank():
            #     outputs = {}
            #     if self.gather_context_logits:
            #         outputs['context_logits'] = outputs_context_logits
            #     if self.gather_generation_logits:
            #         outputs['generation_logits'] = outputs_generation_logits
            #     return outputs
            # else:
            #     return None

        profile_fn(benchmark_profiler, generation_phase_step_count)

        return {
            'embeddings_output': self.buffer['embeddings_output']
        }

    # def decode_stream(self,
    #                   batch_size: int,
    #                   scfg: SamplingConfig,
    #                   sequence_lengths: torch.Tensor,
    #                   context_lengths: torch.Tensor,
    #                   host_context_lengths,
    #                   max_context_length: int,
    #                   beam_width: int,
    #                   cache_indirections: list,
    #                   input_ids: torch.Tensor,
    #                   hidden_states: torch.Tensor,
    #                   prompt_embedding_table: torch.Tensor,
    #                   tasks: torch.Tensor,
    #                   prompt_vocab_size: torch.Tensor,
    #                   ite: int,
    #                   sequence_limit_lengths: torch.Tensor,
    #                   stop_words_data,
    #                   bad_words_data,
    #                   output_sequence_lengths: bool = False,
    #                   return_dict: bool = False,
    #                   encoder_output: torch.Tensor = None,
    #                   encoder_input_lengths: torch.Tensor = None,
    #                   stopping_criteria: StoppingCriteria = None,
    #                   logits_processor: LogitsProcessor = None,
    #                   cross_attention_mask: torch.Tensor = None,
    #                   **kwargs):
    #     kv_cache_block_offsets = None
    #     host_kv_cache_block_offsets = None
    #     cross_kv_cache_block_offsets = None
    #     host_cross_kv_cache_block_offsets = None
    #     attention_mask = None
    #     outputs_context_logits = None

    #     def get_outputs_dict(output_ids):
    #         outputs = {}
    #         outputs['output_ids'] = output_ids
    #         if output_sequence_lengths:
    #             outputs[
    #                 'sequence_lengths'] = self.sequence_length_buffer.reshape(
    #                     [batch_size, beam_width])
    #         if self.gather_context_logits:
    #             outputs['context_logits'] = outputs_context_logits
    #         return outputs

    #     next_step_tensors = None
    #     for step in range(0, self.max_new_tokens):

    #         should_stop, next_step_tensors, tasks, context_lengths, host_context_lengths, attention_mask, context_logits, generation_logits, encoder_input_lengths = self.handle_per_step(
    #             cache_indirections, step, batch_size, max_context_length,
    #             beam_width, input_ids, hidden_states, scfg,
    #             kv_cache_block_offsets, host_kv_cache_block_offsets,
    #             cross_kv_cache_block_offsets, host_cross_kv_cache_block_offsets,
    #             prompt_embedding_table, tasks, context_lengths,
    #             host_context_lengths, attention_mask, cross_attention_mask,
    #             prompt_vocab_size, ite, sequence_limit_lengths,
    #             sequence_lengths, next_step_tensors, stop_words_data,
    #             bad_words_data, encoder_output, encoder_input_lengths,
    #             stopping_criteria, logits_processor)
    #         if step == 0:
    #             outputs_context_logits = context_logits
    #         if should_stop is not None:

    #             final_output_ids = self.finalize_decoder(context_lengths,
    #                                                      batch_size,
    #                                                      beam_width,
    #                                                      scfg,
    #                                                      in_progress=True)

    #             if self.mapping.is_first_pp_rank():
    #                 if return_dict:
    #                     yield get_outputs_dict(final_output_ids)
    #                 else:
    #                     yield final_output_ids
    #             else:
    #                 yield None

    #             if should_stop.item():
    #                 return

    #     final_output_ids = self.finalize_decoder(context_lengths, batch_size,
    #                                              beam_width, scfg)
    #     if self.mapping.is_first_pp_rank():
    #         if return_dict:
    #             yield get_outputs_dict(final_output_ids)
    #         else:
    #             yield final_output_ids
    #     else:
    #         yield None

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
    @cuda_stream_guard
    def decode(self,
               input_ids: torch.Tensor,
               context_lengths: torch.Tensor,
               sampling_config: SamplingConfig,
               prompt_embedding_table: torch.Tensor = None,
               tasks: torch.Tensor = None,
               prompt_vocab_size: torch.Tensor = None,
               stop_words_list=None,
               bad_words_list=None,
               streaming: bool = False,
               output_sequence_lengths: bool = False,
               return_dict: bool = False,
               encoder_output: torch.Tensor = None,
               encoder_input_lengths: torch.Tensor = None,
               stopping_criteria: StoppingCriteria = None,
               logits_processor: LogitsProcessor = None,
               cross_attention_mask: torch.Tensor = None,
               **kwargs):
        scfg = sampling_config
        batch_size = context_lengths.size(0)
        beam_width = scfg.num_beams
        max_context_length = torch.max(context_lengths).item()
        host_context_lengths = context_lengths.cpu()
        assert batch_size == self.batch_size, \
            "Given batch size is different from the one used in setup()," \
            "rerun the setup function with the new batch size to avoid buffer overflow."
        assert max_context_length <= self.max_context_length, \
            "Given input length is large then the one used in setup()," \
            "rerun the setup function with the new max_context_length to avoid buffer overflow."
        assert beam_width == self.beam_width, \
            "Given beam width is different from the one used in setup()," \
            "rerun the setup function with the new beam width to avoid buffer overflow."
        assert self.sink_token_length <= torch.min(context_lengths).item(), \
            "Given sink token length is larger than shortest context length," \
            "rerun the setup function with a smaller sink token length."
        ite = 0  # index of local batches, will always be 0 if pp_size = 1

        if self.remove_input_padding and input_ids.dim() == 2:
            assert input_ids.shape[
                0] == 1, "Packed 2D input must have shape [1, <sum of input lengths>]"
            input_ids = input_ids.squeeze(0)

        self.__setup_decoder(input_ids, scfg, host_context_lengths)
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        sequence_limit_lengths = torch.full((batch_size, 1),
                                            self.max_seq_length,
                                            dtype=torch.int32,
                                            device=self.device)

        # Sequence_lengths for the dynamic decoder still has the input paddings.
        sequence_lengths = torch.full((batch_size * beam_width, 1),
                                      max_context_length,
                                      dtype=torch.int32,
                                      device=self.device)

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                self.max_attention_window_size,
            ),
                       0,
                       dtype=torch.int32,
                       device=self.device),
            torch.full((
                batch_size,
                beam_width,
                self.max_attention_window_size,
            ),
                       0,
                       dtype=torch.int32,
                       device=self.device)
        ]  # ping-pong buffers

        hidden_states = None
        if self.mapping.has_pp():
            max_num_tokens = max(batch_size * beam_width,
                                 batch_size * self.max_seq_length)
            hidden_size = self.hidden_size * self.mapping.tp_size
            hidden_states = torch.zeros((1, max_num_tokens, hidden_size))

        # Init KV cache block manager
        if self.paged_kv_cache and self.has_attn_layers:
            num_blocks, max_blocks_per_seq = self._get_num_paged_blocks(
                self.max_attention_window_size, self.sink_token_length,
                self.use_one_more_block)
            self.buffer[f'host_kv_cache_pool_pointers'] = torch.tensor(
                [self.kv_cache_pool.data_ptr(), 0], dtype=torch.int64)

            block_size = self.num_heads_kv * self.tokens_per_block * self.head_size
            self.kv_cache_manager = KVCacheManager(
                num_layers=self.num_attn_layers,
                num_blocks=num_blocks,
                block_size=block_size,
                tokens_per_block=self.tokens_per_block,
                max_blocks_per_seq=max_blocks_per_seq,
                max_attention_window_size=self.max_attention_window_size,
                sink_token_len=self.sink_token_length,
                beam_width=beam_width,
                use_one_more_block=self.use_one_more_block)

            if self.cross_attention:
                cross_num_blocks, max_cross_blocks_per_seq = self._get_num_paged_blocks(
                    self.encoder_max_input_length,
                    sink_token_length=0,
                    use_one_more_block=False)
                self.buffer[
                    f'host_cross_kv_cache_pool_pointers'] = torch.tensor(
                        [self.cross_kv_cache_pool.data_ptr(), 0],
                        dtype=torch.int64)

                cross_block_size = self.num_heads_kv * self.tokens_per_block * self.head_size
                self.cross_kv_cache_manager = KVCacheManager(
                    num_layers=self.num_layers,
                    num_blocks=cross_num_blocks,
                    block_size=cross_block_size,
                    tokens_per_block=self.tokens_per_block,
                    max_blocks_per_seq=max_cross_blocks_per_seq,
                    max_attention_window_size=self.encoder_max_input_length,
                    sink_token_len=self.sink_token_length,
                    beam_width=beam_width,
                    use_one_more_block=False)

            # Add sequences to the manager
            for bi in range(batch_size):
                generation_sequence = GenerationSequence(seq_idx=bi,
                                                         batch_idx=bi)
                self.kv_cache_manager.add_sequence(generation_sequence,
                                                   max_context_length)
                if self.cross_attention:
                    cross_generation_sequence = GenerationSequence(seq_idx=bi,
                                                                   batch_idx=bi)
                    self.cross_kv_cache_manager.add_sequence(
                        cross_generation_sequence,
                        self.encoder_max_input_length,
                        always_share_across_beam=True)
                    # cross attention paged kv cache should always share the context blocks across beams
                    # due to the fact that we are not adding new key/value cache to cross kv in generation

        stop_words_lens = None
        stop_words_list_ptrs = None
        max_stop_words_len = 0
        if stop_words_list is not None:
            stop_words_list = torch.from_numpy(stop_words_list).contiguous().to(
                'cuda')
            max_stop_words_len = stop_words_list.shape[2]
            stop_words_lens = torch.full((batch_size, ),
                                         max_stop_words_len,
                                         dtype=torch.int32).to('cuda')
            stop_words_list_ptrs = torch.zeros((batch_size), dtype=torch.int64)
            for bi in range(batch_size):
                stop_words_list_ptrs[bi] = stop_words_list.data_ptr(
                ) + bi * 2 * max_stop_words_len * stop_words_list.element_size(
                )
            stop_words_list_ptrs = stop_words_list_ptrs.to('cuda')
        stop_words_data = (stop_words_list_ptrs, stop_words_lens,
                           max_stop_words_len)

        bad_words_lens = None
        bad_words_list_ptrs = None
        max_bad_words_len = 0
        if bad_words_list is not None:
            bad_words_list = torch.from_numpy(bad_words_list).contiguous().to(
                'cuda')
            max_bad_words_len = bad_words_list.shape[2]
            bad_words_lens = torch.full((batch_size, ),
                                        max_bad_words_len,
                                        dtype=torch.int32).to('cuda')
            bad_words_list_ptrs = torch.zeros((batch_size), dtype=torch.int64)
            for bi in range(batch_size):
                bad_words_list_ptrs[bi] = bad_words_list.data_ptr(
                ) + bi * 2 * max_bad_words_len * bad_words_list.element_size()
            bad_words_list_ptrs = bad_words_list_ptrs.to('cuda')
        bad_words_data = (bad_words_list_ptrs, bad_words_lens,
                          max_bad_words_len)

        # start context phase
        if streaming:
            return self.decode_stream(
                batch_size, scfg, sequence_lengths, context_lengths,
                host_context_lengths, max_context_length, beam_width,
                cache_indirections, input_ids, hidden_states,
                prompt_embedding_table, tasks, prompt_vocab_size, ite,
                sequence_limit_lengths, stop_words_data, bad_words_data,
                output_sequence_lengths, return_dict, encoder_output,
                encoder_input_lengths, stopping_criteria, logits_processor,
                cross_attention_mask, **kwargs)
        else:
            return self.decode_regular(
                batch_size, scfg, sequence_lengths, context_lengths,
                host_context_lengths, max_context_length, beam_width,
                cache_indirections, input_ids, hidden_states,
                prompt_embedding_table, tasks, prompt_vocab_size, ite,
                sequence_limit_lengths, stop_words_data, bad_words_data,
                output_sequence_lengths, return_dict, encoder_output,
                encoder_input_lengths, stopping_criteria, logits_processor,
                cross_attention_mask, **kwargs)
