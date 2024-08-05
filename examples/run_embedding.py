# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import csv
import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from data.sample_builder import EvaluationSampleBuilder
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

import transformers

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behavior'
    )
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='engine_outputs')
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["Born in north-east France, Soyer trained as a"])
    parser.add_argument(
        '--no_prompt_template',
        dest='use_prompt_template',
        default=True,
        action='store_false',
        help=
        "Whether or not to use default prompt template to wrap the input text.")
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)

    parser.add_argument('--output_log_probs_npy',
                        type=str,
                        help='Numpy file where the log_probs are stored',
                        default=None)

    parser.add_argument('--output_cum_log_probs_npy',
                        type=str,
                        help='Numpy file where the cum_log_probs are stored',
                        default=None)

    parser.add_argument('--tokenizer_dir',
                        help="HF tokenizer config path",
                        default='gpt2')
    parser.add_argument(
        '--tokenizer_type',
        help=
        'Specify that argument when providing a .model file as the tokenizer_dir. '
        'It allows AutoTokenizer to instantiate the correct tokenizer type.')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams > 1",
                        default=1)
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument('--streaming', default=False, action='store_true')
    parser.add_argument('--streaming_interval',
                        type=int,
                        help="How often to return tokens when streaming.",
                        default=5)
    parser.add_argument(
        '--prompt_table_path',
        type=str,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--prompt_tasks',
        help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument('--lora_dir',
                        type=str,
                        default=None,
                        nargs="+",
                        help="The directory of LoRA weights")
    parser.add_argument(
        '--lora_task_uids',
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")
    parser.add_argument(
        '--num_prepend_vtokens',
        nargs="+",
        type=int,
        help="Number of (default) virtual tokens to prepend to each sentence."
        " For example, '--num_prepend_vtokens=10' will prepend the tokens"
        " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")
    parser.add_argument(
        '--run_profiling',
        default=False,
        action='store_true',
        help="Run several 10 iterations to profile the inference latencies.")
    parser.add_argument(
        '--medusa_choices',
        type=str,
        default=None,
        help="Medusa choice to use, if not none, will use Medusa decoding."
        "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
    )

    return parser.parse_args(args=args)


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    model_name, model_version = read_model_name(args.engine_dir)
    if args.tokenizer_dir is None:
        logger.warning(
            "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
        )
        args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[model_name]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_dir
    )
    pad_id = tokenizer.eos_token_id
    end_id = tokenizer.eos_token_id

    if not PYTHON_BINDINGS and not args.use_py_session:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        args.use_py_session = True
    if args.debug_mode and not args.use_py_session:
        logger.warning(
            "Debug mode is not supported in C++ session for now, fallback to Python session."
        )
        args.use_py_session = True
    runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
    runner_kwargs = dict(engine_dir=args.engine_dir,
                         lora_dir=args.lora_dir,
                         rank=runtime_rank,
                         debug_mode=args.debug_mode,
                         lora_ckpt_source=args.lora_ckpt_source,
                         is_embedding=True)
    if args.medusa_choices is not None:
        args.medusa_choices = ast.literal_eval(args.medusa_choices)
        assert args.use_py_session, "Medusa is only supported by py_session"
        assert args.temperature == 0, "Medusa should use temperature == 0"
        assert args.num_beams == 1, "Medusa should use num_beams == 1"
        runner_kwargs.update(medusa_choices=args.medusa_choices)
    if not args.use_py_session:
        runner_kwargs.update(
            max_batch_size=8,  # TODO(ilyaonoff)
            max_input_len=8182,  # TODO(ilyaonoff)
            max_output_len=args.max_output_len,
            max_beam_width=args.num_beams,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
        )
    runner = runner_cls.from_dir(**runner_kwargs)

    samples = []
    with open(args.input_file) as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            samples.append(sample)

    sample_builder = EvaluationSampleBuilder(tokenizer)

    data = []
    for sample in samples:
        data.append(sample_builder(sample))

    with torch.no_grad():
        all_probs = []
        all_scores = []
        all_pred = []
        all_true = []

        for start_index in tqdm(range(0, len(data), 8)):  # TODO(ilyaonoff)
            batch_data = data[start_index:min(start_index + 8, len(data))]
            batch_output = {}
            for key in ['query_pos_input_ids', 'query_neg_input_ids', 'document_input_ids']:
                batch_input_ids = [d[key] for d in batch_data]

                outputs = runner.embed(
                    batch_input_ids,
                    max_attention_window_size=args.max_attention_window_size,
                    sink_token_length=args.sink_token_length,
                    end_id=end_id,
                    pad_id=pad_id,
                    lora_uids=None, # ["0"] * len(batch_data),  # TODO(ilyaonoff)
                    prompt_table_path=args.prompt_table_path,
                    prompt_tasks=args.prompt_tasks,
                    # output_sequence_lengths=True,
                    # return_dict=True
                )
                batch_output[key[:-len('_input_ids')]] = torch.nn.functional.normalize(outputs['embeddings_outputs'], dim=1)

            pos_scores = torch.sum(batch_output['document'] * batch_output['query_pos'], dim=1, keepdim=True)
            neg_scores = torch.sum(batch_output['document'] * batch_output['query_neg'], dim=1, keepdim=True)
            scores = torch.cat([neg_scores, pos_scores], dim=1)

            batch_probs = torch.nn.functional.softmax(scores, dim=1).cpu().tolist()
            batch_pred = torch.argmax(scores, dim=1).cpu().tolist()
            batch_scores = scores.cpu().tolist()

            print(batch_scores, [d['label'] for d in batch_data])

            all_probs.extend(batch_probs)
            all_scores.extend(batch_scores)
            all_pred.extend(batch_pred)
            all_true.extend([d['label'] for d in batch_data])

        all_probs = np.array(all_probs)
        all_scores = np.array(all_scores)
        all_pred = np.array(all_pred)
        all_true = np.array(all_true)

        metrics = {
            'ROC AUC': roc_auc_score(all_true, all_probs[:, 1]),
            'ACCURACY': np.sum(all_pred == all_true) / all_true.shape[0]
        }

        print(metrics)
        # torch.cuda.synchronize()

    print(outputs['embeddings_outputs'].shape)

    # if args.streaming:
    #     for curr_outputs in throttle_generator(outputs,
    #                                            args.streaming_interval):
    #         if runtime_rank == 0:
    #             output_ids = curr_outputs['output_ids']
    #             sequence_lengths = curr_outputs['sequence_lengths']
    #             cum_log_probs = None
    #             log_probs = None
    #             if args.output_cum_log_probs_npy != None:
    #                 cum_log_probs = outputs['cum_log_probs']
    #             if args.output_log_probs_npy != None:
    #                 log_probs = outputs['log_probs']
    #             print_output(
    #                 tokenizer,
    #                 output_ids,
    #                 input_lengths,
    #                 sequence_lengths,
    #                 output_csv=args.output_csv,
    #                 output_npy=args.output_npy,
    #                 cum_log_probs=cum_log_probs,
    #                 log_probs=log_probs,
    #                 output_cum_log_probs_npy=args.output_cum_log_probs_npy,
    #                 output_log_probs_npy=args.output_log_probs_npy)
    # else:
    #     if runtime_rank == 0:
    #         output_ids = outputs['output_ids']
    #         sequence_lengths = outputs['sequence_lengths']
    #         context_logits = None
    #         generation_logits = None
    #         cum_log_probs = None
    #         log_probs = None
    #         if runner.gather_context_logits:
    #             context_logits = outputs['context_logits']
    #         if runner.gather_generation_logits:
    #             generation_logits = outputs['generation_logits']
    #         if args.output_cum_log_probs_npy != None:
    #             cum_log_probs = outputs['cum_log_probs']
    #         if args.output_log_probs_npy != None:
    #             log_probs = outputs['log_probs']
    #         print_output(tokenizer,
    #                      output_ids,
    #                      input_lengths,
    #                      sequence_lengths,
    #                      output_csv=args.output_csv,
    #                      output_npy=args.output_npy,
    #                      context_logits=context_logits,
    #                      generation_logits=generation_logits,
    #                      output_logits_npy=args.output_logits_npy,
    #                      cum_log_probs=cum_log_probs,
    #                      log_probs=log_probs,
    #                      output_cum_log_probs_npy=args.output_cum_log_probs_npy,
    #                      output_log_probs_npy=args.output_log_probs_npy)

    # if args.run_profiling:
    #     ite = 10
    #     # warmup
    #     for _ in range(ite):
    #         with torch.no_grad():
    #             outputs = runner.generate(
    #                 batch_input_ids,
    #                 max_new_tokens=args.max_output_len,
    #                 max_attention_window_size=args.max_attention_window_size,
    #                 end_id=end_id,
    #                 pad_id=pad_id,
    #                 temperature=args.temperature,
    #                 top_k=args.top_k,
    #                 top_p=args.top_p,
    #                 num_beams=args.num_beams,
    #                 length_penalty=args.length_penalty,
    #                 early_stopping=args.early_stopping,
    #                 repetition_penalty=args.repetition_penalty,
    #                 presence_penalty=args.presence_penalty,
    #                 frequency_penalty=args.frequency_penalty,
    #                 stop_words_list=stop_words_list,
    #                 bad_words_list=bad_words_list,
    #                 lora_uids=args.lora_task_uids,
    #                 prompt_table_path=args.prompt_table_path,
    #                 prompt_tasks=args.prompt_tasks,
    #                 streaming=args.streaming,
    #                 output_sequence_lengths=True,
    #                 return_dict=True)
    #             torch.cuda.synchronize()

    #     tensorrt_llm.profiler.start("tmp")
    #     for _ in range(ite):
    #         with torch.no_grad():
    #             outputs = runner.generate(
    #                 batch_input_ids,
    #                 max_new_tokens=args.max_output_len,
    #                 max_attention_window_size=args.max_attention_window_size,
    #                 end_id=end_id,
    #                 pad_id=pad_id,
    #                 temperature=args.temperature,
    #                 top_k=args.top_k,
    #                 top_p=args.top_p,
    #                 num_beams=args.num_beams,
    #                 length_penalty=args.length_penalty,
    #                 early_stopping=args.early_stopping,
    #                 repetition_penalty=args.repetition_penalty,
    #                 presence_penalty=args.presence_penalty,
    #                 frequency_penalty=args.frequency_penalty,
    #                 stop_words_list=stop_words_list,
    #                 bad_words_list=bad_words_list,
    #                 lora_uids=args.lora_task_uids,
    #                 prompt_table_path=args.prompt_table_path,
    #                 prompt_tasks=args.prompt_tasks,
    #                 streaming=args.streaming,
    #                 output_sequence_lengths=True,
    #                 return_dict=True)
    #             torch.cuda.synchronize()
    #     tensorrt_llm.profiler.stop("tmp")

    #     print(
    #         f"batch_size: {len(batch_input_ids)}, avg latency of {ite} iterations: : {tensorrt_llm.profiler.elapsed_time_in_sec('tmp') / ite} sec"
    #     )


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
