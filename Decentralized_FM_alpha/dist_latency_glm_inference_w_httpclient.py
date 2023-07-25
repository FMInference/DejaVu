import sys
sys.path.append("../GLM-130B")
from quantization import quantize

import argparse
import time
import os
import torch
import stat
import re
from functools import partial
from typing import List, Tuple
import torch.nn.functional as F
import numpy as np
from SwissArmyTransformer import mpu
from SwissArmyTransformer.generation.autoregressive_sampling import update_mems, get_masks_and_position_ids_default
from SwissArmyTransformer.mpu import vocab_parallel_cross_entropy
from SwissArmyTransformer.generation.utils import timed_name, generate_continually
# from SwissArmyTransformer.generation.sampling_strategies.base_strategy import top_k_logits
from SwissArmyTransformer import get_args, get_tokenizer
from SwissArmyTransformer.arguments import initialize_distributed
from SwissArmyTransformer.training import load_checkpoint
from SwissArmyTransformer.model import GLM130B
from coordinator.coordinator_client import LocalCoordinatorClient
import traceback
from loguru import logger
import torch.distributed as dist
from time import sleep
from datetime import datetime
import requests
import math
import random


debug_print=False


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-65504):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
    #    # Remove all tokens with a probability less than the last token of the top-k
        if debug_print and dist.get_rank() == 0:
            print(f"<top_k_logits>: Nan? {torch.isnan(torch.topk(logits, top_k)[0][..., -1, None]).any()}")
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        if debug_print and dist.get_rank() == 0:
            print(f"<top_k_logits> indices_to_remove: {indices_to_remove}")
        logits[indices_to_remove] = filter_value
        if debug_print and  dist.get_rank() == 0:
            print(f"<top_k_logits> top_k handled! ")

    elif top_p > 0.0:
        batch_size = logits.shape[0]
        for i in range(batch_size):
            # convert to 1D
            current_logits = logits[i].view(-1).contiguous()
            sorted_logits, sorted_indices = torch.sort(current_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1).clamp(0, 1).nan_to_num(), dim=-1)
            if debug_print and dist.get_rank() == 0:
                print(f"<top_k_logits> cumulative_probs: {torch.isnan(cumulative_probs).any()}")

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            if debug_print and dist.get_rank() == 0:
                print(f"<top_k_logits> sorted_indices_to_remove1: {torch.isnan(sorted_indices_to_remove).any()}, {sorted_indices_to_remove.shape}")
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            if debug_print and dist.get_rank() == 0:
                print(f"<top_k_logits> sorted_indices_to_remove2: {torch.isnan(sorted_indices_to_remove).any()}, {sorted_indices_to_remove.shape}")
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            if debug_print and dist.get_rank() == 0:
                print(f"<top_k_logits> indices_to_remove: {torch.isnan(indices_to_remove).any()}, {indices_to_remove.shape}, {torch.max(indices_to_remove)}")
            logits[i, indices_to_remove] = filter_value
        if debug_print and dist.get_rank() == 0:
            print(f"<top_k_logits> top_k handled! ")

    return logits


class BaseStrategy:
    def __init__(self, batch_size, invalid_slices=[], temperature=1., top_k=200, eps=1e-4, top_p=0.0, end_tokens=None):
        self.batch_size = batch_size
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = top_k
        self.top_p = top_p
        self.eps = eps
        if end_tokens is None:
            end_tokens = []
        self.end_tokens = end_tokens
        self._is_done = np.zeros(self.batch_size, dtype=np.bool_)

    @property
    def is_done(self) -> bool:
        return self._is_done.all()

    def forward(self, logits, tokens, mems, temperature=None):
        # if dist.get_rank() == 0:
        #    print(f"<BaseStrategy.forward>1 logits {logits.shape}, tokens {tokens.shape}, mems {mems.shape}")
        logits = logits.view(-1, logits.size(-1))
        # if dist.get_rank() == 0:
        #    print(f"<BaseStrategy.forward>2 logits {logits.shape}")
        batch_size = tokens.shape[0]
        if temperature is None:
            temperature = self.temperature
        
        if debug_print and dist.get_rank() == 0:
            print(f"<BaseStrategy.forward>2 temperature: {temperature}, topk: {self.topk}, top_p: {self.top_p}")
        
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
            
        if debug_print and dist.get_rank() == 0:
            print(f"<BaseStrategy.forward>3 logits {logits.shape}")
        
        logits = logits.float().nan_to_num()
        logits = top_k_logits(logits, self.topk, self.top_p)
        
        if debug_print and dist.get_rank() == 0:
            print(f"<BaseStrategy.forward>4 logits {logits.shape},  is there Nan: {torch.isnan(logits).any()}")
        
        probs = F.softmax(logits, dim=-1)  # float is essetial, due to a bug in Pytorch
        probs = probs.clamp(0, 1).nan_to_num()
        
        if debug_print and dist.get_rank() == 0:
            print(f"<BaseStrategy.forward>5 logits {probs.shape}, is there Nan: {torch.isnan(probs).any()}")
        
        pred = torch.multinomial(probs, num_samples=1)
        if debug_print and dist.get_rank() == 0:
            print(f"<BaseStrategy.forward>6 logits {pred.shape}")
        for i in range(self.batch_size):
            if i >= batch_size:
                self._is_done[i] = True
            elif self._is_done[i]:
                pred[i] = -1
            elif pred[i].item() in self.end_tokens:
                self._is_done[i] = True
        tokens = torch.cat((tokens, pred.view(tokens.shape[:-1] + (1,))), dim=-1)
        return tokens, mems

    def finalize(self, tokens, mems):
        self._is_done = np.zeros(self.batch_size, dtype=np.bool_)
        return tokens, mems


def batch_filling_sequence(
        model,
        seqs,
        context_lengths,
        strategy,
        max_memory_length=100000,
        get_masks_and_position_ids=get_masks_and_position_ids_default,
        mems=None,
        get_last_layer_embedding=False,
        get_logprobs=0,
        **kw_args
):
    # print("<batch_filling_sequence> I am here 1")
    # if dist.get_rank() == 0:
    #    print(f"<batch_filling_sequence> seqs: {seqs}")
    assert len(seqs.shape) == 2
    # building the initial tokens, attention_mask, and position_ids
    batch_size, context_length = seqs.shape

    seqs, attention_mask, position_ids = get_masks_and_position_ids(seqs)
    tokens = seqs[..., :context_length]
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.type_as(next(model.parameters()))  # if fp16
    # initialize generation
    counter = context_length - 1  # Last fixed index is ``counter''
    index = 0 if mems is None else mems.shape[2]  # Next forward starting index, also the length of cache.
    num_beams = 1
    # step-by-step generation
    # print("<batch_filling_sequence> I am here 2")
    output_embedding = None
    logprobs = {'topk_indices': [], 'topk_logprobs': []}
    while counter < seqs.shape[1] - 1:
        if dist.get_rank() == 0:
            print(f"<batch_filling_sequence> counter:{counter}/{seqs.shape[1] - 1}")
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.
        # forward
        tokens = tokens.reshape(batch_size * num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size * num_beams, mems.shape[-2],
                            mems.shape[-1]) if mems is not None else None

        output_embedding_flag = get_last_layer_embedding and counter == context_length - 1
        logits, *output_per_layers = model(
            tokens[:, index:],
            position_ids[..., index: counter + 1],
            attention_mask[..., index: counter + 1, :counter + 1],  # TODO memlen
            mems=mems,
            output_hidden_states=output_embedding_flag
        )
        if output_embedding_flag:
            output_embedding = output_per_layers[-1]['hidden_states'].detach()
        mem_kv = [o['mem_kv'] for o in output_per_layers]
        mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
        if counter == context_length - 1:
            logits = logits[torch.arange(batch_size), context_lengths - 1]
        else:
            logits = logits[:, -1]
        counter += 1
        index = counter
        # if torch.distributed.get_rank() == 0:
        #     print(f"counter: {counter}: logits: {logits.float().abs().mean()}")
        # sampling
        logits = logits.reshape(batch_size, num_beams, -1)
        tokens = tokens.reshape(batch_size, num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size, num_beams, mems.shape[-2], mems.shape[-1])
        tokens, mems = strategy.forward(logits, tokens, mems)

        if debug_print and dist.get_rank() == 0:
            print(f"<batch_filling_sequence> logits.shape: {logits.shape}")

        if get_logprobs != 0:  # TODO: currently this encoding assumes num_completion=1
            logits = F.log_softmax(logits.detach(), -1)
            current_topk_logprobs, current_topk_indices = logits.topk(1 + get_logprobs, dim=-1)
            if debug_print and dist.get_rank() == 0:
                print(f"<batch_filling_sequence> current_topk_logprobs.shape: {current_topk_logprobs.shape}")
                print(f"<batch_filling_sequence> current_topk_indices.shape: {current_topk_indices.shape}")

            logprobs['topk_logprobs'].append(current_topk_logprobs)
            logprobs['topk_indices'].append(current_topk_indices)

        if len(tokens.shape) == 3 and num_beams == 1:
            num_beams = tokens.shape[1]
            position_ids = position_ids.unsqueeze(1).expand(batch_size, num_beams, -1).reshape(batch_size * num_beams,
                                                                                               -1)
            attention_mask_shape = attention_mask.shape[-3:]
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1, -1, -1).reshape(
                batch_size * num_beams, *attention_mask_shape)
        if strategy.is_done:
            break

    tokens, mems = strategy.finalize(tokens, mems)
    results = {}
    results['tokens'] = tokens
    results['mems'] = mems
    if get_last_layer_embedding:
        results['output_embedding'] = output_embedding
    if get_logprobs != 0:
        results['logprobs'] = logprobs
    return results


def add_bminf_args(parser):
    """Arguments for BMInf"""
    group = parser.add_argument_group("BMInf")

    group.add_argument("--bminf", action="store_true", help="Use BMInf to support low resource evaluation")
    group.add_argument("--bminf-memory-limit", type=int, default=20, help="Max memory for model per GPU (in GB)")
    return parser


def add_quantization_args(parser):
    group = parser.add_argument_group("Quantization")

    group.add_argument("--quantization-bit-width", type=int, default=None)
    group.add_argument("--from-quantized-checkpoint", action="store_true", help="Loading from a quantized checkpoint")


def foo_port_add_coordinator_args(parser):
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--working-directory', type=str,
                        default='/root/fm/working_dir', metavar='S',
                        help='Working_dir for file cache.')


def initialize(extra_args_provider):
    parser = argparse.ArgumentParser(add_help=False)
    add_bminf_args(parser)
    add_quantization_args(parser)
    GLM130B.add_model_specific_args(parser)
    extra_args_provider(parser)
    foo_port_add_coordinator_args(parser)
    known, args_list = parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.do_train = False
    initialize_distributed(args)
    return args


def initialize_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)

    # Initialize model
    model = GLM130B(args).half()

    # Load checkpoint
    torch.distributed.barrier()
    start = time.time()
    
    if args.from_quantized_checkpoint:
        assert args.quantization_bit_width is not None
        # Quantize model before moving to GPU
        model = quantize(model, args.quantization_bit_width)
                
    load_checkpoint(model, args)
    
    if args.quantization_bit_width is not None and not args.from_quantized_checkpoint:
        # Quantize model before moving to GPU
        model = quantize(model, args.quantization_bit_width)
    
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(f"> Checkpoint loaded in {time.time() - start:.1f}s")

    if args.bminf:
        import bminf

        if torch.distributed.get_rank() == 0:
            print(f"> BMInf activated, memory limit: {args.bminf_memory_limit} GB")
        with torch.cuda.device(args.device):
            model = bminf.wrapper(model, quantization=False, memory_limit=args.bminf_memory_limit << 30)
    else:
        model = model.to(args.device)

    torch.cuda.empty_cache()
    model.eval()

    # generate rotary embedding cache
    original_parallel_output = model.transformer.parallel_output
    model.transformer.parallel_output = True
    with torch.no_grad():
        _, *_ = model(
            torch.ones(1, args.max_sequence_length, device=torch.cuda.current_device(), dtype=torch.int64),
            torch.arange(args.max_sequence_length, device=torch.cuda.current_device(), dtype=torch.int64).view(1, -1),
            torch.randn(
                1,
                1,
                args.max_sequence_length,
                args.max_sequence_length,
                device=torch.cuda.current_device(),
            )
            < 0.5,
        )
    model.transformer.parallel_output = original_parallel_output
    torch.distributed.barrier()

    return model, tokenizer


def add_generation_specific_args(parser):
    parser.add_argument("--sampling-strategy", type=str, default="BaseStrategy", help="Type of sampling strategy.")
    parser.add_argument("--min-gen-length", type=int, default=0, help="The minimum length each blank should generate.")
    parser.add_argument(
        "--print-all-beams", action="store_true", help="Print all output generated by beam search strategy."
    )


def isEnglish(s):
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_masks_and_position_ids(seq, mask_position, max_gen_length, gmask=False):
    context_length = seq.shape[1]
    tokens = torch.nn.functional.pad(seq, (0, max_gen_length), mode='constant', value=-1)
    attention_mask = torch.ones((1, tokens.shape[-1], tokens.shape[-1]), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., : context_length - 1] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    position_ids = torch.arange(tokens.shape[-1], dtype=torch.long, device=tokens.device)
    if not gmask:
        position_ids[context_length - 1:] = mask_position

    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids


def get_masks_and_position_ids_batch(seqs, mask_position, max_gen_length, pad_pos, gmask=False):
    batch_size = seqs.shape[0]
    context_length = seqs.shape[1]
    tokens = torch.nn.functional.pad(seqs, (0, max_gen_length), mode='constant', value=-1)
    # TODO This might be wrong, double check.
    attention_mask = torch.ones((batch_size, tokens.shape[-1], tokens.shape[-1]), device=tokens.device)
    attention_mask.tril_()

    for i in range(batch_size):
        attention_mask[i, :, 0:pad_pos[i]] = 0
        attention_mask[i, :, pad_pos[i]: context_length - 1] = 1

    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    position_ids = torch.zeros((batch_size, tokens.shape[-1]), dtype=torch.long, device=tokens.device)

    for i in range(batch_size):
        position_ids[i] = torch.arange(tokens.shape[-1], dtype=torch.long, device=tokens.device) - pad_pos[i]
        position_ids[i, 0:pad_pos[i]] = 0

    if not gmask:
        position_ids[:, context_length - 1:] = mask_position

    # if dist.get_rank() == 0:
    #    print(f"<get_masks_and_position_ids_batch> tokens: {tokens}")
    #    print(f"<get_masks_and_position_ids_batch> attention_mask: {attention_mask}")
    #    print(f"<get_masks_and_position_ids_batch> position_ids: {position_ids}")

    return tokens, attention_mask, position_ids


def fill_blanks_efficient(raw_texts: str, model, tokenizer, strategy, config=None):
    seqs = []
    generation_mask = "[gMASK]"
    use_gmask = True
    last_layer_embedding = None
    result_logprobs = None
    for raw_text in raw_texts:
        mask_pattern = r"\[g?MASK\]"
        text_list = re.split(mask_pattern, raw_text)
        pattern_list = re.compile(mask_pattern).findall(raw_text)
        seq = []
        for i in range(len(pattern_list)):
            pattern = pattern_list[i]
            sub_text = text_list[i]
            seq.extend(tokenizer.tokenize(sub_text))
            seq.append(tokenizer.get_command(pattern))

        seq.extend(tokenizer.tokenize(text_list[-1]))

        if "MASK]" not in raw_text:
            seq += [tokenizer.get_command(generation_mask)]
            raw_text += " " + generation_mask
        if not raw_text.endswith("MASK]"):
            seq = seq + [tokenizer.get_command("eos")]
        if mpu.get_model_parallel_rank() == 0:
            print("\nInput: {}\n".format(raw_text))
        if config is None and len(seq) > args.max_sequence_length:
            raise ValueError("text too long.")
        seqs.append(seq)

    # for ii in range(len(seqs)-1):
    #    assert len(seqs[ii]) == len(seqs[ii+1])

    # generation
    num_output = args.num_beams if args.sampling_strategy == "BeamSearchStrategy" else 1

    # print("<fill_blanks> I am here 1")
    # continually detect the first mark position

    # if dist.get_rank() == 0:
    #    print(f"<fill_blanks_efficient> seqs :{seqs}")

    # detect mask position
    mask_token = tokenizer.get_command(generation_mask)

    batch_size = len(seqs)
    context_length = 0
    for seq in seqs:
        if len(seq) > context_length:
            context_length = len(seq)
    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> batch_size: {batch_size}, context_length: {context_length}")

    padding_pos = []
    for seq in seqs:
        padding_pos.append(context_length - len(seq))

    # if dist.get_rank() == 0:
    #    print(f"<fill_blanks_efficient> padding_pos {padding_pos}")

    input_seqs = torch.cuda.LongTensor(
        [[0] * (context_length - len(seq)) + seq + [tokenizer.get_command("sop")] for seq in seqs],
        device=args.device,
    )

    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> input_seqs.shape :{input_seqs.shape}")
    mask_position = context_length - 1

    # if dist.get_rank() == 0:
    #    print(f"<fill_blanks_efficient> input_seqs {input_seqs}")

    # print("<fill_blanks> I am here 2")
    if config is not None and config['prompt_embedding']:
        get_last_layer_embedding = True
    else:
        get_last_layer_embedding = False

    if config is not None and config['logprobs'] != 0:
        logprobs_n = config['logprobs']
    else:
        logprobs_n = 0

    results = batch_filling_sequence(
        model,
        input_seqs,
        torch.cuda.LongTensor([input_seqs.shape[-1] for _ in range(input_seqs.shape[0])], device=args.device),
        strategy=strategy,
        get_masks_and_position_ids=partial(
            get_masks_and_position_ids_batch,
            mask_position=mask_position,
            max_gen_length=config['max_tokens'] if config else args.out_seq_length - input_seqs.shape[-1],
            pad_pos=padding_pos,
            gmask=use_gmask,
        ),
        get_last_layer_embedding=get_last_layer_embedding,
        get_logprobs=logprobs_n
    )
    outputs = results['tokens']

    if get_last_layer_embedding:
        last_layer_embedding = results['output_embedding']

    if logprobs_n != 0:
        raw_top_indices = torch.cat(results['logprobs']['topk_indices'], dim=1)
        raw_top_logprobs = torch.cat(results['logprobs']['topk_logprobs'], dim=1)
        result_logprobs = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(raw_top_indices.shape[1]):
                current_sample_pos_prob = raw_top_logprobs[i, j, :].tolist()
                # current_sample_pos_tokens = tokenizer.convert_ids_to_tokens(raw_top_indices[i,j,:].tolist())
                current_sample_pos_tokens = [tokenizer.IdToToken(id) for id in raw_top_indices[i, j, :].tolist()]
                result_logprobs[i].append(list(zip(current_sample_pos_tokens, current_sample_pos_prob)))
        if dist.get_rank() == 0:
            print(f"<fill_blanks_efficient> result_logprobs: {result_logprobs}")

    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> outputs:{outputs.shape}")
    answers = []
    for i in range(outputs.shape[0]):
        answers_per_seq = []
        for j in range(num_output):
            output = outputs[i][j]
            if output[-1] == tokenizer.get_command("eos"):
                output = output[:-1]
            if dist.get_rank() == 0:
                print(f"<fill_blanks_efficient> output :{output.shape}")
            answers_per_seq.append(tokenizer.detokenize(output[padding_pos[i]:].tolist()))
        answers.append(answers_per_seq)
    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> answers: {answers}")

    if last_layer_embedding is not None:
        last_layer_embedding = torch.transpose(last_layer_embedding, 0, 1)
        last_layer_embeddings = []
        for i in range(batch_size):
            current_sample_embedding = last_layer_embedding[i, padding_pos[i]:, :]
            if dist.get_rank() == 0:
                print(f"<fill_blanks_efficient> current_sample_embedding_{i} .shape: {current_sample_embedding.shape}")
            last_layer_embeddings.append(current_sample_embedding)
    else:
        last_layer_embeddings = None

    return answers, last_layer_embeddings, result_logprobs


def post_processing_text(output_text, query, prompt_str_length):
    if query.get('max_tokens') == 0:
        return ""
    elif query.get('echo', False):
        text = output_text.replace("[[gMASK]][sop]", " ")
    else:
        text = output_text[prompt_str_length:].replace("[[gMASK]][sop]", "")
    end_pos = len(text)
    print(f"<post_processing_text>1 end_pos: {end_pos}.")

    # stop_tokens = []
    # if isinstance(query.get('stop_words', ""), str):
    #    if query.get('stop_words', "") != "":
    #        stop_tokens.extend(query.get('stop_words', "").split(";"))
    # stop_tokens.extend(query.get('stop', []))
    stop_tokens = []
    if query.get('stop', []) is not None:
        for token in query.get('stop', []):
            if token != '':
                stop_tokens.append(token)

    print(f"<post_processing_text> stop_tokens: {stop_tokens}.")

    for stop_token in stop_tokens:
        if query.get('echo', False):
            if text[prompt_str_length:].find(stop_token) != -1:
                end_pos = min(text[prompt_str_length:].find(stop_token) + len(stop_token), end_pos)
        else:
            if text.find(stop_token) != -1:
                end_pos = min(text.find(stop_token) + len(stop_token), end_pos)
        print(f"<post_processing_text>2 end_pos: {end_pos}.")

    print(f"<post_processing_text> text: {text}, end_pos: {end_pos}")
    post_processed_text = text[:end_pos]
    print(f"<post_processing_text> input: {output_text}")
    print(f"<post_processing_text> output: {post_processed_text}")
    
    post_processed_text = post_processed_text.replace("[gMASK]][sop]", "")
    return post_processed_text


def to_result(output, query, prompt_str_length, last_layer_embedding, top_logprobs, job_id=None,
              working_directory=None):
    print(f"<to_result> output: {output}")
    # TODO, Lots of missing attributes here!!!!
    '''
    if len(output) == 1:
        item={'choices':[],}
        choice = {
            "text": post_processing_text(output[0][0], query, prompt_str_length[0]),
            "index": 0,
            "finish_reason": "length"
        }
        if top_logprobs is not None:
            choice['logprobs'] = top_logprobs[0]
        if last_layer_embedding is not None:
            print(f"serialize last layer embedding, shape {last_layer_embedding} ")
            tensor_filename = working_directory+'/'+job_id+'_embedding.pt'
            torch.save(last_layer_embedding, tensor_filename)
            with open(tensor_filename, "rb") as fp:
                files = {"file": fp}
                res = requests.post("https://planetd.shift.ml/file", files=files).json()
                choice['embedding'] = res["filename"]
                os.remove(tensor_filename)
        item['choices'].append(choice)
        return item
    else:
    '''
    result = {}
    items = []
    if last_layer_embedding is not None:
        # last_layer_embedding = torch.transpose(last_layer_embedding, 0, 1)
        # print(f"serialize last layer embeddings {last_layer_embedding} ")
        tensor_filename = working_directory + '/' + job_id + '_embedding.pt'
        torch.save(last_layer_embedding, tensor_filename)
        with open(tensor_filename, "rb") as fp:
            files = {"file": fp}
            res = requests.post("https://planetd.shift.ml/file", files=files).json()
            result['embedding'] = res["filename"]
            os.remove(tensor_filename)

    for i in range(len(output)):
        item = {'choices': [], }
        print(f"<to_result> output{i}: {prompt_str_length[i]} / {len(output[i][0])}")
        choice = {
            "text": post_processing_text(output[i][0], query, prompt_str_length[i]),
            "index": 0,
            "finish_reason": "length"
        }
        if top_logprobs is not None:
            choice['token_logprobs'] = top_logprobs[i]
        item['choices'].append(choice)
        items.append(item)
    result['inference_result'] = items
    return result


def main(args):
    if dist.get_rank() == 0:
        print(args)

    local_cord_client = LocalCoordinatorClient(
        working_directory=args.working_directory,
        coordinator_url="http://localhost:5000/eth",
    )
    try:
        model, tokenizer = initialize_model_and_tokenizer(args)
        end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
        local_cord_client.update_status(args.job_id, "running")
    except Exception as e:
        print('Exception in model initialization inference:', e)
        error = traceback.format_exc()
        local_cord_client.update_status(args.job_id, "failed", returned_payload={"message": error})
        print(error)
        raise e

    glm_model_name = 'glm'
    if args.quantization_bit_width is not None:
        glm_model_name = glm_model_name + '-int' + str(args.quantization_bit_width)

    if dist.get_rank()==0:
        print("<Main> GLM name: ", glm_model_name)
    
    
    try:
        while True:
            try:
                has_work = False
                raw_text = ""
                config = {}
                if dist.get_rank() == 0:
                    instructions = local_cord_client.fetch_instructions(glm_model_name, 0)
                    # print(instructions)
                    last_instruction = instructions[-1]

                    if last_instruction["message"] == "break":
                        logger.info(f"Received stop instruction. <{glm_model_name}>")
                        logger.info("# BREAK ")
                        break
                    elif last_instruction["message"] == "continue":
                        logger.info(f"Received keep instruction. <{glm_model_name}>")
                        sleep(1)
                        has_work = False
                    elif last_instruction["message"] == "run":
                        fetched_tasks = [x for x in instructions
                                         if x["message"] == "run" and x['payload']['status'] == 'submitted']

                        if len(fetched_tasks) > 0:
                            instruction = fetched_tasks[0]
                            logger.info("Instruction:")
                            logger.info(str(instruction))
                            job_id = instruction['payload']['id']
                            print(f"Job <{job_id}> has been batched")
                            
                            with open(args.working_directory + "/job_logs.txt", "a") as fp_log:
                                fp_log.write(f"{datetime.now()}: {job_id} start.\n")

                            # TODO: we assume len(payload) is 1, right?
                            query = instruction['payload']['payload'][0]
                            if isinstance(query['prompt'], list):
                                raw_text = query['prompt']
                                for i in range(len(raw_text)):
                                    raw_text[i] = raw_text[i].strip()
                            elif isinstance(query['prompt'], str):
                                raw_text = query['prompt']
                                raw_text = [raw_text.strip()]
                            else:
                                print("wrong prompt format, it can only be str or list of str")
                                print(query['prompt'])

                            config = {
                                'seed': query.get('seed', None),
                                'temperature': query.get('temperature', 0.9),
                                'top_k': query.get('top_k', 50),
                                'top_p': query.get('top_p', 0),
                                'max_tokens': query.get('max_tokens', 10) if query.get('max_tokens', 10) > 0 else 1,
                                'prompt_embedding': query.get('prompt_embedding', False),
                                'logprobs': query.get('logprobs', 0)
                            }
                            has_work = True
                        else:
                            has_work = False

                torch.cuda.empty_cache()

                dist.barrier()
                if dist.get_rank() == 0:
                    dist.broadcast_object_list([raw_text, config, has_work])
                else:
                    info = [raw_text, config, has_work]
                    torch.distributed.broadcast_object_list(info)
                    raw_text, config, has_work = info
                dist.barrier()

                if has_work:
                    print(f"Rank-<{dist.get_rank()}> join inference.")
                    start_time = time.time()
                    # strategy = BaseStrategy(batch_size=1, temperature=args.temperature, top_k=args.top_k,
                    #                        top_p=args.top_p, end_tokens=end_tokens)
                    # Followed Jue's suggestion for temperature

                    batch_size = min(len(raw_text), 32)
                    num_iter = math.ceil(len(raw_text) / batch_size)
                    answers = []
                    last_layer_embedding = []
                    top_logprobs = []
                    if config['seed'] is not None:
                        torch.manual_seed(config['seed'])
                        np.random.seed(config['seed'])
                        random.seed(config['seed'])
                        # if debug_print:
                        print(f"<Main> Rank-<{dist.get_rank()}> setup random seed: {config['seed']}")

                    for iter_i in range(num_iter):
                        current_raw_text = raw_text[iter_i * batch_size: (iter_i + 1) * batch_size]
                        if config['temperature'] == 0:
                            strategy = BaseStrategy(batch_size=len(current_raw_text), temperature=1, top_k=1,
                                                    top_p=config['top_p'], end_tokens=end_tokens)
                        else:
                            strategy = BaseStrategy(batch_size=len(current_raw_text), temperature=config['temperature'],
                                                    top_k=config['top_k'], top_p=config['top_p'], end_tokens=end_tokens)

                        cur_answer, cur_last_layer_embedding, cur_top_logprobs = fill_blanks_efficient(current_raw_text,
                                                                                                       model, tokenizer,
                                                                                                       strategy, config)
                        answers.extend(cur_answer)
                        if cur_last_layer_embedding is None:
                            last_layer_embedding = None
                        else:
                            last_layer_embedding.extend(cur_last_layer_embedding)
                        if cur_top_logprobs is None:
                            top_logprobs = None
                        else:
                            top_logprobs.extend(cur_top_logprobs)
                        if dist.get_rank() == 0:
                            print(f"<Main> Current iter handled: {len(answers)}/{len(raw_text)}")

                    end_time = time.time()
                    # print(f"Rank-<{dist.get_rank()}>: answer:")
                    # print(answers)
                    if dist.get_rank() == 0:
                        print(f"Job-{job_id} GLM Inference takes {end_time - start_time}s")
                        prompt_str_lengths = []
                        for text in raw_text:
                            prompt_str_lengths.append(len(text))
                        result = to_result(answers, query, prompt_str_lengths, last_layer_embedding, top_logprobs,
                                           job_id=job_id,
                                           working_directory=args.working_directory)
                        return_payload = {
                            'request': query,
                            'result': result,
                            'raw_compute_time': end_time - start_time
                        }
                        # local_cord_client.update_status(
                        local_cord_client.update_status_global_coordinator(
                            job_id,
                            "finished",
                            returned_payload=return_payload
                        )
                        local_cord_client.update_status(job_id, "finished", returned_payload={})
                        with open(args.working_directory + "/job_logs.txt", "a") as fp_log:
                            fp_log.write(f"{datetime.now()}: {job_id} end.\n")

            except Exception as e:
                error = traceback.format_exc()
                if dist.get_rank() == 0:
                    local_cord_client.update_status(
                        job_id,
                        "failed",
                        returned_payload={"message": error}
                    )
                print(error)

    except Exception as e:
        error = traceback.format_exc()
        print(error)
        print('Exception in latency inference:', e)


if __name__ == "__main__":
    args = initialize(extra_args_provider=add_generation_specific_args)

    with torch.no_grad():
        main(args)
