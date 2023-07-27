import os
import re
import argparse
import time

import torch
import pytest

from einops import rearrange

from transformers import GPT2Config, GPT2Tokenizer, OPTConfig, AutoTokenizer

# from flash_attn.models.gpt import GPTLMHeadModel
from src.models.gpt_sparse import GPTLMHeadModel
from flash_attn.models.opt import remap_state_dict_opt, opt_config_to_gpt2_config
from flash_attn.utils.pretrained import state_dict_from_pretrained
from flash_attn.utils.generation import update_graph_cache


parser = argparse.ArgumentParser(description='OPT generation benchmarking')
parser.add_argument('--promptlen', type=int, default=128)
parser.add_argument('--genlen', type=int, default=100)
parser.add_argument('--attn-density', type=float, default=1.0)
parser.add_argument('--mlp-density', type=float, default=1.0)
args = parser.parse_args()


os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
torch.distributed.init_process_group(backend='nccl', init_method='env://')
device = f'cuda:{torch.distributed.get_rank()}'
world_size = torch.distributed.get_world_size()
# Need this, otherwise when we capture the graph the process for GPU 1 would run on both
# GPU0 and GPU1 and things would hang
torch.cuda.set_device(device)

repeats = 3
dtype = torch.float16
device = 'cuda'
rtol, atol = 3e-3, 3e-1
fused_ft_kernel = True
config = OPTConfig.from_pretrained("facebook/opt-66b")
config.hidden_size = 12 * 1024
config.word_embed_proj_dim = config.hidden_size
config.ffn_dim = 4 * config.hidden_size
config.num_attention_heads = 96
config.num_hidden_layers = 96
# config.num_hidden_layers = 48
config = opt_config_to_gpt2_config(config)
# Only prenorm supports residual_in_fp32
# config.residual_in_fp32 = getattr(config, 'prenorm', True)
config.use_flash_attn = True
config.fused_bias_fc = True
config.fused_mlp = True
# config.fused_dropout_add_ln = True
config.pad_vocab_size_multiple = 8 * world_size
config.sequence_parallel = False  # Need to set this to False for generation

from apex.transformer import parallel_state
parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
rank = parallel_state.get_tensor_model_parallel_rank()
process_group = parallel_state.get_tensor_model_parallel_group()

model = GPTLMHeadModel(config, device=device, dtype=dtype, process_group=process_group)
model.eval()

if args.attn_density < 1.0:
    num_active_heads = int(args.attn_density * config.num_attention_heads / world_size)
    # from flash_attn.modules.mha import ParallelMHA
    from src.models.modules.mha_sparse import ParallelMHA
    for module in model.modules():
        if isinstance(module, ParallelMHA):
            module.num_active_heads = num_active_heads

if args.mlp_density < 1.0:
    num_active_coordinates = int(args.mlp_density * config.n_inner / world_size)
    # from flash_attn.ops.fused_dense import ParallelFusedMLP
    from src.ops.fused_dense_sparse import ParallelFusedMLP
    for module in model.modules():
        if isinstance(module, ParallelFusedMLP):
            module.num_active_coordinates = num_active_coordinates

print('Num params', sum(p.numel() for p in model.parameters()) * world_size)

torch.manual_seed(0)
# OPT tokenizer requires use_fast=False
# https://huggingface.co/docs/transformers/model_doc/opt
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b", use_fast=False)

input_ids = torch.randint(0, 100, (1, args.promptlen), dtype=torch.long, device='cuda')
max_length = input_ids.shape[1] + args.genlen
# input_ids = tokenizer("Hello, my dog is cute and",
#                        return_tensors="pt").input_ids.to(device=device)
# max_length = 60

from torch.profiler import profile, record_function, ProfilerActivity


print('Without CUDA graph')
fn = lambda: model.generate(input_ids=input_ids, max_length=max_length, fused_ft_kernel=True,
                       vocab_size=config.vocab_size, return_dict_in_generate=True, output_scores=True,
                       timing=False)

fn()
torch.cuda.synchronize()
torch.distributed.barrier()
start = time.time()
for _ in range(repeats):
    out = model.generate(input_ids=input_ids, max_length=max_length, fused_ft_kernel=True,
                         vocab_size=config.vocab_size, return_dict_in_generate=True, output_scores=True,
                         timing=False)
    # with profile(activities=[ProfilerActivity.CUDA]) as prof:
    # fn()
torch.cuda.synchronize()
# if rank == 0:
#     prof.export_chrome_trace(f"opt-175b_generation_tp{world_size}.json")
print(f'Prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms')
if rank == 0:
    print(tokenizer.batch_decode(out.sequences.tolist()))

# Capture graph outside the timing loop
batch_size, seqlen_og = input_ids.shape
# We need to pass tensor_parallel here, otherwise the kv_cache will have the wrong shape
model._decoding_cache = update_graph_cache(
    model, None, batch_size, seqlen_og, max_length, tensor_parallel=world_size
)

# model._decoding_cache = update_graph_cache(
#     model, None, batch_size, seqlen_og, max_length, tensor_parallel=world_size, n_warmups=11
# )

print('With CUDA graph')
torch.cuda.synchronize()
torch.distributed.barrier()
start = time.time()
for _ in range(repeats):
    out_cg = model.generate(input_ids=input_ids, max_length=max_length, fused_ft_kernel=True, cg=True,
                            vocab_size=config.vocab_size, return_dict_in_generate=True,
                            output_scores=True, timing=False)
torch.cuda.synchronize()
print(f'Prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms')
# print(tokenizer.batch_decode(out_cg.sequences.tolist()))
if rank == 0:
    print(tokenizer.batch_decode(out_cg.sequences.tolist()))

# If we don't delete the cache, it would hang and not exit. Maybe because the CUDA graph is still
# around and the NCCL connection is not closed?
del model._decoding_cache
