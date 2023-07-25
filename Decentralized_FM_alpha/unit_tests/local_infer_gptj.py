#!/usr/bin/env python
# coding: utf-8

import torch
from time import time
from transformers import AutoTokenizer, GPTJForCausalLM, GPTJConfig
from deepspeed.profiling.flops_profiler import FlopsProfiler

def main():

    batch_size = 32
    prompt_length = 512
    generate_length = 32

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')

    config = GPTJConfig.from_pretrained('EleutherAI/gpt-j-6B')
    
    config.n_layer = 3

    model = GPTJForCausalLM(config).eval().half().cuda()

    tokenizer.pad_token = tokenizer.eos_token

    model.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer(['hello world']*batch_size, padding='max_length', max_length=prompt_length, return_tensors='pt')

    inputs = {
        k:v.cuda() for k,v in inputs.items()
    }

    model.generate(inputs['input_ids'], max_length=prompt_length+generate_length)
    tic = time()
    for _ in range(10):
        model.generate(inputs['input_ids'], max_length=prompt_length+generate_length)
    toc = time()
    avg_t = (toc - tic)/10

    prof = FlopsProfiler(model)

    prof.start_profile()

    inputs = tokenizer(['hello world']*batch_size, padding='max_length', max_length=prompt_length, return_tensors='pt')

    inputs = {
        k:v.cuda() for k,v in inputs.items()
    }

    model.generate(inputs['input_ids'], max_length=prompt_length+generate_length)

    prof.stop_profile()
    flops = prof.get_total_flops()
    params = prof.get_total_params()
    prof.print_model_profile()
    prof.end_profile()

    print(flops / 1e12 / avg_t, 'TFLOP/s')


if __name__ == '__main__':
    main()


