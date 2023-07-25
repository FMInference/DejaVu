#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
from time import time
# import intel_extension_for_pytorch as ipex


def _create_layers(args, dtype=torch.float16):
    if args.model_type == 'gptj':
        from modules.hf_gptj_module import GPTBlock
    else:
        raise Exception(f'unknown model type {args.model_type}')
    cpu_layers = []
    start_time = time()
    for layer_index in range(args.num_layers):
        print(f'loading layer {layer_index}')
        current_layer = GPTBlock.from_pretrained(args.model_name, layer_index=layer_index).to(dtype).eval()
        # current_layer = current_layer.to(memory_format=torch.channels_last)
        # current_layer = ipex.optimize(current_layer)
        cpu_layers.append(current_layer)
    end_time = time()
    print("Init model takes: {:3.2f}s".format(end_time-start_time))
    return cpu_layers


def _get_embedding_size(args):
    if args.model_type == 'gpt2':
        from modules.hf_gpt2_module import GPTConfig
        config = GPTConfig.from_pretrained(args.model_name)
        return config.n_embd
    elif args.model_type == 'gptj':
        from modules.hf_gptj_module import GPTConfig
        config = GPTConfig.from_pretrained(args.model_name)
        return config.n_embd
    else:
        raise Exception(f'unknown model type {args.model_type}')


def _get_num_heads(args):
    if args.model_type == 'gpt2':
        from modules.hf_gpt2_module import GPTConfig
        config = GPTConfig.from_pretrained(args.model_name)
        return config.n_head
    elif args.model_type == 'gptj':
        from modules.hf_gptj_module import GPTConfig
        config = GPTConfig.from_pretrained(args.model_name)
        return config.n_head
    else:
        raise Exception(f'unknown model type {args.model_type}')


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    parser.add_argument('--skip-prompt', action='store_true',
                        help='Skip the computation of prompt phase.')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
    parser.add_argument('--model-name', type=str, default='./pretrained_models/gpt-j-6B', metavar='S',
                        help='trained model path')
    parser.add_argument('--model-type', type=str, default='gptj', metavar='S',
                        help='trained model path')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--num-layers', type=int, default=3, metavar='N',
                        help='-')
    parser.add_argument('--prompt-seq-length', type=int, default=512, metavar='N',
                        help='-')
    parser.add_argument('--gen-seq-length', type=int, default=50, metavar='N',
                        help='-')
    args = parser.parse_args()

    assert args.fp16
    dtype = torch.bfloat16
    # dtype = torch.bfloat16 if args.fp16 else torch.float32
    model = _create_layers(args, dtype=dtype)
    emb_dim = _get_embedding_size(args)
    num_heads = _get_num_heads(args)
    head_dim = emb_dim // num_heads

    # inputs = torch.empty((args.batch_size, args.prompt_seq_length, 12288),
    #                     requires_grad=False, dtype=dtype).normal_(mean=0.1, std=0.2)
    inputs = torch.ones((args.batch_size, args.prompt_seq_length, emb_dim), requires_grad=False, dtype=dtype)
    # inputs = inputs.to(memory_format=torch.channels_last)

    if args.skip_prompt:
        cached_tuples = []
        fill_start_time = time()
        for i in range(args.num_layers):
            # cached_key = torch.empty((args.batch_size, 96, args.prompt_seq_length, 128),
            #                          requires_grad=False, dtype=dtype).normal_(mean=0.1, std=0.2)
            # cached_value = torch.empty((args.batch_size, 96, args.prompt_seq_length, 128),
            #                           requires_grad=False, dtype=dtype).normal_(mean=0.1, std=0.2)
            cached_key = torch.ones((args.batch_size, num_heads, args.prompt_seq_length, head_dim),
                                    requires_grad=False, dtype=dtype)
            cached_value=torch.ones((args.batch_size, num_heads, args.prompt_seq_length, head_dim),
                                    requires_grad=False, dtype=dtype)
            cached_tuples.append((cached_key, cached_value))
            print("Fill key value for layer <{}>".format(i))
        fill_end_time = time()
        print("Fill Key value takes {:3.2f}s".format(fill_end_time-fill_start_time))
    else:
        cached_tuples = [None for _ in range(args.num_layers)]
        with torch.no_grad():
            with torch.autocast(device_type='cpu', dtype=dtype):
                start_time = time()
                # prompt phase
                for layer_index in range(args.num_layers):
                    if layer_index == 0:
                        embeddings, cached_tuples[layer_index] = model[layer_index](inputs, skip_ln=False)
                    else:
                        embeddings, cached_tuples[layer_index] = model[layer_index](embeddings, skip_ln=False)
                    embeddings = embeddings.to(dtype)
                prompt_end_time = time()
                print("Prompt <{}> takes {:3.2f}s".format(args.prompt_seq_length, prompt_end_time-start_time))
                print("Shape of key:", cached_tuples[0][0].shape, "Shape of value:", cached_tuples[0][1].shape)

    with torch.no_grad():
        with torch.autocast(device_type='cpu', dtype=dtype):
            total_time = 0
            for i in range(args.gen_seq_length):
                inputs = torch.empty((args.batch_size, 1, emb_dim),
                                     requires_grad=False, dtype=dtype)
                # inputs = inputs.to(memory_format=torch.channels_last)
                token_start_time = time()
                embeddings = torch.zeros((args.batch_size, 1, emb_dim), dtype=dtype)
                # print(inputs.shape)
                for layer_index in range(args.num_layers):
                    if layer_index == 0:
                        embeddings, cached_tuples[layer_index] = model[layer_index](inputs, cached_tuples[layer_index],
                                                                                    skip_ln=False)
                    else:
                        embeddings, cached_tuples[layer_index] = model[layer_index](embeddings, cached_tuples[layer_index],
                                                                                    skip_ln=False)
                    embeddings = embeddings.to(dtype)
                    # print(embeddings.dtype)
                token_end_time = time()
                print("Token <{}> takes {:3.2f}s".format(i, token_end_time - token_start_time))
                if i > 1:
                    total_time += (token_end_time - token_start_time)
            avg_time = total_time/(args.gen_seq_length-2)
            print("Averaged token generate time: {:3.2f}s".format(avg_time))


if __name__ == '__main__':
    main()


