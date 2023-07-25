#!/usr/bin/env python
# coding: utf-8
import torch
import intel_extension_for_pytorch as ipex


def main():

    dtype = torch.bfloat16
    # dtype = torch.bfloat16 if args.fp16 else torch.float32
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim).to(dtype)
    print("input type:", embedding.dtype)
    layer_norm = torch.nn.LayerNorm(embedding_dim, dtype=dtype).eval()
    print("layer norm weight type:", layer_norm.weight.data.dtype)
    layer_norm = layer_norm.to(memory_format=torch.channels_last)
    layer_norm = ipex.optimize(layer_norm, dtype=dtype)

    with torch.no_grad():
        with torch.autocast(device_type='cpu', dtype=dtype):
            output = layer_norm(embedding)
            print(output)



if __name__ == '__main__':
    main()


