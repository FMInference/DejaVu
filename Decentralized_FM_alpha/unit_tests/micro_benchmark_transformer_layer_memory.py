import argparse
import time
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from modules.gpt_modules import GPTTransformerLayer
import GPUtil


def benchmark_iter(layers, fake_batch, output, external_gradient):
    output.data = layers(fake_batch)
    # torch.cuda.current_stream().synchronize()
    output.backward(gradient=external_gradient)
    # torch.cuda.current_stream().synchronize()


def benchmark_transformer_layer(args, device):
    module_list = []
    for _ in range(args.num_layers):
        module_list.append(GPTTransformerLayer(args.embedding_dim, args.num_heads,
                                               4 * args.embedding_dim, use_checkpoint=args.use_checkpoint))
    layers = nn.Sequential(*module_list).to(device).half()
    print(layers)

    batch_shape = (args.batch_size, args.seq_length, args.embedding_dim)

    # activities=[ProfilerActivity.CUDA]
    n = 96
    fake_input_batches = [torch.zeros(batch_shape, requires_grad=True, device=device, dtype=torch.float16)
                          for _ in range(n)]
    for micro_batch in fake_input_batches:
        if micro_batch.grad is None:
            micro_batch.grad = torch.zeros_like(micro_batch.data)

    output_batches = [torch.zeros(batch_shape, requires_grad=True, device=device, dtype=torch.float16)
                      for _ in range(n)]
    external_gradients = [torch.full(batch_shape, 1.0, requires_grad=False, device=device, dtype=torch.float16)
                          for _ in range(n)]
    for i in range(n):
        if i == 10:
            torch.cuda.current_stream().synchronize()
            with profile(profile_memory=True, record_shapes=True, with_flops=True, with_modules=True) as prof:
                benchmark_iter(layers, fake_input_batches[i], output_batches[i], external_gradients[i])
                torch.cuda.current_stream().synchronize()
                GPUtil.showUtilization()
        else:
            benchmark_iter(layers, fake_input_batches[i], output_batches[i], external_gradients[i])

        print("Max memory allocated: {} GB.".format(torch.cuda.max_memory_allocated(device=device)/1024/1024/1024))
        print("Max memory reserved: {} GB.".format(torch.cuda.max_memory_reserved(device=device)/1024/1024/1024))
        print(torch.cuda.memory_summary())
        print(torch.cuda.memory_stats())
        print(torch.cuda.memory_snapshot())

    trace_file = "../trace_json/gpt3_gpipe_local_benchmark_iter.json"
    prof.export_chrome_trace(trace_file)
    print(prof.key_averages(group_by_input_shape=True).table())


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    parser.add_argument('--use-checkpoint', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use check point for activation recompute')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--batch-size', type=int, default=3, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--num-layers', type=int, default=10, metavar='N',
                        help='-')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=16, metavar='N',
                        help='-')
    args = parser.parse_args()
    assert args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda', args.cuda_id)
    benchmark_transformer_layer(args, device)


if __name__ == '__main__':
    main()
