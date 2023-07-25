import argparse
import time
import torch
from torch import nn
import apex
from optimizer.grad_scalar import *
from optimizer.optimizer import Fp16Optimizer
from modules.gpt_modules import GPTTransformerLayer


def benchmark_transformer_layer(args, device):
    module_list = []
    for _ in range(args.num_layers):
        module_list.append(GPTTransformerLayer(args.embedding_dim, args.num_heads,
                                               4 * args.embedding_dim, use_checkpoint=args.use_checkpoint))
    layers = nn.Sequential(*module_list).to(device)
    print(layers)
    # summary(one_layer, (args.seq_length, args.embedding_size), batch_dim=0)
    layers.half()
    optimizer = torch.optim.SGD(layers.parameters(), lr=args.lr)
    # optimizer = apex.optimizers.FusedSGD(layers.parameters(), lr=args.lr)
    grad_scaler = ConstantGradScaler(0.1, offload=args.offload)
    fp16_optimizer = Fp16Optimizer(optimizer, grad_scaler, device, offload=args.offload)

    batch_shape = (args.batch_size, args.seq_length, args.embedding_dim)

    forward_start_event = torch.cuda.Event(enable_timing=True)
    backward_start_event = torch.cuda.Event(enable_timing=True)
    optimizer_start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(11):
        fake_batch = torch.zeros(batch_shape, requires_grad=True, dtype=torch.float16).to(device)
        fake_batch.uniform_(-0.05, 0.05)
        external_gradient = torch.full(batch_shape, 0.1, dtype=torch.float16).to(device)
        external_gradient.uniform_(-0.01, 0.01)

        print("Current memory allocated: {:2.3f} MB, peak memory: {:2.3f} MB".format(
            torch.cuda.memory_allocated(device) / 1048576, torch.cuda.max_memory_allocated(device) / 1048576))

        forward_start_event.record()
        output = layers(fake_batch)

        backward_start_event.record()
        output.backward(gradient=external_gradient)

        optimizer_start_event.record()
        fp16_optimizer.step()
        end_event.record()

        torch.cuda.current_stream().synchronize()

        print("Iter ", i, "Current iter forward takes:", forward_start_event.elapsed_time(backward_start_event))
        print("Iter ", i, "Current iter backward takes:", backward_start_event.elapsed_time(optimizer_start_event))
        print("Iter ", i, "Current iter optimizer step takes:", optimizer_start_event.elapsed_time(end_event))

        print("Current memory allocated: {:2.3f} MB, peak memory: {:2.3f} MB".format(
            torch.cuda.memory_allocated(device) / 1048576, torch.cuda.max_memory_allocated(device) / 1048576))


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    parser.add_argument('--use-checkpoint', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use check point for activation recompute')
    parser.add_argument('--offload', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use CPU to store model.')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--num-layers', type=int, default=5, metavar='N',
                        help='-')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                        help='-')
    args = parser.parse_args()
    assert args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda', args.cuda_id)
    benchmark_transformer_layer(args, device)


if __name__ == '__main__':
    main()