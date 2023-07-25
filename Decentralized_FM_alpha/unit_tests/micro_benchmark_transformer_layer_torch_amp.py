import argparse
import time
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from modules.gpt_modules import GPTTransformerLayer


def benchmark_transformer_layer(args, device):
    module_list = []
    for _ in range(args.num_layers):
        module_list.append(GPTTransformerLayer(args.embedding_dim, args.num_heads,
                                               4 * args.embedding_dim, use_checkpoint=args.use_checkpoint))
    layers = nn.Sequential(*module_list).to(device)
    print(layers)
    # summary(one_layer, (args.seq_length, args.embedding_size), batch_dim=0)

    batch_shape = (args.batch_size, args.seq_length, args.embedding_dim)
    forward_time = 0
    backward_time = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    scaler = GradScaler()

    for i in range(11):
        if args.use_amp:
            print("Use AMP")
            fake_batch = torch.zeros(batch_shape, requires_grad=True, dtype=torch.float16).to(device)
            external_gradient = torch.full(batch_shape, 1.0, dtype=torch.float16).to(device)
        else:
            print("Not use AMP")
            fake_batch = torch.zeros(batch_shape, requires_grad=True).to(device)
            external_gradient = torch.full(batch_shape, 1.0).to(device)

        print("Current memory allocated: {:2.3f} MB, peak memory: {:2.3f} MB".format(
            torch.cuda.memory_allocated(device) / 1048576, torch.cuda.max_memory_allocated(device) / 1048576))

        start_time = time.time()
        start_event.record()

        if args.use_amp:
            with autocast():
                output = layers(fake_batch)
        else:
            output = layers(fake_batch)

        end_event.record()
        torch.cuda.current_stream().synchronize()
        forward_end_time = time.time()
        print("Iter ", i, "Current iter forward takes:", forward_end_time-start_time, " by event timer:",
              start_event.elapsed_time(end_event))

        output.backward(gradient=external_gradient)
        torch.cuda.current_stream().synchronize()
        backward_end_time = time.time()
        print("Iter ", i, "Current iter backward takes:", backward_end_time-forward_end_time)
        print("Current memory allocated: {:2.3f} MB, peak memory: {:2.3f} MB".format(
            torch.cuda.memory_allocated(device) / 1048576, torch.cuda.max_memory_allocated(device) / 1048576))


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    parser.add_argument('--use-checkpoint', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use check point for activation recompute')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--use-amp', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use amp to train')
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
    args = parser.parse_args()
    assert args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda', args.cuda_id)
    benchmark_transformer_layer(args, device)


if __name__ == '__main__':
    main()