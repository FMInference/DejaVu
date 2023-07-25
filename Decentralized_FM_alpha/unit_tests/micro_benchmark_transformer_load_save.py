import argparse
import time
import torch
from torch import nn
from modules.gpt_modules import GPTTransformerLayer


def benchmark_transformer_layer_load_save(args, device):
    module_list = []
    for _ in range(args.num_layers):
        module_list.append(GPTTransformerLayer(args.embedding_dim, args.num_heads,
                                               4 * args.embedding_dim, use_checkpoint=args.use_checkpoint))
    layers = nn.Sequential(*module_list).to(device).half()
    print(layers)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    save_time = 0
    # save_time_event = 0
    load_time = 0
    # load_time_event = 0
    model_name = 'gpt_m' + str(args.embedding_dim) + '_l' + str(args.num_layers) + '-'
    for i in range(11):
        save_start_time = time.time()
        start_event.record()
        torch.save(layers, args.path+model_name + str(i) + '.pth')
        end_event.record()
        torch.cuda.synchronize()
        save_end_time = time.time()
        print(i, " save model takes:", save_end_time-save_start_time, " cuda event time:",
              start_event.elapsed_time(end_event))
        load_start_time = time.time()
        start_event.record()
        layers = torch.load(args.path+model_name + str(i) + '.pth')
        end_event.record()
        torch.cuda.synchronize()
        load_end_time = time.time()
        print(i, " load model takes:", load_end_time - load_start_time, " cuda event time:",
              start_event.elapsed_time(end_event))
        if i != 0:
            save_time += (save_end_time-save_start_time)
            load_time += (load_end_time-load_start_time)
    save_time /= 10
    load_time /= 10
    print("==========================================================")
    print("Average save model takes: ", save_time)
    print("Average load model takes: ", load_time)


def main():
    parser = argparse.ArgumentParser(description='Load and Save.')
    parser.add_argument('--use-checkpoint', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use check point for activation recompute')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--num-layers', type=int, default=1, metavar='N',
                        help='-')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=5120, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=40, metavar='N',
                        help='-')
    parser.add_argument('--path', type=str, default='../logs/model_checkpoint/', metavar='S',
                        help='-')
    args = parser.parse_args()
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
    benchmark_transformer_layer_load_save(args, device)


if __name__ == '__main__':
    main()