import argparse
import time
import torch
from torch import nn
import torch.autograd.profiler as profiler
from modules.gpt_modules import GPTTransformerLayer
from deepspeed.profiling.flops_profiler import FlopsProfiler


def benchmark_transformer_layer_train(args, device):
    module_list = []
    for _ in range(args.num_layers):
        module_list.append(GPTTransformerLayer(args.embedding_dim, args.num_heads,
                                               4 * args.embedding_dim, use_checkpoint=args.use_checkpoint))
    layers = nn.Sequential(*module_list).to(device).half()
    print(layers)
    # summary(one_layer, (args.seq_length, args.embedding_size), batch_dim=0)

    batch_shape = (args.batch_size, args.seq_length, args.embedding_dim)
    forward_time = 0
    backward_time = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    ds_prof = FlopsProfiler(layers)

    with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
        for i in range(11):
            if i == 1:
                ds_prof.start_profile()
            if i != 0:
                start_time = time.time()
                start_event.record()
            fake_batch = torch.zeros(size=batch_shape, requires_grad=True, dtype=torch.float16, device=device)
            output = layers(fake_batch)
            if i != 0 :
                end_event.record()
            if args.use_cuda:
                torch.cuda.current_stream().synchronize()
            if i != 0:
                forward_end_time = time.time()
                print("Iter ", i, "Current iter forward takes:", forward_end_time-start_time, " (by event timer:",
                      start_event.elapsed_time(end_event))
                forward_time += (forward_end_time-start_time)
            external_gradient1 = torch.full(batch_shape, 1.0).to(device)
            output.backward(gradient=external_gradient1)
            if args.use_cuda:
                torch.cuda.current_stream().synchronize()
            if i != 0:
                backward_end_time = time.time()
                print("Iter ", i, "Current iter backward takes:", backward_end_time-forward_end_time)
                backward_time += (backward_end_time-forward_end_time)

            if i == 1:
                ds_prof.stop_profile()
                flops = ds_prof.get_total_flops()
                macs = ds_prof.get_total_macs()
                params = ds_prof.get_total_params()
                ds_prof.print_model_profile()
                ds_prof.end_profile()
                print("Flop raw: {}, PFlop: {} for a batch of 1024".format(flops, flops * 1024 / 10 ** 15))
                print("Macs:", macs)
                print("Params:", params)

        print("Average forward time: ", forward_time/10, " average backward time: ", backward_time/10)
    trace_file = "../trace_json/gpt3_gpipe_local_benchmark.json"
    prof.export_chrome_trace(trace_file)


def benchmark_transformer_layer_inference(args, device):
    module_list = []
    for _ in range(args.num_layers):
        module_list.append(GPTTransformerLayer(args.embedding_dim, args.num_heads,
                                               4 * args.embedding_dim, use_checkpoint=args.use_checkpoint))
    layers = nn.Sequential(*module_list).to(device).half()
    print(layers)
    # summary(one_layer, (args.seq_length, args.embedding_size), batch_dim=0)

    batch_shape = (args.batch_size, args.seq_length, args.embedding_dim)
    forward_time = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    ds_prof = FlopsProfiler(layers)

    with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
        for i in range(11):
            if i == 1:
                ds_prof.start_profile()
            if i != 0:
                start_time = time.time()
                start_event.record()
            fake_batch = torch.zeros(size=batch_shape, requires_grad=True, dtype=torch.float16, device=device)
            output = layers(fake_batch)
            if i != 0 :
                end_event.record()
            if args.use_cuda:
                torch.cuda.current_stream().synchronize()
            if i != 0:
                forward_end_time = time.time()
                print("Iter ", i, "Current iter forward takes:", forward_end_time-start_time, " (by event timer:",
                      start_event.elapsed_time(end_event))
                forward_time += (forward_end_time-start_time)
            print(torch.cuda.memory_summary())
        print("Average inference time: ", forward_time/10)
    trace_file = "../trace_json/gpt3_gpipe_local_benchmark.json"
    prof.export_chrome_trace(trace_file)


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    # Notice you cannot use checkpoint to compute FLOPS!!!!
    parser.add_argument('--use-checkpoint', default=False, type=lambda x: (str(x).lower() == 'true'),
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
    parser.add_argument('--embedding-dim', type=int, default=4096, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=32, metavar='N',
                        help='-')
    args = parser.parse_args()
    print(hasattr(args, 'foo'))
    assert hasattr(args, 'cuda_id')
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
    benchmark_transformer_layer_train(args, device)
    #benchmark_transformer_layer_inference(args, device)


if __name__ == '__main__':
    main()