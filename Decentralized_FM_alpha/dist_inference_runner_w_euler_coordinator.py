import argparse
import torch.autograd.profiler as profiler
from pipeline_parallel.dist_pp_utils import get_pp_inference_module
from utils.dist_args_utils import *
from utils.dist_inference_utils import *
from comm.comm_utils import *
from task_datasets.inference_data import get_request_processor
from coordinator.lsf.lsf_coordinate_client_deprecated import CoordinatorInferenceClient


def main():
    parser = argparse.ArgumentParser(description='Inference Runner with coordinator.')
    add_device_arguments(parser)
    add_torch_distributed_inference_w_euler_coordinator_arguments(parser)
    add_inference_arguments(parser)
    add_inference_details_arguments(parser)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    args = parser.parse_args()
    #args.infer_data = ''
    print_arguments(args)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    coord_client = CoordinatorInferenceClient(args)
    # keep beating during loading model
    print(get_pp_inference_module)

    prime_ip, rank, port = coord_client.notify_inference_join()
    print("<====Coordinator assigned prime-IP:", prime_ip, " and my assigned rank", rank, "====>")
    
    init_inference_communicators_with_coordinator(args, prime_ip, rank, port=port)

    request_processor = get_request_processor(args)
    request_processor.set_arguments(args)
    
    # all ranks heart beating during model loading
    pipe = coord_client.decorate_run_heart_beating_during(get_pp_inference_module)(args, device, rank=rank)

    # rank0: beat before inference
    if rank == 0:
        pipe.inference_batch = coord_client.decorate_run_heart_beating_before(pipe.inference_batch)

    if args.profiling == 'no-profiling':
        avg_iter_time = distributed_inference_mask_iter(args, pipe, device, request_processor)
    else:
        prefix = './trace_json/inference_' + args.pp_mode
        trace_file = prefix + get_inference_arguments_str(args, rank=rank) + '_' + args.profiling + '_' + args.trace_postfix + \
                     '.json'
        if args.profiling == 'tidy_profiling':
            avg_iter_time = distributed_inference_mask_iter(args, pipe, device, request_processor)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                avg_iter_time = distributed_inference_mask_iter(args, pipe, device, request_processor)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False
    
    # train_finish_msg = str(rank) + '#' + str(round(avg_iter_time, 3))
    coord_client.notify_inference_finish(rank=rank, iter_time=round(avg_iter_time, 3))


if __name__ == '__main__':
    main()
