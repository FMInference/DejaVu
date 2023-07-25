import argparse
import torch.autograd.profiler as profiler
from task_datasets.qqp import get_glue_qqp_train_data_loader
from task_datasets.tokenizer import build_tokenizer
from pipeline_parallel.dist_pp_utils import get_pp_module
from utils.dist_args_utils import *
from utils.dist_train_utils import *
from comm.comm_utils import *


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_training_model_arguments(parser)
    add_qqp_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_parallel_schema_arguments(parser)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    init_communicators(args)

    if get_pipeline_parallel_rank() == 0 or get_pipeline_parallel_rank() == args.pipeline_group_size-1:
        tokenizer = build_tokenizer(args)
        print("token vocab size:", tokenizer.vocab_size)
        train_data_loader = get_glue_qqp_train_data_loader(args, tokenizer)
        num_classes = 2
        vocab_size = tokenizer.vocab_size
    else:
        train_data_loader = None
        num_classes = 2
        vocab_size = -1

    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        print("Running ", args.pp_mode, " with data parallel.")
    else:
        print("Running ", args.pp_mode, " without data parallel.")

    pipe = get_pp_module(args, vocab_size, num_classes, device, use_dp)

    if args.profiling == 'no-profiling':
        distributed_train_foo_iter(args, pipe, device, train_data_loader)
    else:
        prefix = './trace_json/gpt3_' + args.pp_mode
        if use_dp:
            prefix = prefix + '_' + args.dp_mode
        trace_file = prefix + get_learning_arguments_str(args) + get_model_arguments_str(args) + \
                     get_dist_arguments_str(args) + get_mixed_precision_arguments_str(args) + '_' + \
                     args.profiling + '_' + args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            distributed_train_foo_iter(args, pipe, device, train_data_loader)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                distributed_train_foo_iter(args, pipe, device, train_data_loader)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False


if __name__ == '__main__':
    main()
