import argparse
import time
import torch.distributed as dist
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from task_datasets.qqp import get_glue_qqp_train_data_loader
from task_datasets.tokenizer import build_tokenizer
from utils.dist_args_utils import *
from utils.dist_debug_utils import *
from modules.dist_gpt_fsdp_module import GPTGlueFsdpModel
from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap


def main():
    parser = argparse.ArgumentParser(description='Fairscale-ZeRO_S3-GPT3')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_training_model_arguments(parser)
    add_qqp_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    parser.add_argument('--fsdp-degree', type=str, default="recursive", metavar='S',
                        help='how to use FSDP (default: recursive)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            rank=args.rank, world_size=args.world_size)

    tokenizer = build_tokenizer(args)
    print("token vocab size:", tokenizer.vocab_size)
    print("batch size: ", args.batch_size)
    train_dataloader = get_glue_qqp_train_data_loader(args, tokenizer)
    vocab_size = tokenizer.vocab_size
    num_classes = 2
    model = GPTGlueFsdpModel(args, vocab_size, num_classes).to(device)
    # model = checkpoint_wrapper(model, offload_to_cpu=True)
    # disable my own checkpoint, notice that FSDP checkpoint cannot be combined with flatten_parameters
    if args.fsdp_degree == 'simple':
        fsdp_model = FSDP(model, reshard_after_forward=True, move_params_to_cpu=False, mixed_precision=False,
                          flatten_parameters=False)
    elif args.fsdp_degree == 'recursive':
        fsdp_config = dict(reshard_after_forward=True, move_params_to_cpu=False, mixed_precision=False,
                           flatten_parameters=False)
        with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
            fsdp_model = auto_wrap(model, auto_wrap_policy=default_auto_wrap_policy)
            fsdp_model = FSDP(fsdp_model, **fsdp_config)
    else:
        print("Illegal FSDP degree!")
        assert False

    torch.cuda.set_device(args.cuda_id)

    print_cuda_memory(args, "Declared FSDP model-"+args.fsdp_degree, device)
    # dist_model = checkpoint_wrapper(dist_model, offload_to_cpu=True)
    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=args.lr)
    print_cuda_memory(args, "Declared optimizer for FSDP model", device)
    fsdp_model.train()

    total_time = 0
    multi_iter = 5
    for i, data in enumerate(train_dataloader):
        if i % multi_iter == 0:
            fsdp_model.zero_grad(set_to_none=True)
            start_time = time.time()

        cur_start_time = time.time()
        input_ids = data['text'].to(device)
        # input_ids.require_grad = True
        # position_ids = get_position_id(args.seq_length, args.batch_size, device)
        labels = data['label'].to(device)

        # output = fsdp_model(input_ids, position_ids)
        output = fsdp_model(input_ids)
        loss = torch.nn.functional.cross_entropy(output, labels)
        forward_time = time.time()
        print("{}/{} Forward pass takes {:3.2f}s, loss: ".format(i%multi_iter, multi_iter, forward_time-cur_start_time),
              loss.item())
        print_cuda_memory(args, "FSDP forward iter is done", device)
        loss.backward()
        backward_time = time.time()
        print("{}/{} Backward pass takes {:3.2f}s".format(i%multi_iter, multi_iter, backward_time-forward_time))
        print_cuda_memory(args, "FSDP backward iter is done", device)

        if (i+1) % multi_iter == 0:
            optimizer.step()

            end_time = time.time()
            iter_time = end_time - start_time
            print("Whole iteration takes {:3.2f}s".format(iter_time))
            print_cuda_memory(args, "FSDP optimizer step is done")
            total_time += iter_time
        if i >= args.num_iters - 1:
            break
    averaged_time = total_time / (args.num_iters // multi_iter)
    print("Finished running ", args.num_iters, " iters, averaged run time:", averaged_time)


if __name__ == '__main__':
    main()
