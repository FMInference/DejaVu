import argparse
import time
import tempfile
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe
from task_datasets.qqp import get_glue_qqp_train_data_loader
from task_datasets.tokenizer import build_tokenizer
from utils.dist_args_utils import *
from utils.dist_debug_utils import *
from modules.dist_gpt_fsdp_module import GPTFsdpStageLast, GPTFsdpStageMiddle, GPTFsdpStageFirst


def main():
    parser = argparse.ArgumentParser(description='PyTorch-Pipe-GPT3')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_training_model_arguments(parser)
    add_qqp_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    assert args.use_cuda and torch.cuda.is_available() and args.cuda_num <= torch.cuda.device_count()

    tokenizer = build_tokenizer(args)
    print("token vocab size:", tokenizer.vocab_size)
    print("batch size: ", args.batch_size)
    train_dataloader = get_glue_qqp_train_data_loader(args, tokenizer)
    vocab_size = tokenizer.vocab_size
    num_classes = 2

    assert(args.num_layers % args.cuda_num == 0)

    num_stage_layers = args.num_layers // args.cuda_num
    stages_list = []
    for local_cuda_rank in range(args.cuda_num):
        device = torch.device('cuda', local_cuda_rank)
        if local_cuda_rank == 0:
            stages_model = GPTFsdpStageFirst(args, num_stage_layers, vocab_size, num_classes, device,
                                             use_checkpoint=False, explicit_fsdp=False)
        elif local_cuda_rank == args.cuda_num - 1:
            stages_model = GPTFsdpStageLast(args, num_stage_layers, vocab_size, num_classes, device,
                                            use_checkpoint=False, explicit_fsdp=False)
        else:
            stages_model = GPTFsdpStageMiddle(args, num_stage_layers, vocab_size, num_classes, device,
                                              use_checkpoint=False, explicit_fsdp=False)
        stages_list.append(stages_model)

    model = torch.nn.Sequential(*stages_list)
    chunks = args.batch_size // args.micro_batch_size

    tmp_file = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmp_file.name),
            # Specifying _transports and _channels is a workaround and we no longer
            # will have to specify _transports and _channels for PyTorch
            # versions >= 1.8.1
            _transports=["ibv", "uv"],
            _channels=["cuda_ipc", "cuda_basic"],
        )
    )

    pipe_model = Pipe(model, chunks=chunks, checkpoint='except_last')

    print_multi_cuda_memory(args, "Declared Pipe model")
    # dist_model = checkpoint_wrapper(dist_model, offload_to_cpu=True)
    optimizer = torch.optim.SGD(pipe_model.parameters(), lr=args.lr)
    print_multi_cuda_memory(args, "Declared optimizer for Pipe model")
    pipe_model.train()

    total_time = 0

    for i, data in enumerate(train_dataloader):

        pipe_model.zero_grad(set_to_none=True)
        start_time = time.time()

        input_ids = data['text'].to(torch.device('cuda', 0))
        # input_ids.require_grad = True
        # position_ids = get_position_id(args.seq_length, args.batch_size, torch.device('cuda', 0))
        labels = data['label'].to(torch.device('cuda', args.cuda_num-1))

        # output = pipe_model(input_ids, position_ids)
        output = pipe_model(input_ids).local_value()
        loss = torch.nn.functional.cross_entropy(output, labels)
        forward_time = time.time()
        print("{} Forward pass takes {:3.2f}s, loss: ".format(i, forward_time-start_time), loss.item())
        print_multi_cuda_memory(args, "Pipe forward iter is done")
        loss.backward()
        backward_time = time.time()
        print("{} Backward pass takes {:3.2f}s".format(i, backward_time-forward_time))
        print_multi_cuda_memory(args, "Pipe backward iter is done")
        optimizer.step()
        end_time = time.time()
        iter_time = end_time - start_time
        print("Whole iteration takes {:3.2f}s".format(iter_time))
        print_multi_cuda_memory(args, "Pipe optimizer step is done")
        if i > 0:
            total_time += iter_time
        if i >= args.num_iters - 1:
            break
    averaged_time = total_time / (args.num_iters - 1)
    print("Finished running ", args.num_iters, " iters, averaged run time:", averaged_time)


if __name__ == '__main__':
    main()
