import os

import deepspeed
import argparse
import time
import torch
from task_datasets.qqp import QQPDataset
from task_datasets.tokenizer import build_tokenizer
from utils.dist_args_utils import *
from modules.gpt_modules import GlueSeqClassificationModel, get_position_id


def main():
    parser = argparse.ArgumentParser(description='ZeRO-GPT3')
    add_training_model_arguments(parser)
    add_qqp_task_arguments(parser)
    # add_torch_distributed_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='rank of the node')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    device = torch.device('cuda', args.local_rank)

    # print(args.master_addr)
    # print(args.master_port)

    # os.environ['RANK'] = str(args.rank)
    print(os.environ)
    # deepspeed.init_distributed(init_method='tcp://'+os.environ['MASTER_ADDR']+':'+os.environ['MASTER_PORT'])
    # deepspeed.init_distributed(init_method="env://")
    # dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    tokenizer = build_tokenizer(args)
    print("token vocab size:", tokenizer.vocab_size)
    train_dataset = QQPDataset('training', args.train_data, tokenizer, args.seq_length)

    num_classes = 2
    model = GlueSeqClassificationModel(args, tokenizer.vocab_size, num_classes).half()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    ds_config = {
        "train_batch_size": args.batch_size,
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "zero_allow_untested_optimizer": True,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3
        },
        "offload_param": {
            "device": "cpu"
        },
        "offload_optimizer": {
            "device": "cpu",
            "buffer_count": 4,
            "fast_init": False
        }
    }

    model_engine, optimizer, train_dataloader, _ = \
        deepspeed.initialize(args=args, model=model, model_parameters=model.parameters(), optimizer=optimizer,
                             training_data=train_dataset, config=ds_config)

    if deepspeed.comm.get_rank() == 0:
        print("World size: {}, Batch size: {}/{}.".format(deepspeed.comm.get_world_size(), args.micro_batch_size,
                                                            args.batch_size))
        print("Model dim:{}, Num of Layers:{}, Seq length: {}"
              .format(args.embedding_dim, args.num_layers, args.seq_length))

    ga_steps = args.batch_size // args.micro_batch_size // deepspeed.comm.get_world_size()

    for i, data in enumerate(train_dataloader):
        start_time = time.time()
        for j in range(ga_steps):
            micro_start_time = time.time()
            input_ids = data['text'].to(device)
            position_ids = get_position_id(args.seq_length, input_ids.size(0), device)
            labels = data['label'].to(device)
            output = model_engine(input_ids, position_ids)
            loss = torch.nn.functional.cross_entropy(output, labels)
            micro_forward_time = time.time()
            if deepspeed.comm.get_rank() == 0:
                print("Input shape: ", input_ids.shape)
                print("{}/{} Forward pass takes {:3.2f}s".format(j, ga_steps, micro_forward_time - micro_start_time))
            model_engine.backward(loss)
            micro_backward_time = time.time()
            if deepspeed.comm.get_rank() == 0:
                print("{}/{} Backward pass takes {:3.2f}s".format(j, ga_steps, micro_backward_time - micro_forward_time))
            model_engine.step()
            # torch.cuda.synchronize()
            # deepspeed.comm.barrier()
        end_time = time.time()
        if deepspeed.comm.get_rank() == 0:
            print("========<{}> Whole iteration takes {:3.2f}s========".format(i, end_time - start_time))
        # print(data)
        if i >= args.num_iters - 1:
            break


if __name__ == '__main__':
    main()
