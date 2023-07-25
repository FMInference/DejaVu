import os

import deepspeed
import argparse
import time
import torch
from task_datasets.qqp import QQPDataset
from task_datasets.tokenizer import build_tokenizer
from utils.dist_args_utils import *
from modules.dist_gpt_ds_module import GlueSeqClassificationPipeModel
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader


def main():
    parser = argparse.ArgumentParser(description='Deepspeed pipeline-GPT3')
    add_training_model_arguments(parser)
    add_qqp_task_arguments(parser)
    # add_torch_distributed_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='rank of the node')
    parser.add_argument('--pipeline-parallel-size', type=int, default=8, help='pipeline parallelism')
    parser.add_argument('--dp-zero-stage', type=int, default=1, help='pipeline parallelism')
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

    deepspeed.init_distributed("nccl")
    torch.cuda.set_device(args.local_rank)

    tokenizer = build_tokenizer(args)
    print("token vocab size:", tokenizer.vocab_size)
    train_dataset = QQPDataset('training', args.train_data, tokenizer, args.seq_length, data_as_tuple=True)

    num_classes = 2
    model = GlueSeqClassificationPipeModel(args, tokenizer.vocab_size, num_classes).half()
    model = PipelineModule(
        layers=model.to_layers_for_deepspeed_pipeline(),
        loss_fn = torch.nn.CrossEntropyLoss(),
        num_stages=args.pipeline_parallel_size,
        partition_method='uniform'
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # deepspeed only allows ZeRO-1 to combine with pipeline parallelism.
    assert args.dp_zero_stage == 1

    ds_config = {
        "train_batch_size": args.batch_size,
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "zero_allow_untested_optimizer": True,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": args.dp_zero_stage
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
        deepspeed.initialize(args=args,
                             model=model,
                             model_parameters=[p for p in model.parameters() if p.requires_grad],
                             optimizer=optimizer,
                             training_data=train_dataset,
                             config=ds_config)

    if deepspeed.comm.get_rank() == 0:
        print("<===========World size: {}, Batch size: {}/{}.===========>".format(deepspeed.comm.get_world_size(), args.micro_batch_size,
                                                            args.batch_size))
        print("<===========Model dim:{}, Num of Layers:{}, Seq length: {}, gradient_accumulation_steps: {}.===========>"
              .format(args.embedding_dim, args.num_layers, args.seq_length, model_engine.gradient_accumulation_steps()))

    print("<++++++++++++++++++ Training start, my rank: {} ++++++++++++++++++>".format(deepspeed.comm.get_rank()))

    for i in range(args.num_iters):
        start_time = time.time()
        _ = model_engine.train_batch()
        end_time = time.time()
        if deepspeed.comm.get_rank() == 0:
            print("========<{}> Whole iteration takes {:3.2f}s========".format(i, end_time - start_time))


if __name__ == '__main__':
    main()
