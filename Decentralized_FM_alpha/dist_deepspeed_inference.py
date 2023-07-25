import os
import deepspeed
import argparse
import time
import torch
from utils.dist_args_utils import *
from transformers import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Deepspeed Inference-GPT3')

    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='rank of the node')
    parser.add_argument('--mp-size', type=int, default=8, help='size of tensor model parallelism')
    parser.add_argument('--dp-zero-stage', type=int, default=1, help='pipeline parallelism')
    parser.add_argument('--prompt-seq-length', type=int, default=512, help='seq length in prompt phase')
    parser.add_argument('--token-seq-length', type=int, default=50, help='seq length in token phase')
    parser.add_argument('--batch-size', type=int, default=8, help='size of tensor model parallelism')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed("nccl")
    device = torch.device('cuda', args.local_rank)

    checkpoint_json = {
    }
    config = OPTConfig.from_pretrained('/home/fsuser/GPT-home-private/opt-175b-new')
    model = OPTForCausalLM(config)
    # model = OPTForCausalLM.from_pretrained('facebook/opt-125m')
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

    # model = model_class.from_pretrained(args.model_name_or_path)
    # Initialize the DeepSpeed-Inference engine
    ds_engine = deepspeed.init_inference(model,
                                         mp_size=deepspeed.comm.get_world_size(),
                                         dtype=torch.half,
                                         checkpoint=None,
                                         replace_method='auto',
                                         replace_with_kernel_inject=True)
    model = ds_engine.module

    with torch.no_grad():
        for i in range(10):
            start_time = time.time()
            input_ids = \
                tokenizer(['hello world!'] * args.batch_size, padding='max_length', max_length=args.prompt_seq_length,
                          return_tensors='pt')[
                    'input_ids'].to(device)
            output = model.generate(input_ids, max_new_tokens=args.token_seq_length)
            end_time = time.time()
            print("======== Inference iter {} takes  {:3.2f}s========".format(i, end_time - start_time))

if __name__ == '__main__':
    main()
