'''
Please ensure deepspeed version >= 0.7.1.
Launch with the following command:

$ deepspeed --num_gpus 8 infer_deepspeed.py

'''

import os
import torch
import deepspeed
import time

from transformers import OPTForCausalLM, AutoTokenizer, OPTConfig
from transformers.models.opt.modeling_opt  import OPTDecoderLayer

def main():

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    print(f'rank: {local_rank}/{world_size}')
    
    batch_size = 4
    prompt_length = 512
    token_length = 50
    model_name_or_path = 'facebook/opt-1.3b'
    fp16 = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = OPTConfig.from_pretrained(model_name_or_path)
    model = OPTForCausalLM(config) # not load checkpoint

    model = deepspeed.init_inference(
        model,
        mp_size=world_size,
        dtype=torch.float16 if fp16 else torch.float32,
        replace_with_kernel_inject=False,
        ##########################################################################
        # This parameter defines which part of the model needs to be parallelized.
        injection_policy={OPTDecoderLayer: ('self_attn.out_proj', '.fc2')}
        ##########################################################################
    )
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i in range(5+1):
            if i == 1:
                # skip first
                tic = time.time()
            input_ids = tokenizer(['hello'] * batch_size, max_length=prompt_length, padding='max_length', return_tensors='pt')['input_ids'].cuda()
            model.generate(input_ids, max_new_tokens=token_length)

        toc = time.time()

    print((toc - tic) / 5)


if __name__ == '__main__':
    main()
