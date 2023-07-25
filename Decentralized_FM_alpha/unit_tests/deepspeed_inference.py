import os
import deepspeed
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import time

# use the latest version of deepspeed 0.67 (pip install deepspeed==0.6.7)


local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
# generator = pipeline('text-generation', model='gpt2', device=local_rank)
print("?")
batch_size = 16
prompt_length = 512
generate_length = 32

tokenizer = AutoTokenizer.from_pretrained('gpt2')

model = GPT2LMHeadModel.from_pretrained('gpt2').half().eval()

tokenizer.pad_token = tokenizer.eos_token

model.pad_token_id = tokenizer.pad_token_id

model = model.cuda()


ds_inf = deepspeed.init_inference(model,
                                  mp_size=world_size,
                                  dtype=torch.float,
                                  # checkpoint='./deepspeed_inference.json',
                                  replace_method='auto',
                                  replace_with_kernel_inject=False,
                                  enable_cuda_graph=False,
                                  cpu_offload=True)

model = ds_inf.module


print("?")
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    inputs = tokenizer(['hello world'] * batch_size, padding='max_length', max_length=prompt_length,
                       return_tensors='pt')

    inputs = {
        k: v.cuda() for k, v in inputs.items()
    }

    print(inputs['input_ids'].shape)

    model.generate(inputs['input_ids'], max_length=prompt_length + generate_length)
    tic = time.time()
    for _ in range(10):
        model.generate(inputs['input_ids'], max_length=prompt_length + generate_length)
    toc = time.time()
    avg_t = (toc - tic) / 10
    print("Avg_t:", avg_t)