import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, batch_size=None):
        
        self.tokenizer = tokenizer
        self.data = data
        self.idx = list(range(len(data)))
        
        if batch_size is not None:
            n_dummy = batch_size - len(data) % batch_size
            if n_dummy < batch_size:
                self.idx += [-1]*n_dummy
                self.data = self.data + ['dummy']*n_dummy
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        item = self.tokenizer(
            self.data[idx], return_tensors='pt', 
            padding='max_length', truncation=True,
        )
        item = {k: v.squeeze(0) for k, v in item.items()}
        item['text'] = item['input_ids']
        item['idx'] = self.idx[idx]

        return item
    
class DummyRequestProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        print("<DummyRequestProcessor>")
        
    def set_arguments(self, args):
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.temperature = args.temperature
        self.echo_prompt = args.echo_prompt
        self.top_k_per_token = args.top_k_per_token
        self.num_completions = args.num_completions
        self.max_tokens = args.generate_seq_length
        self.stop = args.stop
        
        if (args.echo_prompt and args.input_seq_length == self.tokenizer.model_max_length+1 and args.generate_seq_length==0):
            # special case! to support 2049 tokens
            args.input_seq_length = self.tokenizer.model_max_length + 1
            self.tokenizer.model_max_length = args.input_seq_length
        else:
            self.tokenizer.model_max_length = min(args.input_seq_length, self.tokenizer.model_max_length - args.generate_seq_length)
    
    def get_dataloader(self, batch_size, num_workers=0):
        
        dataset = JsonDataset(
            ['you are', 'hello world', '1 2 3 4', 'a b c d']*1000, 
            self.tokenizer, batch_size=batch_size,
        )
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=None
        )
        
        return data_loader
        
    def add_result(self, inputs, outputs, batch_time=None):
        batch_size = len(inputs['idx'])
        tokenizer = self.tokenizer
        for i in range(batch_size):
            idx = inputs['idx'][i]
            if idx < 0:
                continue
                
            if self.echo_prompt:
                n_pads = (1-inputs['attention_mask'][i]).sum()
            else:
                n_pads = 0
                
            print(f'>>>>>> batch_time: {batch_time:.4f}s, batch_size: {batch_size} <<<<<<')
            item = {
                'choices': [], 
                'request_time': {
                'batch_time': batch_time,
                'batch_size': batch_size,
                }
            }
            
            for i_ret, output_dict in enumerate(outputs):
                choice = {
                    "text": (tokenizer.decode(output_dict['token_ids'][i][n_pads:]) if 'token_ids' in output_dict else ''),
                    "index": i_ret,
                    "logprobs": {
                        "tokens": (tokenizer.convert_ids_to_tokens(output_dict['token_ids'][i][n_pads:] if 'token_ids' in output_dict else [])),
                        "token_logprobs": (output_dict['token_logprobs'][i][n_pads:].tolist() if 'token_logprobs' in output_dict else []),
                        "top_logprobs": ([
                            {
                                tokenizer.convert_ids_to_tokens(topk_id.item()): top_logprob.item()  for topk_id, top_logprob in zip(topk_ids, top_logprobs)
                            } \
                            for topk_ids, top_logprobs in zip(
                                output_dict['topk_ids'][i][n_pads:],
                                output_dict['topk_logprobs'][i][n_pads:]
                            )
                        ] if self.top_k_per_token > 0 else None),
                        "text_offset": [],
                    },
                    "finish_reason": "length",
                }
                if self.echo_prompt:
                    if len(choice['logprobs']['token_logprobs']) > 0:
                        choice['logprobs']['token_logprobs'][0] = None
                        if choice['logprobs']['top_logprobs'] is not None:
                            choice['logprobs']['top_logprobs'][0] = None    
                item['choices'].append(choice)

            # handle early stop
            for c in item['choices']:
                c['finish_reason'] = 'length'

            if self.stop is not None:
                for c in item['choices']:
                    for stop in self.stop:
                        if stop in c['text']:
                            c['text'] = c['text'][:c['text'].find(stop)]
                            c['finish_reason'] = 'stop'
                            
            for choice in item['choices']:
                print([choice['text']])
        
    def write_scenario_state(self):
        pass
    

class RequestProcessor:
    def __init__(self, request_path, tokenizer):
        
        self.tokenizer = tokenizer
        self.request_path = request_path
        dirname = os.path.dirname(request_path)
        basename = os.path.basename(request_path)

        self.output_path = os.path.join(dirname, 'output_'+basename)
        print("<RequestProcessor> dir:", dirname)
        print("<RequestProcessor> file:", basename)
#         print("<RequestProcessor>, output file:", self.output_path)
        if basename.endswith('jsonl'):
            with open(self.request_path) as f:
                self.data = []
                for line in f:
                    if line.strip() != '':
                        self.data.append({'request': json.loads(line)})
        elif basename.endswith('json'):
            with open(self.request_path) as f:
                self.data = [
                    {'request': line} for line in json.load(f)
                ]
        else:
            assert False, "Not supported file format"
        first_request = self.data[0]['request']
        """
            A line of request:
            request_type: ClassVar[str] = "language-model-inference"
            model: str
            prompt: str
            max_tokens: Optional[int] # Maximum number of tokens to generate
            temperature: Optional[float] # Annealing temperature
            top_p: Optional[float] # Fraction of probability mass to keep (in top-p sampling)
            n: Optional[int] # Number of samples to generate
            logprobs: Optional[int] # Number of tokens to show logprobs
            echo: Optional[bool] # Include the input as part of the output (e.g., for language modeling)
            best_of: Optional[int] # Produce This many candidates per token
            stop: Optional[List[str]]  # Stop when any of these strings are generated
        """
        self.top_k = first_request.get('top_k', None)
        self.top_p = first_request.get('top_p', None)
        self.temperature = first_request.get('temperature', 0)
        self.echo_prompt = first_request.get('echo', 0)
        self.top_k_per_token = first_request.get('logprobs', 0)
        self.num_completions = first_request.get('n', 1)
        self.max_tokens = first_request.get('max_tokens', 1)
        self.best_of = first_request.get('best_of', 1)
        self.stop = first_request.get('stop', None)
        self.is_glm = False
        
    def set_arguments(self, args):
        
        if hasattr(args, 'output_path') and args.output_path is not None:
            self.output_path = args.output_path
        
        if hasattr(args, 'overwrite_request_args') and args.overwrite_request_args:
            self.top_k = args.top_k
            self.top_p = args.top_p
            self.temperature = args.temperature
            self.echo_prompt = args.echo_prompt
            self.top_k_per_token = args.top_k_per_token
            self.num_completions = args.num_completions
            self.max_tokens = args.generate_seq_length
            self.best_of = args.best_of
            max_input_seq_length = args.input_seq_length
        else:
            args.top_k = self.top_k
            args.top_p = self.top_p
            args.temperature = self.temperature
            args.echo_prompt = self.echo_prompt
            args.top_k_per_token = self.top_k_per_token
            args.num_completions = self.num_completions
            args.generate_seq_length = self.max_tokens
            args.best_of = self.best_of
            args.stop = self.stop
            
            if args.echo_prompt and args.model_type in ['glm']:
                self.tokenizer.echo_prompt = args.echo_prompt
                self.is_glm = True
        
            max_input_seq_length = 1
            for i, x in enumerate(self.data):
                seq_length = self.tokenizer(x['request']['prompt'], return_tensors='pt', padding=True,
                                            truncation=False)['input_ids'].size(1)
                
                if seq_length > max_input_seq_length:
                    max_input_seq_length = seq_length
                #if i > 100:
                    # first 100 is enough
                    #break

            if args.model_type != 't5':
                if self.tokenizer.model_max_length > 10000:
                    self.tokenizer.model_max_length = 2048
                if args.model_type == 'bloom':
                    # hf's default value for bloom is wrong
                    self.tokenizer.model_max_length = 2048
                args.input_seq_length = min(
                    max_input_seq_length + 1,
                    self.tokenizer.model_max_length - args.generate_seq_length,
                )

                if args.budget is not None:
                    budget = args.budget
                else:
                    print('warn: budget is not set, will set batch size to 1')
                    budget = 1

                #args.token_micro_batch_size = 2 # TODO: hard code
                args.batch_size = max(budget // ((args.input_seq_length + args.generate_seq_length)*self.num_completions), args.token_micro_batch_size) // args.token_micro_batch_size * args.token_micro_batch_size
                args.batch_size = min(args.batch_size, 64) # TODO: if batch size is too large, the comm will stuck.
                #args.token_micro_batch_size = args.batch_size

            else:
                if self.tokenizer.model_max_length > 10000:
                    self.tokenizer.model_max_length = 512
                
                # T5 does not have length limit
                args.input_seq_length = min(
                    max_input_seq_length + 1,
                    self.tokenizer.model_max_length,
                ) # max_input_seq_length

                if args.budget is not None:
                    budget = args.budget
                else:
                    print('warn: budget is not set, will set batch size to 1')
                    budget = 1

                args.batch_size = max(budget // (args.input_seq_length + args.generate_seq_length), 1)
                args.batch_size = min(args.batch_size, 64)
                args.token_micro_batch_size = args.batch_size
        
        if (args.echo_prompt and max_input_seq_length == self.tokenizer.model_max_length+1 and args.generate_seq_length==0):
            # special case! to support 2049 tokens
            args.input_seq_length = self.tokenizer.model_max_length + 1
            
        self.tokenizer.model_max_length = args.input_seq_length

        print('input seq length:', args.input_seq_length)
            
        
    def get_dataloader(self, batch_size, num_workers=0):
        
        dataset = JsonDataset(
            [x['request']['prompt'] for x in self.data], 
            self.tokenizer, batch_size=batch_size,
        )
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False, # make it explicit
            pin_memory=True,
            collate_fn=None
        )
        
        return data_loader
    
    def add_result(self, inputs, outputs, batch_time=None):
        batch_size = len(inputs['idx'])
        tokenizer = self.tokenizer
        for i in range(batch_size):
            idx = inputs['idx'][i]
            if idx < 0:
                continue
                
            if self.echo_prompt:
                n_pads = (1-inputs['attention_mask'][i]).sum()
                if self.is_glm:
                    n_pads += 2 # [gMASK] and sop
            else:
                n_pads = 0
                
            item = {
                'choices': [], 
                'request_time': {
                'batch_time': batch_time,
                'batch_size': batch_size,
                }
            }
            
            # print(tokenizer.decode(outputs[0]['token_ids'][0]))
            for i_ret, output_dict in enumerate(outputs):
                choice = {
                    "text": (tokenizer.decode(output_dict['token_ids'][i][n_pads:]) if 'token_ids' in output_dict else ''),
                    "index": i_ret,
                    "logprobs": {
                        "tokens": (tokenizer.convert_ids_to_tokens(output_dict['token_ids'][i][n_pads:] if 'token_ids' in output_dict else [])),
                        "token_logprobs": (output_dict['token_logprobs'][i][n_pads:].tolist() if 'token_logprobs' in output_dict else []),
                        "top_logprobs": ([
                            {
                                tokenizer.convert_ids_to_tokens(topk_id.item()): top_logprob.item()  for topk_id, top_logprob in zip(topk_ids, top_logprobs)
                            } \
                            for topk_ids, top_logprobs in zip(
                                output_dict['topk_ids'][i][n_pads:],
                                output_dict['topk_logprobs'][i][n_pads:]
                            )
                        ] if self.top_k_per_token > 0 else None),
                        "text_offset": [],
                    },
                    "finish_reason": "length",
                }
                if self.echo_prompt:
                    if len(choice['logprobs']['token_logprobs']) > 0:
                        choice['logprobs']['token_logprobs'][0] = None
                        if choice['logprobs']['top_logprobs'] is not None:
                            choice['logprobs']['top_logprobs'][0] = None
                item['choices'].append(choice)
            self.data[idx]['result'] = item
            
            try:
                if self.num_completions > 1:
                    self.data[idx]['result']['choices'] = sorted(
                        self.data[idx]['result']['choices'],
                        key=lambda c: -np.mean(c['logprobs']['token_logprobs']),
                    )
                    self.data[idx]['result']['choices'][:self.best_of]
                    for _i, c in enumerate(self.data[idx]['result']['choices']):
                        c['index'] = _i
            except:
                print('fail to sort choices')
                
            # handle early stop
            for c in item['choices']:
                c['finish_reason'] = 'length'

            if self.stop is not None:
                for c in item['choices']:
                    for stop in self.stop:
                        if stop in c['text']:
                            c['text'] = c['text'][:c['text'].find(stop)]
                            c['finish_reason'] = 'stop'
        
    def write_scenario_state(self):
        with open(self.output_path, 'w') as f:
            for line in self.data:
                f.write(json.dumps(line) + '\n')
            
            
def get_tokenizer(args):
    
    if args.model_type in ['yalm']:
        from modules.yalm_tokenizer import YalmTokenizer
        
        tokenizer = YalmTokenizer.from_pretrained(args.model_name)
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        return tokenizer
    
    if args.model_type in ['glm']:
        from modules.glm_tokenizer import GLMTokenizer
        
        tokenizer = GLMTokenizer.from_pretrained(args.model_name)
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        return tokenizer
    
    # default: huggingface's implementation
    # TODO, a dirty fix, for GPT-66B, we find the default implementation has some issue:
    # See: https://github.com/huggingface/tokenizers/pull/1005. so that the fast tokenizer works correctly.
    if args.model_name == '/home/ubuntu/fm/models/opt-66b-new':
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('/home/ubuntu/fm/models/opt-66b-new')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if args.model_type in ['t5']:
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'left'
    else:
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
    # tokenizer.model_max_length = args.input_seq_length
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def get_request_processor(args, infer_data=None):
    print("<get_request_processor>:", infer_data)
    tokenizer = get_tokenizer(args)
    if infer_data is None:
        assert args.infer_data is not None
        infer_data = args.infer_data

    if infer_data.strip() == '':
        return DummyRequestProcessor(tokenizer)
    else:
        return RequestProcessor(infer_data, tokenizer)
