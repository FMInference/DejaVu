import time
import json
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_processor, get_logits_warper


from .dist_pipeline_inference_mask_greedy_token_pipe_sync import DistGreedyInferenceMaskTokenPipeSync


class DistSampleInferenceMaskTokenPipeSync(DistGreedyInferenceMaskTokenPipeSync):
    
    def __init__(self, args, device, rank=None, be_coordinated=False):
        super().__init__(args, device, rank=rank, be_coordinated=be_coordinated)
        self.update_processors(args)
        
    def update_processors(self, args):
        self.logits_processor = get_logits_processor()
        self.logits_warper = get_logits_warper(
            top_k = (None if args.top_k is None or args.top_k == 0 else args.top_k),
            top_p = (None if args.top_p is None or args.top_p <= 0 else args.top_p),
            temperature = args.temperature,
            num_beams = 1,
        )

    def _generate_new_token(self, index):
        assert self.pp_rank == self.pipeline_group_size - 1
        z = self.layers['lm'](self.output_token_emb[index])
        if torch.isnan(z).any():
            print('containing nan, setting them to zero!')
            print(z)
        z = z.float().nan_to_num() # test if fp32 whether cause inf
        z = torch.nn.functional.log_softmax(z, -1)
        
        if self.top_k_per_token > 0:
            logprobs, indices = z.topk(k=self.top_k_per_token, dim=-1)
            self.ret_topk_tokens[
                index*self.token_micro_batch_size*self.num_completions:(index+1)*self.token_micro_batch_size*self.num_completions,
                self.i_current_token
            ] = indices.squeeze(1)
            self.ret_topk_token_logprobs[
                index*self.token_micro_batch_size*self.num_completions:(index+1)*self.token_micro_batch_size*self.num_completions,
                self.i_current_token
            ] = logprobs.squeeze(1)
            
        # [:, -1] because multinomial only accept 1/2d tensors
        z_to_sample = z[:, -1] # bs, vocab
        z_to_sample = self.logits_warper(None, z_to_sample)
        p_to_sample = z_to_sample.softmax(-1).clamp(0, 1).nan_to_num() 
        indices = torch.multinomial(p_to_sample, num_samples=1) # bs, 1
        logprobs = torch.gather(z[:, -1], -1, indices) # bs, 1
        self.send_new_tokens[index] = indices
        
        self.ret_tokens[
            index*self.token_micro_batch_size*self.num_completions:(index+1)*self.token_micro_batch_size*self.num_completions,
            self.i_current_token
        ] = indices.squeeze(-1)
        self.ret_token_logprobs[
            index*self.token_micro_batch_size*self.num_completions:(index+1)*self.token_micro_batch_size*self.num_completions,
            self.i_current_token
        ] = logprobs.squeeze(-1)
        
        if index == self.token_micro_batch_num - 1:
            self.i_current_token += 1