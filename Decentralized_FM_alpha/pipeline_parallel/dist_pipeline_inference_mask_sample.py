import time
import json
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_processor, get_logits_warper


from .dist_pipeline_inference_mask_greedy import DistGreedyInferenceMaskAsync


class DistSampleInferenceMaskAsync(DistGreedyInferenceMaskAsync):
    
    def __init__(self, args, device, rank=None):
        super().__init__(args, device, rank=rank)
        self.num_completions = args.num_completions
        self.logits_processor = get_logits_processor()
        self.logits_warper = get_logits_warper(
            top_k = (None if args.top_k is None or args.top_k == 0 else args.top_k),
            top_p = (None if args.top_p is None or args.top_p <= 0 else args.top_p),
            temperature = args.temperature,
            num_beams = 1,
        )

    def _generate_new_token(self, step):
        assert self.pp_rank == self.pipeline_group_size - 1
        if step >= 0:
            z = self.layers['lm'](self.output_token_emb[step])
            save_step = step+1
        else:
            # generate from prompt
            z = self.layers['lm'](self.initial_output_token_emb)
            # reuse 0-th new tokens
            step = 0
            save_step = 0
            
        z = torch.nn.functional.log_softmax(z, -1)
        
        if self.top_k_per_token > 0:
            logprobs, indices = z.topk(k=self.top_k_per_token, dim=-1)
            self.ret_topk_tokens[:, save_step] = indices.squeeze(1)
            self.ret_topk_token_logprobs[:, save_step] = logprobs.squeeze(1)
            
        # [:, -1] because multinomial only accept 1/2d tensors
        z_to_sample = z[:, -1] # bs, vocab
        z_to_sample = self.logits_warper(None, z_to_sample)
        indices = torch.multinomial(z_to_sample.softmax(-1), num_samples=1) # bs, 1
        logprobs = torch.gather(z[:, -1], -1, indices) # bs, 1
        self.send_new_tokens[step] = indices
        
        self.ret_tokens[:, save_step] = indices.squeeze(-1)
        self.ret_token_logprobs[:, save_step] = logprobs.squeeze(-1)

    def _init_cached_seqs_and_attentions(self):
        self.merged = False
        self.cached_attention.clear()
        for _ in range(self.num_layers):
            self.cached_attention.append([None for _ in range(self.seq_num)])
        
    def _merge_cached_seqs_and_attentions(self):
        if self.merged:
            for layer_index in range(self.num_layers):
                key, value = self.cached_attention[layer_index]
                self.cached_attention[layer_index] = (key[:, :, :self.input_seq_length], value[:, :, :self.input_seq_length])
        else:
            for layer_index in range(self.num_layers):
                key = torch.cat([kv[0] for kv in self.cached_attention[layer_index]], dim=0)
                value = torch.cat([kv[1] for kv in self.cached_attention[layer_index]], dim=0)
                self.cached_attention[layer_index] = (key, value)
            self.merged = True
        
    def inference_batch(self, input_=None, output_=None, attention_mask=None):
        self.comm.barrier()
        start_time = time.time()
        self._init_cached_seqs_and_attentions() # TODO: should I put here
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.init_time_stamp = time.time() * 1e+6
            self.init_event.record()

        with torch.no_grad():
            self.forward_seq_pipeline_stage(input_data=input_, attention_mask=attention_mask)
            for nc in range(self.num_completions):
                self.forward_new_token_pipeline_stage(attention_mask=attention_mask)
                self.comm.barrier()
                if self.pp_rank == self.pipeline_group_size - 1 and output_ is not None:
                    assert isinstance(output_, list)
                    item = {}
                    if self.generate_seq_length > 0:
                        item = {
                            'token_ids': self.ret_tokens.cpu(),
                            'token_logprobs': self.ret_token_logprobs.cpu(),
                        }
                        if self.top_k_per_token > 0:
                            item['topk_ids'] = self.ret_topk_tokens.cpu()
                            item['topk_logprobs'] = self.ret_topk_token_logprobs.cpu()
                    output_.append(item)

        end_time = time.time()
        iter_time = end_time - start_time
        print("Rank {} node whole INFERENCE iteration takes {:3.2f}s".format(self.global_rank, iter_time))
        print("-------------------------------------------")
        
        return iter_time