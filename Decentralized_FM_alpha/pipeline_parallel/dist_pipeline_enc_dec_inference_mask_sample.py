import time
import json
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_processor, get_logits_warper


from .dist_pipeline_inference_mask_sample import DistSampleInferenceMaskAsync


class DistSampleEncDecInferenceMaskAsync(DistSampleInferenceMaskAsync):
    #TODO
    
    def __init__(self, args, device, rank=None):
        super().__init__(args, device, rank=rank)
        self.num_completions = args.num_completions
        self.update_processors(args)
        
        self.encoder_seq_emb = torch.zeros(
            (self.seq_num * self.micro_batch_size, self.input_seq_length, self.embedding_dim),
            requires_grad=False, device=self.device, dtype=self.dtype
        )
        
    def update_processors(self, args):
        self.logits_processor = get_logits_processor()
        self.logits_warper = get_logits_warper(
            top_k = (None if args.top_k is None or args.top_k == 0 else args.top_k),
            top_p = (None if args.top_p is None or args.top_p <= 0 else args.top_p),
            temperature = args.temperature,
            num_beams = 1,
        )

    def change_buffer_size(self):
        self._init_events()
        self._init_buffers()
        self.encoder_seq_emb = torch.zeros(
            (self.seq_num * self.micro_batch_size, self.input_seq_length, self.embedding_dim),
            requires_grad=False, device=self.device, dtype=self.dtype
        )

    def _get_embedding_size(self):
        if self.model_type == 't5':
            from modules.hf_t5_module import EncDecConfig
            config = EncDecConfig.from_pretrained(self.model_name)
            self.config = config # keep config here
            return config.d_model
        else:
            raise Exception(f'unknown model type {self.model_type}')
            
    def _create_layers(self):
        if self.model_type == 't5':
            from modules.hf_t5_module import (
                EncDecEmbeddings,
                EncBlock,
                DecBlock,
                EncHead,
                DecHead,
            )
        else:
            raise Exception(f'unknown model type {self.model_type}')
        
        if self.pp_rank == 0:
            self.layers['emb'] = EncDecEmbeddings.from_pretrained(
                self.model_name
            ).to(self.dtype).eval().to(self.device)
        for layer_index in range(self.num_layers):
            # global layer indexing could be an argument
            global_layer_index = self.num_layers * self.pp_rank + layer_index
            print(f'loading layer {global_layer_index}')
            self.layers['enc_block'+str(layer_index)] = EncBlock.from_pretrained(
                self.model_name, layer_index=global_layer_index
            ).to(self.dtype).eval().to(self.device)
            self.layers['dec_block'+str(layer_index)] = DecBlock.from_pretrained(
                self.model_name, layer_index=global_layer_index
            ).to(self.dtype).eval().to(self.device)
        if self.pp_rank == self.pipeline_group_size - 1:
            self.layers['enc_head'] = EncHead.from_pretrained(
                self.model_name
            ).to(self.dtype).eval().to(self.device)
            self.layers['dec_head'] = DecHead.from_pretrained(
                self.model_name
            ).to(self.dtype).eval().to(self.device)
            
    def _init_cached_seqs_and_attentions(self):
        self.merged = False
        if not hasattr(self, 'cached_cross_attention'):
            self.cached_cross_attention = []
        self.cached_cross_attention.clear()
        self.cached_attention.clear()
        for _ in range(self.num_layers):
            self.cached_cross_attention.append(None)
        for _ in range(self.num_layers):
            self.cached_attention.append(None)
            
    def _forward_compute_prompt_seq(self, index, seq=None, mask=None):
        print("Compute prompt seq<", index, ">.")
        if self.pp_rank == 0:
            self.input_seq_emb[index] = self.layers['emb'](seq)
        current_emb = None
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                current_emb = self.layers['enc_block' + str(layer_index)](self.input_seq_emb[index], mask=mask)
            else:
                current_emb = self.layers['enc_block' + str(layer_index)](current_emb, mask=mask)
                
        if self.pp_rank == self.pipeline_group_size - 1:
            # layer norm
            self.output_seq_emb[index] = self.layers['enc_head'](current_emb)
        else:
            self.output_seq_emb[index] = current_emb
            
    def _forward_compute_generate_token(self, step, mask=None):
        print("Compute generate seq<", step, ">.")
        if self.pp_rank == 0:
            current_emb = self.layers['emb'](self.recv_new_token[step])
        else:
            current_emb = self.input_token_emb[step]
        for layer_index in range(self.num_layers):
            if layer_index != self.num_layers - 1:
                current_emb, self.cached_attention[layer_index], self.cached_cross_attention[layer_index] = \
                    self.layers['dec_block' + str(layer_index)](
                    current_emb, 
                    layer_past=self.cached_attention[layer_index],
                    encoder_hidden_states=self.encoder_seq_emb,
                    encoder_layer_past=self.cached_cross_attention[layer_index],
                    encoder_mask=mask,
                )
            else:
                self.output_token_emb[step], self.cached_attention[layer_index], self.cached_cross_attention[layer_index] = \
                    self.layers['dec_block' + str(layer_index)](
                    current_emb, 
                    layer_past=self.cached_attention[layer_index],
                    encoder_hidden_states=self.encoder_seq_emb,
                    encoder_layer_past=self.cached_cross_attention[layer_index],
                    encoder_mask=mask,
                )
        if self.pp_rank == self.pipeline_group_size - 1:
            self._generate_new_token(step)

    def _generate_new_token(self, step):
        assert self.pp_rank == self.pipeline_group_size - 1
        if step >= 0:
            z = self.layers['dec_head'](self.output_token_emb[step])
            if torch.isnan(z).any():
                print('containing nan, setting them to zero!')
                print(z)
            z = z.float().nan_to_num() # test if fp32 whether cause inf
            z = torch.nn.functional.log_softmax(z, -1)

            if self.top_k_per_token > 0:
                logprobs, indices = z.topk(k=self.top_k_per_token, dim=-1)
                self.ret_topk_tokens[:, step] = indices.squeeze(1)
                self.ret_topk_token_logprobs[:, step] = logprobs.squeeze(1)

            # [:, -1] because multinomial only accept 1/2d tensors
            z_to_sample = z[:, -1] # bs, vocab
            z_to_sample = self.logits_warper(None, z_to_sample)
            indices = torch.multinomial(z_to_sample.softmax(-1).clamp(0, 1).nan_to_num() , num_samples=1) # bs, 1
            logprobs = torch.gather(z[:, -1], -1, indices) # bs, 1
            self.send_new_tokens[step] = indices

            self.ret_tokens[:, step] = indices.squeeze(-1)
            self.ret_token_logprobs[:, step] = logprobs.squeeze(-1)
        
        else:
            # first token is always pad
            self.send_new_tokens[0][:] = self.config.pad_token_id
    
    def _process_mask_during_generation(self, attention_mask):
        # encoder mask do not need to update
        return attention_mask
        
    def _merge_cached_seqs_and_attentions(self):
        # enc dec dont need to merge
        for i in range(self.num_layers):
            self.cached_attention[i] = None

        if not self.merged:
            torch.cuda.synchronize()
            if self.pp_rank == self.pipeline_group_size - 1:
                self.encoder_seq_emb[:] = torch.cat(self.output_seq_emb, 0)

            # sync broadcast
            self.comm.broadcast(self.encoder_seq_emb, src=self.pipeline_group_size - 1,)

            torch.cuda.synchronize()
            
            self.merged = True
