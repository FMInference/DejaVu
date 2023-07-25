import time
import json
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_processor, get_logits_warper


from .dist_pipeline_inference_mask_sample_token_pipe_sync import DistSampleInferenceMaskTokenPipeSync


class DistSampleEncDecInferenceMaskSync(DistSampleInferenceMaskTokenPipeSync):
    #TODO
    
    def __init__(self, args, device, rank=None):
        super().__init__(args, device, rank=rank)
        
        self.encoder_seq_emb = torch.zeros(
            (self.seq_num * self.micro_batch_size, self.input_seq_length, self.embedding_dim),
            requires_grad=False, device=self.device, dtype=self.dtype
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
            # alias
            self.layers['lm'] = self.layers['dec_head']
            
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
            
    def _forward_compute_generate_token(self, index, mask=None):
        print("Compute generate token batch<", index, ">.")
        
        if mask is not None and self.num_completions > 1:
            # repeat n times
            mask = mask.repeat(self.num_completions, 1)
            
        if self.pp_rank == 0:
            current_emb = self.layers['emb'](self.recv_new_token[index])
        else:
            current_emb = self.input_token_emb[index]
        for layer_index in range(self.num_layers):
            if layer_index != self.num_layers - 1:
                current_emb, self.cached_attention[layer_index][index], self.cached_cross_attention[layer_index][index] = \
                    self.layers['dec_block' + str(layer_index)](
                    current_emb, 
                    layer_past=self.cached_attention[layer_index][index],
                    encoder_hidden_states=self.encoder_seq_emb[index*self.token_micro_batch_size:(index+1)*self.token_micro_batch_size].repeat(self.num_completions, 1, 1),
                    encoder_layer_past=self.cached_cross_attention[layer_index][index],
                    encoder_mask=mask,
                )
            else:
                self.output_token_emb[index], self.cached_attention[layer_index][index], self.cached_cross_attention[layer_index][index] = \
                    self.layers['dec_block' + str(layer_index)](
                    current_emb, 
                    layer_past=self.cached_attention[layer_index][index],
                    encoder_hidden_states=self.encoder_seq_emb[index*self.token_micro_batch_size:(index+1)*self.token_micro_batch_size].repeat(self.num_completions, 1, 1),
                    encoder_layer_past=self.cached_cross_attention[layer_index][index],
                    encoder_mask=mask,
                )
        if self.pp_rank == self.pipeline_group_size - 1:
            self._generate_new_token(index)
    
    def _process_mask_during_generation(self, attention_mask):
        # encoder mask do not need to update
        return attention_mask
        
    def _merge_cached_seqs_and_attentions(self):
        
        self.i_current_token = 0
        
        if self.stop is not None:
            self.stop_flag[:] = 0
                
        if self.pp_rank == self.pipeline_group_size - 1:
            for i in range(self.token_micro_batch_num):
                self.send_new_tokens[i].data[:] = self.config.pad_token_id
            
        # enc dec dont need to merge
        for i in range(self.num_layers):
            self.cached_attention[i] = [None for _ in range(self.token_micro_batch_num)]
        for i in range(self.num_layers):
            self.cached_cross_attention[i] = [None for _ in range(self.token_micro_batch_num)]

        if not self.merged:
            torch.cuda.synchronize()
            if self.pp_rank == self.pipeline_group_size - 1:
                self.encoder_seq_emb[:] = torch.cat(self.output_seq_emb, 0)

            # sync broadcast
            self.comm.broadcast(self.encoder_seq_emb, src=self.pipeline_group_size - 1,)
                
            torch.cuda.synchronize()
            
            self.merged = True
