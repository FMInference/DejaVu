import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock as _GPTNeoBlock
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig as GPTConfig

# @torch.jit.script
def gpt_loss_func(input, target):
    lm_logits, labels = input, target
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embed_dropout)
        
    def forward(self, input_ids):
        
        device = input_ids.device
        
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        
        # position ids
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)
        
        return hidden_states
    

class GPTBlock(_GPTNeoBlock):
    def __init__(self, config, layer_id, *args, use_checkpoint=True, **kargs):
        super().__init__(config=config, layer_id=layer_id, *args, **kargs)
        self.config = config
        self.use_checkpoint = use_checkpoint
        
        def attn_res(x: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_1(x)
            x = self.attn(x)[0]
            return x + res
        self.attn_res = attn_res
        
        def mlp_res(x: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_2(x)
            x = self.mlp(x)
            return x + res
        self.mlp_res = mlp_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if not self.training:
            x = self.attn_res(x)
            x = self.mlp_res(x)
            return x
        
        if self.use_checkpoint:
            x.requires_grad_(True)
            x = checkpoint(self.attn_res, x)
        else:
            x = self.attn_res(x)
            
        if self.use_checkpoint:
            x.requires_grad_(True)
            x = checkpoint(self.mlp_res, x)
        else:
            x = self.mlp_res(x)
        return x
    
    
class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, x, input_ids=None):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
    
        
class GPTClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        
    def forward(self, hidden_states, input_ids=None):
        
        batch_size, sequence_length = hidden_states.shape[:2]
        if input_ids is not None:
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        else:
            sequence_lengths = -1
        
        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        logits = self.score(self.ln_f(pooled_hidden_states))
        
        return logits
        