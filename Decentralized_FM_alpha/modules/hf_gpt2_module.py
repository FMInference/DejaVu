import os
import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as _GPT2Block
from transformers.models.gpt2.configuration_gpt2 import GPT2Config as GPTConfig


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_embs.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, input_ids, past_layer=None, mask=None):
        
        if mask is None:
            if past_layer is not None:
                past_length = past_layer[0].size(2)
            else:
                past_length = 0
        else:
            # masked tokens
            past_length = (mask-1).sum(-1, keepdims=True)
            if past_layer is not None:
                past_length += past_layer[0].size(2)
                
        device = input_ids.device
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        # position ids
        position_ids = torch.arange(
            0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_ids = position_ids + past_length
        position_ids[position_ids<0] = 0

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # hidden_states = self.drop(hidden_states)

        return hidden_states


class GPTBlock(_GPT2Block):
    def __init__(self, config, *args, use_checkpoint=True, **kargs):
        super().__init__(config=config, *args, **kargs)
        self.config = config
        self.use_checkpoint = use_checkpoint
        
    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, f'pytorch_{layer_index}.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, x: torch.Tensor, layer_past=None, mask=None) -> torch.Tensor:
        
        if mask is not None:
            # bool -> float
            attention_mask = 1000*(mask[:, None, None, :]-1)
        else:
            attention_mask = None
        
        res = x
        x = self.ln_1(x)
        x, present = self.attn(x, use_cache=True, layer_past=layer_past, attention_mask=attention_mask)
        x = res + x
        
        res = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = res + x

        return x, present


class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_lm_head.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, x, input_ids=None):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x

