import os
import torch
from torch import nn
import copy
from transformers.models.t5.modeling_t5 import T5Block as _T5Block
from transformers.models.t5.modeling_t5 import T5LayerNorm as _T5LayerNorm
from transformers.models.t5.configuration_t5 import T5Config as EncDecConfig
from transformers.modeling_utils import ModuleUtilsMixin


class EncDecEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = EncDecConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_embs.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, input_ids, **kargs,):
        
        hidden_states = self.shared(input_ids)

        return hidden_states
    
    
class EncBlock(_T5Block):
    def __init__(self, config, *args, use_checkpoint=True, **kargs):
        # always true for distributed inference
        has_relative_attention_bias = True
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        super().__init__(config=encoder_config, has_relative_attention_bias=has_relative_attention_bias, *args, **kargs)
        self.config = config
        self.use_checkpoint = use_checkpoint
        
    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = EncDecConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(
                {k.replace('encoder.block.0.', ''): v for k, v in torch.load(os.path.join(
                    model_path, f'pytorch_enc_{layer_index}.pt',
                )).items()}
            )
        except Exception as e:
            raise e
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, 
                hidden_states: torch.Tensor, 
                mask=None,
    ) -> torch.Tensor:
        
        if mask is not None:
            # bool -> float
            attention_mask = 1e9*(mask[:, None, None, :]-1)
        else:
            attention_mask = None
        
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=None, # compute on demand
            past_key_value=None,
            use_cache=False,
        )
        hidden_states = self_attention_outputs[0]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        return hidden_states
        


class DecBlock(_T5Block):
    def __init__(self, config, *args, use_checkpoint=True, **kargs):
        # always true for distributed inference
        has_relative_attention_bias = True
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        # assert num_layers == num_decoder_layers
        # decoder_config.num_layers = config.num_decoder_layers 
        super().__init__(config=decoder_config, has_relative_attention_bias=has_relative_attention_bias, *args, **kargs)
        self.config = config
        self.use_checkpoint = use_checkpoint
        
    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = EncDecConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(
                {k.replace('decoder.block.0.', ''): v for k, v in torch.load(os.path.join(
                    model_path, f'pytorch_dec_{layer_index}.pt',
                )).items()}
            )
        except Exception as e:
            raise e
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, 
                hidden_states: torch.Tensor, 
                mask=None,
                layer_past=None, 
                encoder_hidden_states=None,
                encoder_mask=None,
                encoder_layer_past=None,
    ) -> torch.Tensor:
        
        input_shape = hidden_states.shape[:2]
        if mask is None:
            mask = torch.ones(*input_shape).to(hidden_states.device)
        attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
            input_shape, mask,
        )
        attention_mask = 1e9*(attention_mask-1)
        
            
        if encoder_mask is not None:
            # bool -> float
            encoder_attention_mask = 1e9*(encoder_mask[:, None, None, :]-1)
        else:
            encoder_attention_mask = None
        
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=None, # compute on demand
            past_key_value=layer_past,
            use_cache=True,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        
        assert encoder_hidden_states is not None
        # cross attn
        # the actual query length is unknown for cross attention
        # if using past key value states. Need to inject it here
        if present_key_value_state is not None:
            query_length = present_key_value_state[0].shape[2]
        else:
            query_length = None

        cross_attention_outputs = self.layer[1](
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=None, # compute on demand
            past_key_value=encoder_layer_past,
            query_length=query_length,
            use_cache=True,
        )
        hidden_states = cross_attention_outputs[0]
        cross_attn_past_key_value = cross_attention_outputs[1]

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        return hidden_states, present_key_value_state, cross_attn_past_key_value
       
        
class EncHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_layer_norm = _T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = EncDecConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_enc_head.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, x, input_ids=None):
        x = self.final_layer_norm(x)
        return x

class DecHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_layer_norm = _T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.config = config
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = EncDecConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_dec_head.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, x, input_ids=None):
        x = self.final_layer_norm(x)
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            x = x * (self.config.d_model**-0.5)
        x = self.lm_head(x)
        return x

