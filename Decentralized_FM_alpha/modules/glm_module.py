import os
import math
import torch
from torch import nn
import torch.nn.functional as F


class GPTConfig:
    
    def __init__(self):
        
        self.num_layers = 70
        self.hidden_size = 12288
        self.inner_hidden_size = 32768
        self.vocab_size = 150528
        self.num_attention_heads = 96
        self.max_sequence_length = 2048
        self.layernorm_epsilon = 1e-5
        self.tokenizer_type = 'icetk-glm-130B'
        self.layernorm_order = 'post'
    
    @classmethod
    def from_pretrained(cls, *args, **kargs):
        return cls()
    
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions
    
@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k
    
def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half, learnable=False, device=torch.device('cpu')):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        inv_freq = inv_freq.to(precision)
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq, False)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]
    
    
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, layer_id, hidden_size_per_attention_head=None):
        super().__init__()
        
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = num_attention_heads
        self.hidden_size_per_partition = hidden_size
            
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head

        # Strided linear layer.
        self.query_key_value = nn.Linear(
            hidden_size,
            3 * self.inner_hidden_size,
        )

        self.dense = nn.Linear(
            self.inner_hidden_size,
            hidden_size,
        )
        
        self.rotary_emb = RotaryEmbedding(
            self.hidden_size_per_attention_head,
            base=10000,
            precision=torch.float32,
            learnable=False,
        )
        
        self.scale_mask_softmax = None # TODO: can optimize

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)
    
    def attention_fn(
        self, query_layer, key_layer, value_layer, attention_mask, mem=None, 
        scaling_attention_score=True,
        **kw_args
    ):

        # seqlen, batch, head, hidden_size
        seq_len, b, nh, hidden_size = key_layer.shape

        # b, seqlen, stack, head, hidden
        # cache_kv = (
        #     torch.stack((key_layer, value_layer))
        #     .permute(2, 1, 0, 3, 4)
        #     .detach()
        #     .contiguous()
        #     .view(b, seq_len, nh * hidden_size * 2)
        # )
        
        if mem is not None:  # the first time, mem is None
            # might change batch_size
            # b, seqlen, head, hidden -> seqlen, b, head, hidden
            memk, memv = mem[0], mem[1] # (bs, nhead, seq_len, xxx)
            memk = memk.permute(2, 0, 1, 3)
            memv = memv.permute(2, 0, 1, 3)
            key_layer = torch.cat((memk, key_layer), dim=0)
            value_layer = torch.cat((memv, value_layer), dim=0)
            
        present = (
            key_layer.permute(1, 2, 0, 3),
            value_layer.permute(1, 2, 0, 3),
        )

        query_key_layer_scaling_coeff = float(self.layer_id + 1)
        if scaling_attention_score:
            query_layer = query_layer / (math.sqrt(self.hidden_size_per_attention_head) * query_key_layer_scaling_coeff)

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        matmul_result = torch.zeros( # torch.empty sometimes yields nan
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # if log_attention_weights is not None:
        #     attention_scores += log_attention_weights

        if self.scale_mask_softmax:
            self.scale_mask_softmax.scale = query_key_layer_scaling_coeff
            attention_probs = self.scale_mask_softmax(attention_scores, attention_mask.contiguous())
        else:
            if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
                # if auto-regressive, skip
                attention_scores.masked_fill_(attention_mask, -10000.0)

            attn_dtype = attention_scores.dtype
                
            attention_scores = attention_scores.float()
            attention_scores = attention_scores * query_key_layer_scaling_coeff

            attention_probs = F.softmax(attention_scores, dim=-1)

            attention_probs = attention_probs.to(attn_dtype)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, present

    def forward(self, hidden_states, layer_past=None, mask=None, *args, **kw_args):

        attn = self
        attention_fn = self.attention_fn

        # [seq, b, 3 * hn * np]
        mixed_raw_layer = attn.query_key_value(hidden_states)

        # [seq, b, (np * 3 * hn)] --> [seq, b, np, 3 * hn]
        new_tensor_shape = mixed_raw_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        kw_args["position_ids"] = kw_args["position_ids"].transpose(0, 1)

        cos, sin = self.rotary_emb(value_layer, seq_len=kw_args["position_ids"].max() + 1)
        
        query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, kw_args["position_ids"])
        
        context_layer, present = attention_fn(query_layer, key_layer, value_layer, mask, layer_past)

        output = attn.dense(context_layer)

        return output, present
    
    
class GEGLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_fn = F.gelu

    def forward(self, x):
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)
    
    
class MLP(torch.nn.Module):
    def __init__(self, hidden_size, inner_hidden_size=None, layer_id=None):
        super().__init__()
        self.layer_id = layer_id
        self.activation_func = GEGLU()
        # Set output layer initialization if not provided.
        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = nn.Linear(
            self.hidden_size,
            2 * self.inner_hidden_size, # for geglu
        )
        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            self.inner_hidden_size,
            self.hidden_size,
        )

    def forward(self, hidden_states, **kw_args):
        
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        
        return hidden_states
    

class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig()
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_embs.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, input_ids, past_layer=None, mask=None):
    
        hidden_states = self.word_embeddings(input_ids)

        return hidden_states
    

class GPTBlock(nn.Module):
    
    # TODO: should be object's attribute
    echo_prompt = False
    
    def __init__(self, config, layer_number, *args, use_checkpoint=True, device='cpu', **kargs):
        super().__init__()
        self.config = config
        self.use_checkpoint = use_checkpoint
        self.layer_number = layer_number
        
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.inner_hidden_size = config.inner_hidden_size
        self.num_layers = config.num_layers
        self.alpha = (2 * self.num_layers) ** 0.5
        self.layernorm_epsilon = config.layernorm_epsilon
        
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)
        
        self.attention = SelfAttention(self.hidden_size, num_attention_heads=self.num_attention_heads, layer_id=layer_number)
        
        self.mlp = MLP(self.hidden_size, inner_hidden_size=self.inner_hidden_size, layer_id=layer_number)
        
        if self.echo_prompt:
            max_positions = 2048
            self.register_buffer(
                "bias",
                (1 - torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                    1, 1, max_positions, max_positions
                )).bool(),
                persistent=False,
            )
        
    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = GPTConfig()
            
        _reset_parameters = nn.Linear.reset_parameters
        def dummy(*args, **kargs):
            pass
        nn.Linear.reset_parameters = dummy # disable init
        module = cls(config, layer_number=layer_index).eval()
        nn.Linear.reset_parameters = _reset_parameters
        
        # !!! cannot use skip_init, it will skip init non-persisitent buffer, e.g. causal mask and sin/cos
        # module = torch.nn.utils.skip_init(cls, config, layer_index).eval() # fast init
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, f'pytorch_{layer_index}.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
            
        return module

    
    def forward(self, hidden_states: torch.Tensor, layer_past=None, mask=None) -> torch.Tensor:
           
        if mask is None:
            mask = torch.ones(hidden_states.size(0), hidden_states.size(1), device=hidden_states.device, dtype=torch.bool)
        else:
            mask = mask.bool()
        position_ids = (mask.cumsum(-1) - 1).relu() # avoid negative id
        
        if not self.echo_prompt:
            if layer_past is None:
                extend_mask = ~mask
                extend_mask = extend_mask.unsqueeze(1) | extend_mask.unsqueeze(2)
                extend_mask[:, :-1, -1] = 1 # [sop] is always the first token
            else:
                extend_mask = ~mask
                extend_mask = extend_mask.unsqueeze(1) | extend_mask[:, -1:].unsqueeze(2)
                position_ids = position_ids[:, layer_past[0].size(2):]
            extend_mask = extend_mask.unsqueeze(1) # head dim
        else:
            assert layer_past is None
            extend_mask = ~mask
            extend_mask = extend_mask.unsqueeze(1) | extend_mask.unsqueeze(2)
            extend_mask = extend_mask.unsqueeze(1) # head dim
            
            causal_mask = self.bias[:, :, :hidden_states.size(1), :hidden_states.size(1)]
            # print(mask[0].sum())
            # print(extend_mask[0, 0, :8, :8])
            # print(causal_mask[0, 0, :8, :8])
            extend_mask = extend_mask | causal_mask
            # print(extend_mask[0, 0, :8, :8])
        
        hidden_states = hidden_states.transpose(0, 1) # transpose
                
        attention_input = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, present = self.attention(attention_input, layer_past=layer_past, mask=extend_mask, position_ids=position_ids) # TODO kv

        # Residual connection.
        hidden_states = attention_input * self.alpha + attention_output

        mlp_input = self.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.mlp(mlp_input)

        # Second residual connection.
        output = mlp_input * self.alpha + mlp_output
        
        output = output.transpose(0, 1)
                   
        return output, present
    
    
class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.layernorm_epsilon = config.layernorm_epsilon
        
        self.final_layernorm = nn.LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=None,
        )
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig()
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_lm_head.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, hidden_states, *args, **kargs):
        
        output = self.final_layernorm(hidden_states)
        output = self.lm_head(output)

        return output

