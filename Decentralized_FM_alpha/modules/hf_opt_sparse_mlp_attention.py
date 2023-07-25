from typing import List, Optional, Tuple, Union
import numpy as np
import os
import glob
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.opt.modeling_opt import ACT2FN
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.opt.modeling_opt import OPTAttention as _OPTAttention
from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding
from transformers.models.opt.configuration_opt import OPTConfig as GPTConfig


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(float("-inf")), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def _prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class GPTEmbeddings(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.word_embed_proj_dim,
            self.padding_idx,
            device=device,
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
                device=device,
            )
        else:
            self.project_in = None

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = torch.nn.utils.skip_init(cls, config).eval()  # fast init
        try:
            module.load_state_dict(
                torch.load(
                    os.path.join(
                        model_path,
                        "pytorch_embs.pt",
                    )
                )
            )
        except:
            print("Cannot load from <model_name>. The model is randomly initialized.")
        return module

    def forward(self, input_ids, past_layer=None, mask=None, **kargs):
        if mask is None:
            if past_layer is not None:
                past_length = past_layer[0].size(2)
            else:
                past_length = 0
        else:
            # masked tokens
            past_length = (mask - 1).sum(-1, keepdims=True)
            if past_layer is not None:
                past_length += past_layer[0].size(2)

        device = input_ids.device
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        inputs_embeds = self.embed_tokens(input_ids)
        # position ids
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_ids = position_ids + past_length + self.embed_positions.offset
        position_ids[position_ids < 0] = 0

        position_embeds = F.embedding(
            position_ids,
            self.embed_positions.weight,
            self.embed_positions.padding_idx,
            self.embed_positions.max_norm,
            self.embed_positions.norm_type,
            self.embed_positions.scale_grad_by_freq,
            self.embed_positions.sparse,
        )

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + position_embeds
        return hidden_states


class OPTAttention(_OPTAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        device="cpu",
    ):
        super(_OPTAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def prepare_head_mask(
        self,
        hidden_states: torch.Tensor,
    ):
        """This function uses sparsity predictor to create an attention head mask for simulation."""
        self.predictor = self.predictor.float()
        bsz, tgt_len, _ = hidden_states.size()

        with torch.no_grad():
            _logit = self.predictor(hidden_states.reshape(-1, self.embed_dim).float())

            _, _top_k_indices = _logit.topk(int(self.topk), dim=1)
            _top_k_indices = _top_k_indices[:, : int(self.topk)].reshape(
                bsz, tgt_len, int(self.topk)
            )
            _top_k_indices = _top_k_indices.transpose(1, 2)
            _head_mask = torch.zeros(
                bsz,
                self.num_heads,
                tgt_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).scatter_(1, _top_k_indices, 1)
        return _head_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        previous_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # get activated head from sparsity predictor
        hmask = None
        if previous_emb != None and self.predictor != None:
            hmask = self.prepare_head_mask(previous_emb)
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            dtype_attn_weights = attn_weights.dtype

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if dtype_attn_weights == torch.float16:
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(dtype_attn_weights)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)

        if hmask is not None:
            attn_output = hmask.unsqueeze(-1) * attn_output

        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class GPTBlock(OPTDecoderLayer):
    def __init__(self, config, *args, use_checkpoint=True, device="cpu", **kargs):
        # super().__init__(config=config, *args, **kargs)
        super(OPTDecoderLayer, self).__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            device=device,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, device=device)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, device=device)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, device=device)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, device=device)

        self.config = config
        self.use_checkpoint = use_checkpoint

        self.default_comp_stream = torch.cuda.default_stream()
        self.sparse_comp_stream = torch.cuda.Stream(
            device=torch.device("cuda"), priority=-1
        )
        self.event_done = torch.cuda.Event(enable_timing=False, blocking=False)

    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = GPTConfig.from_pretrained(model_path)

        module = torch.nn.utils.skip_init(cls, config).eval()  # fast init
        try:
            module.load_state_dict(
                torch.load(
                    os.path.join(
                        model_path,
                        f"pytorch_{layer_index}.pt",
                    )
                )
            )
        except:
            print("Cannot load from <model_name>. The model is randomly initialized.")
        # initialize and loading mlp predictor
        if layer_index < int(os.environ["LAYER"]):
            module.predictor = nn.Sequential(
                nn.Linear(module.embed_dim, 1000, bias=None),
                nn.Linear(1000, config.ffn_dim, bias=None),
            )
            predictor_path = os.environ["SPRARSE_PATH"]
            module.topk = int(os.environ["TOPK"])
            try:
                predictor_path = glob.glob(
                    f"{predictor_path}/c4_layer{layer_index}*.pt"
                )[0]
                print(f"loading mlp sparse predictor from {predictor_path}")
                module.predictor.load_state_dict(torch.load(predictor_path))
            except:
                print(
                    f"Cannot mlp sparse predictor {layer_index}. The model is randomly initialized."
                )
        else:
            module.predictor = None

        # initialize and loading attn predictor
        if 5 <= layer_index <= 33 or 63 <= layer_index < 95:
            module.self_attn.predictor = nn.Sequential(
                nn.Linear(module.embed_dim, 1000, bias=None),
                nn.Linear(1000, config.num_attention_heads, bias=None),
            )
            predictor_path = os.environ["SPRARSE_PATH"]
            module.self_attn.topk = float(os.environ["ATTN_TOPK_1"])
            try:
                predictor_path = glob.glob(
                    f"{predictor_path}/c4_att_layer{layer_index}*.pt"
                )[0]
                print(f"loading attnetion sparse predictor from {predictor_path}")
                module.self_attn.predictor.load_state_dict(torch.load(predictor_path))
            except:
                print(
                    f"Cannot load attnetion sparse predictor {layer_index}. The model is randomly initialized."
                )
        elif layer_index > 33 and layer_index < 63:
            module.self_attn.predictor = nn.Sequential(
                nn.Linear(module.embed_dim, 1000, bias=None),
                nn.Linear(1000, config.num_attention_heads, bias=None),
            )
            predictor_path = os.environ["SPRARSE_PATH"]
            module.self_attn.topk = float(os.environ["ATTN_TOPK_2"])
            try:
                predictor_path = glob.glob(
                    f"{predictor_path}/c4_att_layer{layer_index}*.pt"
                )[0]
                print(f"loading attnetion sparse predictor from {predictor_path}")
                module.self_attn.predictor.load_state_dict(torch.load(predictor_path))
            except:
                print(
                    f"Cannot load attnetion sparse predictor {layer_index}. The model is randomly initialized."
                )
        else:
            module.self_attn.predictor = None

        module.layer_index = layer_index
        module.fc2.weight.data = module.fc2.weight.data.t().contiguous()
        module.self_attn.layer_index = layer_index

        return module

    def prepare_fc_weights(self, hidden_states: torch.Tensor):
        with torch.no_grad():
            self.predictor = self.predictor.float()

            _logit = self.predictor(hidden_states.reshape(-1, self.embed_dim).float())
            _, _top_indices = _logit.topk(self.topk, dim=1)
            _top_k_indices = _top_indices[:, : self.topk]
            self._mask = torch.zeros_like(_logit)
            self._mask = self._mask.scatter(1, _top_k_indices, 1).bool().half()

    def forward(
        self, x: torch.Tensor, layer_past=None, mask=None, previous_emb=None
    ) -> torch.Tensor:
        if layer_past is not None:
            past_length = layer_past[0].size(2)
        else:
            past_length = 0
        if mask is None:
            mask = torch.ones(
                (x.size(0), x.size(1) + past_length), dtype=torch.bool, device=x.device
            )
        attention_mask = _prepare_decoder_attention_mask(
            mask, x.shape[:2], x, past_length
        )

        hidden_states = x  # alias
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, _, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=layer_past,
            previous_emb=previous_emb,
        )
        hidden_states = residual + hidden_states

        # use mlp sparsity predictor to get selected neurons
        self._mask = None
        if self.predictor != None:
            self.prepare_fc_weights(residual)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        if self.predictor != None:
            hidden_states = hidden_states * self._mask
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = torch.nn.functional.linear(
            hidden_states, self.fc2.weight.data.T, bias=self.fc2.bias.data
        )

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        return hidden_states, present


class GPTLMHead(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size, device=device)
        else:
            self.final_layer_norm = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
                device=device,
            )
        else:
            self.project_out = None

        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False, device=device
        )

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        # module = cls(config).eval()
        module = torch.nn.utils.skip_init(cls, config).eval()  # fast init
        try:
            module.load_state_dict(
                torch.load(
                    os.path.join(
                        model_path,
                        "pytorch_lm_head.pt",
                    )
                )
            )
        except:
            print("Cannot load from <model_name>. The model is randomly initialized.")
        return module

    def forward(self, x, input_ids=None):
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        if self.project_out is not None:
            x = self.project_out(x)
        x = self.lm_head(x)
        return x
