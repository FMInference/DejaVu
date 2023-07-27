import math
import re

from collections import OrderedDict

import torch
import torch.nn.functional as F

from einops import rearrange


def remap_state_dict_opt(state_dict, config):
    def key_mapping_model(key):
        key = re.sub(r"^model.decoder.", "transformer.", key)
        # The OPT-350m model uses '^decoder' instead of '^model.decoder'
        key = re.sub(r"^decoder.", "transformer.", key)
        return key

    state_dict = OrderedDict((key_mapping_model(k), v) for k, v in state_dict.items())

    # Word embedding and position embedding
    def key_mapping_emb(key):
        key = re.sub(
            r"^transformer.embed_tokens.",
            "transformer.embeddings.word_embeddings.",
            key,
        )
        # The OPT-350m model uses has project_in and project_out
        key = re.sub(
            r"^transformer.project_in.", "transformer.embeddings.project_in.", key
        )
        key = re.sub(r"^transformer.project_out.", "project_out.", key)
        key = re.sub(
            r"^transformer.position_embeddings.",
            "transformer.embeddings.position_embeddings.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    if "transformer.embed_positions.weight" in state_dict.keys():
        # OPT uses the first 2 indices of pos_emb for padding tokens
        pos_embeddings = state_dict.pop("transformer.embed_positions.weight")
        state_dict[
            "transformer.embeddings.position_embeddings.weight"
        ] = pos_embeddings[2:]
        word_embeddings = state_dict.pop(
            "transformer.embeddings.word_embeddings.weight"
        )
        # It's possible that vocab_size is padded to be a multiple of 8, for example.
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple)
            * pad_vocab_size_multiple
        )
        state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
            word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
        )
        state_dict["lm_head.weight"] = state_dict[
            "transformer.embeddings.word_embeddings.weight"
        ]
    # if "transformer.embeddings.position_embeddings.weight" in state_dict.keys():
    #     # OPT uses the first 2 indices of pos_emb for padding tokens
    #     pos_embeddings = state_dict.pop(
    #         "transformer.embeddings.position_embeddings.weight"
    #     )
    #     state_dict[
    #         "transformer.embeddings.position_embeddings.weight"
    #     ] = pos_embeddings[2:]
    #     word_embeddings = state_dict.pop(
    #         "transformer.embeddings.word_embeddings.weight"
    #     )
    #     # It's possible that vocab_size is padded to be a multiple of 8, for example.
    #     pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    #     vocab_size = (
    #         math.ceil(config.vocab_size / pad_vocab_size_multiple)
    #         * pad_vocab_size_multiple
    #     )
    #     state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
    #         word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    #     )
    #     state_dict["lm_head.weight"] = state_dict[
    #         "transformer.embeddings.word_embeddings.weight"
    #     ]

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^final_layer_norm.", r"transformer.ln_f.", key)
        key = re.sub(
            r"^transformer.layers.(\d+).self_attn_layer_norm.",
            r"transformer.layers.\1.norm1.",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).final_layer_norm.",
            r"transformer.layers.\1.norm2.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        return re.sub(
            r"^transformer.layers.(\d+).fc(1|2).",
            r"transformer.layers.\1.mlp.fc\2.",
            key,
        )

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())
    # Attention
    for l in range(config.num_hidden_layers):
        Wq = state_dict.pop(f"transformer.layers.{l}.self_attn.q_proj.weight")
        Wk = state_dict.pop(f"transformer.layers.{l}.self_attn.k_proj.weight")
        Wv = state_dict.pop(f"transformer.layers.{l}.self_attn.v_proj.weight")
        bq = state_dict.pop(f"transformer.layers.{l}.self_attn.q_proj.bias")
        bk = state_dict.pop(f"transformer.layers.{l}.self_attn.k_proj.bias")
        bv = state_dict.pop(f"transformer.layers.{l}.self_attn.v_proj.bias")
        state_dict[f"transformer.layers.{l}.mixer.Wqkv.weight"] = torch.cat(
            [Wq, Wk, Wv], dim=0
        )
        state_dict[f"transformer.layers.{l}.mixer.Wqkv.bias"] = torch.cat(
            [bq, bk, bv], dim=0
        )

    def key_mapping_attn(key):
        return re.sub(
            r"^transformer.layers.(\d+).self_attn.out_proj.",
            r"transformer.layers.\1.mixer.out_proj.",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def shard_state_dict_tp(state_dict, config, world_size, rank):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    shared_state_dict = {}

    def shard_first_dim(new, old, key):
        x = old[key]
        dim = x.shape[0] // world_size
        new[key] = x[rank * dim : (rank + 1) * dim]

    def shard_last_dim(new, old, key):
        x = old[key]
        dim = x.shape[-1] // world_size
        new[key] = x[..., rank * dim : (rank + 1) * dim]

    def shard_qkv_headdim(new, old, key):
        x = rearrange(old[key], "(three d) ... -> three d ...", three=3)
        dim = x.shape[1] // world_size
        new[key] = rearrange(
            x[:, rank * dim : (rank + 1) * dim], "three d ... -> (three d) ..."
        )

    shard_first_dim(
        shared_state_dict, state_dict, "transformer.embeddings.word_embeddings.weight"
    )
    if "lm_head.weight" in state_dict:
        shard_first_dim(shared_state_dict, state_dict, "lm_head.weight")
    if "transformer.embeddings.position_embeddings.weight" in state_dict:
        shard_last_dim(
            shared_state_dict,
            state_dict,
            "transformer.embeddings.position_embeddings.weight",
        )
    for i in range(config.num_hidden_layers):
        shard_qkv_headdim(
            shared_state_dict, state_dict, f"transformer.layers.{i}.mixer.Wqkv.weight"
        )
        shard_qkv_headdim(
            shared_state_dict, state_dict, f"transformer.layers.{i}.mixer.Wqkv.bias"
        )
        shard_last_dim(
            shared_state_dict,
            state_dict,
            f"transformer.layers.{i}.mixer.out_proj.weight",
        )
        shard_first_dim(
            shared_state_dict, state_dict, f"transformer.layers.{i}.mlp.fc1.weight"
        )
        shard_first_dim(
            shared_state_dict, state_dict, f"transformer.layers.{i}.mlp.fc1.bias"
        )
        shard_last_dim(
            shared_state_dict, state_dict, f"transformer.layers.{i}.mlp.fc2.weight"
        )
        if rank == 0:
            shared_state_dict[f"transformer.layers.{i}.mlp.fc2.bias"] = state_dict[
                f"transformer.layers.{i}.mlp.fc2.bias"
            ]
            shared_state_dict[
                f"transformer.layers.{i}.mixer.out_proj.bias"
            ] = state_dict[f"transformer.layers.{i}.mixer.out_proj.bias"]
        shared_state_dict[f"transformer.layers.{i}.norm1.weight"] = state_dict[
            f"transformer.layers.{i}.norm1.weight"
        ]
        shared_state_dict[f"transformer.layers.{i}.norm1.bias"] = state_dict[
            f"transformer.layers.{i}.norm1.bias"
        ]
        shared_state_dict[f"transformer.layers.{i}.norm2.weight"] = state_dict[
            f"transformer.layers.{i}.norm2.weight"
        ]
        shared_state_dict[f"transformer.layers.{i}.norm2.bias"] = state_dict[
            f"transformer.layers.{i}.norm2.bias"
        ]
        # shared_state_dict[f"transformer.ln_f.weight"] = state_dict[
        #     "transformer.ln_f.weight"
        # ]
        # shared_state_dict[f"transformer.ln_f.bias"] = state_dict[
        #     "transformer.ln_f.bias"
        # ]
        shared_state_dict[f"transformer.ln_f.weight"] = state_dict[
            "transformer.final_layer_norm.weight"
        ]
        shared_state_dict[f"transformer.ln_f.bias"] = state_dict[
            "transformer.final_layer_norm.bias"
        ]
    return shared_state_dict
