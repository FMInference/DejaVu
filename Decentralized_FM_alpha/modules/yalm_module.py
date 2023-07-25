import os
import math
import torch
from torch import nn

class GPTConfig:
    def __init__(self):
        # hard code for yalm
        self.num_layers = 80
        self.embedding_size = 2048
        self.hidden_size = 10240
        self.num_attention_heads = 128
        self.intermediate_size = 27308
        self.padded_vocab_size = 128000
        self.layernorm_epsilon = 1e-5
        self.max_position_embeddings = 2050 # for rotation embeddings
        
        # self.num_layers = 3
        # self.embedding_size = 512
        # self.hidden_size = 1024
        # self.num_attention_heads = 16
        # self.intermediate_size = 4096
        # self.padded_vocab_size = 50304
        # self.layernorm_epsilon = 1e-5
        # self.max_position_embeddings = 1024 # for rotation embeddings
        
        ## share
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        self.hidden_size_per_partition = self.hidden_size
        self.num_attention_heads_per_partition = self.num_attention_heads
        
        '''
        _intermediate_pad ............... 4
        activation_type ................. geglu
        adam_beta1 ...................... 0.9
        adam_beta2 ...................... 0.999
        adam_eps ........................ 1e-08
        adlr_autoresume ................. False
        adlr_autoresume_interval ........ 1000
        apply_query_key_layer_scaling ... False
        apply_residual_connection_post_layernorm  False
        attention_dropout ............... 0.1
        attention_softmax_in_fp32 ....... False
        batch_size ...................... 1
        bert_load ....................... None
        bias_dropout_fusion ............. False
        bias_gelu_fusion ................ False
        block_data_path ................. None
        checkpoint_activations .......... False
        checkpoint_in_cpu ............... False
        checkpoint_num_layers ........... 1
        clip_grad ....................... 1.0
        contigious_checkpointing ........ False
        cpu_optimizer ................... False
        cpu_torch_adam .................. False
        data_impl ....................... infer
        data_path ....................... None
        DDP_impl ........................ local
        deepscale ....................... False
        deepscale_config ................ None
        deepspeed ....................... False
        deepspeed_activation_checkpointing  False
        deepspeed_config ................ None
        deepspeed_mpi ................... False
        distribute_checkpointed_activations  False
        distributed_backend ............. nccl
        dynamic_loss_scale .............. True
        embedding_size .................. 2048
        eod_mask_loss ................... False
        eval_interval ................... 1000
        eval_iters ...................... 100
        exit_interval ................... None
        faiss_use_gpu ................... False
        finetune ........................ False
        fp16 ............................ True
        fp16_lm_cross_entropy ........... False
        fp32_allreduce .................. False
        genfile ......................... None
        greedy .......................... True
        hidden_dropout .................. 0.1
        hidden_size ..................... 10240
        hysteresis ...................... 2
        ict_head_size ................... None
        ict_load ........................ None
        indexer_batch_size .............. 128
        indexer_log_interval ............ 1000
        init_method_std ................. 0.02
        intermediate_size ............... 27312
        layernorm_epsilon ............... 1e-05
        lazy_mpu_init ................... None
        load ............................ yalm100b_checkpoint/weights
        load_release_checkpoint ......... True
        local_rank ...................... None
        log_interval .................... 100
        loss_scale ...................... None
        loss_scale_window ............... 1000
        lr .............................. None
        lr_decay_iters .................. None
        lr_decay_style .................. linear
        lr_decay_tokens ................. None
        make_vocab_size_divisible_by .... 1
        mask_prob ....................... 0.15
        max_position_embeddings ......... 1024
        memory_centric_tiled_linear ..... False
        merge_file ...................... None
        min_lr .......................... 0.0
        min_scale ....................... 1
        mmap_warmup ..................... False
        model_parallel_size ............. 8
        no_load_optim ................... False
        no_load_rng ..................... False
        no_save_optim ................... False
        no_save_rng ..................... False
        num_attention_heads ............. 128
        num_layers ...................... 80
        num_samples ..................... 0
        num_unique_layers ............... None
        num_workers ..................... 2
        out_seq_length .................. 128
        override_lr_scheduler ........... False
        param_sharing_style ............. grouped
        params_dtype .................... torch.float16
        partition_activations ........... False
        pos_encoding_type ............... rotary
        profile_backward ................ False
        query_in_block_prob ............. 0.1
        rank ............................ 0
        recompute ....................... False
        remote_device ................... none
        report_topk_accuracies .......... []
        reset_attention_mask ............ False
        reset_position_ids .............. False
        sample_context_field ............ prefix
        sample_generated_field .......... suffix
        sample_input_file ............... examples/example_cond_input.json
        sample_output_file .............. cond_output.json
        save ............................ None
        save_interval ................... None
        scaled_masked_softmax_fusion .... False
        scaled_upper_triang_masked_softmax_fusion  False
        scattered_embeddings ............ False
        seed ............................ 1234
        seq_length ...................... 256
        short_seq_prob .................. 0.1
        split ........................... 969, 30, 1
        split_transformers .............. False
        synchronize_each_layer .......... False
        temperature ..................... 1.0
        tensorboard_dir ................. None
        tile_factor ..................... 1
        titles_data_path ................ None
        tokenizer_type .................. SentencePiece
        tokens .......................... 0
        top_k ........................... 0
        top_p ........................... 0.0
        train_iters ..................... None
        train_tokens .................... None
        use_checkpoint_lr_scheduler ..... False
        use_cpu_initialization .......... False
        use_one_sent_docs ............... False
        use_pin_memory .................. False
        vocab_file ...................... yalm100b_checkpoint/vocab/voc_100b.sp
        warmup .......................... 0.01
        warmup_iters .................... None
        weight_decay .................... 0.01
        world_size ...................... 8
        zero_allgather_bucket_size ...... 0.0
        zero_contigious_gradients ....... False
        zero_reduce_bucket_size ......... 0.0
        zero_reduce_scatter ............. False
        zero_stage ...................... 1.0
        '''

    @classmethod
    def from_pretrained(cls, *args, **kargs):
        return cls()

        
        
        
class RotaryPositionEncoding(nn.Module):
    def __init__(self, max_seq_length, hidden_size_per_attention_head, dtype):
        super().__init__()
        cos_cached, sin_cached = RotaryPositionEncoding.get_cache_multipliers(
            max_seq_length, hidden_size_per_attention_head, dtype
        )
        self.register_buffer("cos_cached", cos_cached.unsqueeze(1).unsqueeze(2), persistent=False)
        self.register_buffer("sin_cached", sin_cached.unsqueeze(1).unsqueeze(2), persistent=False)

    def forward(self, hidden_state, offset):
        seq_length = hidden_state.shape[0]
        if isinstance(offset, torch.Tensor):
            realidx = torch.arange(seq_length, device=hidden_state.device).view(1, seq_length) + offset[:, None]
            cos = self.cos_cached[realidx].view(offset.size(0), seq_length, 1, self.cos_cached.size(-1))
            cos = cos.transpose(0, 1)
            sin = self.sin_cached[realidx].view(offset.size(0), seq_length, 1, self.cos_cached.size(-1))
            sin = sin.transpose(0, 1)
        else:
            # cos = cos[..., offset : q.shape[-2] + offset, :]
            cache_slice = slice(offset, offset + seq_length)
            cos, sin = self.cos_cached[cache_slice], self.sin_cached[cache_slice]
        return self.apply_rotary_position_encoding(
            hidden_state, cos, sin,
        )

    @staticmethod
    def get_cache_multipliers(max_seq_length, hidden_size, dtype):
        inv_freqs = 1e-4 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size)
        positions = torch.arange(max_seq_length, dtype=torch.float)
        angles = positions.unsqueeze(-1) * inv_freqs

        cos, sin = torch.cos(angles), torch.sin(angles)
        return cos.to(dtype), sin.to(dtype)
    
    @staticmethod
    def apply_rotary_position_encoding(hidden_state, cos_cached, sin_cached):
        sq, b, np, hn = hidden_state.shape
        half_hn = hn // 2
        left, right = hidden_state[..., :half_hn], hidden_state[..., half_hn:]
        encoded_left = cos_cached * left - sin_cached * right
        encoded_right = sin_cached * left + cos_cached * right
        return torch.cat((encoded_left, encoded_right), dim=3)
    

class SelfAttention(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()

        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size*3, device=device)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, device=device)

        self.hidden_size_per_partition = config.hidden_size_per_partition
        self.num_attention_heads_per_partition = config.num_attention_heads_per_partition
        self.hidden_size_per_attention_head = config.hidden_size_per_attention_head
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        
        self.rotary_position_encoding = RotaryPositionEncoding(
            config.max_position_embeddings,
            self.hidden_size_per_attention_head,
            torch.float32,
        )
        
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

    def forward(self, hidden_states, attention_mask, layer_past=None,
            offset=None,
            get_key_value=True):
        
        # do transpose 
        hidden_states = hidden_states.transpose(0, 1)
        if layer_past is not None:
            layer_past = (layer_past[0].permute(2, 0, 1, 3), layer_past[1].permute(2, 0, 1, 3))
        
        # hidden_states: [sq, b, h]
    
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = mixed_x_layer.chunk(3, dim=-1)

        # if self.pos_encoding_type == 'rotary':
        
        if offset is None:
            offset = 0 if layer_past is None else layer_past[0].size(0)
        
        query_layer = self.rotary_position_encoding(query_layer, offset)
        key_layer = self.rotary_position_encoding(key_layer, offset)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        
        present = (key_layer, value_layer)
        
        key_length = key_layer.size(0)
        query_length = query_layer.size(0)
        
        # [b, np, sq, sk]
        output_size = (query_layer.size(1), 
                       query_layer.size(2), 
                       query_layer.size(0), 
                       key_layer.size(0))
        
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.zeros( # empty sometimes gives nan
            output_size[0]*output_size[1], 
            output_size[2], 
            output_size[3],
            dtype=query_layer.dtype, 
            device=hidden_states.device)

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(matmul_result, 
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0,1).transpose(1, 2),  #[b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        
        # update mask for inference
        # if layer_past is not None:
        #     # keep last item at dim=2
        #     attention_mask = attention_mask[
        #         ...,
        #         attention_scores.size(3) - 1,
        #         :attention_scores.size(3)].unsqueeze(2)
        # else:
        #     attention_mask = attention_mask[
        #         ...,
        #         :attention_scores.size(3),
        #         :attention_scores.size(3)]
            
        # causal
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        mask_value = torch.finfo(attention_scores.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attention_scores.dtype, device=attention_scores.device)
        attention_scores = torch.where(causal_mask, attention_scores, mask_value)
        # e causal
            
        attention_scores = attention_scores + attention_mask
        
        dtype_attn_weights = attention_scores.dtype
        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if dtype_attn_weights == torch.float16:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), 
                       value_layer.size(2), 
                       query_layer.size(0), 
                       value_layer.size(3)) 

        # change view [sk, b * np, hn] 
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)
        
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)
        
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0,1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # print(context_layer)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)
        
        # transpose back
        output = output.transpose(0, 1)
        present = (present[0].permute(1, 2, 0, 3), present[1].permute(1, 2, 0, 3))

        return output, present
    
    
class MLP(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.dense_ffn_hidden = nn.Linear(config.hidden_size, config.intermediate_size, device=device)
        self.dense_ffn_gate = nn.Linear(config.hidden_size, config.intermediate_size, device=device)
        self.dense_ffn_output = nn.Linear(config.intermediate_size, config.hidden_size, device=device)
        self.act = nn.functional.gelu
            
    def forward(self, hidden_states):
        
        intermediate_states = self.dense_ffn_hidden(hidden_states)
        intermediate_states = self.act(intermediate_states)
        gate = self.dense_ffn_gate(hidden_states)
        intermediate_states = intermediate_states * gate
        hidden_states = self.dense_ffn_output(intermediate_states)
        
        return hidden_states
    

class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.padded_vocab_size, config.embedding_size)
        self.input_layernorm = nn.LayerNorm(
            config.embedding_size,
            eps=config.layernorm_epsilon
        )
        
        if config.embedding_size != config.hidden_size:
            self.register_buffer(
                "projector",
                torch.eye(config.embedding_size, config.hidden_size),
                persistent=False,
            )
        
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
        hidden_states = self.input_layernorm(hidden_states)
        
        if self.embedding_size != self.hidden_size:
            hidden_states = hidden_states @ self.projector

        return hidden_states


class GPTBlock(nn.Module):
    def __init__(self, config, layer_number, *args, use_checkpoint=True, device='cpu', **kargs):
        super().__init__()
        self.config = config
        self.use_checkpoint = use_checkpoint
        self.layer_number = layer_number
        
        if self.layer_number >= 1:
            self.input_layernorm = nn.LayerNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                device=device,
            )
            
        self.attention = SelfAttention(config, device=device)
        
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            device=device,
        )
        
        self.mlp = MLP(config, device=device)
        
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
        
        if mask is not None:
            # bool -> float
            attention_mask = (1e4 *(mask[:, None, None, :]-1)).to(hidden_states.dtype)
        else:
            attention_mask = torch.zeros(
                (hidden_states.size(0), 1, 1, hidden_states.size(1)), 
                dtype=hidden_states.dtype, device=hidden_states.device)
           
        offset = None
        if mask is not None:                
            # masked tokens
            offset = (mask-1).sum(-1, keepdims=False).long()
            if layer_past is not None:
                offset += layer_past[0].size(2)
                
        residual = hidden_states
        
        if self.layer_number >= 1:
            attention_input = self.input_layernorm(hidden_states)
        else:
            attention_input = hidden_states
            
        # Self attention.
        attention_output, present = self.attention(
            attention_input,
            attention_mask,
            layer_past=layer_past,
            offset=offset,
            get_key_value=True
        )
        
        layernorm_input = attention_output + residual
        
        ######
        
        residual = layernorm_input
        
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        
        mlp_output = self.mlp(layernorm_output)
        
        hidden_states = mlp_output + residual
        
        #######
        
        return hidden_states, present

class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        
        self.input_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.dense = nn.Linear(
            config.hidden_size,
            config.embedding_size,
        )

        self.activation_func = nn.functional.gelu

        self.output_layer_norm = nn.LayerNorm(
            config.embedding_size,
            eps=config.layernorm_epsilon
        )

        self.output_bias = torch.nn.Parameter(
            torch.zeros(config.padded_vocab_size)
        )
        
        self.lm_head = nn.Linear(
            config.embedding_size, config.padded_vocab_size, bias=None,
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

        output = self.input_layer_norm(hidden_states)
        output = self.dense(output)
        output = self.activation_func(output)
        output = self.output_layer_norm(output)
        
        output = self.lm_head(output) + self.output_bias

        return output

