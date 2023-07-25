import time
import torch.nn.functional as func
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_warper
from coordinator.http_coordinate_client import get_coordinator_client


class DistInferenceMaskTokenPipeHomoBatch:
    def __init__(self, args, device):
        print("=======Initialize Dist Inference(DistInferenceMaskTokenPipeHomoBatch).=======")
        self.device = device
        self.coord_client = get_coordinator_client()
        # Model info for pipeline
        self.max_layers = args.max_layers
        self.num_layers = args.num_layers
        self.model_name = args.model_name
        self.model_type = args.model_type
        self.embedding_dim = self._get_embedding_size()
        self._layer_begin = args.num_layers * get_pipeline_parallel_rank()
        self._layer_end = args.num_layers * (get_pipeline_parallel_rank() + 1)
        if args.fp16:
            self.use_fp16 = True
            print("=======Gpipe use FP16=======")
        else:
            self.use_fp16 = False
            print("=======Gpipe use FP32=======")
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        # Inference setting hyper-parameters
        self.echo_prompt = args.echo_prompt
        self.num_completions = args.num_completions
        self.top_k_per_token = args.top_k_per_token
        self.stop = args.stop
        if self.stop is not None:
            from task_datasets.inference_data import get_tokenizer
            self.tokenizer = get_tokenizer(args)
            self.stop_flag = torch.zeros(1, requires_grad=False, device=device).long()
        self.input_seq_length = args.input_seq_length
        self.generate_seq_length = args.generate_seq_length
        self.logits_warper = get_logits_warper(
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_beams=1,
        )
        self.update_processors(args)

        # Pipeline info
        self.pipeline_group_size = args.pipeline_group_size
        self.pp_rank = get_pipeline_parallel_rank()  # Rank is the pipeline rank by default.
        self.pre_node_rank = self.pp_rank - 1
        self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1
        self.comm = get_pipeline_parallel_comm()

        self.micro_batch_size = 1
        self.seq_num = args.batch_size

        assert (self.seq_num % args.token_micro_batch_size == 0)
        self.token_micro_batch_size = args.token_micro_batch_size
        self.token_micro_batch_num = self.seq_num // self.token_micro_batch_size

        self.layers = {}
        self._create_layers()

        self._init_buffers()
        self._print_buffers()

        self.cached_attention = []
        self._init_cached_seqs_and_attentions()

    def _create_layers(self):
        if self.model_type == 'gpt2':
            from modules.hf_gpt2_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'gptj':
            from modules.hf_gptj_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'gptneox':
            from modules.hf_gptneox_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt':
            from modules.hf_opt_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'bloom':
            from modules.hf_bloom_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'yalm':
            from modules.yalm_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'glm':
            from modules.glm_module import GPTEmbeddings, GPTBlock, GPTLMHead
        else:
            raise Exception(f'unknown model type {self.model_type}')

        if self.pp_rank == 0:
            self.layers['emb'] = GPTEmbeddings.from_pretrained(self.model_name).to(self.dtype).eval().to(self.device)
        for layer_index in range(self.num_layers):
            # global layer indexing could be an argument
            global_layer_index = self.num_layers * self.pp_rank + layer_index
            # in case the total number of layers are not dividable by pipeline group size.
            if self.max_layers is not None and global_layer_index >= self.max_layers:
                self.num_layers = layer_index
                break
            print(f'loading layer {global_layer_index}')
            self.layers['block' + str(layer_index)] = GPTBlock.from_pretrained(
                self.model_name, layer_index=global_layer_index).to(self.dtype).eval().to(self.device)
            if self.coord_client:
                self.coord_client.update_status('running', returned_payload={
                    'rank': self.pp_rank, 'loaded_layer': layer_index, 'total_layer': self.num_layers})
        if self.pp_rank == self.pipeline_group_size - 1:
            self.layers['lm'] = GPTLMHead.from_pretrained(
                self.model_name
            ).to(self.dtype).eval().to(self.device)

    def _get_embedding_size(self):
        if self.model_type == 'gpt2':
            from modules.hf_gpt2_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_embd
        elif self.model_type == 'gptj':
            from modules.hf_gptj_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_embd
        elif self.model_type == 'gptneox':
            from modules.hf_gptneox_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type == 'opt':
            from modules.hf_opt_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type == 'bloom':
            from modules.hf_bloom_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type == 'yalm':
            from modules.yalm_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type == 'glm':
            from modules.glm_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        else:
            raise Exception(f'unknown model type {self.model_type}')

    def _init_buffers(self):
        if self.pp_rank == self.pipeline_group_size - 1:
            ret_seq_length = self.generate_seq_length if not self.echo_prompt else \
                self.input_seq_length + self.generate_seq_length
            self.ret_tokens = torch.zeros((self.seq_num * self.num_completions, ret_seq_length),
                                          requires_grad=False, device=self.device, dtype=torch.int64)
            self.ret_token_logprobs = torch.zeros((self.seq_num * self.num_completions, ret_seq_length),
                                                  requires_grad=False, device=self.device, dtype=self.dtype)
            if self.top_k_per_token > 0:
                self.ret_topk_tokens = torch.zeros(
                    (self.seq_num * self.num_completions, ret_seq_length, self.top_k_per_token),
                    requires_grad=False, device=self.device, dtype=torch.int64)
                self.ret_topk_token_logprobs = torch.zeros(
                    (self.seq_num * self.num_completions, ret_seq_length, self.top_k_per_token),
                    requires_grad=False, device=self.device, dtype=self.dtype)
        if self.pp_rank == 0:
            self.recv_new_token = [torch.zeros((self.token_micro_batch_size * self.num_completions, 1),
                                               requires_grad=False, device=self.device, dtype=torch.int64)
                                   for _ in range(self.token_micro_batch_num)]
        if self.pp_rank == self.pipeline_group_size - 1:
            self.send_new_tokens = [torch.zeros((self.token_micro_batch_size * self.num_completions, 1),
                                                requires_grad=False, device=self.device, dtype=torch.int64)
                                    for _ in range(self.token_micro_batch_num)]

        if self.generate_seq_length == 0:
            self.input_seq_emb = [torch.zeros((1, self.input_seq_length - 1, self.embedding_dim),
                                              requires_grad=False, device=self.device, dtype=self.dtype)
                                  for _ in range(self.seq_num)]
            self.output_seq_emb = [torch.zeros((1, self.input_seq_length - 1, self.embedding_dim),
                                               requires_grad=False, device=self.device, dtype=self.dtype)
                                   for _ in range(self.seq_num)]
        else:
            self.input_seq_emb = [torch.zeros((1, self.input_seq_length, self.embedding_dim),
                                              requires_grad=False, device=self.device, dtype=self.dtype)
                                  for _ in range(self.seq_num)]
            self.output_seq_emb = [torch.zeros((1, self.input_seq_length, self.embedding_dim),
                                               requires_grad=False, device=self.device, dtype=self.dtype)
                                   for _ in range(self.seq_num)]
        self.input_token_emb = [torch.zeros((self.token_micro_batch_size * self.num_completions, 1, self.embedding_dim),
                                            requires_grad=False, device=self.device, dtype=self.dtype)
                                for _ in range(self.token_micro_batch_num)]
        self.output_token_emb = [torch.zeros((self.token_micro_batch_size*self.num_completions, 1, self.embedding_dim),
                                             requires_grad=False, device=self.device, dtype=self.dtype)
                                 for _ in range(self.token_micro_batch_num)]

    def _print_buffers(self):
        if self.pp_rank == 0:
            if self.use_fp16:
                print("=======Rank-(0) recv_new_token: {} KB (fp16)======="
                      .format(self.token_micro_batch_size * self.token_micro_batch_num * 2 // 1024))
            else:
                print("=======Rank-(0) recv_new_token: {} KB (fp32)======="
                      .format(self.token_micro_batch_size * self.token_micro_batch_num * 4 // 1024))
        if self.pp_rank == self.pipeline_group_size - 1:
            if self.use_fp16:
                print("=======Rank-(N-1) send_new_token: {} KB (fp16)======="
                      .format(self.token_micro_batch_size * self.token_micro_batch_num * 2 // 1024))
            else:
                print("=======Rank-(N-1) send_new_token: {} KB (fp32)======="
                      .format(self.token_micro_batch_size * self.token_micro_batch_num * 4 // 1024))
        seq_emb_num = self.seq_num * self.input_seq_length * self.embedding_dim
        if self.use_fp16:
            print("=======input_seq_emb: {} MB shape: {} X {} (fp16)======="
                  .format(seq_emb_num * 2 // 1024 // 1024, self.input_seq_emb[0].shape, self.seq_num))
            print("=======output_seq_emb: {} MB shape: {} X {} (fp16)======="
                  .format(seq_emb_num * 2 // 1024 // 1024, self.input_seq_emb[0].shape, self.seq_num))
        else:
            print("=======input_seq_emb: {} MB shape: {} X {} (fp32)======="
                  .format(seq_emb_num * 4 // 1024 // 1024, self.input_seq_emb[0].shape, self.seq_num))
            print("=======output_seq_emb: {} MB shape: {} X {} (fp32)======="
                  .format(seq_emb_num * 4 // 1024 // 1024, self.input_seq_emb[0].shape, self.seq_num))
        token_emb_num = self.token_micro_batch_size * self.embedding_dim * self.token_micro_batch_num
        if self.use_fp16:
            print("=======input_token_emb: {} MB shape: {} X {} (fp16)======="
                  .format(token_emb_num * 2 // 1024 // 1024, self.input_token_emb[0].shape, self.token_micro_batch_num))
            print("=======output_seq_emb: {} MB shape: {} X {} (fp16)======="
                  .format(token_emb_num * 2 // 1024 // 1024, self.output_token_emb[0].shape,
                          self.token_micro_batch_num))
        else:
            print("=======input_token_emb: {} MB shape: {} X {} (fp32)======="
                  .format(token_emb_num * 4 // 1024 // 1024, self.input_token_emb[0].shape, self.token_micro_batch_num))
            print("=======output_token_emb: {} MB shape: {} X {} (fp32)======="
                  .format(token_emb_num * 4 // 1024 // 1024, self.output_token_emb[0].shape,
                          self.token_micro_batch_num))

    def change_buffer_size(self):
        self._init_buffers()

    def _init_cached_seqs_and_attentions(self):
        if not self.echo_prompt:
            self.i_current_token = 0
        else:
            self.i_current_token = self.input_seq_length

        self.cached_attention.clear()
        for _ in range(self.num_layers):
            self.cached_attention.append([None for _ in range(self.seq_num)])

        self.token_cached_attention = []
        for _ in range(self.num_layers):
            self.token_cached_attention.append([None for _ in range(self.seq_num)])

        if self.stop is not None:
            self.stop_flag[:] = 0

    def update_processors(self, args):
        self.logits_warper = get_logits_warper(
            top_k=(None if args.top_k is None or args.top_k == 0 else args.top_k),
            top_p=(None if args.top_p is None or args.top_p <= 0 else args.top_p),
            temperature=args.temperature,
            num_beams=1,
        )

    def _generate_new_token(self, index):
        assert self.pp_rank == self.pipeline_group_size - 1
        z = self.layers['lm'](self.output_token_emb[index])
        if torch.isnan(z).any():
            print('containing nan, setting them to zero!')
            print(z)
        z = z.float().nan_to_num()  # test if fp32 whether cause inf
        z = torch.nn.functional.log_softmax(z, -1)

        if self.top_k_per_token > 0:
            logprobs, indices = z.topk(k=self.top_k_per_token, dim=-1)
            self.ret_topk_tokens[index * self.token_micro_batch_size * self.num_completions:
                                 (index + 1) * self.token_micro_batch_size * self.num_completions,
                                 self.i_current_token] = indices.squeeze(1)
            self.ret_topk_token_logprobs[index * self.token_micro_batch_size * self.num_completions:
                                         (index + 1) * self.token_micro_batch_size * self.num_completions,
                                         self.i_current_token] = logprobs.squeeze(1)

        # [:, -1] because multinomial only accept 1/2d tensors
        z_to_sample = z[:, -1]  # bs, vocab
        z_to_sample = self.logits_warper(None, z_to_sample)
        p_to_sample = z_to_sample.softmax(-1).clamp(0, 1).nan_to_num()
        indices = torch.multinomial(p_to_sample, num_samples=1)  # bs, 1
        logprobs = torch.gather(z[:, -1], -1, indices)  # bs, 1
        self.send_new_tokens[index] = indices

        self.ret_tokens[index * self.token_micro_batch_size * self.num_completions:
                        (index + 1) * self.token_micro_batch_size * self.num_completions,
                        self.i_current_token] = indices.squeeze(-1)
        self.ret_token_logprobs[index * self.token_micro_batch_size * self.num_completions:
                                (index + 1) * self.token_micro_batch_size * self.num_completions,
                                self.i_current_token] = logprobs.squeeze(-1)

        if index == self.token_micro_batch_num - 1:
            self.i_current_token += 1

    def _merge_cached_seqs_and_attentions(self):
        if not self.echo_prompt:
            self.i_current_token = 0
        else:
            self.i_current_token = self.input_seq_length

        for layer_index in range(self.num_layers):
            key = torch.split(torch.cat([kv[0] for kv in self.cached_attention[layer_index]], dim=0),
                              self.token_micro_batch_size, dim=0)
            value = torch.split(torch.cat([kv[1] for kv in self.cached_attention[layer_index]], dim=0),
                                self.token_micro_batch_size, dim=0)
            self.cached_attention[layer_index] = list(zip(key, value))
            if self.use_fp16:
                print("=======Layer {} cached key: {} MB shape: {} (fp16)======="
                      .format(layer_index, torch.numel(key[0]) * self.token_micro_batch_num * 2 // 1024 // 1024,
                              key[0].shape))
                print("=======Layer {} cached key: {} MB shape: {} (fp16)======="
                      .format(layer_index, torch.numel(value[0]) * self.token_micro_batch_num * 2 // 1024 // 1024,
                              value[0].shape))
            else:
                print("=======Layer {} cached key: {} MB shape: {} (fp32)======="
                      .format(layer_index, torch.numel(key[0]) * self.token_micro_batch_num * 4 // 1024 // 1024,
                              key[0].shape))
                print("=======Layer {} cached key: {} MB shape: {} (fp32)======="
                      .format(layer_index, torch.numel(value[0]) * self.token_micro_batch_num * 4 // 1024 // 1024,
                              value[0].shape))
        if self.pp_rank == self.pipeline_group_size - 1:
            for i in range(self.token_micro_batch_num):
                self._generate_new_token(i)

        if self.stop is not None:
            self.stop_flag[:] = 0

    def _copy_initial_token_emb(self, index):
        assert self.pp_rank == self.pipeline_group_size - 1
        buff_i = index // self.token_micro_batch_size
        pos = index % self.token_micro_batch_size
        print("_copy_initial_token_emb")
        for k in range(self.num_completions):
            print(f"_copy_initial_token_emb {k}/{self.num_completions}")
            self.output_token_emb[buff_i][pos + k * self.token_micro_batch_size] = self.output_seq_emb[index][:, -1:]

    def _generate_echo_token_logprobs(self, index, indices):
        assert self.pp_rank == self.pipeline_group_size - 1
        assert self.num_completions == 1

        if self.generate_seq_length == 0:
            z = self.layers['lm'](self.output_seq_emb[index])
        else:
            z = self.layers['lm'](self.output_seq_emb[index][:, :-1])
        z = func.log_softmax(z, -1)
        original_indices = indices
        indices = indices[:, 1:]  # skip first

        logprobs = torch.gather(z, -1, indices.unsqueeze(-1)).squeeze(-1)
        self.ret_tokens[
            index * self.micro_batch_size:(index + 1) * self.micro_batch_size, :self.i_current_token
        ] = original_indices
        self.ret_token_logprobs[
            index * self.micro_batch_size:(index + 1) * self.micro_batch_size,
            1:self.i_current_token
        ] = logprobs
        if self.top_k_per_token > 0:
            logprobs, indices = z.topk(k=self.top_k_per_token, dim=-1)
            self.ret_topk_tokens[
                index * self.micro_batch_size:(index + 1) * self.micro_batch_size,
                1:self.i_current_token
            ] = indices
            self.ret_topk_token_logprobs[
                index * self.micro_batch_size:(index + 1) * self.micro_batch_size,
                1:self.i_current_token
            ] = logprobs

    def _forward_compute_prompt_seq(self, index, seq, mask):
        print("Compute prompt seq<", index, ">.")
        if self.pp_rank == 0:
            self.input_seq_emb[index] = self.layers['emb'](seq, mask=mask)
        current_emb = self.input_seq_emb[index]
        caches = [None] * self.num_layers

        for layer_index in range(self.num_layers):
            current_emb, caches[layer_index] = \
                self.layers['block' + str(layer_index)](current_emb, caches[layer_index], mask=mask)
            self.cached_attention[layer_index][index] = caches[layer_index]

        self.output_seq_emb[index] = current_emb

        if self.pp_rank == self.pipeline_group_size - 1:
            self._copy_initial_token_emb(index)

            if self.echo_prompt:
                self._generate_echo_token_logprobs(index, indices=seq)

    def _get_cached_attention(self, layer_index, token_batch_index):
        if self.num_completions == 1:
            return self.cached_attention[layer_index][token_batch_index]
        else:
            # 2* (token_bs, ., seq_len, .)
            prompt_cache = self.cached_attention[layer_index][token_batch_index]
            # 2* (token_bs * num_completion, ., seq_len, .)
            token_cache = self.token_cached_attention[layer_index][token_batch_index]
            # 2*(token_bs * num_completion, ., seq_len, .)
            prompt_cache = [prompt_cache[0].repeat(self.num_completions, 1, 1, 1),
                            prompt_cache[1].repeat(self.num_completions, 1, 1, 1)]
            if token_cache is not None:
                token_cache = [torch.cat([prompt_cache[0], token_cache[0]], dim=2),
                               torch.cat([prompt_cache[1], token_cache[1]], dim=2)]
            else:
                token_cache = prompt_cache
            return token_cache

    def _set_cached_attention(self, cache, layer_index, token_batch_index):
        if self.num_completions == 1:
            self.cached_attention[layer_index][token_batch_index] = cache
        else:
            self.token_cached_attention[layer_index][token_batch_index] = [cache[0][:, :, self.input_seq_length:],
                                                                           cache[1][:, :, self.input_seq_length:]]

    def _forward_compute_generate_token(self, index, mask=None):
        if mask is not None and self.num_completions > 1:
            # repeat n times
            mask = mask.repeat(self.num_completions, 1)
        # print("Compute generate seq micro-batch <", index, ">.")
        if self.pp_rank == 0:
            current_emb = self.layers['emb'](self.recv_new_token[index], self.cached_attention[0][index], mask=mask)
        else:
            current_emb = self.input_token_emb[index]

        for layer_index in range(self.num_layers):
            cache = self._get_cached_attention(layer_index, index)
            current_emb, cache = \
                self.layers['block' + str(layer_index)](current_emb, cache, mask=mask)
            self._set_cached_attention(cache, layer_index, index)
        self.output_token_emb[index] = current_emb

        if self.pp_rank == self.pipeline_group_size - 1:
            self._generate_new_token(index)

    def _process_mask_during_generation(self, attention_mask):
        if attention_mask is not None:
            # increase one for the new token
            attention_mask = func.pad(attention_mask, pad=(0, 1), mode='constant', value=1)
        return attention_mask

    def forward_seq_pipeline_stage(self, input_data=None, attention_mask=None):
        if self.pp_rank == 0 or self.pp_rank == self.pipeline_group_size - 1:
            assert (input_data is not None)
            if self.pp_rank == 0 and self.generate_seq_length == 0:
                # input reduce 1 for first node
                input_data = input_data[:, :-1]

        if input_data is not None:
            input_seqs = torch.chunk(input_data, self.seq_num, dim=0)
        else:
            input_seqs = [None] * self.seq_num

        if attention_mask is not None:
            if self.generate_seq_length == 0:
                # attention reduce 1
                attention_mask = attention_mask[:, :-1]
            attention_mask = torch.chunk(attention_mask, self.seq_num, dim=0)
        else:
            attention_mask = [None] * self.seq_num

        for i in range(self.seq_num):
            if self.pp_rank == 0:  # Only send output to next node, do not receive
                # Compute
                self._forward_compute_prompt_seq(index=i, seq=input_seqs[i], mask=attention_mask[i])
                # Send
                self.comm.send(self.output_seq_emb[i], dst=self.post_node_rank)
            elif self.pp_rank == self.pipeline_group_size - 1:  # Only receive input from last node, do not send
                # Receive
                self.comm.recv(self.input_seq_emb[i], src=self.pre_node_rank)
                # Compute
                self._forward_compute_prompt_seq(index=i, seq=input_seqs[i], mask=attention_mask[i])
            else:  # receive, compute, and send
                # Receive
                self.comm.recv(self.input_seq_emb[i], src=self.pre_node_rank)
                # Compute
                self._forward_compute_prompt_seq(index=i, seq=input_seqs[i], mask=attention_mask[i])
                # Send
                self.comm.send(self.output_seq_emb[i], dst=self.post_node_rank)

    def forward_new_token_pipeline_stage(self, attention_mask=None):
        if self.generate_seq_length == 0:
            # handle seq_length == 0
            return
        self._merge_cached_seqs_and_attentions()

        if self.generate_seq_length == 1:
            # skip token pipelin when generate_seq_length == 1
            return

        for step in range(self.generate_seq_length):

            # check early stop
            if self.stop is not None:
                self._check_stop(step)

            print("Compute generate token step <", step, ">.")
            # for last node, the first step does not compute
            if step != 0 or self.pp_rank != self.pipeline_group_size - 1:
                attention_mask = self._process_mask_during_generation(attention_mask)
            self.forward_new_token_pipeline_step(step, attention_mask=attention_mask)

            # sync and check early stop
            if self.stop is not None:
                if self.stop_flag.item():
                    break

    def _check_stop(self, step):

        if step % 4 == 0 and step > 0:  # check every 4 tokens
            # check
            if self.pp_rank == self.pipeline_group_size - 1:
                self.stop_flag[:] = 1
                for tokens in self.ret_tokens:
                    tokens = tokens[:self.i_current_token]
                    text = self.tokenizer.decode(tokens)
                    is_stopped = False
                    for _stop in self.stop:
                        if _stop in text:
                            is_stopped = True
                            break
                    if not is_stopped:
                        self.stop_flag[:] = 0
                        break
            # sync
            self.comm.broadcast(self.stop_flag, src=self.pipeline_group_size - 1)

    def forward_new_token_pipeline_step(self, step: int, attention_mask=None):
        attention_masks = torch.split(
            attention_mask, self.token_micro_batch_size, dim=0
        )
        for i in range(self.token_micro_batch_num):
            # Last node:
            if self.pp_rank == self.pipeline_group_size - 1:
                if step == 0:
                    # Send
                    self.comm.send(self.send_new_tokens[i], dst=0)
                else:
                    # Receive
                    self.comm.recv(self.input_token_emb[i], src=self.pre_node_rank)
                    # Compute
                    self._forward_compute_generate_token(i, mask=attention_masks[i])
                    if step != self.generate_seq_length - 1 and (self.stop is None or self.stop_flag.item() == 0):
                        # Send
                        self.comm.send(self.send_new_tokens[i], dst=0)
            # Rank-0 node:
            elif self.pp_rank == 0:
                if step != self.generate_seq_length - 1 and (self.stop is None or self.stop_flag.item() == 0):
                    # Receive
                    self.comm.recv(self.recv_new_token[i], src=self.pipeline_group_size - 1)
                    # Compute
                    self._forward_compute_generate_token(i, mask=attention_masks[i])
                    # Send
                    self.comm.send(self.output_token_emb[i], dst=self.post_node_rank)
            else:  # Middle nodes:
                if step != self.generate_seq_length - 1 and (self.stop is None or self.stop_flag.item() == 0):
                    # Receive
                    self.comm.recv(self.input_token_emb[i], src=self.pre_node_rank)
                    # Compute
                    self._forward_compute_generate_token(i, mask=attention_masks[i])
                    # Send
                    self.comm.send(self.output_token_emb[i], dst=self.post_node_rank)

    def inference_batch(self, input_=None, output_=None, attention_mask=None):
        print(f"<inference_batch> rank-<{self.pp_rank}> Enter!")
        self.comm.barrier()
        print(f"<inference_batch> rank-<{self.pp_rank}> after first barrier!")
        self._init_cached_seqs_and_attentions()  # TODO: should I put here
        print(f"<inference_batch> rank-<{self.pp_rank}> after first _init_cached_seqs_and_attentions!")
        self.comm.barrier()
        print(f"<inference_batch> rank-<{self.pp_rank}> after second barrier!")
        start_time = time.time()

        print(f"<inference_batch> rank-<{self.pp_rank}> enter computation!")
        with torch.no_grad():
            self.forward_seq_pipeline_stage(input_data=input_, attention_mask=attention_mask)
            print(f"<inference_batch> rank-<{self.pp_rank}> forward_seq_pipeline_stage is done!")
            self.forward_new_token_pipeline_stage(attention_mask=attention_mask)
            print(f"<inference_batch> rank-<{self.pp_rank}> forward_seq_pipeline_stage is done!")

        self.comm.barrier()
        print(f"<inference_batch> rank-<{self.pp_rank}> after third barrier!")

        if self.pp_rank == self.pipeline_group_size - 1 and output_ is not None:
            ret_topk_tokens = None
            ret_topk_token_logprobs = None
            # token_micro_batch_num * num_completions
            ret_tokens = self.ret_tokens[:, :self.i_current_token].cpu().split(self.token_micro_batch_size)
            ret_token_logprobs = self.ret_token_logprobs[:, :self.i_current_token].cpu().split(
                self.token_micro_batch_size)
            if self.top_k_per_token > 0:
                ret_topk_tokens = self.ret_topk_tokens[:, :self.i_current_token].cpu().split(
                    self.token_micro_batch_size)
                ret_topk_token_logprobs = self.ret_topk_token_logprobs[:, :self.i_current_token].cpu().split(
                    self.token_micro_batch_size)

            print(f"<inference_batch> rank-<{self.pp_rank}> after marker1 !")

            for i in range(self.num_completions):
                item = {
                    'token_ids': torch.cat(ret_tokens[i::self.num_completions], 0),
                    'token_logprobs': torch.cat(ret_token_logprobs[i::self.num_completions], 0),
                }
                if self.top_k_per_token > 0:
                    item['topk_ids'] = torch.cat(ret_topk_tokens[i::self.num_completions], 0)
                    item['topk_logprobs'] = torch.cat(ret_topk_token_logprobs[i::self.num_completions], 0)
                output_.append(item)
            print(f"<inference_batch> rank-<{self.pp_rank}> after marker2 !")

        end_time = time.time()
        iter_time = end_time - start_time
        print("Rank {} node whole INFERENCE iteration takes {:3.2f}s".format(self.pp_rank, iter_time))
        print("-------------------------------------------")
        return iter_time
