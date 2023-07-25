import time
import torch.nn.functional as func
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_warper
import json
from typing import List, Dict
from coordinator.coordinator_client import LocalCoordinatorClient


class DistInferenceMaskTokenPipeAutoBatch:
    def __init__(self, args, device, coord_client: LocalCoordinatorClient = None):
        print("=======Initialize Dist Inference(DistInferenceMaskTokenPipeAutoBatch).=======")
        self.has_work = torch.zeros(1, dtype=torch.int, device=device)
        self.task_settings = []
        self.dist_store = dist.distributed_c10d._get_default_store()
        self.task_count = 0
        self.current_job_ids = []
        self.device = device
        if coord_client is not None:
            self.coord_client = coord_client
        else:
            self.coord_client = None  # get_coordinator_client()
        self.init_job_id = args.job_id  # This should be changed to model ID later.
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
        from task_datasets.inference_data import get_tokenizer
        self.tokenizer = get_tokenizer(args)

        # Inference setting hyper-parameters
        self.top_k = args.top_k  # This variable cannot be updated interactively yet.
        self.echo_prompt = []
        self.num_completions = []
        self.top_k_per_token = []
        self.input_seq_length = []
        self.generate_seq_length = []
        self.logits_warpers = []
        self.i_current_token = []
        # self._init_batch_settings(task_settings)

        # Pipeline info
        self.pipeline_group_size = args.pipeline_group_size
        self.pp_rank = get_pipeline_parallel_rank()  # Rank is the pipeline rank by default.
        self.pre_node_rank = self.pp_rank - 1
        self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1
        self.comm = get_pipeline_parallel_comm()
        self.seq_micro_batch_size = 1
        self.token_micro_batch_size = 1

        self.layers = {}
        self._create_layers()

        if self.pp_rank == self.pipeline_group_size - 1:
            self.ret_tokens = []
            self.ret_token_logprobs = []
            self.ret_topk_tokens = []
            self.ret_topk_token_logprobs = []
        if self.pp_rank == 0:
            self.recv_new_token = []
        if self.pp_rank == self.pipeline_group_size - 1:
            self.send_new_tokens = []
        self.input_seq_emb = []
        self.output_seq_emb = []
        self.input_token_emb = []
        self.output_token_emb = []
        # self._init_buffers()
        self.cached_attention = []
        # self._init_cached_seqs_and_attentions()

    def sync_has_work(self):
        self.comm.broadcast(self.has_work, src=0)
        print("sync_has_work:", self.has_work.item())
        self.comm.barrier()

    def _init_batch_settings(self):
        self.echo_prompt.clear()
        self.num_completions.clear()
        self.top_k_per_token.clear()
        self.input_seq_length.clear()
        self.generate_seq_length.clear()
        self.logits_warpers.clear()

        self.seq_num = len(self.task_settings)
        for i in range(self.seq_num):
            self.echo_prompt.append(self.task_settings[i].get('echo', False))
            self.num_completions.append(self.task_settings[i].get('n', 1))
            self.top_k_per_token.append(self.task_settings[i].get('logprobs', 0))
            current_seq_length = self.tokenizer(self.task_settings[i]['prompt'], return_tensors='pt', padding=True,
                                                truncation=False)['input_ids'].size(1)
            # 2048 is hardcoded.
            current_seq_length = min(current_seq_length, 2048 - self.task_settings[i].get('max_tokens', 1))
            self.input_seq_length.append(current_seq_length)
            self.generate_seq_length.append(self.task_settings[i].get('max_tokens', 1))

            current_top_p = self.task_settings[i].get('top_p', 1.0)
            current_temperature = self.task_settings[i].get('temperature', 0)
            current_logits_warper = get_logits_warper(
                top_k=(None if self.top_k is None or self.top_k == 0 else self.top_k),
                top_p=(None if current_top_p is None or current_top_p <= 0 else current_top_p),
                temperature=current_temperature,
                num_beams=1,
            )
            self.logits_warpers.append(current_logits_warper)
            self.i_current_token.append(None)

    def _print_batch_settings(self):
        print("<DistInferenceMaskTokenPipeAutoBatch-_print_batch_settings>:")
        for i in range(self.seq_num):
            print("-----------------------------------------------------------")
            print(f"Seq index <{i}>")
            print(f"echo_prompt: <{self.echo_prompt[i]}>")
            print(f"num_completions: <{self.num_completions[i]}>")
            print(f"top_k_per_token: <{self.top_k_per_token[i]}>")
            print(f"input_seq_length: <{self.input_seq_length[i]}>")
            print(f"generate_seq_length: <{self.generate_seq_length[i]}>")

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
                self.coord_client.update_status(self.init_job_id, 'running', returned_payload={
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
            self.ret_tokens.clear()
            self.ret_token_logprobs.clear()
            self.ret_topk_tokens.clear()
            self.ret_topk_token_logprobs.clear()
        if self.pp_rank == 0:
            self.recv_new_token.clear()
        if self.pp_rank == self.pipeline_group_size - 1:
            self.send_new_tokens.clear()
        self.input_seq_emb.clear()
        self.output_seq_emb.clear()
        self.input_token_emb.clear()
        self.output_token_emb.clear()

        if self.pp_rank == self.pipeline_group_size - 1:
            for i in range(self.seq_num):
                current_ret_seq_length = self.generate_seq_length[i] if not self.echo_prompt[i] else \
                    self.input_seq_length[i] + self.generate_seq_length[i]
                current_ret_tokens = torch.zeros((self.num_completions[i], current_ret_seq_length),
                                                 requires_grad=False, device=self.device, dtype=torch.int64)
                self.ret_tokens.append(current_ret_tokens)
                current_ret_token_logprobs = torch.zeros((self.num_completions[i], current_ret_seq_length),
                                                         requires_grad=False, device=self.device, dtype=self.dtype)
                self.ret_token_logprobs.append(current_ret_token_logprobs)
                if self.top_k_per_token[i] > 0:
                    current_ret_topk_tokens = torch.zeros((self.num_completions[i], current_ret_seq_length,
                                                           self.top_k_per_token[i]), requires_grad=False,
                                                          device=self.device, dtype=torch.int64)
                    self.ret_topk_tokens.append(current_ret_topk_tokens)
                    current_ret_topk_token_logprobs = torch.zeros((self.num_completions[i], current_ret_seq_length,
                                                                   self.top_k_per_token[i]), requires_grad=False,
                                                                  device=self.device, dtype=self.dtype)
                    self.ret_topk_token_logprobs.append(current_ret_topk_token_logprobs)
                else:
                    self.ret_topk_tokens.append(None)
                    self.ret_topk_token_logprobs.append(None)
        if self.pp_rank == 0:
            for i in range(self.seq_num):
                current_recv_new_token = torch.zeros((self.num_completions[i], 1), requires_grad=False,
                                                     device=self.device, dtype=torch.int64)
                self.recv_new_token.append(current_recv_new_token)
        if self.pp_rank == self.pipeline_group_size - 1:
            for i in range(self.seq_num):
                current_send_new_tokens = torch.zeros((self.num_completions[i], 1), requires_grad=False,
                                                      device=self.device, dtype=torch.int64)
                self.send_new_tokens.append(current_send_new_tokens)
        for i in range(self.seq_num):
            if self.generate_seq_length[i] == 0:
                current_input_seq_emb = torch.zeros((1, self.input_seq_length[i] - 1, self.embedding_dim),
                                                    requires_grad=False, device=self.device, dtype=self.dtype)
                current_output_seq_emb = torch.zeros((1, self.input_seq_length[i] - 1, self.embedding_dim),
                                                     requires_grad=False, device=self.device, dtype=self.dtype)
            else:
                current_input_seq_emb = torch.zeros((1, self.input_seq_length[i], self.embedding_dim),
                                                    requires_grad=False, device=self.device, dtype=self.dtype)
                current_output_seq_emb = torch.zeros((1, self.input_seq_length[i], self.embedding_dim),
                                                     requires_grad=False, device=self.device, dtype=self.dtype)
            self.input_seq_emb.append(current_input_seq_emb)
            self.output_seq_emb.append(current_output_seq_emb)
            current_input_token_emb = torch.zeros((self.num_completions[i], 1, self.embedding_dim),
                                                  requires_grad=False, device=self.device, dtype=self.dtype)
            self.input_token_emb.append(current_input_token_emb)
            current_output_token_emb = torch.zeros((self.num_completions[i], 1, self.embedding_dim),
                                                   requires_grad=False, device=self.device, dtype=self.dtype)
            self.output_token_emb.append(current_output_token_emb)

    def _print_buffers(self):
        print(f"<DistInferenceMaskTokenPipeAutoBatch-_print_buffers>: rank-<{self.pp_rank}>=======")
        for i in range(self.seq_num):
            print("-----------------------------------------------------------")
            print(f"Seq index <{i}>")
            if self.pp_rank == self.pipeline_group_size - 1:
                print(f"ret_tokens: <{self.ret_tokens[i].shape}>")
                print(f"ret_token_logprobs: <{self.ret_token_logprobs[i].shape}>")
                if self.ret_topk_tokens[i] is not None:
                    print(f"ret_topk_tokens: <{self.ret_topk_tokens[i].shape}>")
                else:
                    print(f"ret_topk_tokens: <None>")
                if self.ret_topk_token_logprobs[i] is not None:
                    print(f"ret_topk_token_logprobs: <{self.ret_topk_token_logprobs[i].shape}>")
                else:
                    print(f"ret_topk_token_logprobs: <None>")
            if self.pp_rank == 0:
                print(f"recv_new_token: <{self.recv_new_token[i].shape}>")
            if self.pp_rank == self.pipeline_group_size - 1:
                print(f"send_new_tokens: <{self.send_new_tokens[i].shape}>")
            print(f"input_seq_emb: <{self.input_seq_emb[i].shape}>")
            print(f"output_seq_emb: <{self.output_seq_emb[i].shape}>")
            print(f"input_token_emb: <{self.input_token_emb[i].shape}>")
            print(f"output_token_emb: <{self.output_token_emb[i].shape}>")

    def update_batch_setting(self, task_settings: List[Dict] = None, job_ids: List[str] = None):
        print(f"<DistInferenceMaskTokenPipeAutoBatch-update_batch_setting>: rank-<{self.pp_rank}>=======")
        print(task_settings)
        print(job_ids)
        self.task_count += 1
        if self.pp_rank == 0:
            assert task_settings is not None and job_ids is not None
            task_settings_str = json.dumps({'task_settings': task_settings, 'job_ids': job_ids})
            self.dist_store.set('current_task_'+str(self.task_count), task_settings_str)
            print(task_settings_str)
            self.task_settings.clear()
            self.task_settings.extend(task_settings)
            self.current_job_ids.clear()
            self.current_job_ids.extend(job_ids)
        else:
            assert task_settings is None
            task_settings_str = self.dist_store.get('current_task_'+str(self.task_count))
            print(task_settings_str)
            task_dict = json.loads(task_settings_str)
            task_settings = task_dict['task_settings']
            self.task_settings.clear()
            self.task_settings.extend(task_settings)
            self.current_job_ids.clear()
            self.current_job_ids.extend(task_dict['job_ids'])
        self._init_batch_settings()
        self._print_batch_settings()
        self._init_buffers()
        self._print_buffers()
        self._init_cached_seqs_and_attentions()

    def _init_cached_seqs_and_attentions(self):
        for i in range(self.seq_num):
            if not self.echo_prompt[i]:
                self.i_current_token[i] = 0
            else:
                self.i_current_token[i] = self.input_seq_length[i]
        self.cached_attention.clear()
        for _ in range(self.num_layers):
            self.cached_attention.append([None for _ in range(self.seq_num)])
        self.token_cached_attention = []
        for _ in range(self.num_layers):
            self.token_cached_attention.append([None for _ in range(self.seq_num)])

    def _generate_new_token(self, index):
        assert self.pp_rank == self.pipeline_group_size - 1
        z = self.layers['lm'](self.output_token_emb[index])
        if torch.isnan(z).any():
            print('containing nan, setting them to zero!')
            print(z)
        z = z.float().nan_to_num()  # test if fp32 whether cause inf
        z = torch.nn.functional.log_softmax(z, -1)

        if self.top_k_per_token[index] > 0:
            logprobs, indices = z.topk(k=self.top_k_per_token[index], dim=-1)
            self.ret_topk_tokens[index][:, self.i_current_token[index]] = indices.squeeze(1)
            self.ret_topk_token_logprobs[index][:, self.i_current_token[index]] = logprobs.squeeze(1)
        # [:, -1] because multinomial only accept 1/2d tensors
        z_to_sample = z[:, -1]  # bs, vocab
        z_to_sample = self.logits_warpers[index](None, z_to_sample)
        p_to_sample = z_to_sample.softmax(-1).clamp(0, 1).nan_to_num()
        indices = torch.multinomial(p_to_sample, num_samples=1)  # bs, 1
        logprobs = torch.gather(z[:, -1], -1, indices)  # bs, 1
        self.send_new_tokens[index] = indices

        self.ret_tokens[index][:, self.i_current_token[index]] = indices.squeeze(-1)
        self.ret_token_logprobs[index][:, self.i_current_token[index]] = logprobs.squeeze(-1)
        self.i_current_token[index] += 1

    def _merge_cached_seqs_and_attentions(self):
        for i in range(self.seq_num):
            if not self.echo_prompt[i]:
                self.i_current_token[i] = 0
            else:
                self.i_current_token[i] = self.input_seq_length[i]
        if self.pp_rank == self.pipeline_group_size - 1:
            for i in range(self.seq_num):
                if self.generate_seq_length[i] != 0:
                    self._generate_new_token(i)

    def _copy_initial_token_emb(self, index):
        assert self.pp_rank == self.pipeline_group_size - 1
        print("_copy_initial_token_emb")
        for k in range(self.num_completions[index]):
            print(f"_copy_initial_token_emb {k}/{self.num_completions[index]}")
            self.output_token_emb[index][k] = self.output_seq_emb[index][:, -1:]

    def _generate_echo_token_logprobs(self, index, indices):
        assert self.pp_rank == self.pipeline_group_size - 1
        assert self.num_completions[index] == 1

        if self.generate_seq_length[index] == 0:
            z = self.layers['lm'](self.output_seq_emb[index])
        else:
            z = self.layers['lm'](self.output_seq_emb[index][:, :-1])
        z = func.log_softmax(z, -1)
        original_indices = indices
        indices = indices[:, 1:]  # skip first

        logprobs = torch.gather(z, -1, indices.unsqueeze(-1)).squeeze(-1)
        self.ret_tokens[index][:, :self.i_current_token[index]] = original_indices
        self.ret_token_logprobs[index][:, 1:self.i_current_token[index]] = logprobs
        if self.top_k_per_token[index] > 0:
            logprobs, indices = z.topk(k=self.top_k_per_token[index], dim=-1)
            self.ret_topk_tokens[index][:, 1:self.i_current_token[index]] = indices
            self.ret_topk_token_logprobs[index][:, 1:self.i_current_token[index]] = logprobs

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
            if self.echo_prompt[index]:
                self._generate_echo_token_logprobs(index, indices=seq)

    def _get_cached_attention(self, layer_index, index):
        if self.num_completions[index] == 1:
            return self.cached_attention[layer_index][index]
        else:
            # 2* (token_bs, ., seq_len, .)
            prompt_cache = self.cached_attention[layer_index][index]
            # 2* (token_bs * num_completion, ., seq_len, .)
            token_cache = self.token_cached_attention[layer_index][index]
            # 2*(token_bs * num_completion, ., seq_len, .)
            prompt_cache = [prompt_cache[0].repeat(self.num_completions[index], 1, 1, 1),
                            prompt_cache[1].repeat(self.num_completions[index], 1, 1, 1)]
            if token_cache is not None:
                token_cache = [torch.cat([prompt_cache[0], token_cache[0]], dim=2),
                               torch.cat([prompt_cache[1], token_cache[1]], dim=2)]
            else:
                token_cache = prompt_cache
            return token_cache

    def _set_cached_attention(self, cache, layer_index, index):
        if self.num_completions[index] == 1:
            self.cached_attention[layer_index][index] = cache
        else:
            self.token_cached_attention[layer_index][index] = [cache[0][:, :, self.input_seq_length[index]:],
                                                               cache[1][:, :, self.input_seq_length[index]:]]

    def _forward_compute_generate_token(self, index, mask=None):
        if mask is not None and self.num_completions[index] > 1:
            # repeat n times
            mask = mask.repeat(self.num_completions[index], 1)
        # print("Compute generate seq micro-batch <", index, ">.")
        if self.pp_rank == 0:
            current_emb = self.layers['emb'](self.recv_new_token[index], self.cached_attention[0][index], mask=mask)
        else:
            current_emb = self.input_token_emb[index]
        for layer_index in range(self.num_layers):
            cache = self._get_cached_attention(layer_index, index)
            current_emb, cache = self.layers['block' + str(layer_index)](current_emb, cache, mask=mask)
            self._set_cached_attention(cache, layer_index, index)
        self.output_token_emb[index] = current_emb
        if self.pp_rank == self.pipeline_group_size - 1:
            self._generate_new_token(index)

    def _process_mask_during_generation(self, attention_mask):
        for i in range(self.seq_num):
            if attention_mask[i] is not None:
                # increase one for the new token
                attention_mask[i] = func.pad(attention_mask[i], pad=(0, 1), mode='constant', value=1)
        return attention_mask

    def forward_seq_pipeline_stage(self, input_data=None, attention_mask=None):
        # if self.pp_rank == 0 or self.pp_rank == self.pipeline_group_size - 1:
        if self.pp_rank == 0:
            assert (input_data is not None)
            if self.pp_rank == 0:
                for i in range(self.seq_num):
                    if self.generate_seq_length[i] == 0:
                        # input reduce 1 for first node
                        input_data[i] = input_data[i][:-1]
        if input_data is not None:
            input_seqs = input_data
        else:
            input_seqs = [None] * self.seq_num

        if attention_mask is not None:
            for i in range(self.seq_num):
                if attention_mask[i] is not None:
                    if self.generate_seq_length[i] == 0:
                        # attention reduce 1
                        attention_mask[i] = attention_mask[i][:-1]
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
        self._merge_cached_seqs_and_attentions()
        for step in range(max(self.generate_seq_length)):
            print("Compute generate token step <", step, ">.")
            # for last node, the first step does not compute
            if step != 0 or self.pp_rank != self.pipeline_group_size - 1:
                if attention_mask:
                    attention_mask = self._process_mask_during_generation(attention_mask)
            self.forward_new_token_pipeline_step(step, attention_masks=attention_mask)

    def forward_new_token_pipeline_step(self, step: int, attention_masks=None):
        for i in range(self.seq_num):
            # This seq does not need these much seq.
            if step >= self.generate_seq_length[i]:
                continue

            # Last node:
            if self.pp_rank == self.pipeline_group_size - 1:
                if step == 0:
                    # Send
                    self.comm.send(self.send_new_tokens[i], dst=0)
                else:
                    # Receive
                    self.comm.recv(self.input_token_emb[i], src=self.pre_node_rank)
                    # Compute
                    self._forward_compute_generate_token(i, mask=attention_masks[i] if attention_masks else None)
                    if step != self.generate_seq_length[i] - 1:
                        # Send
                        self.comm.send(self.send_new_tokens[i], dst=0)
            # Rank-0 node:
            elif self.pp_rank == 0:
                if step != self.generate_seq_length[i] - 1:
                    # Receive
                    self.comm.recv(self.recv_new_token[i], src=self.pipeline_group_size - 1)
                    # Compute
                    self._forward_compute_generate_token(i, mask=attention_masks[i] if attention_masks else None)
                    # Send
                    self.comm.send(self.output_token_emb[i], dst=self.post_node_rank)
            else:  # Middle nodes:
                if step != self.generate_seq_length[i] - 1:
                    # Receive
                    self.comm.recv(self.input_token_emb[i], src=self.pre_node_rank)
                    # Compute
                    self._forward_compute_generate_token(i, mask=attention_masks[i] if attention_masks else None)
                    # Send
                    self.comm.send(self.output_token_emb[i], dst=self.post_node_rank)

    def inference_batch(self, input_=None, output_=None, attention_mask=None):
        print(f"<AutoBatch inference_batch> rank-<{self.pp_rank}> Enter!")
        self.comm.barrier()
        print(f"<AutoBatch inference_batch> rank-<{self.pp_rank}> after first barrier!")
        self._init_cached_seqs_and_attentions()  # TODO: should I put here
        print(f"<AutoBatch inference_batch> rank-<{self.pp_rank}> after first _init_cached_seqs_and_attentions!")
        self.comm.barrier()
        print(f"<AutoBatch inference_batch> rank-<{self.pp_rank}> after second barrier!")
        start_time = time.time()

        print(f"<AutoBatch inference_batch> rank-<{self.pp_rank}> enter computation!")
        with torch.no_grad():
            self.forward_seq_pipeline_stage(input_data=input_, attention_mask=attention_mask)
            print(f"<AutoBatch inference_batch> rank-<{self.pp_rank}> forward_seq_pipeline_stage is done!")
            self.forward_new_token_pipeline_stage(attention_mask=attention_mask)
            print(f"<AutoBatch inference_batch> rank-<{self.pp_rank}> forward_new_token_pipeline_stage is done!")

        self.comm.barrier()
        print(f"<AutoBatch inference_batch> rank-<{self.pp_rank}> after third barrier!")

        if self.pp_rank == self.pipeline_group_size - 1 and output_ is not None:
            ret_tokens = []
            ret_token_logprobs = []
            ret_topk_tokens = []
            ret_topk_token_logprobs = []
            for i in range(self.seq_num):
                ret_tokens.append(self.ret_tokens[i][:, :self.i_current_token[i]].cpu())
                ret_token_logprobs.append(self.ret_token_logprobs[i][:, :self.i_current_token[i]].cpu())
                if self.top_k_per_token[i] > 0:
                    ret_topk_tokens.append(self.ret_topk_tokens[i][:, :self.i_current_token[i]].cpu())
                    ret_topk_token_logprobs.append(self.ret_topk_token_logprobs[i][:, :self.i_current_token[i]].cpu())
                else:
                    ret_topk_tokens.append(None)
                    ret_topk_token_logprobs.append(None)
            print(f"<AutoBatch inference_batch> rank-<{self.pp_rank}> after marker1 !")

            for i in range(self.seq_num):
                item = {
                    'token_ids': ret_tokens[i],
                    'token_logprobs': ret_token_logprobs[i],
                    'topk_ids': ret_topk_tokens[i],
                    'topk_logprobs': ret_topk_token_logprobs[i]
                }
                output_.append(item)
            print(f"<inference_batch> rank-<{self.pp_rank}> after marker2 !")

        end_time = time.time()
        iter_time = end_time - start_time
        print("<AutoBatch inference_batch> rank-{} INFERENCE iteration takes {:3.2f}s".format(self.pp_rank, iter_time))
        print("-------------------------------------------")
        return iter_time
