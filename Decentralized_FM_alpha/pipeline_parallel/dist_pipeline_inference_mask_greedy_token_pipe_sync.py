import time
import json
import torch
import torch.nn.functional as F
from comm.comm_utils import *
from modules.generation_utils import get_logits_processor, get_logits_warper
from .dist_pipeline_inference_greedy_token_pipe_sync import (
    DistGreedyInferenceTokePipeSync,
)
from .share_prefix import SharePrefix
from coordinator.http_coordinate_client import get_coordinator_client

import os

if "SPARSE_ATT" in os.environ:
    SPARSE_ATT = True
else:
    SPARSE_ATT = False

if "SPARSE" in os.environ:
    SPARSE = True
else:
    SPARSE = False


class DistGreedyInferenceMaskTokenPipeSync(DistGreedyInferenceTokePipeSync):
    r"""
    Async implementation of Distributed Inference.
    The current implementation leave the computation on the PyTorch default stream and the communication on a different
    stream, there is:
        a group of events to check if recv (from rank i-1) finishes in the forward propagation;
        a group of events to check if computation finishes in the forward propagation.
    """

    def __init__(self, args, device, rank=None, be_coordinated=False):
        if be_coordinated:
            self.coord_client = get_coordinator_client()
        else:
            self.coord_client = None
        self.echo_prompt = args.echo_prompt
        if args.num_completions % 2 == 0:
            self.num_completions = 2
            self.num_completion_loops = args.num_completions // 2
        else:
            self.num_completions = 1
            self.num_completion_loops = args.num_completions
        self.top_k_per_token = args.top_k_per_token
        self.micro_batch_size = 1

        self.max_layers = args.max_layers
        self._layer_begin = args.num_layers * get_pipeline_parallel_rank()
        self._layer_end = args.num_layers * (get_pipeline_parallel_rank() + 1)

        ##########
        self.stop = args.stop
        # self.stop = None

        if self.stop is not None:
            # from transformers import AutoTokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            from task_datasets.inference_data import get_tokenizer

            self.tokenizer = get_tokenizer(args)
            self.stop_flag = torch.zeros(1, requires_grad=False, device=device).long()
        ##########

        ##########
        self.share_prefix = SharePrefix()  # not applicable for echo
        if not args.share_prefix:
            self.share_prefix.disable()
        elif args.echo_prompt:
            print(
                "Warn: share-prefix with echo-prompt is not supported. Disabling share-prefix.."
            )
            self.share_prefix.disable()
        ##########

        super().__init__(args, device, rank=rank)

        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_send_stream = torch.cuda.Stream(device=device, priority=0)

        if self.pp_rank == self.pipeline_group_size - 1:
            ret_seq_length = (
                self.generate_seq_length
                if not self.echo_prompt
                else self.input_seq_length + self.generate_seq_length
            )

            self.ret_tokens = torch.zeros(
                (self.seq_num * self.num_completions, ret_seq_length),
                requires_grad=False,
                device=self.device,
                dtype=torch.int64,
            )

            self.ret_token_logprobs = torch.zeros(
                (self.seq_num * self.num_completions, ret_seq_length),
                requires_grad=False,
                device=self.device,
                dtype=self.dtype,
            )

            if self.top_k_per_token > 0:
                self.ret_topk_tokens = torch.zeros(
                    (
                        self.seq_num * self.num_completions,
                        ret_seq_length,
                        self.top_k_per_token,
                    ),
                    requires_grad=False,
                    device=self.device,
                    dtype=torch.int64,
                )

                self.ret_topk_token_logprobs = torch.zeros(
                    (
                        self.seq_num * self.num_completions,
                        ret_seq_length,
                        self.top_k_per_token,
                    ),
                    requires_grad=False,
                    device=self.device,
                    dtype=self.dtype,
                )

        if self.generate_seq_length == 0:
            # reduce 1
            self.input_seq_emb = [
                torch.zeros(
                    (1, self.input_seq_length - 1, self.embedding_dim),
                    requires_grad=False,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.seq_num)
            ]
            self.output_seq_emb = [
                torch.zeros(
                    (1, self.input_seq_length - 1, self.embedding_dim),
                    requires_grad=False,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.seq_num)
            ]

    def change_buffer_size(self):
        if self.pp_rank == self.pipeline_group_size - 1:
            ret_seq_length = (
                self.generate_seq_length
                if not self.echo_prompt
                else self.input_seq_length + self.generate_seq_length
            )
            self.ret_tokens = torch.zeros(
                (self.seq_num * self.num_completions, ret_seq_length),
                requires_grad=False,
                device=self.device,
                dtype=torch.int64,
            )
            self.ret_token_logprobs = torch.zeros(
                (self.seq_num * self.num_completions, ret_seq_length),
                requires_grad=False,
                device=self.device,
                dtype=self.dtype,
            )
            if self.top_k_per_token > 0:
                self.ret_topk_tokens = torch.zeros(
                    (
                        self.seq_num * self.num_completions,
                        ret_seq_length,
                        self.top_k_per_token,
                    ),
                    requires_grad=False,
                    device=self.device,
                    dtype=torch.int64,
                )
                self.ret_topk_token_logprobs = torch.zeros(
                    (
                        self.seq_num * self.num_completions,
                        ret_seq_length,
                        self.top_k_per_token,
                    ),
                    requires_grad=False,
                    device=self.device,
                    dtype=self.dtype,
                )
        if self.pp_rank == 0:
            self.recv_new_token = [
                torch.zeros(
                    (self.token_micro_batch_size * self.num_completions, 1),
                    requires_grad=False,
                    device=self.device,
                    dtype=torch.int64,
                )
                for _ in range(self.token_micro_batch_num)
            ]
        if self.pp_rank == self.pipeline_group_size - 1:
            self.send_new_tokens = [
                torch.zeros(
                    (self.token_micro_batch_size * self.num_completions, 1),
                    requires_grad=False,
                    device=self.device,
                    dtype=torch.int64,
                )
                for _ in range(self.token_micro_batch_num)
            ]

        self.input_seq_emb = [
            torch.zeros(
                (1, self.input_seq_length, self.embedding_dim),
                requires_grad=False,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.seq_num)
        ]
        self.output_seq_emb = [
            torch.zeros(
                (1, self.input_seq_length, self.embedding_dim),
                requires_grad=False,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.seq_num)
        ]
        self.input_token_emb = [
            torch.zeros(
                (
                    self.token_micro_batch_size * self.num_completions,
                    1,
                    self.embedding_dim,
                ),
                requires_grad=False,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.token_micro_batch_num)
        ]
        self.output_token_emb = [
            torch.zeros(
                (
                    self.token_micro_batch_size * self.num_completions,
                    1,
                    self.embedding_dim,
                ),
                requires_grad=False,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.token_micro_batch_num)
        ]

    def _print_buffers(self):
        if self.generate_seq_length == 0:
            # don't print when seq_length == 0
            return
        super()._print_buffers()

    def _get_embedding_size(self):
        if self.model_type == "gpt2":
            from modules.hf_gpt2_module import GPTConfig

            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_embd
        elif self.model_type == "gptj":
            from modules.hf_gptj_module import GPTConfig

            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_embd
        elif self.model_type == "gptneox":
            from modules.hf_gptneox_module import GPTConfig

            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type in [
            "opt",
            "opt-save",
            "opt-ml-att-sparse",
        ]:
            from modules.hf_opt_module import GPTConfig

            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type == "yalm":
            from modules.yalm_module import GPTConfig

            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type == "glm":
            from modules.glm_module import GPTConfig

            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        else:
            raise Exception(f"unknown model type {self.model_type}")

    def _create_layers(self):
        if self.model_type == "gpt2":
            from modules.hf_gpt2_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == "gptj":
            from modules.hf_gptj_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == "gptneox":
            from modules.hf_gptneox_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == "opt":
            from modules.hf_opt_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == "opt-save":
            from modules.hf_opt_module_save import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == "opt-ml-att-sparse":
            from modules.hf_opt_sparse_mlp_attention import (
                GPTEmbeddings,
                GPTBlock,
                GPTLMHead,
            )
        elif self.model_type == "yalm":
            from modules.yalm_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == "glm":
            from modules.glm_module import GPTEmbeddings, GPTBlock, GPTLMHead

            GPTBlock.echo_prompt = self.echo_prompt
        else:
            raise Exception(f"unknown model type {self.model_type}")

        if self.pp_rank == 0:
            self.layers["emb"] = (
                GPTEmbeddings.from_pretrained(self.model_name)
                .to(self.dtype)
                .eval()
                .to(self.device)
            )
        for layer_index in range(self.num_layers):
            # global layer indexing could be an argument
            global_layer_index = self.num_layers * self.pp_rank + layer_index
            if self.max_layers is not None and global_layer_index >= self.max_layers:
                # TODO: this is a hack
                self.num_layers = layer_index
                break
            print(f"loading layer {global_layer_index}")
            self.layers["block" + str(layer_index)] = (
                GPTBlock.from_pretrained(
                    self.model_name, layer_index=global_layer_index
                )
                .to(self.dtype)
                .eval()
                .to(self.device)
            )

            if self.coord_client:
                self.coord_client.update_status(
                    "running",
                    returned_payload={
                        "rank": self.pp_rank,
                        "loaded_layer": layer_index,
                        "total_layer": self.num_layers,
                    },
                )
        if self.pp_rank == self.pipeline_group_size - 1:
            self.layers["lm"] = (
                GPTLMHead.from_pretrained(self.model_name)
                .to(self.dtype)
                .eval()
                .to(self.device)
            )

    def _init_cached_seqs_and_attentions(self):
        self._is_merged = False
        if not self.echo_prompt:
            self.i_current_token = 0
        else:
            self.i_current_token = self.input_seq_length

        self.cached_attention.clear()
        for _ in range(self.num_layers):
            self.cached_attention.append([None for _ in range(self.seq_num)])

        # useful for num_completions > 1
        self.token_cached_attention = []
        for _ in range(self.num_layers):
            self.token_cached_attention.append([None for _ in range(self.seq_num)])

        if self.stop is not None:
            self.stop_flag[:] = 0

        self.share_prefix.clear()

    def _merge_cached_seqs_and_attentions(self):
        if not self.echo_prompt:
            self.i_current_token = 0
        else:
            self.i_current_token = self.input_seq_length

        if self._is_merged:
            if self.pp_rank == self.pipeline_group_size - 1:
                for i in range(self.seq_num):
                    self._copy_initial_token_emb(i)
                    if (
                        i % self.token_micro_batch_size
                        == self.token_micro_batch_size - 1
                    ):
                        self._generate_new_token(i // self.token_micro_batch_size)

            for layer_index in range(self.num_layers):
                for index in range(self.token_micro_batch_num):
                    key, value = self.cached_attention[layer_index][index]
                    self.cached_attention[layer_index][index] = (
                        key[:, :, : self.input_seq_length],
                        value[:, :, : self.input_seq_length],
                    )

            self.token_cached_attention = []
            for _ in range(self.num_layers):
                self.token_cached_attention.append([None for _ in range(self.seq_num)])
        else:
            super()._merge_cached_seqs_and_attentions()
            self._is_merged = True

        if self.stop is not None:
            self.stop_flag[:] = 0

        self.share_prefix.clear()  # token generation do not need this

    def _forward_compute_prompt_seq(self, index, seq, mask):
        print("Compute prompt seq<", index, ">.")
        if self.pp_rank == 0:
            self.input_seq_emb[index] = self.layers["emb"](seq, mask=mask)
        current_emb = self.input_seq_emb[index]
        caches = [None] * self.num_layers
        previous_emb = None

        # when disabled, this will do nothing
        mask, current_emb, caches = self.share_prefix.process_inputs(
            seq, mask, current_emb, caches
        )

        if SPARSE_ATT:
            for layer_index in range(self.num_layers):
                input_emb = current_emb.clone()
                current_emb, caches[layer_index] = self.layers[
                    "block" + str(layer_index)
                ](
                    current_emb,
                    caches[layer_index],
                    mask=mask,
                    previous_emb=previous_emb,
                )
                self.cached_attention[layer_index][index] = caches[layer_index]

                previous_emb = input_emb
        elif SPARSE:
            # TODO: previous MLP block input as input to MLP sparse predictor
            pass
        else:
            for layer_index in range(self.num_layers):
                current_emb, caches[layer_index] = self.layers[
                    "block" + str(layer_index)
                ](current_emb, caches[layer_index], mask=mask)
                self.cached_attention[layer_index][index] = caches[layer_index]

        #'''
        # when disabled, this will do nothing
        current_emb = self.share_prefix.process_outputs(seq, mask, current_emb, caches)

        self.output_seq_emb[index] = current_emb

        if self.pp_rank == self.pipeline_group_size - 1:
            self._copy_initial_token_emb(index)

            if self.echo_prompt:
                self._generate_echo_token_logprobs(index, indices=seq)

    def _generate_echo_token_logprobs(self, index, indices):
        assert self.pp_rank == self.pipeline_group_size - 1
        assert self.num_completions == 1
        if self.generate_seq_length == 0:
            z = self.layers["lm"](self.output_seq_emb[index])
        else:
            z = self.layers["lm"](self.output_seq_emb[index][:, :-1])

        z = F.log_softmax(z, -1)
        original_indices = indices
        indices = indices[:, 1:]  # skip first

        logprobs = torch.gather(z, -1, indices.unsqueeze(-1)).squeeze(-1)
        self.ret_tokens[
            index * self.micro_batch_size : (index + 1) * self.micro_batch_size,
            : self.i_current_token,
        ] = original_indices
        self.ret_token_logprobs[
            index * self.micro_batch_size : (index + 1) * self.micro_batch_size,
            1 : self.i_current_token,
        ] = logprobs
        if self.top_k_per_token > 0:
            logprobs, indices = z.topk(k=self.top_k_per_token, dim=-1)
            self.ret_topk_tokens[
                index * self.micro_batch_size : (index + 1) * self.micro_batch_size,
                1 : self.i_current_token,
            ] = indices
            self.ret_topk_token_logprobs[
                index * self.micro_batch_size : (index + 1) * self.micro_batch_size,
                1 : self.i_current_token,
            ] = logprobs

    def _copy_initial_token_emb(self, index):
        assert self.pp_rank == self.pipeline_group_size - 1
        buff_i = index // self.token_micro_batch_size
        pos = index % self.token_micro_batch_size
        print("_copy_initial_token_emb")
        for k in range(self.num_completions):
            print(f"_copy_initial_token_emb {k}/{self.num_completions}")
            self.output_token_emb[buff_i][
                pos + k * self.token_micro_batch_size
            ] = self.output_seq_emb[index][:, -1:]

    def _get_cached_attention(self, layer_index, token_batch_index):
        if self.num_completions == 1:
            return self.cached_attention[layer_index][token_batch_index]

        else:
            prompt_cache = self.cached_attention[layer_index][
                token_batch_index
            ]  # 2* (token_bs, ., seq_len, .)
            token_cache = self.token_cached_attention[layer_index][
                token_batch_index
            ]  # 2* (token_bs * num_compl, ., seq_len, .)
            prompt_cache = [
                prompt_cache[0].repeat(self.num_completions, 1, 1, 1),
                prompt_cache[1].repeat(self.num_completions, 1, 1, 1),
            ]  # 2*(token_bs * num_compl, ., seq_len, .)
            if token_cache is not None:
                token_cache = [
                    torch.cat([prompt_cache[0], token_cache[0]], dim=2),
                    torch.cat([prompt_cache[1], token_cache[1]], dim=2),
                ]
            else:
                token_cache = prompt_cache

            return token_cache

    def _set_cached_attention(self, cache, layer_index, token_batch_index):
        # self.cached_attention[layer_index][token_batch_index] = cache

        if self.num_completions == 1:
            self.cached_attention[layer_index][token_batch_index] = cache
        else:
            self.token_cached_attention[layer_index][token_batch_index] = [
                cache[0][:, :, self.input_seq_length :],
                cache[1][:, :, self.input_seq_length :],
            ]

    def _forward_compute_generate_token(self, index, mask=None):
        if mask is not None and self.num_completions > 1:
            # repeat n times
            mask = mask.repeat(self.num_completions, 1)

        # print("Compute generate seq micro-batch <", index, ">.")
        if self.pp_rank == 0:
            cache = self._get_cached_attention(0, index)
            current_emb = self.layers["emb"](
                self.recv_new_token[index], self.cached_attention[0][index], mask=mask
            )
        else:
            current_emb = self.input_token_emb[index]

        for layer_index in range(self.num_layers):
            cache = self._get_cached_attention(layer_index, index)
            current_emb, cache = self.layers["block" + str(layer_index)](
                current_emb, cache, mask=mask
            )
            self._set_cached_attention(cache, layer_index, index)
        self.output_token_emb[index] = current_emb

        if self.pp_rank == self.pipeline_group_size - 1:
            self._generate_new_token(index)

    def _generate_new_token(self, index):
        assert self.pp_rank == self.pipeline_group_size - 1
        z = self.layers["lm"](self.output_token_emb[index])
        z = z.float()
        z = F.log_softmax(z, -1)
        logprobs, indices = z.topk(k=1, dim=-1)
        self.send_new_tokens[index] = indices
        self.ret_tokens[
            index
            * self.token_micro_batch_size
            * self.num_completions : (index + 1)
            * self.token_micro_batch_size
            * self.num_completions,
            self.i_current_token,
        ] = indices.squeeze(-1).squeeze(-1)
        self.ret_token_logprobs[
            index
            * self.token_micro_batch_size
            * self.num_completions : (index + 1)
            * self.token_micro_batch_size
            * self.num_completions,
            self.i_current_token,
        ] = logprobs.squeeze(-1).squeeze(-1)
        if self.top_k_per_token > 0:
            logprobs, indices = z.topk(k=self.top_k_per_token, dim=-1)
            self.ret_topk_tokens[
                index
                * self.token_micro_batch_size
                * self.num_completions : (index + 1)
                * self.token_micro_batch_size
                * self.num_completions,
                self.i_current_token,
            ] = indices.squeeze(1)
            self.ret_topk_token_logprobs[
                index
                * self.token_micro_batch_size
                * self.num_completions : (index + 1)
                * self.token_micro_batch_size
                * self.num_completions,
                self.i_current_token,
            ] = logprobs.squeeze(1)

        if index == self.token_micro_batch_num - 1:
            self.i_current_token += 1

    def _process_mask_during_generation(self, attention_mask):
        if attention_mask is not None:
            # increase one for the new token
            attention_mask = F.pad(attention_mask, pad=(0, 1), mode="constant", value=1)

        return attention_mask

    def forward_seq_pipeline_stage(self, input_data=None, attention_mask=None):
        if self.pp_rank == 0 or self.pp_rank == self.pipeline_group_size - 1:
            assert input_data is not None
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
                self.profile_mark_forward_seq_comp_start(i)
                self._forward_compute_prompt_seq(
                    index=i, seq=input_seqs[i], mask=attention_mask[i]
                )
                self.profile_mark_forward_seq_comp_end(i)
                # Send
                self.profile_mark_forward_seq_send_start(i)
                self.comm.send(self.output_seq_emb[i], dst=self.post_node_rank)
                self.profile_mark_forward_seq_send_end(i)
            elif (
                self.pp_rank == self.pipeline_group_size - 1
            ):  # Only receive input from last node, do not send
                # Receive
                self.profile_mark_forward_seq_recv_start(i)
                self.comm.recv(self.input_seq_emb[i], src=self.pre_node_rank)
                self.profile_mark_forward_seq_recv_end(i)
                # Compute
                self.profile_mark_forward_seq_comp_start(i)
                self._forward_compute_prompt_seq(
                    index=i, seq=input_seqs[i], mask=attention_mask[i]
                )
                self.profile_mark_forward_seq_comp_end(i)
            else:  # receive, compute, and send
                # Receive
                self.profile_mark_forward_seq_recv_start(i)
                self.comm.recv(self.input_seq_emb[i], src=self.pre_node_rank)
                self.profile_mark_forward_seq_recv_end(i)
                # Compute
                self.profile_mark_forward_seq_comp_start(i)
                self._forward_compute_prompt_seq(
                    index=i, seq=input_seqs[i], mask=attention_mask[i]
                )
                self.profile_mark_forward_seq_comp_end(i)
                # Send
                self.profile_mark_forward_seq_send_start(i)
                self.comm.send(self.output_seq_emb[i], dst=self.post_node_rank)
                self.profile_mark_forward_seq_send_end(i)

        if self.enable_tidy_profiling:
            self.profile_seq_pipeline_stage()

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
                    tokens = tokens[: self.i_current_token]
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
                    self.profile_mark_forward_token_send_start(i)
                    self.comm.send(self.send_new_tokens[i], dst=0)
                    self.profile_mark_forward_token_send_end(i)
                else:
                    # Receive
                    self.profile_mark_forward_token_recv_start(i)
                    self.comm.recv(self.input_token_emb[i], src=self.pre_node_rank)
                    self.profile_mark_forward_token_recv_end(i)
                    # Compute
                    self.profile_mark_forward_token_comp_start(i)
                    self._forward_compute_generate_token(i, mask=attention_masks[i])
                    self.profile_mark_forward_token_comp_end(i)
                    if step != self.generate_seq_length - 1 and (
                        self.stop is None or self.stop_flag.item() == 0
                    ):
                        # Send
                        self.profile_mark_forward_token_send_start(i)
                        self.comm.send(self.send_new_tokens[i], dst=0)
                        self.profile_mark_forward_token_send_end(i)
            # Rank-0 node:
            elif self.pp_rank == 0:
                if step != self.generate_seq_length - 1 and (
                    self.stop is None or self.stop_flag.item() == 0
                ):
                    # Receive
                    self.profile_mark_forward_token_recv_start(i)
                    self.comm.recv(
                        self.recv_new_token[i], src=self.pipeline_group_size - 1
                    )
                    self.profile_mark_forward_token_recv_end(i)
                    # Compute
                    self.profile_mark_forward_token_comp_start(i)
                    self._forward_compute_generate_token(i, mask=attention_masks[i])
                    self.profile_mark_forward_token_comp_end(i)
                    # Send
                    self.profile_mark_forward_token_send_start(i)
                    self.comm.send(self.output_token_emb[i], dst=self.post_node_rank)
                    self.profile_mark_forward_token_send_end(i)
            else:  # Middle nodes:
                if step != self.generate_seq_length - 1 and (
                    self.stop is None or self.stop_flag.item() == 0
                ):
                    # Receive
                    self.profile_mark_forward_token_recv_start(i)
                    self.comm.recv(self.input_token_emb[i], src=self.pre_node_rank)
                    self.profile_mark_forward_token_recv_end(i)
                    # Compute
                    self.profile_mark_forward_token_comp_start(i)
                    self._forward_compute_generate_token(i, mask=attention_masks[i])
                    self.profile_mark_forward_token_comp_end(i)
                    # Send
                    self.profile_mark_forward_token_send_start(i)
                    self.comm.send(self.output_token_emb[i], dst=self.post_node_rank)
                    self.profile_mark_forward_token_send_end(i)

        if self.enable_tidy_profiling:
            self.profile_token_pipeline_step(step)

    def inference_batch(self, input_=None, output_=None, attention_mask=None):
        print(f"<inference_batch> rank-<{self.pp_rank}> Enter!")
        self.comm.barrier()
        print(f"<inference_batch> rank-<{self.pp_rank}> after first barrier!")
        self._init_cached_seqs_and_attentions()  # TODO: should I put here
        print(
            f"<inference_batch> rank-<{self.pp_rank}> after first _init_cached_seqs_and_attentions!"
        )
        self.comm.barrier()
        print(f"<inference_batch> rank-<{self.pp_rank}> after second barrier!")
        start_time = time.time()
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.init_time_stamp = time.time() * 1e6
            self.init_event.record()

        print(f"<inference_batch> rank-<{self.pp_rank}> enter computation!")
        with torch.no_grad():
            self.forward_seq_pipeline_stage(
                input_data=input_, attention_mask=attention_mask
            )
            print(
                f"<inference_batch> rank-<{self.pp_rank}> forward_seq_pipeline_stage is done!"
            )

            for nc in range(self.num_completion_loops):
                self.forward_new_token_pipeline_stage(attention_mask=attention_mask)
                print(
                    f"<inference_batch> rank-<{self.pp_rank}> forward_seq_pipeline_stage is done!"
                )

                self.comm.barrier()
                print(f"<inference_batch> rank-<{self.pp_rank}> after third barrier!")

                if self.pp_rank == self.pipeline_group_size - 1 and output_ is not None:
                    # token_micro_batch_num * num_completions
                    ret_tokens = (
                        self.ret_tokens[:, : self.i_current_token]
                        .cpu()
                        .split(self.token_micro_batch_size)
                    )
                    ret_token_logprobs = (
                        self.ret_token_logprobs[:, : self.i_current_token]
                        .cpu()
                        .split(self.token_micro_batch_size)
                    )
                    if self.top_k_per_token > 0:
                        ret_topk_tokens = (
                            self.ret_topk_tokens[:, : self.i_current_token]
                            .cpu()
                            .split(self.token_micro_batch_size)
                        )
                        ret_topk_token_logprobs = (
                            self.ret_topk_token_logprobs[:, : self.i_current_token]
                            .cpu()
                            .split(self.token_micro_batch_size)
                        )

                    print(f"<inference_batch> rank-<{self.pp_rank}> after marker1 !")

                    for i in range(self.num_completions):
                        item = {
                            "token_ids": torch.cat(
                                ret_tokens[i :: self.num_completions], 0
                            ),
                            "token_logprobs": torch.cat(
                                ret_token_logprobs[i :: self.num_completions], 0
                            ),
                        }
                        if self.top_k_per_token > 0:
                            item["topk_ids"] = torch.cat(
                                ret_topk_tokens[i :: self.num_completions], 0
                            )
                            item["topk_logprobs"] = torch.cat(
                                ret_topk_token_logprobs[i :: self.num_completions], 0
                            )
                        output_.append(item)
                    print(f"<inference_batch> rank-<{self.pp_rank}> after marker2 !")

        end_time = time.time()
        iter_time = end_time - start_time
        print(
            "Rank {} node whole INFERENCE iteration takes {:3.2f}s".format(
                self.global_rank, iter_time
            )
        )
        print("-------------------------------------------")

        return iter_time
