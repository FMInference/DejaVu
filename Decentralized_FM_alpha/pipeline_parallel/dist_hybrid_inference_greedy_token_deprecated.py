import time
import json
import torch.nn.functional
from comm.hybrid_comm_utils import *


class DistHybridGreedyInference:
    r"""
    Hybrid implementation of Distributed Inference.
    GPU: prompt
    CPU: token generation
    """
    def __init__(self, args, device, rank=None):
        print("=======Initialize Hybrid Dist Inference(Sync).=======")
        if args.fp16:
            self.use_fp16 = True
            print("=======Hybrid use FP16=======")
        else:
            self.use_fp16 = False
            print("=======Hybrid use FP32=======")

        self.node_type = args.node_type
        self.dtype = torch.bfloat16 if self.use_fp16 else torch.float32

        self.model_name = args.model_name
        self.model_type = args.model_type
        self.cpu_comm = get_hybrid_dispatch_comm()  # This is the default pytorch gloo backend.
        self.input_seq_length = args.input_seq_length
        self.emb_dim = self._get_embedding_size()
        self.num_head = self._get_num_heads()
        self.head_dim = self.emb_dim // self.num_head
        self.prompt_micro_batch_size = args.prompt_micro_batch_size
        self.device = device
        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')

        if rank is None:
            self.global_rank = args.rank
        else:
            self.global_rank = rank

        self.stage_num_layers = args.stage_num_layers
        if self.node_type == 'GPU':
            self.pipeline_group_size = args.pipeline_group_size
            self.cpu_pool_size = args.world_size - args.pipeline_group_size
            self.pp_rank = get_gpu_pipeline_rank()  # Rank is the pipeline rank by default.
            self.pre_node_rank = self.pp_rank - 1
            self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1
            self.gpu_comm = get_gpu_pipeline_comm()
            temp_shape = (self.prompt_micro_batch_size, self.input_seq_length, self.emb_dim)
            self.input_seq_emb = torch.zeros(temp_shape, requires_grad=False, device=self.device, dtype=self.dtype)
            self.output_seq_emb = torch.zeros(temp_shape, requires_grad=False, device=self.device, dtype=self.dtype)
            self.producer_buffer_size = args.producer_buffer_size
            key_value_shape = (self.prompt_micro_batch_size, self.num_head, self.input_seq_length, self.head_dim)
            self.producer_key = [[torch.zeros(key_value_shape, requires_grad=False, device='cpu', dtype=self.dtype)
                                  for _ in range(self.stage_num_layers)]
                                 for _ in range(self.producer_buffer_size)]
            self.producer_value = [[torch.zeros(key_value_shape, requires_grad=False, device='cpu', dtype=self.dtype)
                                    for _ in range(self.stage_num_layers)]
                                   for _ in range(self.producer_buffer_size)]
            if self.pp_rank == self.pipeline_group_size - 1:
                self.producer_output = [torch.zeros((self.prompt_micro_batch_size, 1, self.emb_dim),
                                                    requires_grad=False, device='cpu', dtype=self.dtype)
                                        for _ in range(self.producer_buffer_size)]
            self.gpu_layers = {}
            self._create_gpu_layers()
            self._print_buffers_gpu_node()
            self.dispatch_ranks = get_cpu_ranks()
            self.current_dispatch_index = 0
            assert args.producer_buffer_size == self.cpu_pool_size * args.consumer_buffer_size, \
                f"Producer and consumer buffer size are set incorrectly. " \
                f"CPU pool: {self.cpu_pool_size}, " \
                f"producer buffer size: {args.producer_buffer_size}, consumer buffer size: {args.consumer_buffer_size}"

        elif self.node_type == 'CPU':
            self.generate_seq_length = args.generate_seq_length
            self.token_micro_batch_size = args.token_micro_batch_size
            self.global_num_layers = args.global_num_layers
            self.consumer_buffer_size = args.consumer_buffer_size
            self.consumer_prompt_output = [torch.zeros((self.prompt_micro_batch_size, 1, self.emb_dim),
                                                       requires_grad=False, device='cpu', dtype=self.dtype)
                                           for _ in range(self.consumer_buffer_size)]
            key_value_shape = (self.prompt_micro_batch_size, self.num_head, self.input_seq_length, self.head_dim)
            self.consumer_key = [[torch.zeros(key_value_shape, requires_grad=False, device='cpu', dtype=self.dtype)
                                  for _ in range(self.global_num_layers)]
                                 for _ in range(self.consumer_buffer_size)]
            self.consumer_value = [[torch.zeros(key_value_shape, requires_grad=False, device='cpu', dtype=self.dtype)
                                    for _ in range(self.global_num_layers)]
                                   for _ in range(self.consumer_buffer_size)]
            self.cpu_layers = {}
            self._create_cpu_layers()
            self._print_buffers_cpu_node()
        else:
            assert False
        if self.enable_tidy_profiling:
            self.profiling_log = []
            self.init_time_stamp = None
            if self.node_type == 'GPU':
                self.init_event = torch.cuda.Event(enable_timing=True, blocking=False)
                self.forward_seq_recv_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
                self.forward_seq_recv_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
                self.forward_seq_comp_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
                self.forward_seq_comp_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
                self.forward_seq_send_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
                self.forward_seq_send_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
                self.forward_gpu2cpu_send_start_time = None
                self.forward_gpu2cpu_send_end_time = None
            elif self.node_type == 'CPU':
                self.forward_token_recv_start_time = None
                self.forward_token_recv_end_time = None
                self.forward_token_comp_start_time = None
                self.forward_token_comp_end_time = None
            else:
                assert False

    def _get_cpu_dst_rank(self):
        return self.dispatch_ranks[self.current_dispatch_index]

    def _get_gpu_src_rank(self, layer_index):
        return layer_index // self.stage_num_layers

    def _print_buffers_gpu_node(self):
        print("Rank-{} Print buffers meta-info on GPU-node.".format(self.global_rank))
        seq_emb_num = self.prompt_micro_batch_size * self.input_seq_length * self.emb_dim
        if self.use_fp16:
            print("=======input_seq_emb: {} MB shape: {} X 1 (fp16)======="
                  .format(seq_emb_num * 2 // 1024 // 1024, self.input_seq_emb.shape))
            print("=======output_seq_emb: {} MB shape: {} X 1 (fp16)======="
                  .format(seq_emb_num * 2 // 1024 // 1024, self.input_seq_emb.shape))
        else:
            print("=======input_seq_emb: {} MB shape: {} X 1 (fp32)======="
                  .format(seq_emb_num * 4 // 1024 // 1024, self.input_seq_emb.shape, 1))
            print("=======output_seq_emb: {} MB shape: {} X 1 (fp32)======="
                  .format(seq_emb_num * 4 // 1024 // 1024, self.input_seq_emb.shape, 1))
        kv_tensor_dim = self.prompt_micro_batch_size * self.input_seq_length * self.emb_dim
        kv_tensor_num = self.producer_buffer_size * self.stage_num_layers
        kv_tensor_total = kv_tensor_num * kv_tensor_dim
        if self.use_fp16:
            print("=======key_tensor_emb: {} MB shape: {} X {} (fp16)======="
                  .format(kv_tensor_total // 524288, self.input_seq_emb.shape, kv_tensor_num))
            print("=======value_seq_emb: {} MB shape: {} X {} (fp16)======="
                  .format(kv_tensor_total // 524288, self.input_seq_emb.shape, kv_tensor_num))
        else:
            print("=======key_tensor_emb: {} MB shape: {} X {} (fp32)======="
                  .format(kv_tensor_total // 262144, self.input_seq_emb.shape, kv_tensor_num))
            print("=======value_seq_emb: {} MB shape: {} X {} (fp32)======="
                  .format(kv_tensor_total // 262144, self.input_seq_emb.shape, kv_tensor_num))

    def _print_buffers_cpu_node(self):
        print("Rank-{} Print buffers meta-info on CPU-node.".format(self.global_rank))
        kv_tensor_dim = self.prompt_micro_batch_size * self.input_seq_length * self.emb_dim
        kv_tensor_num = self.consumer_buffer_size * self.global_num_layers
        kv_tensor_total = kv_tensor_num * kv_tensor_dim
        if self.use_fp16:
            print("=======key_tensor_emb: {} MB shape: {} X {} (fp16)======="
                  .format(kv_tensor_total // 524288, self.consumer_prompt_output[0].shape, kv_tensor_num))
            print("=======value_seq_emb: {} MB shape: {} X {} (fp16)======="
                  .format(kv_tensor_total // 524288, self.consumer_prompt_output[0].shape, kv_tensor_num))
        else:
            print("=======input_token_emb: {} MB shape: {} X {} (fp32)======="
                  .format(kv_tensor_total // 262144, self.consumer_prompt_output[0].shape, kv_tensor_num))
            print("=======output_token_emb: {} MB shape: {} X {} (fp32)======="
                  .format(kv_tensor_total * 262144, self.consumer_prompt_output[0].shape, kv_tensor_num))

    def _get_embedding_size(self):
        if self.model_type == 'gpt2':
            from modules.hf_gpt2_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_embd
        elif self.model_type == 'gptj':
            from modules.hf_gptj_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_embd
        else:
            raise Exception(f'unknown model type {self.model_type}')

    def _get_num_heads(self):
        if self.model_type == 'gpt2':
            from modules.hf_gpt2_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_head
        elif self.model_type == 'gptj':
            from modules.hf_gptj_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_head
        else:
            raise Exception(f'unknown model type {self.model_type}')

    def _create_gpu_layers(self):
        if self.model_type == 'gpt2':
            from modules.hf_gpt2_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'gptj':
            from modules.hf_gptj_module import GPTEmbeddings, GPTBlock, GPTLMHead
        else:
            raise Exception(f'unknown model type {self.model_type}')

        if self.pp_rank == 0:
            self.gpu_layers['emb'] = GPTEmbeddings.from_pretrained(
                self.model_name
            ).to(self.dtype).eval().to(self.device)
        for layer_index in range(self.stage_num_layers):
            # global layer indexing could be an argument
            global_layer_index = self.stage_num_layers * self.pp_rank + layer_index
            print(f'loading layer {global_layer_index}')
            self.gpu_layers['block' + str(layer_index)] = GPTBlock.from_pretrained(
                self.model_name, layer_index=global_layer_index
            ).to(self.dtype).eval().to(self.device)

    def _create_cpu_layers(self):
        if self.model_type == 'gpt2':
            from modules.hf_gpt2_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'gptj':
            from modules.hf_gptj_module import GPTEmbeddings, GPTBlock, GPTLMHead
        else:
            raise Exception(f'unknown model type {self.model_type}')
        '''
        import intel_extension_for_pytorch as ipex
        self.cpu_layers['emb'] = ipex.optimize(GPTEmbeddings.from_pretrained(self.model_name)
                                               .to(dtype=self.dtype, memory_format=torch.channels_last).eval())
        for global_layer_index in range(self.global_num_layers):
            # global layer indexing could be an argument
            print(f' loading layer {global_layer_index}')
            self.cpu_layers['block' + str(global_layer_index)] = ipex.optimize(GPTBlock.from_pretrained(
                self.model_name, layer_index=global_layer_index
            ).to(dtype=self.dtype, memory_format=torch.channels_last).eval())
        self.cpu_layers['lm'] = ipex.optimize(GPTLMHead.from_pretrained(self.model_name)
                                              .to(dtype=self.dtype, memory_format=torch.channels_last).eval())
        '''
        self.cpu_layers['emb'] = GPTEmbeddings.from_pretrained(self.model_name).to(dtype=self.dtype).eval()
        for global_layer_index in range(self.global_num_layers):
            # global layer indexing could be an argument
            print(f'loading layer {global_layer_index}')
            self.cpu_layers['block' + str(global_layer_index)] = GPTBlock.from_pretrained(
                self.model_name, layer_index=global_layer_index
            ).to(dtype=self.dtype).eval()
        self.cpu_layers['lm'] = GPTLMHead.from_pretrained(self.model_name).to(dtype=self.dtype).eval()

    def _add_producer_cached_tuples(self, layer_index, buf_index, key_value_tuple):
        # print(self.producer_key[buf_index][layer_index].shape)
        # print(key_value_tuple[0].shape)
        self.producer_key[buf_index][layer_index].copy_(key_value_tuple[0], non_blocking=True)
        self.producer_value[buf_index][layer_index].copy_(key_value_tuple[1], non_blocking=True)

    def _add_producer_output_emb(self, buf_index):
        self.producer_output[buf_index].copy_(self.output_seq_emb[:, -1:], non_blocking=True)

    def _get_consumer_cached_tuples(self, layer_index, buf_index):
        return self.consumer_key[buf_index][layer_index], self.consumer_value[buf_index][layer_index]

    def _update_consumer_cached_tuples(self, layer_index, buf_index, key_value_tuple):
        self.consumer_key[buf_index][layer_index] = key_value_tuple[0]
        self.consumer_value[buf_index][layer_index] = key_value_tuple[1]

    def _gpu_forward_compute_prompt_seq(self, buf_index, seq=None):
        print("Compute prompt seq micro-batch <", buf_index, ">.")
        with torch.no_grad():
            if self.pp_rank == 0:
                self.input_seq_emb = self.gpu_layers['emb'](seq)
            current_emb = None
            for layer_index in range(self.stage_num_layers):
                if layer_index == 0:
                    current_emb, key_value_tuple = \
                        self.gpu_layers['block' + str(layer_index)](self.input_seq_emb)
                elif layer_index == self.stage_num_layers - 1:
                    self.output_seq_emb, key_value_tuple = \
                        self.gpu_layers['block' + str(layer_index)](current_emb)
                else:
                    current_emb, key_value_tuple = \
                        self.gpu_layers['block' + str(layer_index)](current_emb)
                self._add_producer_cached_tuples(layer_index, buf_index, key_value_tuple)
            if self.pp_rank == self.pipeline_group_size - 1:
                self._add_producer_output_emb(layer_index)

    def _cpu_forward_compute_generate_token(self, buf_index, last_token):
        with torch.no_grad():
            current_emb = self.cpu_layers['emb'](last_token)  # TODO, modify the emb interface to pass length.
            for layer_index in range(self.global_num_layers):
                if layer_index != self.global_num_layers - 1:
                    current_emb, key_value_tuple = self.cpu_layers['block' + str(layer_index)](
                        current_emb, self._get_consumer_cached_tuples(layer_index, buf_index))
                else:
                    output_emb, key_value_tuple = self.cpu_layers['block' + str(layer_index)](
                        current_emb, self._get_consumer_cached_tuples(layer_index, buf_index))
                current_emb = current_emb.to(self.dtype)
                self._update_consumer_cached_tuples(layer_index, buf_index, key_value_tuple)
            return self._cpu_generate_new_token(output_emb)

    def _cpu_generate_new_token(self, output_emb):
        with torch.no_grad():
            z = self.cpu_layers['lm'](output_emb)
            new_token = z.argmax(-1)
            print("Generate new token: ", new_token.shape)
            return new_token

    def _gpu_send_key_value(self, buf_index):
        torch.cuda.synchronize()
        for layer_index in range(self.stage_num_layers):
            print("Rank-{} GPU node send Local Layer-{} key to Rank-{} CPU node (Buffer-index: {})."
                  .format(self.global_rank, layer_index, self._get_cpu_dst_rank(), buf_index))
            self.cpu_comm.send(self.producer_key[buf_index][layer_index], self._get_cpu_dst_rank())
            print("Rank-{} GPU node send Local Layer-{} value to Rank-{} CPU node (Buffer-index: {})."
                  .format(self.global_rank, layer_index, self._get_cpu_dst_rank(), buf_index))
            self.cpu_comm.send(self.producer_value[buf_index][layer_index], self._get_cpu_dst_rank())
        if self.pp_rank == self.pipeline_group_size - 1:
            print("Rank-{} GPU node send output-emb to Rank-{} CPU node (Buffer-index: {})."
                  .format(self.global_rank, self._get_cpu_dst_rank(), buf_index))
            self.cpu_comm.send(self.producer_output[buf_index], self._get_cpu_dst_rank())

    def _cpu_recv_key_value(self, buf_index):
        for layer_index in range(self.global_num_layers):
            print("Rank-{} CPU node recv Layer-{} key from Rank-{} GPU node (Buffer-index: {})"
                  .format(self.global_rank, layer_index, self._get_gpu_src_rank(layer_index), buf_index))
            self.cpu_comm.recv(self.consumer_key[buf_index][layer_index], self._get_gpu_src_rank(layer_index))
            print("Rank-{} CPU node recv Layer-{} value from Rank-{} GPU node (Buffer-index: {})"
                  .format(self.global_rank, layer_index, self._get_gpu_src_rank(layer_index), buf_index))
            self.cpu_comm.recv(self.consumer_value[buf_index][layer_index], self._get_gpu_src_rank(layer_index))
        print("Rank-{} CPU node recv output-emb from Rank-{} GPU node (Buffer-index: {})"
              .format(self.global_rank, self._get_gpu_src_rank(self.global_num_layers-1), buf_index))
        self.cpu_comm.recv(self.consumer_prompt_output[buf_index], self._get_gpu_src_rank(self.global_num_layers-1))

    def profile_gpu_mark_forward_seq_recv_start(self):
        if self.enable_tidy_profiling:
            self.forward_seq_recv_start_event.record()

    def profile_gpu_mark_forward_seq_recv_end(self):
        if self.enable_tidy_profiling:
            self.forward_seq_recv_end_event.record()

    def profile_gpu_mark_forward_seq_comp_start(self):
        if self.enable_tidy_profiling:
            self.forward_seq_comp_start_event.record()

    def profile_gpu_mark_forward_seq_comp_end(self):
        if self.enable_tidy_profiling:
            self.forward_seq_comp_end_event.record()

    def profile_gpu_mark_forward_seq_send_start(self):
        if self.enable_tidy_profiling:
            self.forward_seq_send_start_event.record()

    def profile_gpu_mark_forward_seq_send_end(self):
        if self.enable_tidy_profiling:
            self.forward_seq_send_end_event.record()

    def profile_gpu2cpu_mark_forward_seq_send_start(self):
        if self.enable_tidy_profiling:
            self.forward_gpu2cpu_send_start_time = time.time()

    def profile_gpu2cpu_mark_forward_seq_send_end(self):
        if self.enable_tidy_profiling:
            self.forward_gpu2cpu_send_end_time = time.time()

    def _get_gpu_event_ts(self, event: torch.cuda.Event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def _get_cpu_ts(self, ts: float):
        return ts * 1e+6

    def gpu_forward_seq_pipeline_stage(self, input_data=None):
        if self.pp_rank == 0:
            assert (input_data is not None)
            input_seqs = torch.chunk(input_data, self.producer_buffer_size, dim=0)
        else:
            input_seqs = None

        for i in range(self.producer_buffer_size):
            if self.pp_rank == 0:  # Only send output to next node, do not receive
                # Compute
                self.profile_gpu_mark_forward_seq_comp_start()
                self._gpu_forward_compute_prompt_seq(buf_index=i, seq=input_seqs[i])
                self.profile_gpu_mark_forward_seq_comp_end()
                # Send
                self.profile_gpu_mark_forward_seq_send_start()
                self.gpu_comm.send(self.output_seq_emb, dst=self.post_node_rank)
                self.profile_gpu_mark_forward_seq_send_end()
            elif self.pp_rank == self.pipeline_group_size - 1:  # Only receive input from last node, do not send
                # Receive
                self.profile_gpu_mark_forward_seq_recv_start()
                self.gpu_comm.recv(self.input_seq_emb, src=self.pre_node_rank)
                self.profile_gpu_mark_forward_seq_recv_end()
                # Compute
                self.profile_gpu_mark_forward_seq_comp_start()
                self._gpu_forward_compute_prompt_seq(buf_index=i, seq=None)
                self.profile_gpu_mark_forward_seq_comp_end()
            else:  # receive, compute, and send
                # Receive
                self.profile_gpu_mark_forward_seq_recv_start()
                self.gpu_comm.recv(self.input_seq_emb, src=self.pre_node_rank)
                self.profile_gpu_mark_forward_seq_recv_end()
                # Compute
                self.profile_gpu_mark_forward_seq_comp_start()
                self._gpu_forward_compute_prompt_seq(buf_index=i, seq=None)
                self.profile_gpu_mark_forward_seq_comp_end()
                # Send
                self.profile_gpu_mark_forward_seq_send_start()
                self.gpu_comm.send(self.output_seq_emb, dst=self.post_node_rank)
                self.profile_gpu_mark_forward_seq_send_end()

            self.profile_gpu2cpu_mark_forward_seq_send_start()
            self._gpu_send_key_value(buf_index=i)
            self.profile_gpu2cpu_mark_forward_seq_send_end()
            self.current_dispatch_index = (self.current_dispatch_index + 1) % self.cpu_pool_size
            if self.enable_tidy_profiling:
                self._profile_seq_pipeline_stage(buf_index=i)

    def _profile_seq_pipeline_stage(self, buf_index):
        torch.cuda.synchronize()
        if self.pp_rank != 0:
            recv_slot = self.forward_seq_recv_start_event.elapsed_time(self.forward_seq_recv_end_event) * 1e+3
            recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. GPU-recv",
                        "ts": self._get_gpu_event_ts(self.forward_seq_recv_start_event), "dur": recv_slot,
                        "args": {"buf-index": buf_index}, "cname": "startup"}  # cname is for color, a little silly.
            # print(recv_log)
            self.profiling_log.append(recv_log)

        comp_slot = self.forward_seq_comp_start_event.elapsed_time(self.forward_seq_comp_end_event) * 1e+3
        comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. GPU-compute",
                    "ts": self._get_gpu_event_ts(self.forward_seq_comp_start_event), "dur": comp_slot,
                    "args": {"buf-index": buf_index}, "cname": "good"}
        # print(comp_log)
        self.profiling_log.append(comp_log)

        if self.pp_rank != self.pipeline_group_size - 1:
            send_slot = self.forward_seq_send_start_event.elapsed_time(self.forward_seq_send_end_event) * 1e+3
            send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. GPU-send",
                        "ts": self._get_gpu_event_ts(self.forward_seq_send_start_event), "dur": send_slot,
                        "args": {"buf-index": buf_index}, "cname": "thread_state_iowait"}
            # print(send_log)
            self.profiling_log.append(send_log)

        dispatch_slot = (self.forward_gpu2cpu_send_end_time - self.forward_gpu2cpu_send_start_time) * 1e+6
        dispatch_log = {"name": "dispatch", "ph": "X", "pid": self.global_rank, "tid": "4. GPU2CPU-dispatch",
                    "ts": self._get_cpu_ts(self.forward_gpu2cpu_send_start_time), "dur": dispatch_slot,
                    "args": {"buf-index": buf_index}, "cname": "thread_state_iowait"}
        self.profiling_log.append(dispatch_log)

    def profile_gpu2cpu_mark_forward_token_recv_start(self):
        if self.enable_tidy_profiling:
            self.forward_token_recv_start_time = time.time()

    def profile_gpu2cpu_mark_forward_token_recv_end(self):
        if self.enable_tidy_profiling:
            self.forward_token_recv_end_time = time.time()

    def profile_cpu_mark_forward_token_comp_start(self):
        if self.enable_tidy_profiling:
            self.forward_token_comp_start_time = time.time()

    def profile_cpu_mark_forward_token_comp_end(self):
        if self.enable_tidy_profiling:
            self.forward_token_comp_end_time = time.time()

    def cpu_forward_new_token_pipeline_step(self):
        for buf_index in range(self.consumer_buffer_size):
            self.profile_gpu2cpu_mark_forward_token_recv_start()
            self._cpu_recv_key_value(buf_index)
            self.profile_gpu2cpu_mark_forward_token_recv_end()
            if self.enable_tidy_profiling:
                self._profile_gpu2cpu_token_pipeline_recv_slot(buf_index)
            print("Rank-{} cpu_forward_new_token_pipeline_step, generate token 0 <buf-index:{}>."
                  .format(self.global_rank, buf_index))
            new_token = self._cpu_generate_new_token(self.consumer_prompt_output[buf_index])
            for step in range(self.generate_seq_length):
                print("Rank-{} cpu_forward_new_token_pipeline_step, generate token {} <buf-index:{}>."
                      .format(self.global_rank, step+1, buf_index))
                self.profile_cpu_mark_forward_token_comp_start()
                new_token = self._cpu_forward_compute_generate_token(buf_index, new_token)
                self.profile_cpu_mark_forward_token_comp_end()
                if self.enable_tidy_profiling:
                    self._profile_cpu_token_pipeline_step_comp_slot(step, buf_index)

    def _profile_gpu2cpu_token_pipeline_recv_slot(self, buf_index: int):
        recv_slot = (self.forward_token_recv_end_time - self.forward_token_recv_start_time) * 1e+6
        recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. GPU2CPU-collect",
                    "ts": self._get_cpu_ts(self.forward_token_recv_start_time), "dur": recv_slot,
                    "args": {"buf-index": buf_index}, "cname": "startup"}
        self.profiling_log.append(recv_log)

    def _profile_cpu_token_pipeline_step_comp_slot(self, step: int, buf_index: int):
        comp_slot = (self.forward_token_comp_end_time - self.forward_token_comp_start_time) * 1e+6
        comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. CPU-compute",
                    "ts": self._get_cpu_ts(self.forward_token_comp_start_time), "dur": comp_slot,
                    "args": {"token-step": step, "buf-index": buf_index}, "cname": "good"}
        self.profiling_log.append(comp_log)

    def export_profiling_result(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.profiling_log, outfile)

    # Sequential debug mode.
    def inference_batch(self, input_=None, output_=None, **kargs):
        self.cpu_comm.barrier()
        start_time = time.time()
        if self.enable_tidy_profiling:
            if self.node_type == 'GPU':
                torch.cuda.synchronize()
                self.init_time_stamp = time.time() * 1e+6
                self.init_event.record()
            else:
                self.init_time_stamp = time.time() * 1e+6

        if self.node_type == 'GPU':
            self.gpu_forward_seq_pipeline_stage(input_data=input_)
        elif self.node_type == 'CPU':
            self.cpu_forward_new_token_pipeline_step()

        self.cpu_comm.barrier()
        prompt_time = time.time()
        print("Rank {} node INFERENCE prompt takes {:3.2f}s".format(self.global_rank, prompt_time - start_time))

        self.cpu_comm.barrier()
        # TODO fix this later.
        # if self.pp_rank == 0 and output_ is not None:
        #    assert isinstance(output_, list)
        #    item = {}
        #    if self.generate_seq_length > 0:
        #        item = {
        #            'token_ids': torch.cat([z.cpu() for z in self.recv_new_token], 1),
        #        }
        #    output_.append(item)
        end_time = time.time()
        iter_time = end_time - start_time
        print("Rank {} node INFERENCE new token takes {:3.2f}s".format(self.global_rank, end_time - prompt_time))
        print("Rank {} node whole INFERENCE iteration takes {:3.2f}s".format(self.global_rank, iter_time))
        print("-------------------------------------------")

        return iter_time
