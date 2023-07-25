import time
import json
import torch.nn.functional
from torch import optim
from comm.comm_utils import *
from modules.dist_gpt_pp_module import *
from data_parallel.dist_dp_utils import get_dp_module
from optimizer.optimizer import get_fp16_optimizer
from offload.offload_utils import *


class GpipeAsyncOffload:
    r"""
    Async implementation of Gpipe.
    The current implementation leave the computation on the PyTorch default stream and the communication on a different
    stream, there is:
        a group of events to check if recv (from rank i-1) finishes in the forward propagation;
        a group of events to check if recv (from rank i+1) finishes in the backward propagation;
        a group of events to check if computation finishes in the forward propagation;
        a group of events to check if computation finishes in the backward propagation.
    """

    def __init__(self, args, vocab_size, num_classes, device, use_dp=False, pp_buffer_size=4):
        print("=======Initialize GpipeAO.")
        if args.fp16:
            self.use_fp16 = True
            print("=======Gpipe use FP16")
        else:
            self.use_fp16 = False
            print("=======Gpipe use FP32")
        self.use_dp = use_dp
        self.pp_buffer_size = pp_buffer_size
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.global_rank = args.rank
        self.pipeline_group_size = args.pipeline_group_size
        self.pp_rank = get_pipeline_parallel_rank()  # Rank is the pipeline rank by default.
        self.pre_node_rank = self.pp_rank - 1
        self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1
        self.comm = get_pipeline_parallel_comm()

        assert (args.batch_size % args.micro_batch_size == 0)
        self.micro_batch_num = args.batch_size // args.micro_batch_size
        self.micro_batch_size = args.micro_batch_size
        self.seq_length = args.seq_length
        self.embedding_dim = args.embedding_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.device = device
        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_send_stream = torch.cuda.Stream(device=device, priority=-1)
        self.cpu_to_gpu_stream = torch.cuda.Stream(device=device, priority=-1)
        self.gpu_to_cpu_stream = torch.cuda.Stream(device=device, priority=-1)

        self.forward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_send_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_offload_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                             for _ in range(self.micro_batch_num)]

        self.backward_load_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_send_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]

        if self.enable_tidy_profiling:
            self.profiling_log = []
            self.forward_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_offload_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                 for _ in range(self.micro_batch_num)]
            self.backward_load_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]

            self.init_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.init_time_stamp = None
            self.optimizer_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.optimizer_end_event = torch.cuda.Event(enable_timing=True, blocking=False)

        self._compute_micro_batch_size()

        assert self.micro_batch_num % self.pp_buffer_size == 0
        if self.pp_rank == 0:
            self.input_micro_batches = None
        else:
            self.input_micro_batch_offload = [
                pin_memory(np.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                     dtype=np.float16 if self.use_fp16 else np.float))
                for _ in range(self.micro_batch_num)]
            self.input_micro_batches = [
                torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                            requires_grad=True, device=self.device, dtype=self.dtype)
                for _ in range(self.pp_buffer_size)]
            self.input_micro_batches_cupy = [
                cupy.asarray(self.input_micro_batches[i].data) for i in range(self.pp_buffer_size)]
        if self.pp_rank == self.pipeline_group_size - 1:
            self.output_micro_batches_grad = None
        else:
            self.output_micro_batches_offload = [
                pin_memory(np.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                     dtype=np.float16 if self.use_fp16 else np.float))
                for _ in range(self.micro_batch_num)]
            self.output_micro_batches = [
                torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                            requires_grad=True, device=self.device, dtype=self.dtype)
                for _ in range(self.pp_buffer_size)]
            self.output_micro_batches_cupy = [
                cupy.asarray(self.output_micro_batches[i].data) for i in range(self.pp_buffer_size)]
            self.output_micro_batches_grad = [
                torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                            requires_grad=False, device=self.device, dtype=self.dtype)
                for _ in range(self.pp_buffer_size)]
            self.output_micro_batches_grad_cupy = [
                cupy.asarray(self.output_micro_batches_grad[i].data) for i in range(self.pp_buffer_size)]

        if self.pp_rank == 0:
            self.model = GPTStageFirst(args, vocab_size, num_classes, device)
        elif self.pp_rank == self.pipeline_group_size - 1:
            self.model = GPTStageLast(args, vocab_size, num_classes, device)
        else:
            self.model = GPTStageMiddle(args, vocab_size, num_classes, device)

        if self.use_fp16:
            self.model.half()
            tmp_optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
            self.optimizer = get_fp16_optimizer(args, tmp_optimizer, device)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

        # Notice that if we use fp16, gradients are aggregated in fp16, this may not be the default in Megatron.
        if use_dp:
            self.dp_optim = get_dp_module(args, device, self.model, self.optimizer)

    def _compute_micro_batch_size(self):
        micro_batch_float_num = self.micro_batch_size * self.seq_length * self.embedding_dim
        if self.use_fp16:
            print("=======Current micro-batch send/recv size: {} MB (fp16)"
                  .format(micro_batch_float_num * 2 // 1024 // 1024))
        else:
            print("=======Current micro-batch send/recv size: {} MB (fp32)"
                  .format(micro_batch_float_num*4//1024//1024))
        print("=======Number of micro-batches: {}.".format(self.micro_batch_num))

    def zero_input_grad(self):
        if self.input_micro_batches:
            for input_micro_batch in self.input_micro_batches:
                if input_micro_batch.grad is not None:
                    input_micro_batch.grad.zero_()

    def profile_mark_forward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.forward_comp_start_events[i])

    def profile_mark_forward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(self.forward_recv_start_events[i])

    def profile_mark_forward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_send_start_events[i])

    def profile_mark_forward_offload_start(self, i):
        if self.enable_tidy_profiling:
            self.gpu_to_cpu_stream.record_event(self.forward_offload_start_events[i])

    def profile_mark_backward_load_start(self, i):
        if self.enable_tidy_profiling:
            self.cpu_to_gpu_stream.record_event(self.backward_load_start_events[i])

    def profile_mark_backward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.backward_comp_start_events[i])

    def profile_mark_backward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(self.backward_recv_start_events[i])

    def profile_mark_backward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.backward_send_start_events[i])

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def forward_stage(self, input_data=None):
        # print("Forward stage start! rank-", self.rank)
        if self.pp_rank == 0:
            assert(input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        elif self.pp_rank == self.pipeline_group_size - 1:
            output_micro_batches = []  # This is small in classification tasks, no need to offload.

        for i in range(self.micro_batch_num):
            if self.pp_rank == 0:  # Only send output to next node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    if i >= self.pp_buffer_size:
                        self.torch_comp_stream.wait_event(self.forward_offload_ready_events[i-self.pp_buffer_size])
                    self.profile_mark_forward_comp_start(i)
                    self.output_micro_batches[i % self.pp_buffer_size].data = self.model(self.input_micro_batches[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    self.comm.send(self.output_micro_batches[i % self.pp_buffer_size].data, dst=self.post_node_rank,
                                   stream=cupy_send_stream)
                    self.torch_send_stream.record_event(self.forward_send_ready_events[i])
                with torch.cuda.stream(self.gpu_to_cpu_stream):
                    self.profile_mark_forward_offload_start(i)
                    cupy_gpu_to_cpu_stream = cupy.cuda.ExternalStream(self.gpu_to_cpu_stream.cuda_stream)
                    self.gpu_to_cpu_stream.wait_event(self.forward_send_ready_events[i])
                    with cupy_gpu_to_cpu_stream:
                        self.output_micro_batches_cupy[i % self.pp_buffer_size].get(
                            out=self.output_micro_batches_offload[i])
                    self.gpu_to_cpu_stream.record_event(self.forward_offload_ready_events[i])
            elif self.pp_rank == self.pipeline_group_size - 1:  # Only receive input from last node, do not send
                with torch.cuda.stream(self.torch_recv_stream):
                    if i >= self.pp_buffer_size:
                        self.torch_recv_stream.wait_event(self.forward_offload_ready_events[i-self.pp_buffer_size])
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_recv_start(i)
                    self.comm.recv(self.input_micro_batches[i % self.pp_buffer_size], src=self.pre_node_rank,
                                   stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    self.profile_mark_forward_comp_start(i)
                    current_micro_output = self.model(self.input_micro_batches[i % self.pp_buffer_size])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                    output_micro_batches.append(current_micro_output)
                with torch.cuda.stream(self.gpu_to_cpu_stream):
                    self.profile_mark_forward_offload_start(i)
                    cupy_gpu_to_cpu_stream = cupy.cuda.ExternalStream(self.gpu_to_cpu_stream.cuda_stream)
                    self.gpu_to_cpu_stream.wait_event(self.forward_comp_ready_events[i])
                    with cupy_gpu_to_cpu_stream:
                        self.input_micro_batches_cupy[i % self.pp_buffer_size].get(
                            out=self.input_micro_batch_offload[i])
                    self.gpu_to_cpu_stream.record_event(self.forward_offload_ready_events[i])
            else:  # receive, compute, and send
                with torch.cuda.stream(self.torch_recv_stream):
                    if i >= self.pp_buffer_size:
                        self.torch_recv_stream.wait_event(self.forward_offload_ready_events[i-self.pp_buffer_size])
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_recv_start(i)
                    self.comm.recv(self.input_micro_batches[i % self.pp_buffer_size], src=self.pre_node_rank,
                                   stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    self.profile_mark_forward_comp_start(i)
                    self.output_micro_batches[i%self.pp_buffer_size].data = \
                        self.model(self.input_micro_batches[i % self.pp_buffer_size])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    self.comm.send(self.output_micro_batches[i%self.pp_buffer_size].data, dst=self.post_node_rank,
                                   stream=cupy_send_stream)
                    self.torch_send_stream.record_event(self.forward_send_ready_events[i])
                with torch.cuda.stream(self.gpu_to_cpu_stream):
                    self.profile_mark_forward_offload_start(i)
                    cupy_gpu_to_cpu_stream = cupy.cuda.ExternalStream(self.gpu_to_cpu_stream.cuda_stream)
                    self.gpu_to_cpu_stream.wait_event(self.forward_send_ready_events[i])
                    with cupy_gpu_to_cpu_stream:
                        self.output_micro_batches_cupy[i%self.pp_buffer_size].get(
                            out=self.input_micro_batch_offload[i])
                        self.input_micro_batches_cupy[i % self.pp_buffer_size].get(
                            out=self.input_micro_batch_offload[i])
                    self.gpu_to_cpu_stream.record_event(self.forward_offload_ready_events[i])
        if self.enable_tidy_profiling:
            self.profiling_forward_stage()
        return output_micro_batches if self.pp_rank == self.pipeline_group_size - 1 else None

    def profiling_forward_stage(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            if self.pp_rank != 0:
                recv_slot = self.forward_recv_start_events[i].elapsed_time(self.forward_recv_ready_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. forward-recv",
                            "ts": self.get_ts(self.forward_recv_start_events[i]), "dur": recv_slot,
                            "args": {"micro-batch": i}, "cname": "startup"}  # cname is for color, a little silly.
                # print(recv_log)
                self.profiling_log.append(recv_log)
            comp_slot = self.forward_comp_start_events[i].elapsed_time(self.forward_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. forward-compute",
                        "ts": self.get_ts(self.forward_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)
            if self.pp_rank != self.pipeline_group_size - 1:
                send_slot = self.forward_send_start_events[i].elapsed_time(self.forward_send_ready_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. forward-send",
                            "ts": self.get_ts(self.forward_send_start_events[i]), "dur": send_slot,
                            "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(send_log)
                self.profiling_log.append(send_log)
            offload_slot = self.forward_offload_start_events[i].elapsed_time(self.forward_offload_ready_events[i])*1e+3
            offload_log = {"name": "offload", "ph": "X", "pid": self.global_rank, "tid": "4. forward-offload",
                           "ts": self.get_ts(self.forward_offload_start_events[i]), "dur": offload_slot,
                           "args": {"micro-batch": i}, "cname": "thread_state_runnable"}
            self.profiling_log.append(offload_log)

    def backward_stage(self, cached_output_micro_batches: List[torch.Tensor] = None, target=None,
                       loss_func=torch.nn.functional.cross_entropy):
        # print("Backward stage start! rank-", self.rank)
        if self.pp_rank == self.pipeline_group_size - 1:
            assert(target is not None)
            assert(cached_output_micro_batches is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert(target is None)
            assert (cached_output_micro_batches is None)
        for i in range(self.micro_batch_num):
            if self.pp_rank == self.pipeline_group_size - 1:  # only send grad back to last node, do not receive
                with torch.cuda.stream(self.cpu_to_gpu_stream):
                    self.profile_mark_backward_load_start(i)
                    if i >= self.pp_buffer_size:
                        self.cpu_to_gpu_stream.wait_event(self.backward_send_ready_events[i-self.pp_buffer_size])
                    cupy_cpu_to_gpu_stream = cupy.cuda.ExternalStream(self.cpu_to_gpu_stream.cuda_stream)
                    with cupy_cpu_to_gpu_stream:
                        self.input_micro_batches_cupy[i % self.pp_buffer_size].set(self.input_micro_batch_offload[i])
                    self.cpu_to_gpu_stream.record_event(self.backward_load_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_load_ready_events[i])
                    self.profile_mark_backward_comp_start(i)
                    loss = loss_func(input=cached_output_micro_batches[i], target=target_as_micro_batches[i])
                    loss.backward()
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.profile_mark_backward_send_start(i)
                    self.comm.send(self.input_micro_batches[i % self.pp_buffer_size].grad, dst=self.pre_node_rank,
                                   stream=cupy_send_stream)
                    self.torch_send_stream.record_event(self.backward_send_ready_events[i])
            elif self.pp_rank == 0:  # only receive grad from previous node, do not send
                with torch.cuda.stream(self.cpu_to_gpu_stream):
                    self.profile_mark_backward_load_start(i)
                    if i >= self.pp_buffer_size:
                        self.cpu_to_gpu_stream.wait_event(self.backward_comp_ready_events[i - self.pp_buffer_size])
                    cupy_cpu_to_gpu_stream = cupy.cuda.ExternalStream(self.cpu_to_gpu_stream.cuda_stream)
                    with cupy_cpu_to_gpu_stream:
                        self.output_micro_batches_cupy[i % self.pp_buffer_size].set(
                            self.output_micro_batches_offload[i])
                    self.cpu_to_gpu_stream.record_event(self.backward_load_ready_events[i])
                with torch.cuda.stream(self.torch_recv_stream):
                    if i >= self.pp_buffer_size:
                        self.torch_recv_stream.wait_event(self.backward_comp_ready_events[i-self.pp_buffer_size])
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_backward_recv_start(i)
                    self.comm.recv(self.output_micro_batches_grad[i % self.pp_buffer_size], src=self.post_node_rank,
                                   stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_load_ready_events[i])
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    self.profile_mark_backward_comp_start(i)
                    self.output_micro_batches[i % self.pp_buffer_size].backward(
                        gradient=self.output_micro_batches_grad[i % self.pp_buffer_size])
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
            else:  # receive, compute and send
                with torch.cuda.stream(self.cpu_to_gpu_stream):
                    self.profile_mark_backward_load_start(i)
                    if i >= self.pp_buffer_size:
                        self.cpu_to_gpu_stream.wait_event(self.backward_send_ready_events[i - self.pp_buffer_size])
                    cupy_cpu_to_gpu_stream = cupy.cuda.ExternalStream(self.cpu_to_gpu_stream.cuda_stream)
                    with cupy_cpu_to_gpu_stream:
                        self.output_micro_batches_cupy[i % self.pp_buffer_size].set(
                            self.output_micro_batches_offload[i])
                        self.input_micro_batches_cupy[i % self.pp_buffer_size].set(
                            self.input_micro_batch_offload[i])
                    self.cpu_to_gpu_stream.record_event(self.backward_load_ready_events[i])
                with torch.cuda.stream(self.torch_recv_stream):
                    if i >= self.pp_buffer_size:
                        self.torch_recv_stream.wait_event(self.backward_send_ready_events[i-self.pp_buffer_size])
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_backward_recv_start(i)
                    self.comm.recv(self.output_micro_batches_grad[i % self.pp_buffer_size], src=self.post_node_rank,
                                   stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_load_ready_events[i])
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    self.profile_mark_backward_comp_start(i)
                    self.output_micro_batches[i % self.pp_buffer_size].backward(
                        gradient=self.output_micro_batches_grad[i % self.pp_buffer_size])
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.profile_mark_backward_send_start(i)
                    self.comm.send(self.input_micro_batches[i % self.pp_buffer_size].grad, dst=self.pre_node_rank,
                                   stream=cupy_send_stream)
                    self.torch_send_stream.record_event(self.backward_send_ready_events[i])
        if self.enable_tidy_profiling:
            self.profiling_backward_stage()

    def profiling_backward_stage(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            load_slot = self.backward_load_start_events[i].elapsed_time(self.backward_load_ready_events[i])
            load_log = {"name": "offload", "ph": "X", "pid": self.global_rank, "tid": "5. backward-load",
                        "ts": self.get_ts(self.backward_load_start_events[i]), "dur": load_slot,
                        "args": {"micro-batch": i}, "cname": "thread_state_runnable"}
            self.profiling_log.append(load_log)

            if self.pp_rank != self.pipeline_group_size - 1:
                recv_slot = self.backward_recv_start_events[i].elapsed_time(self.backward_recv_ready_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "6. backward-recv",
                            "ts": self.get_ts(self.backward_recv_start_events[i]), "dur": recv_slot,
                            "args": {"micro-batch": i}, "cname": "startup"}
                # print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.backward_comp_start_events[i].elapsed_time(self.backward_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "7. backward-compute",
                        "ts": self.get_ts(self.backward_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)
            if self.pp_rank != 0:
                send_slot = self.backward_send_start_events[i].elapsed_time(self.backward_send_ready_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "8. backward-send",
                            "ts": self.get_ts(self.backward_send_start_events[i]), "dur": send_slot,
                            "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(send_log)
                self.profiling_log.append(send_log)

    def optimizer_step(self):
        if self.use_dp:
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.record_event(self.dp_optim.backward_ready_event)
            self.dp_optim.optimizer_step()
        else:
            with torch.cuda.stream(self.torch_comp_stream):
                if self.enable_tidy_profiling:
                    self.optimizer_start_event.record()
                self.optimizer.step()
                if self.enable_tidy_profiling:
                    self.optimizer_end_event.record()
        if self.enable_tidy_profiling:
            self.profiling_optimizer_step()

    def profiling_optimizer_step(self):
        torch.cuda.synchronize()
        if not self.use_dp:
            optimizer_slot = self.optimizer_start_event.elapsed_time(self.optimizer_end_event) * 1e+3
            optimizer_log = {"name": "opt", "ph": "X", "pid": self.global_rank, "tid": "9. optimizer-step",
                             "ts": self.get_ts(self.optimizer_start_event), "dur": optimizer_slot, "cname": "bad"}
            # print(optimizer_log)
            self.profiling_log.append(optimizer_log)
        else:
            self.profiling_log.extend(self.dp_optim.profiling_data_parallel(self.init_time_stamp, self.init_event))

    def export_profiling_result(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.profiling_log, outfile)

    def sgd_iter(self, input_=None, target=None):
        self.comm.barrier()
        start_time = time.time()
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.init_time_stamp = time.time() * 1e+6
            self.init_event.record()
        self.zero_input_grad()
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.forward_stage(input_)
        forward_time = time.time()
        forward_slot = forward_time-start_time
        print("Rank {} node forward pass takes {:3.2f}s".format(self.global_rank, forward_slot))
        self.comm.barrier()  # This is an educated guess that such barrier would make it fair TC (probably required)
        self.backward_stage(outputs, target)
        backward_time = time.time()
        print("Rank {} node backward pass takes {:3.2f}s".format(self.global_rank, backward_time-forward_time))
        optimizer_time = time.time()
        self.optimizer_step()
        torch.cuda.synchronize()
        self.comm.barrier()
        end_time = time.time()
        print("Rank {} node optimizer step takes {:3.2f}s".format(self.global_rank, end_time - optimizer_time))
        iter_time = end_time - start_time
        print("Rank {} node whole iteration takes {:3.2f}s".format(self.global_rank, iter_time))
        print("-------------------------------------------")
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
        return iter_time
