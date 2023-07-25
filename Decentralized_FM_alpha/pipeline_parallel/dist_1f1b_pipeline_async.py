import time
import json
import torch.nn.functional
from torch import optim
from comm.comm_utils import *
from modules.dist_gpt_pp_module import *
from data_parallel.dist_dp_utils import get_dp_module


class Pipe1F1BAsync:
    r"""
    Async implementation of Gpipe.
    The current implementation leave the computation on the PyTorch default stream and the communication on a different
    stream, there is:
        a group of events to check if recv (from rank i-1) finishes in the forward propagation;
        a group of events to check if recv (from rank i+1) finishes in the backward propagation;
        a group of events to check if computation finishes in the forward propagation;
        a group of events to check if computation finishes in the backward propagation.
    """

    def __init__(self, args, vocab_size, num_classes, device, use_dp=False):
        self.global_rank = args.rank
        self.pipeline_group_size = args.pipeline_group_size
        self.pp_rank = get_pipeline_parallel_rank()   # Rank is the pipeline rank by default.
        self.pre_node_rank = self.pp_rank - 1
        self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1
        self.comm = get_pipeline_parallel_comm()
        if use_dp:
            self.dp_comm = get_data_parallel_comm()
            self.dp_rank = get_data_parallel_rank()

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

        self.forward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]

        self.backward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]

        if self.enable_tidy_profiling:
            self.profiling_log = []
            self.forward_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                            for _ in range(self.micro_batch_num)]

            self.backward_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                             for _ in range(self.micro_batch_num)]
            self.init_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.init_time_stamp = None
            self.optimizer_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.optimizer_end_event = torch.cuda.Event(enable_timing=True, blocking=False)

        if self.pp_rank == 0:
            self.input_micro_batches = None
        else:
            self.input_micro_batches = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                    requires_grad=True, device=self.device)
                                        for _ in range(self.micro_batch_num)]

        self.output_micro_batches = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                 requires_grad=True, device=self.device)
                                     for _ in range(self.micro_batch_num)]
        self.output_micro_batches_grad = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                      requires_grad=False, device=self.device)
                                          for _ in range(self.micro_batch_num)]

        self._compute_micro_batch_size()
        if self.pp_rank == 0:
            self.model = GPTStageFirst(args, vocab_size, num_classes, device)
        elif self.pp_rank == self.pipeline_group_size - 1:
            self.model = GPTStageLast(args, vocab_size, num_classes, device)
        else:
            self.model = GPTStageMiddle(args, vocab_size, num_classes, device)

        self.use_dp = use_dp

        if use_dp:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
            self.dp_optim = get_dp_module(args, device, self.model, self.optimizer)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

    def _compute_micro_batch_size(self):
        micro_batch_float_num = self.micro_batch_size * self.seq_length * self.embedding_dim
        print("Current micro-batch send/recv size: {} MB".format(micro_batch_float_num*4//1024//1024))

    def zero_input_grad(self):
        if self.input_micro_batches:
            for input_micro_batch in self.input_micro_batches:
                if input_micro_batch.grad is not None:
                    input_micro_batch.grad.zero_()

    def profile_mark_forward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.forward_comp_start_events[i])

    def profile_mark_backward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.backward_comp_start_events[i])

    def profile_mark_forward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(self.forward_recv_start_events[i])

    def profile_mark_forward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_send_start_events[i])

    def profile_mark_forward_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_send_end_events[i])

    def profile_mark_backward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(self.backward_recv_start_events[i])

    def profile_mark_backward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.backward_send_start_events[i])

    def profile_mark_backward_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.backward_send_end_events[i])

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def forward_micro_batch(self, forward_index):
        if self.pp_rank == 0:  # Only send output to next node, do not receive
            with torch.cuda.stream(self.torch_comp_stream):
                self.profile_mark_forward_comp_start(forward_index)
                self.output_micro_batches[forward_index] = self.model(self.input_micro_batches[forward_index])
                self.torch_comp_stream.record_event(self.forward_comp_ready_events[forward_index])
            with torch.cuda.stream(self.torch_send_stream):
                cupy_forward_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.torch_send_stream.wait_event(self.forward_comp_ready_events[forward_index])
                self.profile_mark_forward_send_start(forward_index)
                self.comm.send(self.output_micro_batches[forward_index].data, dst=self.post_node_rank,
                               stream=cupy_forward_send_stream)
                self.profile_mark_forward_send_end(forward_index)
        elif self.pp_rank == self.pipeline_group_size - 1:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_forward_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.profile_mark_forward_recv_start(forward_index)
                self.comm.recv(self.input_micro_batches[forward_index], src=self.pre_node_rank,
                               stream=cupy_forward_recv_stream)
                self.torch_recv_stream.record_event(self.forward_recv_ready_events[forward_index])
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.wait_event(self.forward_recv_ready_events[forward_index])
                self.profile_mark_forward_comp_start(forward_index)
                self.output_micro_batches[forward_index] = self.model(self.input_micro_batches[forward_index])
                self.torch_comp_stream.record_event(self.forward_comp_ready_events[forward_index])
        else:  # receive, compute, and send
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_forward_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.profile_mark_forward_recv_start(forward_index)
                self.comm.recv(self.input_micro_batches[forward_index], src=self.pre_node_rank,
                               stream=cupy_forward_recv_stream)
                self.torch_recv_stream.record_event(self.forward_recv_ready_events[forward_index])
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.wait_event(self.forward_recv_ready_events[forward_index])
                self.profile_mark_forward_comp_start(forward_index)
                self.output_micro_batches[forward_index] = self.model(self.input_micro_batches[forward_index])
                self.torch_comp_stream.record_event(self.forward_comp_ready_events[forward_index])
            with torch.cuda.stream(self.torch_send_stream):
                cupy_forward_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.torch_send_stream.wait_event(self.forward_comp_ready_events[forward_index])
                self.profile_mark_forward_send_start(forward_index)
                self.comm.send(self.output_micro_batches[forward_index].data, dst=self.post_node_rank,
                               stream=cupy_forward_send_stream)
                self.profile_mark_forward_send_end(forward_index)

    def backward_micro_batch(self, backward_index, target_as_micro_batch=None, loss_func=None):
        if self.pp_rank == self.pipeline_group_size - 1:  # only send grad back to last node, do not receive
            with torch.cuda.stream(self.torch_comp_stream):
                self.profile_mark_backward_comp_start(backward_index)
                loss = loss_func(input= self.output_micro_batches[backward_index], target=target_as_micro_batch)
                loss.backward()
                self.torch_comp_stream.record_event(self.backward_comp_ready_events[backward_index])
            with torch.cuda.stream(self.torch_send_stream):
                cupy_backward_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.torch_send_stream.wait_event(self.backward_comp_ready_events[backward_index])
                self.profile_mark_backward_send_start(backward_index)
                self.comm.send(self.input_micro_batches[backward_index].grad, dst=self.pre_node_rank,
                               stream=cupy_backward_send_stream)
                self.profile_mark_backward_send_end(backward_index)
        elif self.pp_rank == 0:  # only receive grad from previous node, do not send
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_backward_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.profile_mark_backward_recv_start(backward_index)
                self.comm.recv(self.output_micro_batches_grad[backward_index], src=self.post_node_rank,
                               stream=cupy_backward_recv_stream)
                self.torch_recv_stream.record_event(self.backward_recv_ready_events[backward_index])
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.wait_event(self.backward_recv_ready_events[backward_index])
                self.profile_mark_backward_comp_start(backward_index)
                self.output_micro_batches[backward_index].backward(
                    gradient=self.output_micro_batches_grad[backward_index])
                self.torch_comp_stream.record_event(self.backward_comp_ready_events[backward_index])
        else:  # receive, compute and send
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_backward_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.profile_mark_backward_recv_start(backward_index)
                self.comm.recv(self.output_micro_batches_grad[backward_index], src=self.post_node_rank,
                               stream=cupy_backward_recv_stream)
                self.torch_recv_stream.record_event(self.backward_recv_ready_events[backward_index])
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.wait_event(self.backward_recv_ready_events[backward_index])
                self.profile_mark_backward_comp_start(backward_index)
                self.output_micro_batches[backward_index].backward(
                    gradient=self.output_micro_batches_grad[backward_index])
                self.torch_comp_stream.record_event(self.backward_comp_ready_events[backward_index])
            with torch.cuda.stream(self.torch_send_stream):
                cupy_backward_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.torch_send_stream.wait_event(self.backward_comp_ready_events[backward_index])
                self.profile_mark_backward_send_start(backward_index)
                self.comm.send(self.input_micro_batches[backward_index].grad, dst=self.pre_node_rank,
                               stream=cupy_backward_send_stream)
                self.profile_mark_backward_send_end(backward_index)

    def forward_backward_stages(self, input_data=None, target=None, loss_func=torch.nn.functional.cross_entropy):
        # TODO this loading part should be updated later
        if self.pp_rank == 0:
            assert(input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
            target_as_micro_batches = [None for _ in range(self.micro_batch_num)]
        elif self.pp_rank == self.pipeline_group_size - 1:
            assert (target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert (input_data is None and target is None)
            target_as_micro_batches = [None for _ in range(self.micro_batch_num)]

        forward_i = 0
        backward_i = 0
        # Starting phase: to fill the pipeline_parallel.
        while forward_i < self.pipeline_group_size - 1 - self.pp_rank:
            self.forward_micro_batch(forward_index=forward_i)
            forward_i += 1

        # Running phase: 1 forward coupled with 1 backward.
        while forward_i < self.micro_batch_num:
            self.forward_micro_batch(forward_index=forward_i)
            self.backward_micro_batch(backward_index=backward_i,
                                      target_as_micro_batch=target_as_micro_batches[backward_i], loss_func=loss_func)
            forward_i += 1
            backward_i += 1

        # Ending phase: to finish the rest stages in the pipeline_parallel.
        while backward_i < self.micro_batch_num:
            self.backward_micro_batch(backward_index=backward_i,
                                      target_as_micro_batch=target_as_micro_batches[backward_i], loss_func=loss_func)
            backward_i += 1
        if self.enable_tidy_profiling:
            self.profile_forward_backward_stages()

    def profile_forward_backward_stages(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            if self.pp_rank != 0:
                forward_recv_slot = \
                    self.forward_recv_start_events[i].elapsed_time(self.forward_recv_ready_events[i]) * 1e+3
                forward_recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. forward-recv",
                                    "ts": self.get_ts(self.forward_recv_start_events[i]), "dur": forward_recv_slot,
                                    "args": {"micro-batch": i}, "cname": "startup"}
                # print(forward_recv_log)
                self.profiling_log.append(forward_recv_log)

            forward_comp_slot = \
                self.forward_comp_start_events[i].elapsed_time(self.forward_comp_ready_events[i]) * 1e+3
            forward_comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. forward-compute",
                                "ts": self.get_ts(self.forward_comp_start_events[i]), "dur": forward_comp_slot,
                                "args": {"micro-batch": i}, "cname": "good"}
            # print(forward_comp_log)
            self.profiling_log.append(forward_comp_log)

            if self.pp_rank != self.pipeline_group_size - 1:
                forward_send_slot = \
                    self.forward_send_start_events[i].elapsed_time(self.forward_send_end_events[i]) * 1e+3
                forward_send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. forward-send",
                                    "ts": self.get_ts(self.forward_send_start_events[i]), "dur": forward_send_slot,
                                    "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(forward_send_log)
                self.profiling_log.append(forward_send_log)

        for i in range(self.micro_batch_num):
            if self.pp_rank != self.pipeline_group_size - 1:
                backward_recv_slot = \
                    self.backward_recv_start_events[i].elapsed_time(self.backward_recv_ready_events[i]) * 1e+3
                backward_recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "4. backward-recv",
                                     "ts": self.get_ts(self.backward_recv_start_events[i]), "dur": backward_recv_slot,
                                     "args": {"micro-batch": i}, "cname": "startup"}
                # print(backward_recv_log)
                self.profiling_log.append(backward_recv_log)

            backward_comp_slot = \
                self.backward_comp_start_events[i].elapsed_time(self.backward_comp_ready_events[i]) * 1e+3
            backward_comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "5. backward-compute",
                                 "ts": self.get_ts(self.backward_comp_start_events[i]),
                                 "dur": backward_comp_slot, "args": {"micro-batch": i}, "cname": "good"}
            # print(backward_comp_log)
            self.profiling_log.append(backward_comp_log)

            if self.pp_rank != 0:
                backward_send_slot = \
                    self.backward_send_start_events[i].elapsed_time(self.backward_send_end_events[i]) * 1e+3
                backward_send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "6. backward-send",
                                     "ts": self.get_ts(self.backward_send_start_events[i]), "dur": backward_send_slot,
                                     "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(backward_send_log)
                self.profiling_log.append(backward_send_log)

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
            optimizer_log = {"name": "opt", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-step",
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
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self.forward_backward_stages(input_data=input_, target=target)
        self.optimizer_step()
        torch.cuda.synchronize()
        end_time = time.time()
        iter_time = end_time - start_time
        print("Rank {} node 1f1b iteration takes {:3.2f}s".format(self.global_rank, iter_time))
        return iter_time
