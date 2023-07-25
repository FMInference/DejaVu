import time
import json
import torch.nn.functional
from torch import optim
from comm.comm_utils import *
from modules.dist_gpt_pp_module import *
from data_parallel.dist_dp_utils import get_dp_module
from optimizer.optimizer import get_fp16_optimizer


class GpipeAsync:
    r"""
    Async implementation of Gpipe.
    The current implementation leave the computation on the PyTorch default stream and the communication on a different
    stream, there is:
        a group of events to check if recv (from rank i-1) finishes in the forward propagation;
        a group of events to check if recv (from rank i+1) finishes in the backward propagation;
        a group of events to check if computation finishes in the forward propagation;
        a group of events to check if computation finishes in the backward propagation.
    """

    def __init__(self, args, vocab_size, num_classes, device, use_dp=False, rank=None):
        print("=======Initialize Gpipe.")
        if args.fp16:
            self.use_fp16 = True
            print("=======Gpipe use FP16")
        else:
            self.use_fp16 = False
            print("=======Gpipe use FP32")
        self.use_dp = use_dp
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        if rank is None:
            self.global_rank = args.rank
        else:
            self.global_rank = rank
        self.pipeline_group_size = args.pipeline_group_size
        self.pp_rank = get_pipeline_parallel_rank()  # Rank is the pipeline rank by default.
        self.pre_node_rank = self.pp_rank - 1
        self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1
        self.comm = get_pipeline_parallel_comm()
        self.gradient_accumulate_step = args.gradient_accumulate_step
        print("=======Gradient accumulate step: ", self.gradient_accumulate_step)

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

        self._compute_micro_batch_size()
        if self.pp_rank == 0:
            self.input_micro_batches = None
        else:
            self.input_micro_batches = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                    requires_grad=True, device=self.device, dtype=self.dtype)
                                        for _ in range(self.micro_batch_num)]
        if self.pp_rank == self.pipeline_group_size - 1:
            self.output_micro_batches_grad = None
        else:
            self.output_micro_batches_grad = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                          requires_grad=False, device=self.device, dtype=self.dtype)
                                              for _ in range(self.micro_batch_num)]

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
            self.dp_optim = get_dp_module(args, device, self.model, self.optimizer, rank=rank)

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

    def profile_mark_forward_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_send_end_events[i])

    def profile_mark_backward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.backward_comp_start_events[i])

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

    def forward_stage(self, input_data=None, target_data=None):
        # print("Forward stage start! rank-", self.rank)
        if self.pp_rank == 0:
            assert(input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        elif self.pp_rank == self.pipeline_group_size - 1:
            if self.model.task == 'Seq2SeqClassification':
                assert target_data is not None
                target_data_micro_batches = torch.chunk(target_data, self.micro_batch_num, dim=0)
        output_micro_batches = []

        for i in range(self.micro_batch_num):
            if self.pp_rank == 0:  # Only send output to next node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_forward_comp_start(i)
                    current_micro_output = self.model(self.input_micro_batches[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
                    self.profile_mark_forward_send_end(i)
            elif self.pp_rank == self.pipeline_group_size - 1:  # Only receive input from last node, do not send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_recv_start(i)
                    self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    self.profile_mark_forward_comp_start(i)
                    if self.model.task == 'Seq2SeqClassification':
                        current_micro_output = self.model(self.input_micro_batches[i], target_data_micro_batches[i])
                    else:
                        current_micro_output = self.model(self.input_micro_batches[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:  # receive, compute, and send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_recv_start(i)
                    self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    self.profile_mark_forward_comp_start(i)
                    current_micro_output = self.model(self.input_micro_batches[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
                    self.profile_mark_forward_send_end(i)
            output_micro_batches.append(current_micro_output)
        if self.enable_tidy_profiling:
            self.profiling_forward_stage()
        return output_micro_batches

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
                send_slot = self.forward_send_start_events[i].elapsed_time(self.forward_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. forward-send",
                            "ts": self.get_ts(self.forward_send_start_events[i]), "dur": send_slot,
                            "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(send_log)
                self.profiling_log.append(send_log)

    def _loss_compute(self, input_, target):
        print(input_.shape, target.shape)
        if self.model.task == 'SeqClassification':
            return torch.nn.functional.cross_entropy(input=input_, target=target)
        elif self.model.task == 'Seq2SeqClassification':
            # shift_logits = input_[..., :-1, :].contiguous()
            # shift_labels = target[..., 1:].contiguous()
            # return torch.nn.functional.nll_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return self.model(input_)

    def backward_stage(self, cached_output_micro_batches: List[torch.Tensor], target=None):
        # print("Backward stage start! rank-", self.rank)
        if self.pp_rank == self.pipeline_group_size - 1:
            assert(target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert(target is None)
        for i in range(self.micro_batch_num):
            if self.pp_rank == self.pipeline_group_size - 1:  # only send grad back to last node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_backward_comp_start(i)
                    if self.model.task == 'Seq2SeqClassification':
                        cached_output_micro_batches[i].backward()
                    else:
                        loss = torch.nn.functional.cross_entropy(input=cached_output_micro_batches[i],
                                                                 target=target_as_micro_batches[i])
                        loss.backward()
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.profile_mark_backward_send_start(i)
                    self.comm.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank, stream=cupy_send_stream)
                    self.profile_mark_backward_send_end(i)
                # self.input_micro_batches[i].grad = None
                # torch.cuda.synchronize()  # Notice this for memory optimization
            elif self.pp_rank == 0:  # only receive grad from previous node, do not send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_backward_recv_start(i)
                    self.comm.recv(self.output_micro_batches_grad[i], src=self.post_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    self.profile_mark_backward_comp_start(i)
                    cached_output_micro_batches[i].backward(gradient=self.output_micro_batches_grad[i])
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
            else:  # receive, compute and send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_backward_recv_start(i)
                    self.comm.recv(self.output_micro_batches_grad[i], src=self.post_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    self.profile_mark_backward_comp_start(i)
                    cached_output_micro_batches[i].backward(gradient=self.output_micro_batches_grad[i])
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.profile_mark_backward_send_start(i)
                    self.comm.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank, stream=cupy_send_stream)
                    self.profile_mark_backward_send_end(i)
        if self.enable_tidy_profiling:
            self.profiling_backward_stage()

    def profiling_backward_stage(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            if self.pp_rank != self.pipeline_group_size - 1:
                recv_slot = self.backward_recv_start_events[i].elapsed_time(self.backward_recv_ready_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "4. backward-recv",
                            "ts": self.get_ts(self.backward_recv_start_events[i]), "dur": recv_slot,
                            "args": {"micro-batch": i}, "cname": "startup"}
                # print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.backward_comp_start_events[i].elapsed_time(self.backward_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "5. backward-compute",
                        "ts": self.get_ts(self.backward_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)
            if self.pp_rank != 0:
                send_slot = self.backward_send_start_events[i].elapsed_time(self.backward_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "6. backward-send",
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
        self.optimizer.zero_grad(set_to_none=False)

        for step in range(self.gradient_accumulate_step):
            outputs = self.forward_stage(input_, target)
            forward_time = time.time()
            if step == 0:
                forward_slot = forward_time-start_time
            else:
                forward_slot = forward_time-backward_time
            print("Rank {} node forward pass {}/{} takes {:3.2f}s"
                  .format(self.global_rank, step, self.gradient_accumulate_step, forward_slot))
            self.comm.barrier()  # This is an educated guess that such barrier would make it fair TC (probably required)
            self.backward_stage(outputs, target)
            backward_time = time.time()
            print("Rank {} node backward pass {}/{} takes {:3.2f}s"
                  .format(self.global_rank, step, self.gradient_accumulate_step, backward_time-forward_time))
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
