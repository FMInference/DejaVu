import argparse

# This is the 1f1b scheduling from deepspeed.


def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0


class TrainSimulate:
    """
    A schedule for training a batch using parallelism.
    """

    def __init__(self, micro_batches, stages, stage_id):
        self.micro_batches = micro_batches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline_parallel to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append("Recv activation :" + str(curr_buffer))
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append("Recv gradient :" + str(prev_buffer))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append("Send activation :" + str(prev_buffer))
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append("Recv gradient :" + str(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append("Load batch :" + str(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append("Forward computation :" + str(curr_buffer))
                else:
                    cmds.append("Backward computation :" +str(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append("optimizer step")

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def num_pipe_buffers(self):
        buffers = min(self.stages - self.stage_id + 1, self.micro_batches)
        return max(2, buffers)

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id

    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.micro_batches

    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.stages

    def _buffer_idx(self, micro_batch_id):
        assert self._valid_micro_batch(micro_batch_id)
        return micro_batch_id % self.num_pipe_buffers()


def main():
    parser = argparse.ArgumentParser(description='1F1B - simulation')
    parser.add_argument('--micro_batches', type=int, default=16, metavar='S',
                        help='num of micro batches (default: 16)')
    parser.add_argument('--stages', type=int, default=4, metavar='S',
                        help='lengths of pipeline_parallel stages (default: 4)')
    args = parser.parse_args()
    for i in range(args.stages):
        worker = TrainSimulate(args.micro_batches, args.stages, i)
        print("===============Worker {} ===============".format(i))
        for cmd in worker.steps():
            print(cmd)


if __name__ == '__main__':
    main()
