import sys
sys.path.append('../Megatron-LM')


# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLUE finetuning/evaluation."""
from functools import partial
import torch
import time
import torch.nn.functional as F
from deepspeed.profiling.flops_profiler import FlopsProfiler
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import get_timers
from megatron import mpu
from megatron import get_num_microbatches
from megatron.model import ModelType, Float16Module
from megatron.model import DistributedDataParallel as LocalDDP
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.classification import Classification
from megatron.initialize import initialize_megatron
from megatron.utils import average_losses_across_data_parallel_group,unwrap_model
from megatron.training import setup_model_and_optimizer, print_datetime
from megatron.schedules import get_forward_backward_func
from tasks.glue.qqp import QQPDataset
from megatron.data.data_samplers import MegatronPretrainingSampler
import warnings
warnings.filterwarnings("ignore")


def get_qqp_args(parser):
    group = parser.add_argument_group(title='tasks')
    group.add_argument('--train-data-path', type=str, required=True,
                       help='train data path.')
    group.add_argument('--valid-data-path', type=str, required=True,
                       help='train data path.')
    group.add_argument('--test-data-path', type=str, required=True,
                       help='train data path.')
    return parser


def train_dataset_provider():
    """Build train and validation dataset."""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    if mpu.get_tensor_model_parallel_rank() == 0:
        tokenizer = get_tokenizer()
        train_dataset = QQPDataset('training', [args.train_data_path],
                                   tokenizer, args.seq_length)
        train_sampler = MegatronPretrainingSampler(
            total_samples=len(train_dataset),
            consumed_samples=args.consumed_train_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                       batch_sampler=train_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = False
        do_test = False
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None
    valid_data_iterator = None
    test_data_iterator = None
    return train_data_iterator, valid_data_iterator, test_data_iterator


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    num_classes = 2
    print_rank_0('building classification model for QQP ...')
    model = Classification(num_classes=num_classes, num_tokentypes=2,
                           pre_process=pre_process, post_process=post_process)
    return model


def loss_func(labels, output_tensor):
    loss = F.cross_entropy(output_tensor, labels)
    averaged_losses = average_losses_across_data_parallel_group(
            [loss])
    return loss, {'classification loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    keys = ['text', 'types', 'label', 'uid', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)
    tokens = data_b['text'].long()
    # print_rank_0(tokens.shape)
    types = data_b['types'].long()
    lm_labels = data_b['label'].long()
    padding_mask = data_b['padding_mask'].long()
    timers('batch-generator').stop()

    # Forward pass through the model.
    output_tensor = model(tokens, padding_mask, tokentype_ids=types)

    return output_tensor, partial(loss_func, lm_labels)


def megatron_train_step(forward_step_func, data_iterator,
                        model, optimizer, lr_scheduler, profile=False):
    """Single training step."""
    args = get_args()
    timers = get_timers()
    if profile:
        prof = FlopsProfiler(model)
        prof.start_profile()

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in model:
            partition.zero_grad_buffer()
    optimizer.zero_grad()

    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func, data_iterator, model,
        optimizer, timers, forward_only=profile)
    if profile:
        prof.stop_profile()
        if torch.distributed.get_rank() == 0:
            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            params = prof.get_total_params()
            prof.print_model_profile()
            print("Flops:", flops)
            print("Macs:", macs)
            print("Params:", params)
        prof.end_profile()
    # Empty unused memory
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # All-reduce if needed.
    if args.DDP_impl == 'local':
        timers('backward-params-all-reduce').start()
        for model_module in model:
            model_module.allreduce_gradients()
        timers('backward-params-all-reduce').stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    timers('backward-embedding-all-reduce').start()
    if mpu.is_rank_in_embedding_group(ignore_virtual=True) and \
            mpu.get_pipeline_model_parallel_world_size() > 1:
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            unwrapped_model = model[0]
        elif mpu.is_pipeline_last_stage(ignore_virtual=True):
            unwrapped_model = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            unwrapped_model = model[0]
        unwrapped_model = unwrap_model(unwrapped_model, (torchDDP, LocalDDP, Float16Module))

        if unwrapped_model.share_word_embeddings:
            word_embeddings_weight = unwrapped_model.word_embeddings_weight()
            if args.DDP_impl == 'local':
                grad = word_embeddings_weight.main_grad
            else:
                grad = word_embeddings_weight.grad
            torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())

    # All-reduce position_embeddings grad across first (encoder) and split (decoder)
    # stages to ensure that position embeddings parameters stay in sync.
    # This should only run for T5 models with pipeline parallelism
    if mpu.is_rank_in_position_embedding_group() and \
            mpu.get_pipeline_model_parallel_world_size() > 1 and \
            args.pipeline_model_parallel_split_rank is not None:
        unwrapped_model = model[0]
        unwrapped_model = unwrap_model(
            unwrapped_model, (torchDDP, LocalDDP, Float16Module))
        assert args.DDP_impl == 'local', \
            'T5 model is only supported with local DDP mode'
        grad = unwrapped_model.language_model.embedding.position_embeddings.weight.main_grad
        torch.distributed.all_reduce(grad, group=mpu.get_position_embedding_group())
    timers('backward-embedding-all-reduce').stop()

    # Update parameters.
    timers('optimizer').start()
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        lr_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def train_qqp(train_dataset_provider,
              model_provider,
              model_type,
              forward_step_func,
              extra_args_provider=get_qqp_args,
              args_defaults={}):
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    _TRAIN_START_TIME = time.time()
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider, model_type)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate scheduler are built')

    # Data stuff.
    timers('train-data-iterators-setup').start()
    train_data_iterator, valid_data_iterator, test_data_iterator = train_dataset_provider()
    timers('train-data-iterators-setup').stop()
    print_datetime('after data iterators are built')

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train-data-iterators-setup'])
    print_rank_0('training ...')

    print_datetime('First iter start')
    timers('first-train-iter').start()
    print_rank_0('training iter 0')
    # Profiling does not work yet.
    megatron_train_step(forward_step_func, train_data_iterator, model, optimizer, lr_scheduler, profile=False)
    timers('first-train-iter').stop()
    print_datetime('First iter stop')
    timers.log(['first-train-iter'])

    print_datetime('Benchmark iter start')

    for i in range(args.train_iters):
        timers('benchmark-iter-' + str(i)).start()
        print_rank_0('training iter '+str(i+1))
        megatron_train_step(forward_step_func, train_data_iterator, model, optimizer, lr_scheduler, profile=False)
        timers('benchmark-iter-'+str(i)).stop()
    print_datetime('Benchmark iter stop')
    time_log = ['first-train-iter']
    time_log.extend(['benchmark-iter-'+str(i) for i in range(args.train_iters)])
    timers.log(time_log)


if __name__ == '__main__':
    """Finetune/evaluate."""
    print("Start training.")
    train_qqp(train_dataset_provider, model_provider,
              ModelType.encoder_or_decoder, forward_step)
