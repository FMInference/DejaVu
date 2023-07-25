import torch.distributed

from .cupy_nccl_backend import *
from .torch_nccl_backend import *

_DATA_PARALLEL_COMM = None
_DATA_PARALLEL_RANK = -1
_DATA_PARALLEL_WORLD_SIZE = -1

_PIPELINE_PARALLEL_COMM = None
_PIPELINE_PARALLEL_RANK = -1
_PIPELINE_PARALLEL_WORLD_SIZE = -1

_TENSOR_PARALLEL_COMM = None
_TENSOR_PARALLEL_RANK = -1
_TENSOR_PARALLEL_WORLD_SIZE = -1


def get_data_parallel_comm():
    assert _DATA_PARALLEL_COMM is not None
    return _DATA_PARALLEL_COMM


def get_data_parallel_rank() -> int:
    assert _DATA_PARALLEL_RANK is not None
    return _DATA_PARALLEL_RANK


def get_data_parallel_world_size() -> int:
    assert _DATA_PARALLEL_WORLD_SIZE is not None
    return _DATA_PARALLEL_WORLD_SIZE


def get_pipeline_parallel_comm():
    assert _PIPELINE_PARALLEL_COMM is not None
    return _PIPELINE_PARALLEL_COMM


def get_pipeline_parallel_rank() -> int:
    assert _PIPELINE_PARALLEL_RANK is not None
    return _PIPELINE_PARALLEL_RANK


def get_pipeline_parallel_world_size() -> int:
    assert _PIPELINE_PARALLEL_WORLD_SIZE is not None
    return _PIPELINE_PARALLEL_WORLD_SIZE


def get_megatron_tensor_parallel_comm():
    assert _TENSOR_PARALLEL_COMM is not None
    return _TENSOR_PARALLEL_COMM


def get_megatron_tensor_parallel_rank() -> int:
    assert _TENSOR_PARALLEL_RANK is not None
    return _TENSOR_PARALLEL_RANK


def get_megatron_tensor_parallel_world_size() -> int:
    assert _TENSOR_PARALLEL_WORLD_SIZE is not None
    return _TENSOR_PARALLEL_WORLD_SIZE


def _init_communicators(world_size, data_group_size, pipeline_group_size, rank, cuda_id, dist_backend='cupy_nccl'):
    if dist_backend == 'cupy_nccl':
        _init_communicators_cupy(world_size, data_group_size, pipeline_group_size, rank, cuda_id)
    elif dist_backend == 'torch_nccl':
        _init_communicators_torch(world_size, data_group_size, pipeline_group_size, rank, cuda_id)
    else:
        assert False, f'Not legal dist_backend <{dist_backend}>'


def _init_communicators_cupy(world_size, data_group_size, pipeline_group_size, rank, cuda_id):
    assert world_size == data_group_size * pipeline_group_size
    if world_size == data_group_size * pipeline_group_size:
        #    We do the following hard code alignment of communication groups:
        #    Suppose there are 8 instances (world_size), and 4 data parallel groups (data_group_size is 2),
        #    Then there would be 2 pipeline parallel groups (pipeline_group_size is 4), then the groups will look like:
        #    pipeline parallel: <group 0: [0,1,2,3]>, <group 1: [4,5,6,7]>
        #    data parallel: <group 0: [0,4]>, <group 1: [1,5]>, <group 2: [2,6]>, <group 3: [3,7]>
        # assert args.world_size == args.data_group_size * args.pipeline_group_size
        global _DATA_PARALLEL_COMM
        global _PIPELINE_PARALLEL_COMM
        global _DATA_PARALLEL_RANK
        global _PIPELINE_PARALLEL_RANK
        global _DATA_PARALLEL_WORLD_SIZE
        global _PIPELINE_PARALLEL_WORLD_SIZE
        # We use pipeline parallel by default.
        _PIPELINE_PARALLEL_WORLD_SIZE = pipeline_group_size
        _PIPELINE_PARALLEL_RANK = rank % pipeline_group_size
        _PIPELINE_PARALLEL_COMM = CuPyNCCLCommunicator(_PIPELINE_PARALLEL_RANK, cuda_id, pipeline_group_size,
                                                       "pipeline_group_" + str(rank // pipeline_group_size))
        if data_group_size != 1:
            _DATA_PARALLEL_WORLD_SIZE = data_group_size
            _DATA_PARALLEL_RANK = rank // pipeline_group_size
            _DATA_PARALLEL_COMM = CuPyNCCLCommunicator(_DATA_PARALLEL_RANK, cuda_id, data_group_size,
                                                       "data_group_" + str(rank % pipeline_group_size))
    # elif args.world_size == args.data_group_size * args.tensor_group_size:
    #    global _DATA_PARALLEL_COMM
    #    global _TENSOR_PARALLEL_COMM
    #    global _DATA_PARALLEL_RANK
    #    global _TENSOR_PARALLEL_RANK
    #    global _DATA_PARALLEL_WORLD_SIZE
    #    global _TENSOR_PARALLEL_WORLD_SIZE
    # We use megatron tensor parallel by default.
    #    _TENSOR_PARALLEL_WORLD_SIZE = args.tensor_group_size
    #    _TENSOR_PARALLEL_RANK = args.rank % args.tensor_group_size
    #    _TENSOR_PARALLEL_COMM = NCCLCommunicator(_TENSOR_PARALLEL_RANK, args.cuda_id, args.tensor_group_size,
    #                                             "tensor_group_" + str(args.rank // args.tensor_group_size))
    #    if args.data_group_size != 1:
    #        _DATA_PARALLEL_WORLD_SIZE = args.data_group_size
    #        _DATA_PARALLEL_RANK = args.rank // args.tensor_group_size
    #        _DATA_PARALLEL_COMM = NCCLCommunicator(_DATA_PARALLEL_RANK, args.cuda_id, args.data_group_size,
    #                                              "data_group_" + str(args.rank % args.tensor_group_size))
    else:
        print("Not supported yet")
        assert False


def _init_communicators_torch(world_size, data_group_size, pipeline_group_size, rank, cuda_id):
    assert world_size == data_group_size * pipeline_group_size
    torch.cuda.set_device(cuda_id)
    if world_size == data_group_size * pipeline_group_size:
        #    We do the following hard code alignment of communication groups:
        #    Suppose there are 8 instances (world_size), and 4 data parallel groups (data_group_size is 2),
        #    Then there would be 2 pipeline parallel groups (pipeline_group_size is 4), then the groups will look like:
        #    pipeline parallel: <group 0: [0,1,2,3]>, <group 1: [4,5,6,7]>
        #    data parallel: <group 0: [0,4]>, <group 1: [1,5]>, <group 2: [2,6]>, <group 3: [3,7]>
        # assert args.world_size == args.data_group_size * args.pipeline_group_size
        global _DATA_PARALLEL_COMM
        global _PIPELINE_PARALLEL_COMM
        global _DATA_PARALLEL_RANK
        global _PIPELINE_PARALLEL_RANK
        global _DATA_PARALLEL_WORLD_SIZE
        global _PIPELINE_PARALLEL_WORLD_SIZE
        # We use pipeline parallel by default.
        _PIPELINE_PARALLEL_WORLD_SIZE = pipeline_group_size
        _PIPELINE_PARALLEL_RANK = rank % pipeline_group_size

        for i in range(world_size//pipeline_group_size):
            ranks = [rank for rank in range(i*pipeline_group_size, (i+1)*pipeline_group_size)]
            pipeline_group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _PIPELINE_PARALLEL_COMM = TorchNCCLCommunicator(pipeline_group)

        if data_group_size != 1:
            _DATA_PARALLEL_WORLD_SIZE = data_group_size
            _DATA_PARALLEL_RANK = rank // pipeline_group_size

            for i in range(world_size//data_group_size):
                ranks = [rank in rank in range(i, world_size, data_group_size)]
                data_group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _DATA_PARALLEL_COMM = TorchNCCLCommunicator(data_group)
    else:
        print("Not supported yet")
        assert False


def _init_inference_communicators(pipeline_group_size, rank, cuda_id, dist_backend='cupy_nccl'):
    if dist_backend == 'cupy_nccl':
        _init_inference_communicators_cupy(pipeline_group_size, rank, cuda_id)
    elif dist_backend == 'torch_nccl':
        _init_inference_communicators_torch(pipeline_group_size, rank, cuda_id)
    else:
        assert False, f'Not legal dist_backend <{dist_backend}>'


def _init_inference_communicators_cupy(pipeline_group_size, rank, cuda_id):
    global _PIPELINE_PARALLEL_COMM
    global _PIPELINE_PARALLEL_RANK
    global _PIPELINE_PARALLEL_WORLD_SIZE
    _PIPELINE_PARALLEL_WORLD_SIZE = pipeline_group_size
    _PIPELINE_PARALLEL_RANK = rank % pipeline_group_size
    _PIPELINE_PARALLEL_COMM = CuPyNCCLCommunicator(_PIPELINE_PARALLEL_RANK, cuda_id, pipeline_group_size,
                                                   "pipeline_group")


def _init_inference_communicators_torch(pipeline_group_size, rank, cuda_id):
    torch.cuda.set_device(cuda_id)
    global _PIPELINE_PARALLEL_COMM
    global _PIPELINE_PARALLEL_RANK
    global _PIPELINE_PARALLEL_WORLD_SIZE
    _PIPELINE_PARALLEL_WORLD_SIZE = pipeline_group_size
    _PIPELINE_PARALLEL_RANK = rank % pipeline_group_size
    _PIPELINE_PARALLEL_COMM = TorchNCCLCommunicator(torch.distributed.new_group())


# Communicator for training
def init_communicators(args):
    default_init(args)
    _init_communicators(args.world_size, args.data_group_size, args.pipeline_group_size, args.rank, args.cuda_id,
                        args.dist_backend)


def init_inference_communicators(args):
    default_init(args)
    _init_inference_communicators(args.pipeline_group_size, args.rank, args.cuda_id, args.dist_backend)


# Communicator for training with coordinator
def init_communicators_with_coordinator(args, prime_ip, rank):
    init_with_coordinator(args, prime_ip, rank)
    _init_communicators(args.world_size, args.data_group_size, args.pipeline_group_size, rank, args.cuda_id,
                        args.dist_backend)


def init_inference_communicators_with_coordinator(args, prime_ip, rank, port=9999):
    init_with_coordinator(args, prime_ip, rank, port=port)
    _init_inference_communicators(args.pipeline_group_size, rank, args.cuda_id, args.dist_backend)
