from .cupy_nccl_backend import *


_GPU_PIPELINE_COMM = None
_GPU_PIPELINE_RANK = None
_GPU_PIPELINE_WORLD_SIZE = None
_CPU_RANKS = None


def get_gpu_pipeline_comm() -> CuPyNCCLCommunicator:
    assert _GPU_PIPELINE_COMM is not None
    return _GPU_PIPELINE_COMM


def get_gpu_pipeline_rank() -> int:
    assert _GPU_PIPELINE_RANK is not None
    return _GPU_PIPELINE_RANK


def get_gpu_pipeline_world_size() -> int:
    assert _GPU_PIPELINE_WORLD_SIZE is not None
    return _GPU_PIPELINE_WORLD_SIZE


def get_cpu_ranks()-> List[int]:
    assert _CPU_RANKS is not None
    return _CPU_RANKS


def get_hybrid_dispatch_comm():
    return dist


def get_hybrid_dispatch_rank() -> int:
    return dist.get_rank()


def get_hybrid_dispatch_world_size() -> int:
    return dist.get_world_size()


def _init_hybrid_communicators(args, rank=None):
    # world_size, pipeline_group_size, rank, cuda_id = 0
    assert args.world_size > args.pipeline_group_size
    if rank is None:
        dist.init_process_group(backend='gloo', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        rank = args.rank

    global _GPU_PIPELINE_COMM
    global _GPU_PIPELINE_RANK
    global _GPU_PIPELINE_WORLD_SIZE
    global _CPU_RANKS

    _GPU_PIPELINE_WORLD_SIZE = args.pipeline_group_size
    if rank < args.pipeline_group_size:
        _GPU_PIPELINE_RANK = rank
        _GPU_PIPELINE_COMM = CuPyNCCLCommunicator(_GPU_PIPELINE_RANK, args.cuda_id, args.pipeline_group_size,
                                              "pipeline_GPU_group")
    _CPU_RANKS = [i for i in range(args.pipeline_group_size, args.world_size)]


def init_hybrid_communicators(args):
    _init_hybrid_communicators(args)


def init_hybrid_inference_communicators_with_coordinator(args, prime_ip, rank, port=9999):
    init_with_coordinator(args, prime_ip, rank, port=port)
    _init_hybrid_communicators(args, rank=rank)