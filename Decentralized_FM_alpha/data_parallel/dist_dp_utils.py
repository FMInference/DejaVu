from .dist_dp_allreduce import AllReduceDP
from .dist_dp_central_ps import CentralPSDP
from .dist_dp_sharded_ps import ShardedPSDP


def get_dp_module(args, device, module, optimizer, rank=None):
    print("Data parallel implementation: ", args.dp_mode)
    if args.dp_mode == 'allreduce':
        return AllReduceDP(args, device, module, optimizer, rank=rank)
    elif args.dp_mode == 'central_ps':
        return CentralPSDP(args, device, module, optimizer, rank=rank)
    elif args.dp_mode == 'sharded_ps':
        return ShardedPSDP(args, device, module, optimizer, rank=rank)
    else:
        print("Not recognize this data parallel mode.")
        assert False
