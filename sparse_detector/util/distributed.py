"""
Utilities for distributed training
"""
import os
from typing import Any
from collections import namedtuple

import torch
import torch.distributed as dist

DistConfig = namedtuple('DistConfig', ['gpu', 'rank', 'world_size', 'distributed', 'backend', 'dist_url'])

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(dist_url: str) -> Any:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        distributed = False
        return

    distributed = True
    dist_config = DistConfig(gpu=gpu, rank=rank, world_size=world_size, distributed=distributed,
                             backend='nccl', dist_url=dist_url)
    print(dist_config)

    torch.cuda.set_device(dist_config.gpu)
    print('| distributed init (rank {}): {}'.format(dist_config.rank, dist_url), flush=True)
    dist.init_process_group(
        backend=dist_config.backend,
        init_method=dist_config.dist_url,
        world_size=dist_config.world_size,
        rank=dist_config.rank
    )
    dist.barrier()
    setup_for_distributed(rank == 0)

    return dist_config
