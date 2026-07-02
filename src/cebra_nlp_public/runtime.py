import os

import torch
import torch.distributed as dist

from .config_schema import AppConfig


def configure_runtime(cfg: AppConfig, *, enable_ddp: bool) -> bool:
    """Configure device and optional distributed training settings.

    Returns True when running on the main process.
    """
    env = os.environ
    local_rank = int(env.get("LOCAL_RANK", 0))
    cfg.ddp.world_size = int(env.get("WORLD_SIZE", 1))
    cfg.ddp.rank = int(env.get("RANK", 0))
    cfg.ddp.local_rank = local_rank

    is_main_process = cfg.ddp.rank == 0
    use_ddp = enable_ddp and cfg.ddp.world_size > 1
    cuda_available = torch.cuda.is_available()

    if use_ddp:
        if "RANK" not in env or "LOCAL_RANK" not in env:
            print(
                "Warning: WORLD_SIZE > 1 but RANK or LOCAL_RANK not set. "
                "Distributed training may be misconfigured."
            )
        else:
            dist.init_process_group(
                backend="nccl", rank=cfg.ddp.rank, world_size=cfg.ddp.world_size
            )
            if cuda_available:
                torch.cuda.set_device(local_rank)
                cfg.device = f"cuda:{local_rank}"
    elif cuda_available:
        cfg.device = "cuda"

    return is_main_process


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
