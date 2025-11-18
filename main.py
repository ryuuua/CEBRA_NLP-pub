import hydra
import os
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from src.config_schema import AppConfig
from src.pipeline import run_pipeline
from src.utils import prepare_app_config

# .envファイルから環境変数を読み込む
load_dotenv()


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: AppConfig) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp_defaults = {
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "rank": int(os.environ.get("RANK", 0)),
        "local_rank": local_rank,
    }
    cfg = prepare_app_config(cfg, ddp_defaults=ddp_defaults)
    is_main_process = cfg.ddp.rank == 0
    cuda_available = torch.cuda.is_available()
    if cfg.ddp.world_size > 1:
        if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
            print(
                "Warning: WORLD_SIZE > 1 but RANK or LOCAL_RANK not set. "
                "Distributed training may be misconfigured."
            )
        else:
            dist.init_process_group(
                backend="nccl", rank=cfg.ddp.rank, world_size=cfg.ddp.world_size
            )
            torch.cuda.set_device(local_rank)
            if cuda_available:
                cfg.device = f"cuda:{local_rank}"
    else:
        if cuda_available:
            cfg.device = "cuda"
    output_dir = Path(HydraConfig.get().run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run = None
    run_id = None
    wandb_run_active = False
    try:

        # Initialize Weights & Biases
        if is_main_process:
            run_name = HydraConfig.get().job.name
            run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            if run is not None:
                run_id = run.id
                print(f"W&B Run Name: {run_name}, Run ID: {run_id}")
                (output_dir / "wandb_run_id.txt").write_text(run_id)

        wandb_run_active = is_main_process and wandb.run is not None
        run_pipeline(
            cfg,
            log_to_wandb=wandb_run_active,
            is_main_process=is_main_process,
        )
    finally:
        if wandb_run_active:
            wandb.finish()
        if cfg.ddp.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
