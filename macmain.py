import hydra
import torch
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pathlib import Path

from src.config_schema import AppConfig
from src.pipeline import run_pipeline
from src.utils import prepare_app_config, should_log_to_wandb


load_dotenv()


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: AppConfig) -> None:
    cfg = prepare_app_config(
        cfg,
        ddp_defaults={"world_size": 1, "rank": 0, "local_rank": 0},
    )

    if torch.backends.mps.is_available():
        cfg.device = "mps"
    elif torch.cuda.is_available():
        cfg.device = "cuda"
    else:
        cfg.device = "cpu"

    output_dir = Path(HydraConfig.get().run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run_active = False
    try:
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

        wandb_run_active = should_log_to_wandb()
        run_pipeline(cfg, log_to_wandb=wandb_run_active, is_main_process=True)
    finally:
        if wandb_run_active:
            wandb.finish()


if __name__ == "__main__":
    main()
