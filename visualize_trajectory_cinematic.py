import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig

from cebra_nlp_public.config_runtime import to_typed_app_config
from cebra_nlp_public.runtime import configure_runtime, cleanup_distributed


@hydra.main(
    config_path="conf",
    config_name="trajectory_cinematic_viz",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    from cebra_nlp_public.stages.trajectory_cinematic_viz import (
        run as run_trajectory_cinematic_viz,
    )

    load_dotenv()
    typed_cfg = to_typed_app_config(cfg)

    is_main = configure_runtime(typed_cfg, enable_ddp=False)
    output_dir = Path(HydraConfig.get().run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_trajectory_cinematic_viz(typed_cfg, output_dir, is_main_process=is_main)
    cleanup_distributed()


if __name__ == "__main__":
    main()
