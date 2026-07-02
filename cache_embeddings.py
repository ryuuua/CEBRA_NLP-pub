import os
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig

from cebra_nlp_public.config_runtime import to_typed_app_config
from cebra_nlp_public.embedding_parallel import resolve_parallel_strategy
from cebra_nlp_public.runtime import configure_runtime, cleanup_distributed


@hydra.main(config_path="conf", config_name="cache", version_base="1.2")
def main(cfg: DictConfig) -> None:
    from cebra_nlp_public.stages.cache import run as run_cache

    load_dotenv()
    typed_cfg = to_typed_app_config(cfg)

    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    strategy = resolve_parallel_strategy(typed_cfg.embedding)
    if world_size > 1 and strategy == "pipeline2":
        raise SystemExit(
            "parallel_strategy='pipeline2' requires a single process. "
            "Run cache generation without torchrun/DDP (WORLD_SIZE=1), "
            "or pass embedding.parallel_strategy=ddp."
        )

    _is_main = configure_runtime(
        typed_cfg, enable_ddp=(world_size > 1 or strategy == "ddp")
    )
    output_dir = Path(HydraConfig.get().run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cache(typed_cfg, output_dir)
    cleanup_distributed()


if __name__ == "__main__":
    main()
