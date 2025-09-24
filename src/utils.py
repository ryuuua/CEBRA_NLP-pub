import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from src.config_schema import AppConfig

def get_embedding_cache_path(cfg):
    """
    Generates a unique path for a cached text embedding file.
    Reads the base directory from cfg.paths.embedding_cache_dir,
    with a fallback to 'embedding_cache' for backward compatibility.
    """
    # Get base directory from config, with a default for backward compatibility
    base_dir = OmegaConf.select(cfg, 'paths.embedding_cache_dir', default='embedding_cache')
    
    dataset_name = cfg.dataset.name
    embedding_name = cfg.embedding.name
    
    # Create a filename-safe version of the embedding name
    safe_embedding_name = embedding_name.replace('/', '__')


    filename = f"{dataset_name}__{safe_embedding_name}"
    if getattr(cfg.dataset, "shuffle", False):
        seed = getattr(cfg.dataset, "shuffle_seed", None)
        if seed is not None:
            filename += f"__seed{seed}"
        else:
            filename += "__shuffle"

    path = Path(base_dir) / f"{filename}.npz"

    return path

def save_text_embedding(ids, embeddings, shuffle_seed, path: Path):
    """Saves numpy embeddings and their ids to the specified path."""
    print(f"Caching text embeddings to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, embeddings=embeddings, shuffle_seed=shuffle_seed)

    print("...done.")

def load_text_embedding(path: Path):
    """Loads cached ids and embeddings from the specified path if it exists."""
    if path.exists():
        print(f"Found cached text embeddings at {path}. Loading...")

        data = np.load(path, allow_pickle=True)
        ids = data["ids"]
        embeddings = data["embeddings"]
        shuffle_seed = data["shuffle_seed"].item() if "shuffle_seed" in data else None
        print("...done.")
        return ids, embeddings, shuffle_seed

    else:
        print(f"No cached text embeddings found at {path}.")
        return None


def apply_reproducibility(cfg: "AppConfig") -> None:
    """Apply global seeding and deterministic settings based on the config."""

    repro_cfg = getattr(cfg, "reproducibility", None)
    if repro_cfg is None:
        return

    base_seed = int(repro_cfg.seed)

    if dist.is_available() and dist.is_initialized():
        seed_container = [base_seed]
        dist.broadcast_object_list(seed_container, src=0)
        base_seed = seed_container[0]
        rank = dist.get_rank()
    else:
        rank = int(getattr(getattr(cfg, "ddp", None), "rank", 0) or 0)

    seed = base_seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(repro_cfg.deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = repro_cfg.deterministic
        torch.backends.cudnn.benchmark = repro_cfg.cudnn_benchmark
