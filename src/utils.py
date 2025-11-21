import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Dict, Tuple, List

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

CACHE_VERSION = 3

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

def save_text_embedding(
    ids,
    embeddings,
    shuffle_seed,
    path: Path,
    layer_embeddings=None,
    pooling_method: Optional[str] = None,
):
    """
    Saves numpy embeddings and their ids to the specified path.

    Parameters
    ----------
    ids : array-like
        The identifiers for each embedding row.
    embeddings : np.ndarray
        The embeddings associated with the provided ids.
    shuffle_seed : Optional[int]
        Seed used when shuffling the dataset (stored for cache validation).
    path : Path
        Destination file.
    layer_embeddings : Optional[np.ndarray]
        Pooled hidden states for all transformer layers with shape
        (num_samples, num_layers, hidden_dim). Stored when available to avoid
        recomputing heavy transformer passes.
    pooling_method : Optional[str]
        Pooling method used to produce ``embeddings`` (e.g., ``mean`` or ``cls``).
    """
    print(f"Caching text embeddings to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "ids": ids,
        "embeddings": embeddings,
        "shuffle_seed": shuffle_seed,
        "cache_version": CACHE_VERSION,
    }
    if layer_embeddings is not None:
        save_kwargs["layer_embeddings"] = layer_embeddings
    if pooling_method is not None:
        save_kwargs["pooling_method"] = np.asarray(pooling_method)
    np.savez(path, **save_kwargs)

    print("...done.")

def load_text_embedding(path: Path):
    """Loads cached embeddings if compatible with the current cache schema."""
    if not path.exists():
        print(f"No cached text embeddings found at {path}.")
        return None

    print(f"Found cached text embeddings at {path}. Loading...")

    data = np.load(path, allow_pickle=True)
    cache_version = (
        data["cache_version"].item() if "cache_version" in data.files else None
    )
    if cache_version != CACHE_VERSION:
        print(
            f"Cached embeddings version mismatch (found {cache_version}, expected {CACHE_VERSION}). "
            "Recomputing..."
        )
        return None

    ids = data["ids"]
    embeddings = data["embeddings"]
    shuffle_seed = data["shuffle_seed"].item() if "shuffle_seed" in data.files else None
    layer_embeddings = data["layer_embeddings"] if "layer_embeddings" in data.files else None
    pooling_method = (
        data["pooling_method"].item() if "pooling_method" in data.files else None
    )
    print("...done.")
    return ids, embeddings, shuffle_seed, layer_embeddings, pooling_method


def apply_reproducibility(cfg: "AppConfig") -> None:
    """Apply global seeding and deterministic settings based on the config."""

    repro_cfg = getattr(cfg, "reproducibility", None)
    if repro_cfg is None:
        return

    base_seed = int(repro_cfg.seed)

    # Handle distributed training seed synchronization
    if dist.is_available() and dist.is_initialized():
        seed_container = [base_seed]
        dist.broadcast_object_list(seed_container, src=0)
        base_seed = seed_container[0]
        rank = dist.get_rank()
    else:
        ddp_cfg = getattr(cfg, "ddp", None)
        rank = int(getattr(ddp_cfg, "rank", 0) or 0)

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


def prepare_app_config(
    raw_cfg: Any,
    *,
    device_override: Optional[str] = None,
    ddp_defaults: Optional[Dict[str, int]] = None,
) -> "AppConfig":
    """
    Normalize Hydra configs into AppConfig with standard defaults applied.
    """
    from src.config_schema import AppConfig
    from src.cebra_trainer import normalize_model_architecture

    default_cfg = OmegaConf.structured(AppConfig)
    cfg = OmegaConf.merge(default_cfg, raw_cfg)
    OmegaConf.set_struct(cfg, False)

    if hasattr(cfg, "cebra"):
        if getattr(cfg.cebra, "conditional", None) is not None:
            cfg.cebra.conditional = cfg.cebra.conditional.lower()
        cfg.cebra.model_architecture = normalize_model_architecture(
            getattr(cfg.cebra, "model_architecture", "offset1-model")
        )

    if ddp_defaults:
        ddp_cfg = getattr(cfg, "ddp", None)
        if ddp_cfg is not None:
            for key, value in ddp_defaults.items():
                if value is not None and hasattr(ddp_cfg, key):
                    setattr(ddp_cfg, key, value)

    if device_override is not None:
        cfg.device = device_override

    apply_reproducibility(cfg)
    return cfg


def resolve_shuffle_seed(cfg: "AppConfig") -> Optional[int]:
    """Resolve shuffle seed from configuration.
    
    Checks cfg.dataset.shuffle_seed first, then falls back to
    cfg.evaluation.random_state if available.
    """
    if getattr(cfg.dataset, "shuffle_seed", None) is not None:
        return cfg.dataset.shuffle_seed
    if hasattr(cfg, "evaluation"):
        return getattr(cfg.evaluation, "random_state", None)
    return None


def build_id_index_map(
    requested_ids: Sequence,
    cached_ids: Sequence,
) -> Tuple[List[str], Dict[str, int]]:
    """Build string ID to index mapping for efficient cache lookups.
    
    Returns:
        (str_ids, id_to_index): List of string IDs and mapping dict.
    """
    str_ids = [str(i) for i in requested_ids]
    id_to_index = {str(cached_id): idx for idx, cached_id in enumerate(cached_ids)}
    return str_ids, id_to_index


def normalize_binary_labels(data: np.ndarray) -> np.ndarray:
    """Normalize binary labels from {-1, 1} to {0, 1}.
    
    If data contains only -1 and 1, converts to 0 and 1.
    Otherwise returns data unchanged.
    """
    data = np.asarray(data)
    unique_values = set(np.unique(data))
    if unique_values == {-1, 1}:
        return np.where(data == -1, 0, 1)
    return data


def should_log_to_wandb() -> bool:
    """Check if Weights & Biases logging is active."""
    import wandb
    return wandb.run is not None


def find_run_dirs(results_root: Path, run_id: str) -> List[Path]:
    """Locate run directories whose wandb_run_id.txt matches ``run_id``.
    
    Searches recursively under results_root for directories containing
    a wandb_run_id.txt file with content matching run_id.
    """
    matches: List[Path] = []
    for marker in results_root.rglob("wandb_run_id.txt"):
        try:
            if marker.read_text().strip() == run_id:
                matches.append(marker.parent)
        except OSError:
            continue
    return sorted(matches)
