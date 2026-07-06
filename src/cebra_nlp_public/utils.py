import random
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist

from . import cache_utils
from .embedding_cache_adapter import (
    load_embedding_cache as _load_embedding_cache,
    save_embedding_cache as _save_embedding_cache,
)

if TYPE_CHECKING:
    from .config_schema import AppConfig


def get_embedding_cache_path(
    cfg,
    *,
    dataset_key: str | None = None,
    variant_tag: str | None = None,
    include_shuffle_seed: bool | None = None,
):
    """Generate a unique path for a cached text embedding file."""
    return cache_utils.get_embedding_cache_path(
        cfg,
        dataset_key=dataset_key,
        variant_tag=variant_tag,
        include_shuffle_seed=include_shuffle_seed,
    )


def save_text_embedding(
    ids,
    embeddings,
    shuffle_seed,
    path: Path,
    layer_embeddings=None,
    *,
    metadata: object | None = None,
    hidden_state_layer: int | None = None,
    embedding_type: str | None = None,
    pooling: str | None = None,
    rulebook_id: str | None = None,
    registry_key: str | None = None,
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
    """
    resolved_metadata = None
    if metadata is not None:
        if isinstance(metadata, Mapping):
            resolved_metadata = dict(metadata)
        else:
            resolved_metadata = {"repr": repr(metadata)}
    _save_embedding_cache(
        ids,
        embeddings,
        shuffle_seed,
        path,
        layer_embeddings=layer_embeddings,
        hidden_state_layer=hidden_state_layer,
        embedding_type=embedding_type,
        pooling=pooling,
        rulebook_id=rulebook_id,
        registry_key=registry_key,
        metadata_payload=resolved_metadata,
    )


def load_text_embedding(path: Path, *, load_layer_embeddings: bool = True):
    """Loads cached ids and embeddings from the specified path if it exists."""
    loaded = _load_embedding_cache(path, load_layer_embeddings=load_layer_embeddings)
    return None if loaded is None else loaded.payload


def apply_reproducibility(cfg: "AppConfig") -> None:
    """Apply global seeding and deterministic settings based on the config."""

    repro_cfg = getattr(cfg, "reproducibility", None)
    if repro_cfg is None:
        return

    base_seed = int(repro_cfg.seed)
    deterministic = repro_cfg.deterministic

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

    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = repro_cfg.cudnn_benchmark
