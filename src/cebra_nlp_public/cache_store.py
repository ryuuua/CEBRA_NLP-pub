from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ResolvedCache:
    path: Path


def _metadata_matches(path: Path, expected_metadata: Mapping[str, object] | None) -> bool:
    if expected_metadata is None:
        return True
    try:
        with np.load(path, allow_pickle=True) as payload:
            if "metadata_json" not in payload.files:
                return False
            raw = payload["metadata_json"].item()
            metadata = json.loads(str(raw))
    except Exception:
        return False

    for key in ("dataset_key", "variant_tag", "registry_key", "rulebook_id"):
        expected = expected_metadata.get(key)
        if expected is not None and metadata.get(key) != expected:
            return False
    return True


def resolve_cache_with_index(
    expected_path: str | Path,
    *,
    expected_metadata: Mapping[str, object] | None = None,
    dataset_name: str | None = None,
    cfg: object | None = None,
    cache_dir: str | Path | None = None,
) -> ResolvedCache | None:
    del dataset_name, cfg
    expected = Path(expected_path)
    if expected.exists() and _metadata_matches(expected, expected_metadata):
        return ResolvedCache(path=expected)

    search_root = Path(cache_dir) if cache_dir is not None else expected.parent
    if not search_root.exists():
        return None
    for candidate in sorted(search_root.rglob("*.npz")):
        if candidate == expected:
            continue
        if _metadata_matches(candidate, expected_metadata):
            return ResolvedCache(path=candidate)
    return None


def _scalar(payload, key: str):
    if key not in payload.files:
        return None
    value = payload[key]
    if value.shape == ():
        item = value.item()
        return None if item is None else item
    if value.size == 0:
        return None
    item = value.reshape(-1)[0].item()
    return None if item is None else item


def load_text_embedding(path: Path, *, load_layer_embeddings: bool = True):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as payload:
        ids = np.asarray(payload["ids"]).reshape(-1).astype(str)
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        shuffle_seed = _scalar(payload, "shuffle_seed")
        layer_embeddings = (
            np.asarray(payload["layer_embeddings"], dtype=np.float32)
            if load_layer_embeddings and "layer_embeddings" in payload.files
            else None
        )
        hidden_state_layer = _scalar(payload, "hidden_state_layer")
        embedding_type = _scalar(payload, "embedding_type")
        pooling = _scalar(payload, "pooling")
        rulebook_id = _scalar(payload, "rulebook_id")
        registry_key = _scalar(payload, "registry_key")
    return (
        ids,
        embeddings,
        shuffle_seed,
        layer_embeddings,
        hidden_state_layer,
        embedding_type,
        pooling,
        rulebook_id,
        registry_key,
    )


def save_text_embedding(
    ids,
    embeddings,
    shuffle_seed,
    path: Path,
    layer_embeddings=None,
    *,
    hidden_state_layer: int | None = None,
    embedding_type: str | None = None,
    pooling: str | None = None,
    rulebook_id: str | None = None,
    registry_key: str | None = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ids": np.asarray(ids, dtype=str),
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "shuffle_seed": np.asarray(shuffle_seed, dtype=object),
        "hidden_state_layer": np.asarray(hidden_state_layer, dtype=object),
        "embedding_type": np.asarray(embedding_type, dtype=object),
        "pooling": np.asarray(pooling, dtype=object),
        "rulebook_id": np.asarray(rulebook_id, dtype=object),
        "registry_key": np.asarray(registry_key, dtype=object),
        "metadata_json": np.asarray(
            json.dumps(dict(metadata or {}), sort_keys=True),
            dtype=object,
        ),
    }
    if layer_embeddings is not None:
        payload["layer_embeddings"] = np.asarray(layer_embeddings, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path


__all__ = [
    "ResolvedCache",
    "load_text_embedding",
    "resolve_cache_with_index",
    "save_text_embedding",
]
