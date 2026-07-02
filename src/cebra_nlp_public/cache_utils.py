from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path

from .config_schema import AppConfig


ConfigLike = AppConfig | Mapping[str, object]


def _cfg_get(source: ConfigLike | object | None, path: str, default: object = None) -> object:
    current: object | None = source
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def _configured_path(value: object | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _slug(value: object, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text)
    return text.strip("._-") or fallback


def _short_hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _embedding_family(embedding_type: object | None) -> str:
    if embedding_type in {"hf_transformer", "sentence_transformer"}:
        return "lm"
    if embedding_type:
        return _slug(embedding_type)
    return "unknown"


def resolve_embedding_cache_dir(
    cfg: ConfigLike,
    *,
    env_var: str = "CEBRA_NLP_CACHE_DIR",
) -> Path:
    configured = _configured_path(_cfg_get(cfg, "paths.embedding_cache_dir", None))
    if configured is not None:
        return configured

    for name in (env_var, "CEBRA_NLP_CACHE_DIR"):
        configured = _configured_path(os.getenv(name))
        if configured is not None:
            return configured

    return Path("artifacts/cache/embeddings")


def _variant_fingerprint(cfg: ConfigLike, variant_tag: str | None) -> str:
    if variant_tag:
        return _slug(variant_tag)

    fields = [
        _cfg_get(cfg, "embedding.name", ""),
        _cfg_get(cfg, "embedding.type", ""),
        _cfg_get(cfg, "embedding.model_name", ""),
        _cfg_get(cfg, "embedding.pooling", ""),
        _cfg_get(cfg, "embedding.hidden_state_layer", ""),
        _cfg_get(cfg, "embedding.cache_all_layers", ""),
        _cfg_get(cfg, "embedding.torch_dtype", ""),
        _cfg_get(cfg, "embedding.embedding_seed", ""),
    ]
    name = _slug(_cfg_get(cfg, "embedding.name", "embedding"), fallback="embedding")
    return f"{name}__{_short_hash(*fields)}"


def build_embedding_cache_filename(
    cfg: ConfigLike,
    *,
    dataset_key: str | None = None,
    variant_tag: str | None = None,
    include_shuffle_seed: bool | None = None,
) -> str:
    dataset = _slug(dataset_key or _cfg_get(cfg, "dataset.name", "dataset"), fallback="dataset")
    variant = _variant_fingerprint(cfg, variant_tag)
    parts = [dataset, variant]
    if include_shuffle_seed is True:
        seed = _cfg_get(cfg, "dataset.shuffle_seed", None)
        if seed is None:
            seed = _cfg_get(cfg, "evaluation.random_state", None)
        if seed is not None:
            parts.append(f"seed{int(seed)}")
    return "__".join(parts) + ".npz"


def get_embedding_cache_path(
    cfg: ConfigLike,
    *,
    dataset_key: str | None = None,
    variant_tag: str | None = None,
    include_shuffle_seed: bool | None = None,
) -> Path:
    base_dir = resolve_embedding_cache_dir(cfg)
    filename = build_embedding_cache_filename(
        cfg,
        dataset_key=dataset_key,
        variant_tag=variant_tag,
        include_shuffle_seed=include_shuffle_seed,
    )
    model_name = _cfg_get(cfg, "embedding.model_name", None) or _cfg_get(
        cfg,
        "embedding.name",
        "unknown_model",
    )
    return (
        base_dir
        / _embedding_family(_cfg_get(cfg, "embedding.type", None))
        / _slug(model_name, fallback="unknown_model")
        / filename
    )


__all__ = [
    "build_embedding_cache_filename",
    "get_embedding_cache_path",
    "resolve_embedding_cache_dir",
]
