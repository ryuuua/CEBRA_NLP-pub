from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from types import ModuleType
from collections.abc import Mapping
from typing import Any

import numpy as np

from .cache_store import (
    load_text_embedding as _fallback_load_text_embedding,
    save_text_embedding as _fallback_save_text_embedding,
)


_MIN_COMPATIBLE_CACHE_VERSION = (0, 3, 2)
_OPTIONAL_CACHE_DIST_NAME = "labenv-embedding-cache"


@dataclass(frozen=True)
class CacheBackend:
    name: str
    version: str | None
    enabled: bool
    reason: str | None = None


@dataclass(frozen=True)
class LoadedEmbeddingCache:
    payload: tuple[Any, ...]
    metadata: Mapping[str, Any]
    backend: CacheBackend


def _version_tuple(raw: str | None) -> tuple[int, int, int] | None:
    if raw is None:
        return None
    parts: list[int] = []
    for part in raw.split("."):
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
                continue
            break
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    if not parts:
        return None
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _labenv_version(module: ModuleType) -> str | None:
    try:
        return metadata.version(_OPTIONAL_CACHE_DIST_NAME)
    except metadata.PackageNotFoundError:
        raw = getattr(module, "__version__", None)
        return None if raw is None else str(raw)


def _load_labenv_module() -> tuple[ModuleType | None, CacheBackend]:
    try:
        module = import_module("labenv_embedding_cache")
    except Exception as exc:
        return None, CacheBackend(
            name="cebra_nlp_public",
            version=None,
            enabled=False,
            reason=f"labenv_embedding_cache_unavailable:{type(exc).__name__}",
        )

    version = _labenv_version(module)
    parsed = _version_tuple(version)
    if parsed is None or parsed < _MIN_COMPATIBLE_CACHE_VERSION:
        return None, CacheBackend(
            name="cebra_nlp_public",
            version=version,
            enabled=False,
            reason="labenv_embedding_cache_version_too_old",
        )
    return module, CacheBackend(
        name="labenv_embedding_cache",
        version=version,
        enabled=True,
    )


def active_cache_backend() -> CacheBackend:
    _module, backend = _load_labenv_module()
    return backend


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "None", "null"):
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            item = value.item()
            return None if item is None else int(item)
        if value.size == 0:
            return None
        item = value.reshape(-1)[0].item()
        return None if item is None else int(item)
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            item = value.item()
            return None if item is None else str(item)
        if value.size == 0:
            return None
        item = value.reshape(-1)[0].item()
        return None if item is None else str(item)
    text = str(value).strip()
    return text or None


def _metadata_for_labenv(
    metadata_payload: Mapping[str, Any] | None,
    *,
    shuffle_seed: int | None,
    hidden_state_layer: int | None,
    embedding_type: str | None,
    pooling: str | None,
    rulebook_id: str | None,
    registry_key: str | None,
) -> dict[str, Any]:
    payload = dict(metadata_payload or {})
    embedding_model_name = payload.get("embedding_model_name")
    embedding_name = payload.get("embedding_name")
    model_name = _first_present(payload.get("model_name"), embedding_model_name, embedding_name)
    model_id = _first_present(payload.get("model_id"), registry_key, embedding_name, model_name)

    payload.update(
        {
            "dataset_name": payload.get("dataset_name"),
            "dataset_key": payload.get("dataset_key"),
            "dataset_shuffle_seed": shuffle_seed,
            "shuffle_seed": shuffle_seed,
            "model_id": model_id,
            "model_name": model_name,
            "registry_key": registry_key or payload.get("registry_key"),
            "embedding_type": embedding_type or payload.get("embedding_type"),
            "pooling": pooling or payload.get("pooling"),
            "hidden_state_layer": hidden_state_layer,
            "variant_tag": payload.get("variant_tag"),
            "rulebook_id": rulebook_id or payload.get("rulebook_id"),
        }
    )
    return {key: value for key, value in payload.items() if value is not None}


def _tuple_from_labenv_arrays(arrays: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata_payload = _as_mapping(arrays.get("metadata"))
    dataset = _as_mapping(metadata_payload.get("dataset"))
    model = _as_mapping(metadata_payload.get("model"))
    embedding = _as_mapping(metadata_payload.get("embedding"))
    return (
        np.asarray(arrays["ids"]).reshape(-1).astype(str),
        np.asarray(arrays["embeddings"], dtype=np.float32),
        _optional_int(
            _first_present(
                metadata_payload.get("shuffle_seed"),
                metadata_payload.get("dataset_shuffle_seed"),
                dataset.get("shuffle_seed"),
            )
        ),
        (
            np.asarray(arrays["layer_embeddings"], dtype=np.float32)
            if arrays.get("layer_embeddings") is not None
            else None
        ),
        _optional_int(
            _first_present(
                metadata_payload.get("hidden_state_layer"),
                embedding.get("hidden_state_layer"),
            )
        ),
        _optional_str(
            _first_present(metadata_payload.get("embedding_type"), embedding.get("type"))
        ),
        _optional_str(_first_present(metadata_payload.get("pooling"), embedding.get("pooling"))),
        _optional_str(
            _first_present(metadata_payload.get("rulebook_id"), embedding.get("rulebook_id"))
        ),
        _optional_str(
            _first_present(metadata_payload.get("registry_key"), model.get("registry_key"))
        ),
    )


def load_embedding_cache(
    path: Path,
    *,
    load_layer_embeddings: bool = True,
) -> LoadedEmbeddingCache | None:
    path = Path(path)
    module, backend = _load_labenv_module()
    if module is not None and hasattr(module, "load_cache_arrays"):
        if not path.exists():
            return None
        try:
            arrays = module.load_cache_arrays(
                path,
                load_layer_embeddings=load_layer_embeddings,
            )
            return LoadedEmbeddingCache(
                payload=_tuple_from_labenv_arrays(arrays),
                metadata=_as_mapping(arrays.get("metadata")),
                backend=backend,
            )
        except Exception:
            # Legacy/fallback caches should remain readable even when the optional
            # adapter cannot parse a file written before it was introduced.
            pass

    payload = _fallback_load_text_embedding(path, load_layer_embeddings=load_layer_embeddings)
    if payload is None:
        return None
    return LoadedEmbeddingCache(
        payload=payload,
        metadata={},
        backend=CacheBackend(
            name="cebra_nlp_public",
            version=None,
            enabled=False,
            reason=backend.reason,
        ),
    )


def save_embedding_cache(
    ids: Any,
    embeddings: Any,
    shuffle_seed: int | None,
    path: Path,
    *,
    layer_embeddings: Any = None,
    hidden_state_layer: int | None = None,
    embedding_type: str | None = None,
    pooling: str | None = None,
    rulebook_id: str | None = None,
    registry_key: str | None = None,
    metadata_payload: Mapping[str, Any] | None = None,
) -> CacheBackend:
    path = Path(path)
    module, backend = _load_labenv_module()
    if module is not None and hasattr(module, "save_v2_embedding_cache"):
        metadata_for_labenv = _metadata_for_labenv(
            metadata_payload,
            shuffle_seed=shuffle_seed,
            hidden_state_layer=hidden_state_layer,
            embedding_type=embedding_type,
            pooling=pooling,
            rulebook_id=rulebook_id,
            registry_key=registry_key,
        )
        module.save_v2_embedding_cache(
            ids=ids,
            embeddings=embeddings,
            layer_embeddings=layer_embeddings,
            path=path,
            metadata=metadata_for_labenv,
            provenance={"generator": "cebra_nlp_public.embedding_cache_adapter"},
            require_locator=False,
        )
        if hasattr(module, "validate_cache_npz"):
            validation = module.validate_cache_npz(path, verification_mode="fast")
            if validation.get("verification_status") != "passed":
                errors = validation.get("verification_errors") or []
                raise ValueError(f"labenv v2 cache validation failed: {errors}")
        return backend

    _fallback_save_text_embedding(
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
        metadata=metadata_payload,
    )
    return CacheBackend(
        name="cebra_nlp_public",
        version=None,
        enabled=False,
        reason=backend.reason,
    )


__all__ = [
    "CacheBackend",
    "LoadedEmbeddingCache",
    "active_cache_backend",
    "load_embedding_cache",
    "save_embedding_cache",
]
