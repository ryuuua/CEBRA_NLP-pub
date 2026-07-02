from __future__ import annotations

from pathlib import Path
from typing import Any

from .cache_metadata import RULEBOOK_ID


def canonicalize_registry_key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def registry_keys_equivalent(left: Any, right: Any) -> bool:
    return canonicalize_registry_key(left) == canonicalize_registry_key(right)


def enforce_registry_key_on_cache() -> bool:
    return True


def enforce_rulebook_id_on_cache() -> bool:
    return True


def get_rulebook_id() -> str:
    return RULEBOOK_ID


def include_shuffle_seed_in_filename() -> bool:
    return True


def load_embedding_registry() -> dict[str, object]:
    return {}


def load_rulebook() -> dict[str, object]:
    return {
        "schema_version": 1,
        "identity": {"name": "cebra_nlp_public_local_cache", "version": 1},
    }


def resolve_embedding_registry_key(embedding_cfg: Any, *, strict: bool = False) -> str | None:
    del strict
    name = getattr(embedding_cfg, "name", None)
    return canonicalize_registry_key(name)


def resolve_policy_dir() -> Path:
    return Path("conf")


def resolve_registry_path() -> Path:
    return Path("conf/embedding_registry.yaml")


def resolve_rulebook_path() -> Path:
    return Path("conf/embedding_rulebook.yaml")


def resolve_shared_cache_layout() -> str:
    return "hierarchical"


def resolve_shared_cache_tag() -> str | None:
    return None


def resolve_shared_embedding_cache_dir() -> Path:
    return Path("artifacts/cache/embeddings")


def validate_embedding_registry(embedding_cfg: Any | None = None) -> str | None:
    if embedding_cfg is None:
        return None
    return resolve_embedding_registry_key(embedding_cfg, strict=False)


__all__ = [
    "canonicalize_registry_key",
    "enforce_registry_key_on_cache",
    "enforce_rulebook_id_on_cache",
    "get_rulebook_id",
    "include_shuffle_seed_in_filename",
    "load_embedding_registry",
    "load_rulebook",
    "registry_keys_equivalent",
    "resolve_embedding_registry_key",
    "resolve_policy_dir",
    "resolve_registry_path",
    "resolve_rulebook_path",
    "resolve_shared_cache_layout",
    "resolve_shared_cache_tag",
    "resolve_shared_embedding_cache_dir",
    "validate_embedding_registry",
]
