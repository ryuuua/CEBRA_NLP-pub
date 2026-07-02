from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .config_runtime import to_config_container
from .config_schema import AppConfig


ConfigLike = AppConfig | Mapping[str, object]
RULEBOOK_ID = "cebra_nlp_public_local_cache_v1"


def _cfg_get(source: Any, path: str, default: Any = None) -> Any:
    current = source
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def _slug(value: object, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text)
    return text.strip("._-") or fallback


def _hash_payload(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_config_container(value: Any) -> Any:
    return to_config_container(value, resolve=True)


def build_cache_metadata(
    cfg: ConfigLike,
    ids: Sequence[object],
    texts: Sequence[str],
    *,
    labels: Sequence[object] | None = None,
) -> dict[str, object]:
    dataset_name = _slug(_cfg_get(cfg, "dataset.name", "dataset"), fallback="dataset")
    embedding_name = _slug(
        _cfg_get(cfg, "embedding.name", "embedding"),
        fallback="embedding",
    )
    content_hash = _hash_payload(
        {
            "ids": [str(item) for item in ids],
            "texts": [str(item) for item in texts],
            "labels": None if labels is None else [str(item) for item in labels],
        }
    )
    embedding_hash = _hash_payload(
        {
            "embedding": _normalize_config_container(_cfg_get(cfg, "embedding", {})),
            "reproducibility": _normalize_config_container(
                _cfg_get(cfg, "reproducibility", {})
            ),
        }
    )
    registry_key = embedding_name.lower()
    return {
        "dataset_name": dataset_name,
        "dataset_key": f"{dataset_name}__{content_hash}",
        "embedding_name": embedding_name,
        "embedding_type": _cfg_get(cfg, "embedding.type", None),
        "embedding_model_name": _cfg_get(cfg, "embedding.model_name", None),
        "variant_tag": f"{embedding_name}__{embedding_hash}",
        "registry_key": registry_key,
        "rulebook_id": RULEBOOK_ID,
        "num_rows": len(ids),
        "content_hash": content_hash,
    }


__all__ = ["RULEBOOK_ID", "build_cache_metadata"]
