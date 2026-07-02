from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping
from typing import Optional

import torch
import yaml

from .config_schema import EmbeddingConfig


RAW_SIZE_PIPELINE2_THRESHOLD = 200000


def _cfg_get(
    source: object | Mapping[str, object] | None, key: str, default: object = None
) -> object:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _catalog_path() -> Path:
    return (_repo_root() / "conf" / "embedding_model_catalog.yaml").resolve()


@lru_cache(maxsize=1)
def load_embedding_model_catalog() -> dict[str, object]:
    path = _catalog_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None or not isinstance(loaded, Mapping):
        return {}
    return {str(key): value for key, value in loaded.items()}


def lookup_raw_size_value(model_name: Optional[str]) -> Optional[float]:
    if model_name is None:
        return None
    model_name_text = str(model_name).strip()
    if not model_name_text:
        return None

    catalog = load_embedding_model_catalog()
    models_obj = catalog.get("models", {})
    if not isinstance(models_obj, dict):
        return None
    models = models_obj

    def _extract(key: str) -> Optional[float]:
        entry = models.get(key)
        if not isinstance(entry, dict):
            return None
        raw = entry.get("raw_size_value", None)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    candidates = [
        model_name_text,
        model_name_text.split("/")[-1],
    ]
    for cand in candidates:
        found = _extract(cand)
        if found is not None:
            return found

    lower = {str(k).lower(): k for k in models.keys()}
    for cand in candidates:
        key = lower.get(cand.lower())
        if key is None:
            continue
        found = _extract(str(key))
        if found is not None:
            return found
    return None


def normalize_parallel_strategy(value: Optional[object]) -> str:
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if not text:
        return "auto"
    aliases = {
        "pipeshard": "pipeline2",
        "pipeline": "pipeline2",
        "shard": "pipeline2",
        "sharding": "pipeline2",
    }
    return aliases.get(text, text)


def resolve_parallel_strategy(
    embedding_cfg: EmbeddingConfig | Mapping[str, object] | object,
    *,
    cuda_device_count: Optional[int] = None,
) -> str:
    requested = _cfg_get(embedding_cfg, "parallel_strategy", None)
    normalized = normalize_parallel_strategy(requested)
    if normalized in {"single", "ddp", "pipeline2"}:
        return normalized
    if normalized != "auto":
        raise ValueError(
            f"Unsupported parallel strategy '{requested}'. "
            "Use one of: auto, single, ddp, pipeline2."
        )

    if cuda_device_count is None:
        cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if cuda_device_count < 2:
        return "single"

    # Backward-compatible escape hatch: explicit low-level knobs override auto.
    if _cfg_get(embedding_cfg, "device_map", None):
        return "pipeline2"
    if bool(_cfg_get(embedding_cfg, "data_parallel", False)) or bool(
        _cfg_get(embedding_cfg, "multi_process", False)
    ):
        return "ddp"

    embedding_type = str(_cfg_get(embedding_cfg, "type", "") or "")
    model_name = _cfg_get(embedding_cfg, "model_name", None)
    raw_size = lookup_raw_size_value(None if model_name is None else str(model_name))
    if (
        embedding_type == "hf_transformer"
        and raw_size is not None
        and raw_size > RAW_SIZE_PIPELINE2_THRESHOLD
    ):
        return "pipeline2"
    return "ddp"
