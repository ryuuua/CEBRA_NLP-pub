from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, TypedDict

import numpy as np
import torch


ResolvedMaxMemoryKey = int | str
ResolvedMaxMemoryValue = int | str
ResolvedMaxMemoryMap = Dict[ResolvedMaxMemoryKey, ResolvedMaxMemoryValue]


class HFPoolingCapabilities(TypedDict):
    transformer_family: str
    cls_token_id: Optional[int]
    supports_cls_pooling: bool
    supported_poolings: list[str]


def _mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply attention-mask-aware mean pooling to a hidden state tensor."""
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _cls_pool(
    hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    cls_token_id: Optional[int],
    model_name: str,
) -> torch.Tensor:
    """Pool using the hidden state at the CLS token position (usually position 0)."""
    if cls_token_id is None:
        raise ValueError(
            f"Requested pooling='cls' but tokenizer for model '{model_name}' has no cls_token_id."
        )
    matches = input_ids.eq(int(cls_token_id))
    has_cls = matches.any(dim=1)
    if not bool(has_cls.all().item()):
        raise ValueError(
            f"Requested pooling='cls' but some inputs for model '{model_name}' are missing the CLS token."
        )
    cls_positions = matches.int().argmax(dim=1)
    batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_indices, cls_positions, :]


def _normalize_pooling(value: Optional[str]) -> str:
    if value is None:
        return "mean"
    normalized = str(value).strip().lower()
    if normalized in {"mean", "masked_mean", "avg", "average"}:
        return "mean"
    if normalized in {"cls", "cls_token"}:
        return "cls"
    raise ValueError(f"Unsupported pooling method: {value}")


def _is_embedding_l2_normalized(embedding_cfg: object) -> bool:
    for key in ("normalize_embeddings", "normalize", "l2_normalize"):
        raw = getattr(embedding_cfg, key, None)
        if raw is None:
            continue
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"1", "true", "yes", "y", "t", "on"}:
            return True
        if text in {"0", "false", "no", "n", "f", "off", ""}:
            return False
        return bool(raw)
    return False


def _l2_normalize_embeddings(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 2:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return values / norms
    if values.ndim == 3:
        norms = np.linalg.norm(values, axis=2, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return values / norms
    raise ValueError(
        f"L2 normalization expects a 2D/3D embedding tensor, got shape={values.shape}."
    )


def _unwrap_model(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def _infer_hf_transformer_family(model) -> str:
    base_model = _unwrap_model(model)
    config = getattr(base_model, "config", None)
    if config is None:
        return "unknown"
    if bool(getattr(config, "is_encoder_decoder", False)):
        return "encoder_decoder"
    if bool(getattr(config, "is_decoder", False)):
        return "decoder_only"
    return "encoder"


def _infer_hf_pooling_capabilities(model, tokenizer) -> HFPoolingCapabilities:
    family = _infer_hf_transformer_family(model)
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    supports_cls = cls_token_id is not None and family != "decoder_only"
    supported_poolings = ["mean"]
    if supports_cls:
        supported_poolings.append("cls")
    return {
        "transformer_family": family,
        "cls_token_id": cls_token_id,
        "supports_cls_pooling": supports_cls,
        "supported_poolings": supported_poolings,
    }


def _validate_hf_pooling_choice(
    requested_pooling: str, capabilities: HFPoolingCapabilities, model_name: str
) -> str:
    supported = capabilities["supported_poolings"]
    if requested_pooling in supported:
        return requested_pooling
    raise ValueError(
        f"Requested pooling='{requested_pooling}' is not supported for model '{model_name}'. "
        f"Inferred transformer_family={capabilities['transformer_family']}, "
        f"cls_token_id={capabilities['cls_token_id']}, supported_poolings={supported}."
    )


def _validate_pooling_for_embedding_type(embedding_cfg) -> None:
    embedding_type = getattr(embedding_cfg, "type", None)
    requested_pooling = _normalize_pooling(getattr(embedding_cfg, "pooling", None))
    if embedding_type != "hf_transformer" and requested_pooling != "mean":
        raise ValueError(
            f"embedding.pooling='{requested_pooling}' is not configurable for embedding.type='{embedding_type}'. "
            "Only hf_transformer supports pooling override; sentence_transformer uses backend-defined pooling."
        )


def _select_layer(
    hidden_states: Sequence[torch.Tensor], layer_index: int, model_name: str
) -> torch.Tensor:
    """Return the desired hidden state layer, supporting negative indices."""
    total_layers = len(hidden_states)
    if layer_index < 0:
        layer_index += total_layers
    if layer_index < 0 or layer_index >= total_layers:
        raise ValueError(
            f"Layer index {layer_index} is out of bounds for model '{model_name}' "
            f"which exposes {total_layers} hidden states."
        )
    return hidden_states[layer_index]


def _resolve_torch_dtype(dtype: Optional[str]):
    if dtype is None:
        return None
    normalized = str(dtype).strip().lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"Unsupported torch_dtype: {dtype}")


def _ensure_tokenizer_pad_token(tokenizer) -> None:
    if getattr(tokenizer, "pad_token", None) is not None:
        return

    fallback_token = None
    for attr in ("eos_token", "sep_token", "cls_token", "bos_token"):
        token = getattr(tokenizer, attr, None)
        if token is None:
            continue
        fallback_token = token
        break
    if fallback_token is None:
        return

    tokenizer.pad_token = fallback_token
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    if not hasattr(tokenizer, "convert_tokens_to_ids"):
        return

    pad_token_id = tokenizer.convert_tokens_to_ids(fallback_token)
    if pad_token_id is None:
        return
    tokenizer.pad_token_id = pad_token_id


def _resolve_requested_to_args(args, kwargs) -> tuple[object | None, torch.dtype | None]:
    requested_device = None
    requested_dtype = None
    if args:
        first = args[0]
        if isinstance(first, (str, torch.device)):
            requested_device = first
        elif isinstance(first, torch.dtype):
            requested_dtype = first
    if "device" in kwargs:
        requested_device = kwargs.get("device")
    if "dtype" in kwargs:
        requested_dtype = kwargs.get("dtype")
    return requested_device, requested_dtype


def _parse_max_memory(
    value: Optional[object],
) -> Optional[ResolvedMaxMemoryMap]:
    if value is None:
        return None

    def _normalize_key(raw_key: object) -> ResolvedMaxMemoryKey:
        if isinstance(raw_key, int):
            return raw_key
        if isinstance(raw_key, str):
            cleaned = raw_key.strip().strip("'\"")
            if cleaned.isdigit():
                return int(cleaned)
            return cleaned
        return str(raw_key)

    def _normalize_value(raw_val: object) -> ResolvedMaxMemoryValue:
        if isinstance(raw_val, (int, float)):
            return int(raw_val)
        if isinstance(raw_val, str):
            cleaned = raw_val.strip().strip("'\"")
            if cleaned.isdigit():
                return int(cleaned)
            return cleaned
        return str(raw_val)

    if isinstance(value, Mapping):
        parsed: ResolvedMaxMemoryMap = {}
        for key, val in value.items():
            parsed[_normalize_key(key)] = _normalize_value(val)
        return parsed
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "auto":
        return None
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
        if not text:
            return None
    items = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    max_memory: ResolvedMaxMemoryMap = {}
    for item in items:
        if ":" in item:
            key, val = item.split(":", 1)
        elif "=" in item:
            key, val = item.split("=", 1)
        else:
            raise ValueError(
                "max_memory must be a dict or a string like '0:20GiB,1:20GiB,cpu:64GiB'."
            )
        cleaned_key = _normalize_key(key)
        cleaned_val = _normalize_value(val)
        max_memory[cleaned_key] = cleaned_val
    return max_memory


def _resolve_input_device(model) -> Optional[torch.device]:
    if isinstance(model, torch.nn.DataParallel):
        if model.device_ids:
            return torch.device(f"cuda:{model.device_ids[0]}")
    if hasattr(model, "hf_device_map"):
        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, Mapping) and hf_device_map:
            first = next(iter(hf_device_map.values()))
            if isinstance(first, int):
                return torch.device(f"cuda:{first}")
            if isinstance(first, str):
                return torch.device(first)
    if hasattr(model, "device"):
        return model.device
    return None
