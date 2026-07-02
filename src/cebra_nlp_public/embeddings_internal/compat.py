from __future__ import annotations

from typing import Optional

import torch


def _resolve_hf_access_error_type() -> Optional[type]:
    candidates = (
        ("huggingface_hub.errors", "GatedRepoError"),
        ("huggingface_hub.utils", "GatedRepoError"),
        ("huggingface_hub.utils._errors", "GatedRepoError"),
    )
    for module_name, attr_name in candidates:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            error_type = getattr(module, attr_name, None)
            if isinstance(error_type, type) and issubclass(error_type, Exception):
                return error_type
        except Exception:
            continue
    return None


_HF_ACCESS_ERROR = _resolve_hf_access_error_type()


def _is_hf_model_load_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        return True
    return _HF_ACCESS_ERROR is not None and isinstance(exc, _HF_ACCESS_ERROR)


def _ensure_transformers_onnx_config() -> None:
    """
    Ensure `from transformers.onnx import OnnxConfig` succeeds.

    Some Hugging Face models loaded with `trust_remote_code=True` import OnnxConfig
    even when ONNX export is not used (e.g. `jinaai/jina-bert-implementation`).
    Certain environments ship a Transformers build without the `transformers.onnx`
    package, which breaks embedding generation. For our use case, a minimal stub
    base class is sufficient.
    """
    import importlib
    import sys
    import types

    try:
        importlib.import_module("transformers.onnx")
        return
    except Exception:
        try:
            import transformers  # noqa: F401
        except Exception:
            raise

    shim = types.ModuleType("transformers.onnx")
    setattr(shim, "__path__", [])

    class OnnxConfig:  # noqa: D401
        """Compatibility stub for environments missing `transformers.onnx`."""

        pass

    setattr(shim, "OnnxConfig", OnnxConfig)
    setattr(shim, "OnnxConfigWithPast", OnnxConfig)
    setattr(shim, "OnnxSeq2SeqConfigWithPast", OnnxConfig)

    config_shim = types.ModuleType("transformers.onnx.config")
    setattr(config_shim, "OnnxConfig", OnnxConfig)
    setattr(config_shim, "OnnxConfigWithPast", OnnxConfig)
    setattr(config_shim, "OnnxSeq2SeqConfigWithPast", OnnxConfig)
    setattr(shim, "config", config_shim)

    sys.modules["transformers.onnx"] = shim
    sys.modules["transformers.onnx.config"] = config_shim
    transformers_module = None
    try:
        import transformers as transformers_module
    except Exception:
        transformers_module = None
    if transformers_module is not None:
        setattr(transformers_module, "onnx", shim)

    print("[WARN] `transformers.onnx` is missing/unimportable; installed a minimal OnnxConfig shim.")


def _ensure_transformers_pytorch_utils_compat() -> None:
    has_existing_find_pruneable_heads = False
    try:
        from transformers.pytorch_utils import (
            find_pruneable_heads_and_indices as _existing_find_pruneable_heads_and_indices,  # noqa: F401
        )
        _ = _existing_find_pruneable_heads_and_indices
        has_existing_find_pruneable_heads = True
    except Exception:
        has_existing_find_pruneable_heads = False
    if has_existing_find_pruneable_heads:
        return

    try:
        from transformers import pytorch_utils
    except Exception:
        return

    if hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        return

    def _compat_find_pruneable_heads_and_indices(
        heads,
        n_heads: int,
        head_size: int,
        already_pruned_heads,
    ) -> tuple[set, torch.Tensor]:
        heads = set(heads) - set(already_pruned_heads)
        mask = torch.ones(n_heads, head_size)
        for head in heads:
            offset = sum(
                1 if pruned_head < head else 0 for pruned_head in already_pruned_heads
            )
            normalized_head = int(head) - offset
            if normalized_head < 0 or normalized_head >= n_heads:
                continue
            mask[normalized_head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(mask.numel())[mask].long()
        return heads, index

    setattr(
        pytorch_utils,
        "find_pruneable_heads_and_indices",
        _compat_find_pruneable_heads_and_indices,
    )
    print(
        "[WARN] `transformers.pytorch_utils.find_pruneable_heads_and_indices` is missing; installed compatibility shim."
    )


def _ensure_transformers_xlm_roberta_compat() -> None:
    has_existing_create_position_ids = False
    try:
        from transformers.models.xlm_roberta.modeling_xlm_roberta import (
            create_position_ids_from_input_ids as _create_position_ids_from_input_ids,  # noqa: F401
        )
        _ = _create_position_ids_from_input_ids
        has_existing_create_position_ids = True
    except Exception:
        has_existing_create_position_ids = False
    if has_existing_create_position_ids:
        return

    try:
        from transformers.models.xlm_roberta import modeling_xlm_roberta
    except Exception:
        return

    if hasattr(modeling_xlm_roberta, "create_position_ids_from_input_ids"):
        return

    try:
        from transformers.models.roberta.modeling_roberta import (
            create_position_ids_from_input_ids as _create_position_ids_from_input_ids,
        )
    except Exception:
        def _create_position_ids_from_input_ids(
            input_ids: torch.Tensor,
            padding_idx: int,
            past_key_values_length: int = 0,
        ) -> torch.Tensor:
            mask = input_ids.ne(padding_idx).int()
            incremental_indices = (
                torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
            ) * mask
            return incremental_indices.long() + int(padding_idx)

    setattr(
        modeling_xlm_roberta,
        "create_position_ids_from_input_ids",
        _create_position_ids_from_input_ids,
    )
    print(
        "[WARN] `transformers.models.xlm_roberta.modeling_xlm_roberta.create_position_ids_from_input_ids` is missing; installed compatibility shim."
    )


def _ensure_transformers_remote_code_compat() -> None:
    _ensure_transformers_onnx_config()
    _ensure_transformers_pytorch_utils_compat()
    _ensure_transformers_xlm_roberta_compat()


def _ensure_torch_dynamo_compat() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None or hasattr(dynamo, "mark_static_address"):
        return

    def _mark_static_address(*_args, **_kwargs):
        return None

    setattr(dynamo, "mark_static_address", _mark_static_address)
    print(
        "[WARN] `torch._dynamo.mark_static_address` is missing; installed compatibility shim."
    )
