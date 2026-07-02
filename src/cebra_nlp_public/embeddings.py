import numpy as np
import torch
import inspect
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

from .config_schema import AppConfig, EmbeddingConfig  # noqa: F401
from .cache_metadata import build_cache_metadata as _build_shared_cache_metadata
from .cache_store import (
    load_text_embedding,
    resolve_cache_with_index,
    save_text_embedding,
)
from .cache_utils import get_embedding_cache_path
from .embedding_standards import (
    enforce_registry_key_on_cache,
    enforce_rulebook_id_on_cache,
    get_rulebook_id,
    registry_keys_equivalent,
    resolve_embedding_registry_key,
    validate_embedding_registry,
)
from .embedding_parallel import resolve_parallel_strategy
from .embeddings_internal.common import (
    HFPoolingCapabilities,  # noqa: F401
    ResolvedMaxMemoryKey,  # noqa: F401
    ResolvedMaxMemoryMap,
    ResolvedMaxMemoryValue,  # noqa: F401
    _cls_pool,
    _ensure_tokenizer_pad_token,
    _infer_hf_pooling_capabilities,
    _infer_hf_transformer_family,  # noqa: F401
    _is_embedding_l2_normalized,
    _l2_normalize_embeddings,
    _mean_pool,
    _normalize_pooling,
    _parse_max_memory,
    _resolve_input_device,
    _resolve_requested_to_args,
    _resolve_torch_dtype,
    _select_layer,
    _unwrap_model,  # noqa: F401
    _validate_hf_pooling_choice,
    _validate_pooling_for_embedding_type,
)
from .embeddings_internal.compat import (
    _HF_ACCESS_ERROR,  # noqa: F401
    _ensure_torch_dynamo_compat,
    _ensure_transformers_onnx_config,  # noqa: F401
    _ensure_transformers_pytorch_utils_compat,  # noqa: F401
    _ensure_transformers_remote_code_compat,
    _ensure_transformers_xlm_roberta_compat,  # noqa: F401
    _is_hf_model_load_error,
    _resolve_hf_access_error_type,  # noqa: F401
)

from typing import Optional, Sequence, List, Dict, Mapping, TypedDict, cast

_LAST_LAYER_CACHE: Optional[np.ndarray] = None
_LAST_CACHE_USAGE: "CacheUsageInfo | None" = None


class CacheUsageInfo(TypedDict):
    cache_hit: bool
    cache_path: str
    requested_cache_path: str
    dataset_name: str
    dataset_key: str | None
    embedding_name: str
    embedding_model_name: str | None
    registry_key: str | None
    rulebook_id: str | None
    variant_tag: str | None


def _set_last_cache_usage(info: CacheUsageInfo | None) -> None:
    global _LAST_CACHE_USAGE
    _LAST_CACHE_USAGE = info


def get_last_cache_usage() -> CacheUsageInfo | None:
    if _LAST_CACHE_USAGE is None:
        return None
    return cast(CacheUsageInfo, dict(_LAST_CACHE_USAGE))


def get_last_hidden_state_cache() -> Optional[np.ndarray]:
    """Return the cached pooled hidden states for all layers from the most recent transformer run."""
    return _LAST_LAYER_CACHE


def clear_last_hidden_state_cache() -> None:
    """Reset the cached transformer hidden states."""
    global _LAST_LAYER_CACHE
    _LAST_LAYER_CACHE = None


def _validate_embedding_standards(embedding_cfg) -> Optional[str]:
    _validate_pooling_for_embedding_type(embedding_cfg)
    return validate_embedding_registry(embedding_cfg)


def _dist_state() -> tuple[bool, int, int]:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return False, 0, 1
    return True, int(dist.get_rank()), int(dist.get_world_size())


def _configure_sentence_transformer_ddp_fallback(embedding_cfg, *, has_multi_gpu: bool) -> None:
    enable_multi_process = bool(has_multi_gpu)
    embedding_cfg.multi_process = enable_multi_process
    should_assign_default_devices = (
        enable_multi_process
        and getattr(embedding_cfg, "multi_process_devices", None) is None
    )
    if should_assign_default_devices:
        embedding_cfg.multi_process_devices = [0, 1]


def _apply_parallel_strategy(embedding_cfg) -> str:
    strategy = resolve_parallel_strategy(embedding_cfg)
    dist_active, _rank, _world_size = _dist_state()

    embedding_type = str(getattr(embedding_cfg, "type", "") or "")
    has_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1

    if strategy == "single":
        embedding_cfg.device_map = None
        embedding_cfg.data_parallel = False
        embedding_cfg.multi_process = False
        return strategy

    if strategy == "pipeline2":
        if embedding_type not in {"hf_transformer", "sentence_transformer"}:
            raise ValueError(
                "parallel_strategy='pipeline2' is only supported for embedding.type='hf_transformer' or "
                f"'sentence_transformer' (got '{embedding_type}')."
            )
        if dist_active:
            raise RuntimeError(
                "parallel_strategy='pipeline2' requires a single process. "
                "Run without torchrun/DDP (WORLD_SIZE=1), "
                "or pass embedding.parallel_strategy=ddp."
            )
        if getattr(embedding_cfg, "device_map", None) is None:
            embedding_cfg.device_map = "auto"
        embedding_cfg.data_parallel = False
        embedding_cfg.multi_process = False
        embedding_cfg.multi_process_devices = None
        return strategy

    if strategy == "ddp":
        if dist_active:
            embedding_cfg.device_map = None
            embedding_cfg.data_parallel = False
            embedding_cfg.multi_process = False
            return strategy

        # world_size == 1 fallback: use single-process multi-GPU primitives.
        if embedding_type == "hf_transformer":
            embedding_cfg.device_map = None
            embedding_cfg.data_parallel = bool(has_multi_gpu)
        elif embedding_type == "sentence_transformer":
            _configure_sentence_transformer_ddp_fallback(
                embedding_cfg, has_multi_gpu=has_multi_gpu
            )
        return strategy

    raise AssertionError(f"Unhandled parallel strategy: {strategy}")


def resolve_layer_index(layer_count: int, requested: Optional[int]) -> int:
    """
    Resolve the requested hidden state layer index into a non-negative integer.

    Parameters
    ----------
    layer_count : int
        Total number of layers available in the cached tensor.
    requested : Optional[int]
        The layer index specified in the configuration (can be negative or None).
        When None, the final hidden state is selected.
    """
    if layer_count <= 0:
        raise ValueError("Layer cache is empty; cannot select a hidden state layer.")
    index = layer_count - 1 if requested is None else requested
    if index < 0:
        index += layer_count
    if index < 0 or index >= layer_count:
        raise ValueError(
            f"Layer index {requested} is out of bounds for cached tensor with "
            f"{layer_count} layers."
        )
    return index


def get_hf_transformer_embeddings(
    texts,
    model_name,
    device,
    *,
    layer_index: Optional[int] = None,
    pooling: str = "mean",
    trust_remote_code: bool = False,
    device_map: Optional[str] = None,
    max_memory: Optional[ResolvedMaxMemoryMap] = None,
    torch_dtype: Optional[str] = None,
    data_parallel: bool = False,
    batch_size: int = 32,
    cache_all_layers: bool = False,
):
    """Generates embeddings using a Hugging Face Transformer."""
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

    global _LAST_LAYER_CACHE

    def _is_fast_tokenizer_parse_error(exc: Exception) -> bool:
        message = str(exc)
        return "untagged enum ModelWrapper" in message or "TokenizerFast.from_file" in message

    def _is_unrecognized_architecture_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            ("does not recognize this architecture" in message and "model type" in message)
            or ("Unrecognized configuration class" in message and "AutoModel" in message)
        )

    def _raise_access_error(exc: Exception) -> None:
        guidance = (
            f"Unable to download Hugging Face model '{model_name}'. "
            "Check that the model id is public and available, or update your configuration "
            "to use a public embedding model."
        )
        raise RuntimeError(f"{guidance} Original error: {exc}") from exc

    if trust_remote_code:
        _ensure_transformers_remote_code_compat()
    _ensure_torch_dynamo_compat()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        if _is_hf_model_load_error(exc):
            _raise_access_error(exc)
        if not _is_fast_tokenizer_parse_error(exc):
            raise
        print(
            f"[WARN] Fast tokenizer load failed for '{model_name}' ({exc}). "
            "Retrying with use_fast=False."
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )
        except Exception as retry_exc:
            if _is_hf_model_load_error(retry_exc):
                _raise_access_error(retry_exc)
            raise

    _ensure_tokenizer_pad_token(tokenizer)
    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    model_load_kwargs: Dict[str, object] = {}
    if resolved_dtype is not None:
        model_load_kwargs["torch_dtype"] = resolved_dtype

    def _load_model(resolved_trust_remote_code: bool):
        if device_map:
            return AutoModel.from_pretrained(
                model_name,
                trust_remote_code=resolved_trust_remote_code,
                device_map=device_map,
                max_memory=max_memory,
                **model_load_kwargs,
            )
        loaded = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=resolved_trust_remote_code,
            **model_load_kwargs,
        ).to(device)
        if data_parallel and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                loaded = torch.nn.DataParallel(loaded)
        return loaded

    def _load_causal_lm_model(resolved_trust_remote_code: bool):
        if device_map:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=resolved_trust_remote_code,
                device_map=device_map,
                max_memory=max_memory,
                **model_load_kwargs,
            )
        loaded = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=resolved_trust_remote_code,
            **model_load_kwargs,
        ).to(device)
        if data_parallel and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                loaded = torch.nn.DataParallel(loaded)
        return loaded

    loaded_with_causal_lm = False
    try:
        model = _load_model(trust_remote_code)
    except ValueError as exc:
        if _is_unrecognized_architecture_error(exc):
            if not trust_remote_code:
                print(
                    f"[WARN] Transformers could not resolve architecture for '{model_name}' ({exc}). "
                    "Retrying with trust_remote_code=True."
                )
                _ensure_transformers_remote_code_compat()
                try:
                    model = _load_model(True)
                except Exception as retry_exc:
                    if _is_hf_model_load_error(retry_exc):
                        _raise_access_error(retry_exc)
                    if not isinstance(retry_exc, ValueError):
                        raise
                    if not _is_unrecognized_architecture_error(retry_exc):
                        raise
                    print(
                        f"[WARN] AutoModel is not available for '{model_name}' ({retry_exc}). "
                        "Falling back to AutoModelForCausalLM."
                    )
                    model = _load_causal_lm_model(True)
                    loaded_with_causal_lm = True
                else:
                    pass
            else:
                print(
                    f"[WARN] AutoModel is not available for '{model_name}' ({exc}). "
                    "Falling back to AutoModelForCausalLM."
                )
                model = _load_causal_lm_model(True)
                loaded_with_causal_lm = True
        else:
            raise
    except Exception as exc:
        if _is_hf_model_load_error(exc):
            _raise_access_error(exc)
        if _is_unrecognized_architecture_error(exc):
            print(
                f"[WARN] AutoModel is not available for '{model_name}' ({exc}). "
                "Falling back to AutoModelForCausalLM."
            )
            model = _load_causal_lm_model(True)
            loaded_with_causal_lm = True
        else:
            raise
    model.eval()

    resolved_layer_index = -1 if layer_index is None else int(layer_index)
    need_hidden_states = cache_all_layers or resolved_layer_index != -1 or loaded_with_causal_lm

    embeddings_out: Optional[np.ndarray] = None
    layer_cache: Optional[np.ndarray] = None
    requested_pooling = _normalize_pooling(pooling)
    capabilities = _infer_hf_pooling_capabilities(model, tokenizer)
    resolved_pooling = _validate_hf_pooling_choice(
        requested_pooling, capabilities, model_name
    )

    input_device = _resolve_input_device(model)
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Vectorizing with {model_name}"):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            if input_device is not None:
                inputs = inputs.to(input_device)
            outputs = model(**inputs, output_hidden_states=need_hidden_states)
            attention_mask = inputs["attention_mask"]
            hidden_states = outputs.hidden_states if need_hidden_states else None

            if resolved_layer_index == -1:
                hidden_state = getattr(outputs, "last_hidden_state", None)
                if hidden_state is None:
                    if hidden_states is None:
                        raise RuntimeError(
                            f"Model '{model_name}' did not provide last_hidden_state/hidden_states for pooling."
                        )
                    hidden_state = hidden_states[-1]
            else:
                if hidden_states is None:
                    raise RuntimeError(
                        f"Model '{model_name}' did not return hidden states even though "
                        "one was requested."
                    )
                hidden_state = _select_layer(hidden_states, resolved_layer_index, model_name)
            if resolved_pooling == "mean":
                pooled_selected = _mean_pool(hidden_state, attention_mask)
            elif resolved_pooling == "cls":
                pooled_selected = _cls_pool(
                    hidden_state,
                    inputs["input_ids"],
                    getattr(tokenizer, "cls_token_id", None),
                    model_name,
                )
            else:
                raise AssertionError(f"Unhandled pooling method: {resolved_pooling}")
            pooled_selected = pooled_selected.to(dtype=torch.float32)
            batch_embeddings = pooled_selected.cpu().numpy().astype(np.float32, copy=False)

            if embeddings_out is None:
                embeddings_out = np.empty(
                    (len(texts), batch_embeddings.shape[1]), dtype=np.float32
                )
            end = i + batch_embeddings.shape[0]
            embeddings_out[i:end] = batch_embeddings

            if cache_all_layers:
                if hidden_states is None:
                    raise RuntimeError(
                        f"Model '{model_name}' did not return hidden states even though "
                        "cache_all_layers=True was requested."
                    )
                if layer_cache is None:
                    layer_cache = np.empty(
                        (len(texts), len(hidden_states), batch_embeddings.shape[1]),
                        dtype=np.float32,
                    )
                for idx, state in enumerate(hidden_states):
                    if resolved_pooling == "mean":
                        pooled_tensor = _mean_pool(state, attention_mask)
                    elif resolved_pooling == "cls":
                        pooled_tensor = _cls_pool(
                            state,
                            inputs["input_ids"],
                            getattr(tokenizer, "cls_token_id", None),
                            model_name,
                        )
                    else:
                        raise AssertionError(
                            f"Unhandled pooling method: {resolved_pooling}"
                        )
                    pooled_tensor = pooled_tensor.to(dtype=torch.float32)
                    pooled = pooled_tensor.cpu().numpy().astype(np.float32, copy=False)
                    layer_cache[i:end, idx, :] = pooled

    _LAST_LAYER_CACHE = layer_cache
    if embeddings_out is None:
        return np.empty((0, 0), dtype=np.float32)
    return embeddings_out


def get_sentence_transformer_embeddings(
    texts,
    model_name,
    device,
    *,
    batch_size: int = 32,
    multi_process: bool = False,
    multi_process_devices: Optional[List[int]] = None,
    trust_remote_code: bool = False,
    device_map: Optional[str] = None,
    max_memory: Optional[ResolvedMaxMemoryMap] = None,
    torch_dtype: Optional[str] = None,
):
    """Generates embeddings using the SentenceTransformers library."""
    from sentence_transformers import SentenceTransformer

    clear_last_hidden_state_cache()
    _ensure_torch_dynamo_compat()
    device_map_text = None if device_map is None else str(device_map).strip()
    use_device_map = bool(device_map_text) and (
        str(device_map_text).lower() not in {"none", "null"}
    )
    supports_model_kwargs = False
    try:
        supports_model_kwargs = "model_kwargs" in inspect.signature(
            SentenceTransformer.__init__
        ).parameters
    except (TypeError, ValueError):
        supports_model_kwargs = False

    if use_device_map and not supports_model_kwargs:
        print(
            "[WARN] sentence-transformers runtime does not support "
            "SentenceTransformer(..., model_kwargs=...). "
            "Falling back to non-sharded SentenceTransformer loading."
        )
        use_device_map = False

    def _normalize_multi_process_target_devices(
        raw_devices: Optional[Sequence[object]],
    ) -> Optional[List[str]]:
        if raw_devices is None:
            return None
        normalized: List[str] = []
        for raw in raw_devices:
            if raw is None:
                continue
            if isinstance(raw, int):
                normalized.append(f"cuda:{raw}")
                continue
            cleaned = str(raw).strip().strip("'\"")
            if not cleaned:
                continue
            if cleaned.isdigit():
                normalized.append(f"cuda:{int(cleaned)}")
                continue
            normalized.append(cleaned)
        return normalized

    def _raise_access_error(exc: Exception) -> None:
        guidance = (
            f"Unable to download Hugging Face model '{model_name}'. "
            "Check that the model id is public and available, or update your configuration "
            "to use a public embedding model."
        )
        raise RuntimeError(f"{guidance} Original error: {exc}") from exc

    def _is_jina_v3_low_cpu_mem_usage_sort_error(exc: Exception) -> bool:
        message = str(exc)
        if "'<' not supported between instances of 'str' and 'int'" not in message:
            return False
        lowered = str(model_name).lower()
        return "jina" in lowered and "v3" in lowered

    if trust_remote_code:
        _ensure_transformers_remote_code_compat()

    dtype = _resolve_torch_dtype(torch_dtype)
    if multi_process and dtype == torch.bfloat16:
        print(
            "[WARN] multi_process SentenceTransformer does not safely support bfloat16 "
            "in this runtime; falling back to float32 to avoid worker crashes."
        )
        dtype = torch.float32
    force_cpu_init_for_dtype = (
        not supports_model_kwargs and dtype is not None and not use_device_map and not multi_process
    )
    if force_cpu_init_for_dtype:
        print(
            "[WARN] sentence-transformers runtime does not support torch_dtype during load. "
            "Loading on CPU first, casting dtype, then moving to target device."
        )

    model_kwargs: Dict[str, object] = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if use_device_map:
        model_kwargs["device_map"] = str(device_map_text)
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

    def _load_sentence_transformer(load_kwargs: Optional[Dict[str, object]] = None):
        init_kwargs: Dict[str, object] = {
            "device": None if use_device_map and not multi_process else (
                "cpu" if (multi_process or force_cpu_init_for_dtype) else device
            ),
            "trust_remote_code": trust_remote_code,
        }
        if supports_model_kwargs:
            init_kwargs["model_kwargs"] = load_kwargs or None
        elif load_kwargs:
            ignored_keys = ", ".join(sorted(load_kwargs.keys()))
            print(
                "[WARN] Ignoring SentenceTransformer load kwargs unsupported by this "
                f"sentence-transformers version: {ignored_keys}"
            )
        return SentenceTransformer(model_name, **init_kwargs)

    try:
        model = _load_sentence_transformer(model_kwargs)
    except TypeError as exc:
        if not _is_jina_v3_low_cpu_mem_usage_sort_error(exc):
            if _is_hf_model_load_error(exc):
                _raise_access_error(exc)
            raise
        retry_kwargs = dict(model_kwargs)
        retry_kwargs["low_cpu_mem_usage"] = False
        print(
            f"[WARN] Retrying SentenceTransformer load for '{model_name}' with "
            "low_cpu_mem_usage=False due to transformers core_model_loading sort error."
        )
        try:
            model = _load_sentence_transformer(retry_kwargs)
        except Exception as retry_exc:
            if _is_hf_model_load_error(retry_exc):
                _raise_access_error(retry_exc)
            raise
    except Exception as exc:
        if _is_hf_model_load_error(exc):
            _raise_access_error(exc)
        raise

    if dtype is not None and not use_device_map:
        model = model.to(dtype)
        if force_cpu_init_for_dtype:
            model = model.to(device)

    if multi_process:
        target_devices = _normalize_multi_process_target_devices(multi_process_devices)
        if target_devices is None:
            pool = model.start_multi_process_pool()
        else:
            if not target_devices:
                raise ValueError("multi_process_devices is empty; cannot start pool.")
            pool = model.start_multi_process_pool(target_devices=target_devices)
        try:
            encode_kwargs: Dict[str, object] = {"batch_size": batch_size}
            try:
                supports_show_progress_bar = "show_progress_bar" in inspect.signature(
                    model.encode_multi_process
                ).parameters
            except (TypeError, ValueError):
                supports_show_progress_bar = False
            if supports_show_progress_bar:
                encode_kwargs["show_progress_bar"] = True
            return model.encode_multi_process(texts, pool, **encode_kwargs)
        finally:
            model.stop_multi_process_pool(pool)

    if use_device_map:
        transformers_model = getattr(model, "transformers_model", None)
        input_device = None if transformers_model is None else _resolve_input_device(transformers_model)
        if input_device is None:
            try:
                input_device = torch.device(device)
            except Exception:
                input_device = None
        if input_device is not None:
            for name, child in getattr(model, "_modules", {}).items():
                auto_model = getattr(child, "auto_model", None)
                if auto_model is not None and hasattr(auto_model, "hf_device_map"):
                    continue
                try:
                    child.to(input_device)
                except Exception as exc:
                    print(
                        f"[WARN] Failed to move SentenceTransformer submodule '{name}' to {input_device}: {exc}"
                    )

        def _patched_to(*args, _original_to=model.to, **kwargs):
            requested_device, requested_dtype = _resolve_requested_to_args(args, kwargs)
            if requested_device is None:
                return _original_to(*args, **kwargs)
            if requested_dtype is None:
                return model
            try:
                return _original_to(requested_dtype)
            except Exception:
                return model

        model.to = _patched_to  # type: ignore[method-assign]

    return model.encode(texts, show_progress_bar=True, batch_size=batch_size)


def get_embeddings(texts: list, cfg: AppConfig) -> np.ndarray:
    """Factory function to select and run the appropriate embedding model."""
    clear_last_hidden_state_cache()
    emb_cfg = cfg.embedding
    _validate_embedding_standards(emb_cfg)
    strategy = _apply_parallel_strategy(emb_cfg)
    dist_active, rank, _world_size = _dist_state()
    if not dist_active or rank == 0:
        print(f"[embedding.parallel] strategy={strategy}")
    print(f"\n--- Generating embeddings using model: {emb_cfg.name} ---")

    vectors: np.ndarray
    if emb_cfg.type == "hf_transformer":
        # AppConfigとして扱われることで、cfg.deviceに正しくアクセスできる
        max_memory = _parse_max_memory(getattr(emb_cfg, "max_memory", None))
        vectors = get_hf_transformer_embeddings(
            texts,
            emb_cfg.model_name,
            cfg.device,
            layer_index=emb_cfg.hidden_state_layer,
            pooling=getattr(emb_cfg, "pooling", "mean"),
            trust_remote_code=emb_cfg.trust_remote_code,
            device_map=getattr(emb_cfg, "device_map", None),
            max_memory=max_memory,
            torch_dtype=getattr(emb_cfg, "torch_dtype", None),
            data_parallel=getattr(emb_cfg, "data_parallel", False),
            batch_size=getattr(emb_cfg, "batch_size", 32),
            cache_all_layers=getattr(emb_cfg, "cache_all_layers", False),
        )
    elif emb_cfg.type == "sentence_transformer":
        vectors = get_sentence_transformer_embeddings(
            texts,
            emb_cfg.model_name,
            cfg.device,
            batch_size=getattr(emb_cfg, "batch_size", 32),
            multi_process=getattr(emb_cfg, "multi_process", False),
            multi_process_devices=getattr(emb_cfg, "multi_process_devices", None),
            trust_remote_code=getattr(emb_cfg, "trust_remote_code", False),
            device_map=getattr(emb_cfg, "device_map", None),
            max_memory=_parse_max_memory(getattr(emb_cfg, "max_memory", None)),
            torch_dtype=getattr(emb_cfg, "torch_dtype", None),
        )
    else:
        raise ValueError(f"Unknown embedding type: {emb_cfg.type}")

    if getattr(emb_cfg, "output_dim", None) in {None, 0} and vectors.ndim == 2:
        emb_cfg.output_dim = int(vectors.shape[1])

    if _is_embedding_l2_normalized(emb_cfg):
        vectors = _l2_normalize_embeddings(vectors)
        cached_layers = get_last_hidden_state_cache()
        if cached_layers is not None:
            global _LAST_LAYER_CACHE
            _LAST_LAYER_CACHE = _l2_normalize_embeddings(cached_layers)

    return vectors


def _resolve_shuffle_seed(cfg: AppConfig) -> Optional[int]:
    dataset_seed = getattr(cfg.dataset, "shuffle_seed", None)
    if dataset_seed is not None:
        return dataset_seed
    eval_cfg = getattr(cfg, "evaluation", None)
    if eval_cfg is not None:
        return getattr(eval_cfg, "random_state", None)
    return None


def _build_effective_cache_metadata(
    cfg: AppConfig,
    ids: Sequence[object] | np.ndarray,
    texts: Sequence[str],
    labels: Sequence[object] | np.ndarray | None,
) -> Dict[str, object] | None:
    normalized_ids: Sequence[object] = (
        ids.tolist() if isinstance(ids, np.ndarray) else ids
    )
    normalized_labels: Sequence[object] | None
    if labels is None:
        normalized_labels = None
    elif isinstance(labels, np.ndarray):
        normalized_labels = labels.tolist()
    else:
        normalized_labels = labels

    cache_metadata = _build_shared_cache_metadata(
        cfg,
        normalized_ids,
        texts,
        labels=normalized_labels,
    )
    if not isinstance(cache_metadata, dict):
        raise RuntimeError("build_cache_metadata() returned a non-dict payload.")

    required = ("dataset_key", "variant_tag", "registry_key", "rulebook_id")
    missing = [field for field in required if cache_metadata.get(field) is None]
    if missing:
        raise RuntimeError(
            "build_cache_metadata() missing required fields: " + ", ".join(missing)
        )
    return cache_metadata


def _resolve_cache_paths(
    cfg: AppConfig,
    cache_metadata: Mapping[str, object] | None,
) -> tuple[Path, list[Path]]:

    dataset_key = None
    variant_tag = None
    if cache_metadata is not None:
        raw_dataset_key = cache_metadata.get("dataset_key")
        if raw_dataset_key is not None and str(raw_dataset_key).strip():
            dataset_key = str(raw_dataset_key).strip()
        raw_variant_tag = cache_metadata.get("variant_tag")
        if raw_variant_tag is not None and str(raw_variant_tag).strip():
            variant_tag = str(raw_variant_tag).strip()

    try:
        preferred = Path(
            get_embedding_cache_path(
                cfg,
                dataset_key=dataset_key,
                variant_tag=variant_tag,
            )
        )
    except TypeError:
        preferred = Path(get_embedding_cache_path(cfg))
    return preferred, [preferred]


def _select_cached_embeddings(
    cache: Optional[tuple],
    ids: Sequence,
    cfg: AppConfig,
    resolved_seed: Optional[int],
    *,
    require_cache: bool,
) -> Optional[np.ndarray]:
    if cache is None:
        if require_cache:
            raise FileNotFoundError("No cached embeddings found.")
        return None

    def _cache_mismatch(
        message: str,
        *,
        error_type: type[Exception] = ValueError,
    ) -> Optional[np.ndarray]:
        if require_cache:
            raise error_type(message)
        print(f"{message} Recomputing...")
        return None

    (
        cached_ids,
        cached_embeddings,
        cached_seed,
        cached_layer_embeddings,
        cached_hidden_state_layer,
        cached_embedding_type,
        cached_pooling,
        cached_rulebook_id,
        cached_registry_key,
    ) = cache
    if cached_seed != resolved_seed:
        message = (
            "Cached embeddings shuffle_seed mismatch "
            f"(cached={cached_seed}, expected={resolved_seed})."
        )
        return _cache_mismatch(message)

    requested_type = getattr(cfg.embedding, "type", None)
    if cached_embedding_type is not None and requested_type != cached_embedding_type:
        message = (
            "Cached embeddings type mismatch "
            f"(cached={cached_embedding_type}, requested={requested_type})."
        )
        return _cache_mismatch(message)

    if requested_type == "hf_transformer":
        requested_pooling = _normalize_pooling(getattr(cfg.embedding, "pooling", None))
        cached_pooling_norm = _normalize_pooling(cached_pooling)
        if cached_pooling_norm != requested_pooling:
            message = (
                "Cached embeddings pooling mismatch "
                f"(cached={cached_pooling_norm}, requested={requested_pooling})."
            )
            return _cache_mismatch(message)

    if enforce_rulebook_id_on_cache():
        current_rulebook_id = get_rulebook_id()
        if cached_rulebook_id is None:
            message = "Cached embeddings are missing rulebook_id metadata."
            return _cache_mismatch(message)
        if cached_rulebook_id != current_rulebook_id:
            message = (
                "Cached embeddings rulebook_id mismatch "
                f"(cached={cached_rulebook_id}, current={current_rulebook_id})."
            )
            return _cache_mismatch(message)

    if enforce_registry_key_on_cache():
        current_registry_key = resolve_embedding_registry_key(cfg.embedding, strict=False)
        if current_registry_key is not None and cached_registry_key is None:
            message = "Cached embeddings are missing registry_key metadata."
            return _cache_mismatch(message)
        if (
            current_registry_key is not None
            and cached_registry_key is not None
            and not registry_keys_equivalent(cached_registry_key, current_registry_key)
        ):
            message = (
                "Cached embeddings registry_key mismatch "
                f"(cached={cached_registry_key}, current={current_registry_key})."
            )
            return _cache_mismatch(message)

    if cfg.embedding.type == "hf_transformer" and cached_layer_embeddings is None:
        requested_layer = getattr(cfg.embedding, "hidden_state_layer", None)
        resolved_requested_layer = -1 if requested_layer is None else int(requested_layer)
        if cached_hidden_state_layer is None and resolved_requested_layer != -1:
            message = (
                "Cached embeddings are missing hidden_state_layer metadata; cannot "
                "guarantee correctness for non-final layers."
            )
            return _cache_mismatch(message)
        if (
            cached_hidden_state_layer is not None
            and int(cached_hidden_state_layer) != resolved_requested_layer
        ):
            message = (
                "Cached embeddings hidden_state_layer mismatch "
                f"(cached={cached_hidden_state_layer}, requested={resolved_requested_layer})."
            )
            return _cache_mismatch(message)

    id_to_index = {str(i): idx for idx, i in enumerate(cached_ids)}
    try:
        selection_indices = np.asarray([id_to_index[str(i)] for i in ids], dtype=int)
        if cfg.embedding.type == "hf_transformer" and cached_layer_embeddings is not None:
            target_layer = resolve_layer_index(
                cached_layer_embeddings.shape[1],
                getattr(cfg.embedding, "hidden_state_layer", None),
            )
            return cached_layer_embeddings[selection_indices, target_layer, :]
        cached = np.asarray(cached_embeddings)
        return cached[selection_indices]
    except KeyError as exc:
        missing_count = 0
        missing_preview: list[str] = []
        for item in ids:
            key = str(item)
            if key not in id_to_index:
                missing_count += 1
                if len(missing_preview) < 10:
                    missing_preview.append(key)
        message = (
            "Cached embeddings are missing required ids "
            f"(requested={len(ids)}, cached={len(cached_ids)}, missing={missing_count}, preview={missing_preview})."
        )
        if require_cache:
            raise KeyError(message) from exc
        print(f"{message} Recomputing...")
        return None
    except ValueError as exc:
        message = f"{exc}"
        if require_cache:
            raise ValueError(message) from exc
        print(f"{message} Recomputing embeddings...")
        return None


def load_or_generate_embeddings(
    cfg: AppConfig,
    texts: Sequence[str],
    ids: Sequence,
    *,
    labels: Sequence | None = None,
    require_cache: bool = False,
) -> np.ndarray:
    """Load cached embeddings when available; otherwise compute and cache them."""
    import torch.distributed as dist

    _set_last_cache_usage(None)
    registry_key = _validate_embedding_standards(cfg.embedding)
    isinstance(registry_key, str)
    cache_metadata = _build_effective_cache_metadata(cfg, ids, texts, labels)
    write_cache_path, _cache_probe_paths = _resolve_cache_paths(cfg, cache_metadata)

    load_layer_embeddings = bool(getattr(cfg.embedding, "cache_all_layers", False))
    cache = None
    loaded_cache_path = None
    read_cache_path = write_cache_path
    if cache_metadata is not None:
        resolved = resolve_cache_with_index(
            write_cache_path,
            expected_metadata=cache_metadata,
            dataset_name=str(getattr(cfg.dataset, "name", "") or ""),
            cfg=cfg,
        )
        if resolved is not None:
            read_cache_path = resolved.path
    loaded = load_text_embedding(read_cache_path, load_layer_embeddings=load_layer_embeddings)
    if loaded is not None:
        cache = loaded
        loaded_cache_path = read_cache_path
    if loaded_cache_path is not None and loaded_cache_path != write_cache_path:
        print(f"[cache] Using compatible cache path: {loaded_cache_path}")

    resolved_seed = _resolve_shuffle_seed(cfg)
    X_vectors = _select_cached_embeddings(
        cache, ids, cfg, resolved_seed, require_cache=require_cache
    )
    cache_hit = X_vectors is not None

    if X_vectors is None:
        dist_active, rank, world_size = _dist_state()
        strategy = resolve_parallel_strategy(cfg.embedding)

        if dist_active and strategy == "pipeline2":
            raise RuntimeError(
                "parallel_strategy='pipeline2' requires a single process when building the cache. "
                "Run cache generation without torchrun/DDP (WORLD_SIZE=1), "
                "or pass embedding.parallel_strategy=ddp."
            )

        def _save_cache(
            embeddings: np.ndarray, *, layer_embeddings: Optional[np.ndarray]
        ) -> None:
            raw_hidden_state_layer = getattr(cfg.embedding, "hidden_state_layer", None)
            hidden_state_layer = (
                -1 if raw_hidden_state_layer is None else int(raw_hidden_state_layer)
            )
            save_text_embedding(
                ids,
                embeddings,
                resolved_seed,
                write_cache_path,
                layer_embeddings=layer_embeddings,
                hidden_state_layer=(
                    hidden_state_layer if cfg.embedding.type == "hf_transformer" else None
                ),
                embedding_type=getattr(cfg.embedding, "type", None),
                pooling=(
                    getattr(cfg.embedding, "pooling", None)
                    if cfg.embedding.type == "hf_transformer"
                    else None
                ),
                rulebook_id=get_rulebook_id(),
                registry_key=registry_key,
                metadata=cache_metadata if cache_metadata is not None else None,
            )

        def _ddp_gather_first_dim(local: torch.Tensor) -> Optional[torch.Tensor]:
            if not dist_active:
                raise RuntimeError("_ddp_gather_first_dim called without an initialized process group.")
            if local.device.type != "cuda":
                raise ValueError("DDP gather expects CUDA tensors (NCCL backend).")

            size_tensor = torch.tensor([local.shape[0]], device=local.device, dtype=torch.long)
            size_list = [torch.zeros_like(size_tensor) for _ in range(world_size)]
            dist.all_gather(size_list, size_tensor)
            sizes = [int(t.item()) for t in size_list]
            max_size = max(sizes) if sizes else int(local.shape[0])

            if local.shape[0] < max_size:
                pad_shape = (max_size - local.shape[0],) + tuple(local.shape[1:])
                pad = torch.zeros(pad_shape, dtype=local.dtype, device=local.device)
                local = torch.cat([local, pad], dim=0)

            gather_list = (
                [torch.empty_like(local) for _ in range(world_size)]
                if rank == 0
                else None
            )
            dist.gather(local, gather_list=gather_list, dst=0)
            if rank != 0:
                return None

            chunks = []
            if gather_list is None:
                raise RuntimeError("DDP gather did not return tensors on rank 0.")
            for tensor, count in zip(gather_list, sizes):
                if count <= 0:
                    continue
                chunks.append(tensor[:count].detach().cpu())
            if not chunks:
                return torch.empty((0,) + tuple(local.shape[1:]), dtype=local.dtype)
            return torch.cat(chunks, dim=0)

        def _reload_vectors_from_cache() -> Optional[np.ndarray]:
            loaded_cache = load_text_embedding(
                write_cache_path, load_layer_embeddings=load_layer_embeddings
            )
            return _select_cached_embeddings(
                loaded_cache, ids, cfg, resolved_seed, require_cache=True
            )

        def _gather_layer_cache_for_ddp(
            local_layer_cache: Optional[np.ndarray], *, device: torch.device
        ) -> Optional[np.ndarray]:
            if not load_layer_embeddings:
                return None
            if local_layer_cache is None:
                raise RuntimeError(
                    "cache_all_layers=True but no layer cache was produced during embedding generation."
                )
            local_layer_tensor = torch.as_tensor(
                local_layer_cache, dtype=torch.float32, device=device
            )
            gathered_layer_cache_tensor = _ddp_gather_first_dim(local_layer_tensor)
            if rank != 0 or gathered_layer_cache_tensor is None:
                return None
            return gathered_layer_cache_tensor.numpy()

        def _generate_rank0_cache_from_texts() -> None:
            generated_vectors = get_embeddings(list(texts), cfg)
            layer_cache = get_last_hidden_state_cache()
            clear_last_hidden_state_cache()
            _save_cache(generated_vectors, layer_embeddings=layer_cache)

        def _reload_after_distributed_generation() -> Optional[np.ndarray]:
            if rank == 0:
                _generate_rank0_cache_from_texts()
            dist.barrier()
            return _reload_vectors_from_cache()

        if dist_active and strategy == "ddp":
            total = len(texts)
            start = (total * rank) // world_size
            end = (total * (rank + 1)) // world_size

            local_texts = texts[start:end]
            local_vectors = get_embeddings(list(local_texts), cfg)
            local_layer_cache = get_last_hidden_state_cache()
            clear_last_hidden_state_cache()

            device = torch.device(cfg.device)
            local_vectors_tensor = torch.as_tensor(local_vectors, dtype=torch.float32, device=device)
            gathered_vectors = _ddp_gather_first_dim(local_vectors_tensor)
            gathered_layer_cache = _gather_layer_cache_for_ddp(
                local_layer_cache,
                device=device,
            )

            if rank == 0 and gathered_vectors is None:
                raise RuntimeError("DDP gather failed to produce embeddings on rank 0.")
            if rank == 0:
                assert gathered_vectors is not None
                _save_cache(gathered_vectors.numpy(), layer_embeddings=gathered_layer_cache)
            dist.barrier()
            X_vectors = _reload_vectors_from_cache()
        elif dist_active:
            # Non-DDP strategy in a distributed run: only rank 0 generates/saves,
            # then everyone waits and loads from cache.
            X_vectors = _reload_after_distributed_generation()
        else:
            _generate_rank0_cache_from_texts()
            X_vectors = _reload_vectors_from_cache()

    if X_vectors is None:
        raise RuntimeError("Failed to load or generate embeddings.")
    _set_last_cache_usage(
        {
            "cache_hit": cache_hit,
            "cache_path": str(loaded_cache_path or write_cache_path),
            "requested_cache_path": str(write_cache_path),
            "dataset_name": str(getattr(cfg.dataset, "name", "") or ""),
            "dataset_key": (
                str(cache_metadata.get("dataset_key")).strip()
                if cache_metadata is not None and cache_metadata.get("dataset_key") is not None
                else None
            ),
            "embedding_name": str(getattr(cfg.embedding, "name", "") or ""),
            "embedding_model_name": (
                str(getattr(cfg.embedding, "model_name", "") or "").strip() or None
            ),
            "registry_key": registry_key,
            "rulebook_id": get_rulebook_id(),
            "variant_tag": (
                str(cache_metadata.get("variant_tag")).strip()
                if cache_metadata is not None and cache_metadata.get("variant_tag") is not None
                else None
            ),
        }
    )
    return X_vectors


def load_or_generate_embedding_collection(
    cfg: AppConfig,
    texts: Sequence[str],
    ids: Sequence,
    *,
    labels: Sequence | None = None,
    require_cache: bool = False,
) -> np.ndarray:
    """Generate and combine multiple embeddings defined in cfg.embedding_collection."""

    collection_cfg = getattr(cfg, "embedding_collection", None)
    if collection_cfg is None or not collection_cfg.embeddings:
        raise ValueError(
            "embedding_collection.embeddings must contain at least one embedding configuration"
        )

    blocks: List[np.ndarray] = []
    stats = []
    for emb_cfg in collection_cfg.embeddings:
        child_cfg = deepcopy(cfg)
        child_cfg.embedding = emb_cfg
        child_cfg.embedding_collection = None
        vectors = load_or_generate_embeddings(
            child_cfg,
            texts,
            ids,
            labels=labels,
            require_cache=require_cache,
        )
        blocks.append(vectors)
        stats.append((emb_cfg.name, vectors.shape[1]))

    mode = getattr(collection_cfg, "combine_mode", "concat").lower()
    if mode != "concat":
        raise ValueError(
            f"Unsupported embedding combine_mode '{collection_cfg.combine_mode}'. Only 'concat' is supported."
        )

    combined = np.concatenate(blocks, axis=1)
    print(
        "Combined embeddings: "
        + ", ".join(f"{name}:{dim}" for name, dim in stats)
        + f" -> total_dim={combined.shape[1]}"
    )
    return combined
