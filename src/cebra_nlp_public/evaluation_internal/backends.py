from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..config_schema import AppConfig

_CUML_AVAILABLE = False
try:
    import cupy as cp
    from cuml.decomposition import PCA as cuPCA
    from cuml.manifold import UMAP as cuUMAP
    from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
    from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor

    _CUML_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError):
    cp = None  # type: ignore[assignment]
    cuPCA = cuUMAP = cuKNeighborsClassifier = cuKNeighborsRegressor = None

_FAISS_AVAILABLE = False
_FAISS_GPU_AVAILABLE = False
try:
    import faiss  # type: ignore[assignment]

    _FAISS_AVAILABLE = True
    _FAISS_GPU_AVAILABLE = hasattr(faiss, "StandardGpuResources")
except (ImportError, ModuleNotFoundError, OSError):
    # OSError can be raised when GPU builds are present but incompatible with the
    # installed CUDA runtime. Treat this the same as faiss being unavailable so the
    # rest of the pipeline can gracefully fall back to scikit-learn implementations.
    faiss = None  # type: ignore[assignment]


def clear_cuda_cache() -> None:
    """Clear the CUDA cache if running on a GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _should_use_cuml(cfg: Optional[AppConfig] = None, override: Optional[bool] = None) -> bool:
    """Decide whether to use cuML-backed implementations."""
    if not _CUML_AVAILABLE:
        return False
    if override is not None:
        return override

    if cfg is not None:
        device = getattr(cfg, "device", "")
        if isinstance(device, str) and device.lower().startswith("cuda"):
            return True

    return torch.cuda.is_available()


def _to_gpu_array(array):
    if cp is None:  # pragma: no cover - defensive guard
        raise RuntimeError("cuML requested but CuPy is not available.")
    if isinstance(array, cp.ndarray):
        return array
    return cp.asarray(array)


def _to_cpu_numpy(array):
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def _resolve_faiss_backend(use_gpu: bool | None, *, strict: bool = False) -> tuple[bool, bool]:
    """Return (is_available, use_gpu) for FAISS based on requested policy."""
    if not _FAISS_AVAILABLE:
        return False, False
    gpu_possible = bool(_FAISS_GPU_AVAILABLE and torch.cuda.is_available())
    if use_gpu is True:
        if not gpu_possible:
            if strict:
                raise RuntimeError(
                    "FAISS GPU backend requested but no CUDA-enabled FAISS build is available."
                )
            return True, False
        return True, True
    if use_gpu is False:
        return True, False
    # use_gpu is None: pick GPU when possible
    return True, gpu_possible


def _resolve_required_faiss_backend(
    use_gpu: bool | None,
    *,
    missing_message: str,
) -> tuple[bool, bool]:
    faiss_available, faiss_use_gpu = _resolve_faiss_backend(use_gpu, strict=True)
    if not faiss_available:
        raise RuntimeError(missing_message)
    return faiss_available, faiss_use_gpu


def _faiss_knn_search(
    train_matrix: np.ndarray,
    query_matrix: np.ndarray,
    k: int,
    *,
    use_gpu: bool,
    gpu_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not _FAISS_AVAILABLE:
        raise RuntimeError("FAISS backend requested but `faiss` is not installed.")

    train32 = np.asarray(train_matrix, dtype=np.float32, order="C")
    query32 = np.asarray(query_matrix, dtype=np.float32, order="C")
    if train32.shape[1] != query32.shape[1]:
        raise ValueError(
            "Training and query embeddings must have the same dimensionality for FAISS."
        )

    index = faiss.IndexFlatL2(train32.shape[1])  # type: ignore[attr-defined]
    res = None
    if use_gpu:
        if not _FAISS_GPU_AVAILABLE:
            raise RuntimeError(
                "FAISS GPU backend requested but `faiss-gpu` is not available."
            )
        res = faiss.StandardGpuResources()  # type: ignore[attr-defined]
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)  # type: ignore[attr-defined]

    index.add(train32)
    distances, indices = index.search(query32, k)
    return distances, indices


def _faiss_weighted_classification(
    neighbor_labels: np.ndarray,
    weights: np.ndarray,
    all_labels: np.ndarray,
) -> np.ndarray:
    """Compute weighted majority votes given neighbor labels and weights."""
    label_to_pos = {int(label): idx for idx, label in enumerate(all_labels)}
    num_queries, _ = neighbor_labels.shape
    scores = np.zeros((num_queries, len(all_labels)), dtype=np.float64)

    for row in range(num_queries):
        label_indices = [label_to_pos[int(lbl)] for lbl in neighbor_labels[row]]
        np.add.at(scores[row], label_indices, weights[row])

    predicted_indices = scores.argmax(axis=1)
    return all_labels[predicted_indices]


def _faiss_weighted_regression(
    neighbor_targets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return weighted average of neighbor targets."""
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum == 0.0, 1e-12, weight_sum)
    weighted_sum = np.sum(neighbor_targets * weights[..., None], axis=1)
    return weighted_sum / weight_sum
