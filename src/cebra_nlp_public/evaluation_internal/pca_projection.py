from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA

from ..config_schema import AppConfig
from ..label_overlay import (
    LabelOverlaySpec,
    write_label_centroid_points_csv,
    write_label_overlay_points_csv,
)
from .backends import _should_use_cuml, _to_cpu_numpy, _to_gpu_array, cuPCA


def project_with_pca_components(
    embeddings: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    components = np.asarray(components, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {values.shape}.")
    if components.ndim != 2:
        raise ValueError(f"Expected 2D PCA components array, got shape {components.shape}.")
    if values.shape[1] != mean.shape[1] or components.shape[1] != mean.shape[1]:
        raise ValueError(
            "PCA projection inputs have incompatible dimensions: "
            f"embeddings={values.shape}, mean={mean.shape}, components={components.shape}."
        )
    return (values - mean) @ components.T


def _compute_pca_axis_limits(coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"PCA axis limits expect (n_samples, 2) coordinates, got {coords.shape}."
        )
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    padding = span * 0.05
    return np.asarray(
        [
            [mins[0] - padding[0], maxs[0] + padding[0]],
            [mins[1] - padding[1], maxs[1] + padding[1]],
        ],
        dtype=np.float32,
    )


def _compute_aligned_label_centroids(
    embeddings: np.ndarray,
    text_labels: Sequence[str],
    label_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    labels_np = np.asarray(list(text_labels), dtype=str)
    if embeddings_np.ndim != 2:
        raise ValueError(
            f"Label centroid computation expects 2D embeddings, got {embeddings_np.shape}."
        )
    if embeddings_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            "Embeddings and text_labels must have the same number of rows for centroid computation."
        )

    centroids = np.full(
        (len(label_names), embeddings_np.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    counts = np.zeros(len(label_names), dtype=np.int64)
    for label_index, label_name in enumerate(label_names):
        mask = labels_np == str(label_name)
        counts[label_index] = int(np.sum(mask))
        if counts[label_index] == 0:
            continue
        centroids[label_index] = embeddings_np[mask].mean(axis=0)
    return centroids, counts


def fit_pca_projection(
    embeddings: np.ndarray,
    *,
    cfg: Optional[AppConfig] = None,
    n_components: int = 2,
) -> dict[str, np.ndarray]:
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    if n_components <= 0:
        raise ValueError(f"n_components must be > 0, got {n_components}.")
    max_components = min(embeddings_np.shape[0], embeddings_np.shape[1])
    if n_components > max_components:
        raise ValueError(
            "Requested PCA dimensionality exceeds the available sample/feature rank: "
            f"n_components={n_components}, max_supported={max_components}, "
            f"embeddings_shape={embeddings_np.shape}."
        )
    use_gpu_backend = _should_use_cuml(cfg)
    embeddings_gpu = None
    if use_gpu_backend:
        try:
            embeddings_gpu = _to_gpu_array(embeddings_np)
        except Exception as err:  # pragma: no cover - GPU initialisation specific
            print(
                f"cuML backend requested but moving embeddings to GPU failed ({err}); "
                "falling back to CPU PCA."
            )
            use_gpu_backend = False

    X_pca = None
    variance_ratios = None
    components = None
    mean = None
    if use_gpu_backend and cuPCA is not None and embeddings_gpu is not None:
        try:
            pca_gpu = cuPCA(n_components=n_components)
            X_pca = _to_cpu_numpy(pca_gpu.fit_transform(embeddings_gpu))
            variance_ratios = _to_cpu_numpy(pca_gpu.explained_variance_ratio_)
            components = _to_cpu_numpy(pca_gpu.components_)
            mean = _to_cpu_numpy(pca_gpu.mean_)
            print("PCA: using cuML implementation.")
        except Exception as err:  # pragma: no cover - GPU specific failure path
            print(f"cuML PCA failed ({err}); reverting to scikit-learn PCA.")

    if X_pca is None or variance_ratios is None or components is None or mean is None:
        pca_model = PCA(n_components=n_components)
        X_pca = pca_model.fit_transform(embeddings_np)
        variance_ratios = pca_model.explained_variance_ratio_
        components = pca_model.components_
        mean = pca_model.mean_

    return {
        "coordinates": np.asarray(X_pca, dtype=np.float32),
        "explained_variance_ratio": np.asarray(variance_ratios, dtype=np.float32),
        "components": np.asarray(components, dtype=np.float32),
        "mean": np.asarray(mean, dtype=np.float32),
    }


def export_pca_projection_artifacts(
    embeddings: np.ndarray,
    output_dir: Path,
    *,
    scope_name: str,
    cfg: Optional[AppConfig] = None,
    projected_embeddings: np.ndarray | None = None,
    projected_text_labels: Sequence[str] | None = None,
    fit_scope_name: str | None = None,
    axis_limits: np.ndarray | None = None,
    overlay_spec: LabelOverlaySpec | None = None,
    overlay_embeddings: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    pca_result = fit_pca_projection(embeddings, cfg=cfg)
    projected_embeddings_np = np.asarray(
        embeddings if projected_embeddings is None else projected_embeddings,
        dtype=np.float32,
    )
    projected_coordinates = (
        np.asarray(pca_result["coordinates"], dtype=np.float32)
        if projected_embeddings is None
        else project_with_pca_components(
            projected_embeddings_np,
            pca_result["mean"],
            pca_result["components"],
        )
    )
    resolved_fit_scope = str(fit_scope_name or scope_name)
    resolved_axis_limits = (
        np.asarray(axis_limits, dtype=np.float32)
        if axis_limits is not None
        else _compute_pca_axis_limits(pca_result["coordinates"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"pca_model_{scope_name}.npz",
        mean=pca_result["mean"],
        components=pca_result["components"],
        explained_variance_ratio=pca_result["explained_variance_ratio"],
        scope=np.asarray([scope_name]),
        fit_scope=np.asarray([resolved_fit_scope]),
        axis_limits=resolved_axis_limits,
    )

    pca_result["coordinates"] = np.asarray(projected_coordinates, dtype=np.float32)
    pca_result["fit_scope"] = np.asarray([resolved_fit_scope])
    pca_result["axis_limits"] = resolved_axis_limits

    if overlay_spec is None:
        return pca_result

    overlay_pca = None
    overlay_embeddings_np = None
    if overlay_embeddings is not None:
        overlay_embeddings_np = np.asarray(overlay_embeddings, dtype=np.float32)
        overlay_pca = project_with_pca_components(
            overlay_embeddings_np,
            pca_result["mean"],
            pca_result["components"],
        )
        write_label_overlay_points_csv(
            overlay_spec,
            overlay_embeddings_np,
            overlay_pca,
            output_dir / f"label_overlay_points_{scope_name}.csv",
            scope_name=scope_name,
        )
        pca_result["overlay_coordinates"] = overlay_pca

    if projected_text_labels is not None:
        centroid_cebra, centroid_counts = _compute_aligned_label_centroids(
            projected_embeddings_np,
            projected_text_labels,
            overlay_spec.label_names,
        )
        centroid_pca = np.full(
            (len(overlay_spec.label_names), 2),
            np.nan,
            dtype=np.float32,
        )
        valid_mask = centroid_counts > 0
        if np.any(valid_mask):
            centroid_pca[valid_mask] = project_with_pca_components(
                centroid_cebra[valid_mask],
                pca_result["mean"],
                pca_result["components"],
            )
        write_label_centroid_points_csv(
            overlay_spec,
            centroid_cebra,
            centroid_pca,
            output_dir / f"label_centroid_points_{scope_name}.csv",
            scope_name=scope_name,
            fit_scope_name=resolved_fit_scope,
            sample_counts=centroid_counts,
            overlay_cebra_embeddings=overlay_embeddings_np,
            overlay_pca_embeddings=overlay_pca,
        )
        pca_result["centroid_coordinates"] = centroid_pca
        pca_result["centroid_embeddings"] = centroid_cebra
        pca_result["centroid_counts"] = centroid_counts.astype(np.int64)
    return pca_result
