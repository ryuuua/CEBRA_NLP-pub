# results.py
import matplotlib

matplotlib.use("Agg")  # Use Agg backend for headless environments


import json
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Optional, Sequence
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
import os
import tempfile
from .optional_wandb import wandb
from tqdm import tqdm
from .config_schema import AppConfig
from .label_overlay import (
    LabelOverlaySpec,
    write_label_centroid_points_csv,
    write_label_overlay_points_csv,
)
from .evaluation_internal import backends as _evaluation_backends
from .evaluation_internal import pca_projection as _pca_projection
from cebra.integrations.sklearn.metrics import consistency_score
from cebra import plot_consistency
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import gc
import cebra
from .cebra_trainer import normalize_model_architecture

STATIC_PLOT_DPI = 700

_CUML_AVAILABLE = _evaluation_backends._CUML_AVAILABLE
cp = _evaluation_backends.cp
cuPCA = _evaluation_backends.cuPCA
cuUMAP = _evaluation_backends.cuUMAP
cuKNeighborsClassifier = _evaluation_backends.cuKNeighborsClassifier
cuKNeighborsRegressor = _evaluation_backends.cuKNeighborsRegressor

_FAISS_AVAILABLE = _evaluation_backends._FAISS_AVAILABLE
_FAISS_GPU_AVAILABLE = _evaluation_backends._FAISS_GPU_AVAILABLE
faiss = _evaluation_backends.faiss

_UMAP_AVAILABLE = False
_UMAP_IMPORT_ERROR: Exception | None = None
try:
    import umap

    _UMAP_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    umap = None  # type: ignore[assignment]
    _UMAP_IMPORT_ERROR = exc

# train_test_splitはこのファイルで使われていないため削除


def _sync_evaluation_backend_state() -> None:
    _evaluation_backends._CUML_AVAILABLE = _CUML_AVAILABLE
    _evaluation_backends.cp = cp
    _evaluation_backends.cuPCA = cuPCA
    _evaluation_backends.cuUMAP = cuUMAP
    _evaluation_backends.cuKNeighborsClassifier = cuKNeighborsClassifier
    _evaluation_backends.cuKNeighborsRegressor = cuKNeighborsRegressor
    _evaluation_backends._FAISS_AVAILABLE = _FAISS_AVAILABLE
    _evaluation_backends._FAISS_GPU_AVAILABLE = _FAISS_GPU_AVAILABLE
    _evaluation_backends.faiss = faiss
    _evaluation_backends.np = np
    _evaluation_backends.os = os
    _evaluation_backends.torch = torch


def _sync_pca_projection_state() -> None:
    _sync_evaluation_backend_state()
    _pca_projection.np = np
    _pca_projection.PCA = PCA
    _pca_projection._should_use_cuml = _should_use_cuml
    _pca_projection._to_cpu_numpy = _to_cpu_numpy
    _pca_projection._to_gpu_array = _to_gpu_array
    _pca_projection.cuPCA = cuPCA
    _pca_projection.write_label_centroid_points_csv = write_label_centroid_points_csv
    _pca_projection.write_label_overlay_points_csv = write_label_overlay_points_csv


def clear_cuda_cache() -> None:
    """Clear the CUDA cache if running on a GPU."""
    _sync_evaluation_backend_state()
    return _evaluation_backends.clear_cuda_cache()


def _should_use_cuml(cfg: Optional[AppConfig] = None, override: Optional[bool] = None) -> bool:
    """Decide whether to use cuML-backed implementations."""
    _sync_evaluation_backend_state()
    return _evaluation_backends._should_use_cuml(cfg, override)


def _to_gpu_array(array):
    _sync_evaluation_backend_state()
    return _evaluation_backends._to_gpu_array(array)


def _to_cpu_numpy(array):
    _sync_evaluation_backend_state()
    return _evaluation_backends._to_cpu_numpy(array)


def _resolve_faiss_backend(use_gpu: bool | None, *, strict: bool = False) -> tuple[bool, bool]:
    """Return (is_available, use_gpu) for FAISS based on requested policy."""
    _sync_evaluation_backend_state()
    return _evaluation_backends._resolve_faiss_backend(use_gpu, strict=strict)


def _resolve_required_faiss_backend(
    use_gpu: bool | None,
    *,
    missing_message: str,
) -> tuple[bool, bool]:
    _sync_evaluation_backend_state()
    return _evaluation_backends._resolve_required_faiss_backend(
        use_gpu,
        missing_message=missing_message,
    )


def _faiss_knn_search(
    train_matrix: np.ndarray,
    query_matrix: np.ndarray,
    k: int,
    *,
    use_gpu: bool,
    gpu_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    _sync_evaluation_backend_state()
    return _evaluation_backends._faiss_knn_search(
        train_matrix,
        query_matrix,
        k,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
    )


def _faiss_weighted_classification(
    neighbor_labels: np.ndarray,
    weights: np.ndarray,
    all_labels: np.ndarray,
) -> np.ndarray:
    _sync_evaluation_backend_state()
    return _evaluation_backends._faiss_weighted_classification(
        neighbor_labels,
        weights,
        all_labels,
    )


def _faiss_weighted_regression(
    neighbor_targets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    _sync_evaluation_backend_state()
    return _evaluation_backends._faiss_weighted_regression(neighbor_targets, weights)


def project_with_pca_components(
    embeddings: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    return _pca_projection.project_with_pca_components(embeddings, mean, components)


def _compute_pca_axis_limits(coordinates: np.ndarray) -> np.ndarray:
    return _pca_projection._compute_pca_axis_limits(coordinates)


def _compute_aligned_label_centroids(
    embeddings: np.ndarray,
    text_labels: Sequence[str],
    label_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    return _pca_projection._compute_aligned_label_centroids(
        embeddings,
        text_labels,
        label_names,
    )


def fit_pca_projection(
    embeddings: np.ndarray,
    *,
    cfg: Optional[AppConfig] = None,
    n_components: int = 2,
) -> dict[str, np.ndarray]:
    _sync_pca_projection_state()
    return _pca_projection.fit_pca_projection(
        embeddings,
        cfg=cfg,
        n_components=n_components,
    )


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
    _sync_pca_projection_state()
    return _pca_projection.export_pca_projection_artifacts(
        embeddings,
        output_dir,
        scope_name=scope_name,
        cfg=cfg,
        projected_embeddings=projected_embeddings,
        projected_text_labels=projected_text_labels,
        fit_scope_name=fit_scope_name,
        axis_limits=axis_limits,
        overlay_spec=overlay_spec,
        overlay_embeddings=overlay_embeddings,
    )


def save_interactive_plot(
    embeddings,
    text_labels,
    output_dim,
    palette,
    title,
    output_path: Path,
    *,
    overlay_spec: LabelOverlaySpec | None = None,
    overlay_embeddings: np.ndarray | None = None,
):
    """Saves a 2D or 3D interactive plot as an HTML file and a static SVG image."""
    print(
        f"\nGenerating interactive visualization for {output_dim}-dimensional output..."
    )
    if output_dim not in (2, 3):
        print(
            f"Skipping interactive plot: output_dim is {output_dim}, but must be 2 or 3."
        )
        return

    plot_df = pd.DataFrame(
        embeddings[:, :output_dim],
        columns=[f"Dim {i+1}" for i in range(output_dim)],
    )
    plot_df["label"] = text_labels

    axis_style = dict(linecolor="black", gridcolor="gray", mirror=True)
    if output_dim == 2:
        fig = px.scatter(
            plot_df,
            x="Dim 1",
            y="Dim 2",
            color="label",
            hover_name="label",
            title=title,
            color_discrete_map=palette,
        )
    else:  # output_dim == 3
        fig = px.scatter_3d(
            plot_df,
            x="Dim 1",
            y="Dim 2",
            z="Dim 3",
            color="label",
            hover_name="label",
            title=title,
            color_discrete_map=palette,
        )

    fig.update_traces(marker=dict(size=2, opacity=0.6))

    if overlay_spec is not None and overlay_embeddings is not None:
        overlay_coords = np.asarray(overlay_embeddings, dtype=np.float32)
        if overlay_coords.shape[0] != len(overlay_spec.label_names):
            raise ValueError(
                "Overlay embeddings row count does not match label overlay spec: "
                f"{overlay_coords.shape[0]} vs {len(overlay_spec.label_names)}."
            )
        overlay_coords = overlay_coords[:, :output_dim]
        overlay_colors = None
        if palette is not None:
            overlay_colors = [
                palette.get(label_name, "#000000")
                for label_name in overlay_spec.label_names
            ]
        marker_kwargs = dict(
            size=14,
            color=overlay_colors or "#000000",
            line=dict(color="black", width=1),
            opacity=1.0,
        )
        if output_dim == 2:
            marker_kwargs["symbol"] = "x"
            fig.add_scatter(
                x=overlay_coords[:, 0],
                y=overlay_coords[:, 1],
                mode="markers+text",
                text=overlay_spec.label_names,
                textposition="top center",
                marker=marker_kwargs,
                name="Label overlay",
                showlegend=False,
                hovertext=overlay_spec.texts,
                hovertemplate="%{text}<extra>label overlay</extra>",
            )
        else:
            fig.add_scatter3d(
                x=overlay_coords[:, 0],
                y=overlay_coords[:, 1],
                z=overlay_coords[:, 2],
                mode="markers+text",
                text=overlay_spec.label_names,
                textposition="top center",
                marker=marker_kwargs,
                name="Label overlay",
                showlegend=False,
                hovertext=overlay_spec.texts,
                hovertemplate="%{text}<extra>label overlay</extra>",
            )

    # Adjust layout and camera for 3D plots
    base_bg = "rgba(0,0,0,0)"
    if output_dim == 3:
        scene_axis = dict(**axis_style, showbackground=False)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.7, y=1.7, z=0.5),
        )
        fig.update_layout(
            paper_bgcolor=base_bg,
            plot_bgcolor=base_bg,
            scene_camera=camera,
            scene=dict(
                xaxis=scene_axis,
                yaxis=scene_axis,
                zaxis=scene_axis,
                bgcolor=base_bg,
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
    else:
        fig.update_layout(
            paper_bgcolor=base_bg,
            plot_bgcolor=base_bg,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)

    # Save interactive HTML
    fig.write_html(str(output_path))
    print(f"Saved interactive {output_dim}D plot to {output_path}")

    # Save static SVG
    svg_path = output_path.with_suffix(".svg")
    try:
        fig.write_image(str(svg_path), width=1200, height=900)
        print(f"Saved static SVG image to {svg_path}")
    except Exception as e:
        print("\n--- SVG Export Warning ---")
        print(f"Could not save SVG image. Error: {e}")
        print(
            "Please ensure the 'kaleido' package is installed (`pip install kaleido`)"
        )
        print("--------------------------")


def save_static_2d_plots(
    embeddings,
    text_labels,
    palette,
    title_prefix,
    output_dir: Path,
    hue_order: list,
    cfg: Optional[AppConfig] = None,
    log_to_wandb: Optional[bool] = None,
    *,
    scope_name: str = "full",
    pca_projection: Optional[dict[str, np.ndarray]] = None,
    overlay_spec: LabelOverlaySpec | None = None,
):
    """Generates and saves 2D static plots using PCA and UMAP."""
    print("Generating static 2D scatter plots using PCA and UMAP...")

    embeddings_np = np.asarray(embeddings)
    use_gpu_backend = _should_use_cuml(cfg)
    embeddings_gpu = None
    if use_gpu_backend:
        try:
            embeddings_gpu = _to_gpu_array(embeddings_np)
        except Exception as err:  # pragma: no cover - GPU initialisation specific
            print(
                f"cuML backend requested but moving embeddings to GPU failed ({err}); "
                "falling back to CPU implementations."
            )
            use_gpu_backend = False

    reproducibility = getattr(cfg, "reproducibility", None) if cfg is not None else None
    deterministic = bool(getattr(reproducibility, "deterministic", False))
    umap_seed = None
    if deterministic:
        umap_seed = getattr(reproducibility, "seed", None)
        if umap_seed is None and cfg is not None:
            eval_cfg = getattr(cfg, "evaluation", None)
            if eval_cfg is not None:
                umap_seed = getattr(eval_cfg, "random_state", None)

    umap_base_kwargs = dict(n_components=2, n_neighbors=15, min_dist=0.1)
    if deterministic and umap_seed is not None:
        umap_base_kwargs["random_state"] = umap_seed

    if pca_projection is None:
        pca_projection = export_pca_projection_artifacts(
            embeddings_np,
            output_dir,
            scope_name=scope_name,
            cfg=cfg,
        )
    X_pca = np.asarray(pca_projection["coordinates"], dtype=np.float32)
    variance_ratios = np.asarray(
        pca_projection["explained_variance_ratio"],
        dtype=np.float32,
    )
    overlay_pca = None
    if overlay_spec is not None and "overlay_coordinates" in pca_projection:
        overlay_pca = np.asarray(pca_projection["overlay_coordinates"], dtype=np.float32)
    centroid_pca = None
    centroid_counts = None
    if overlay_spec is not None and "centroid_coordinates" in pca_projection:
        centroid_pca = np.asarray(
            pca_projection["centroid_coordinates"],
            dtype=np.float32,
        )
    if overlay_spec is not None and "centroid_counts" in pca_projection:
        centroid_counts = np.asarray(pca_projection["centroid_counts"], dtype=np.int64)
    pca_axis_limits = None
    if "axis_limits" in pca_projection:
        pca_axis_limits = np.asarray(pca_projection["axis_limits"], dtype=np.float32)

    X_umap = None
    if use_gpu_backend and cuUMAP is not None and embeddings_gpu is not None:
        try:
            umap_gpu = cuUMAP(**umap_base_kwargs)
            X_umap = _to_cpu_numpy(umap_gpu.fit_transform(embeddings_gpu))
            print("UMAP: using cuML implementation.")
        except Exception as err:  # pragma: no cover - GPU specific failure path
            print("cuML UMAP failed "
                  f"({err}); reverting to umap-learn CPU implementation.")

    if X_umap is None:
        if not _UMAP_AVAILABLE or umap is None:
            raise RuntimeError(
                "UMAP plotting requires a working `umap-learn` installation. "
                "Set evaluation.enable_plots=false to skip UMAP plots."
            ) from _UMAP_IMPORT_ERROR
        cpu_umap_kwargs = dict(umap_base_kwargs)
        cpu_umap_kwargs["n_jobs"] = 1 if deterministic else -1
        umap_model = umap.UMAP(**cpu_umap_kwargs)
        X_umap = umap_model.fit_transform(embeddings_np)

    print(
        "PCA explained variance ratios:",
        ", ".join(f"{ratio * 100:.2f}%" for ratio in variance_ratios),
    )

    if log_to_wandb is None:
        log_to_wandb = wandb.run is not None

    if log_to_wandb and wandb.run is not None:
        wandb.log(
            {
                f"pca_variance_ratio_dim{i + 1}": float(ratio)
                for i, ratio in enumerate(variance_ratios)
            }
        )

    for X_reduced, name in [(X_pca, "PCA"), (X_umap, "UMAP")]:
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            hue=text_labels,
            palette=palette,
            s=10,
            hue_order=hue_order,
        )
        if name == "PCA" and overlay_spec is not None and overlay_pca is not None:
            overlay_colors = None
            if palette is not None:
                overlay_colors = [
                    palette.get(label_name, "#000000")
                    for label_name in overlay_spec.label_names
                ]
            plt.scatter(
                overlay_pca[:, 0],
                overlay_pca[:, 1],
                c=overlay_colors or "#000000",
                marker="X",
                s=140,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            for label_name, x_coord, y_coord in zip(
                overlay_spec.label_names,
                overlay_pca[:, 0],
                overlay_pca[:, 1],
            ):
                plt.text(
                    float(x_coord),
                    float(y_coord),
                    label_name,
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    zorder=6,
                )
        if (
            name == "PCA"
            and overlay_spec is not None
            and centroid_pca is not None
            and bool(getattr(getattr(cfg, "label_overlay", None), "show_centroids_in_pca", True))
        ):
            centroid_mask = (
                centroid_counts > 0
                if centroid_counts is not None
                else np.all(np.isfinite(centroid_pca), axis=1)
            )
            centroid_colors = None
            if palette is not None:
                centroid_colors = [
                    palette.get(label_name, "#000000")
                    for label_name in overlay_spec.label_names
                ]
            plt.scatter(
                centroid_pca[centroid_mask, 0],
                centroid_pca[centroid_mask, 1],
                facecolors="none",
                edgecolors=(
                    np.asarray(centroid_colors, dtype=object)[centroid_mask].tolist()
                    if centroid_colors is not None
                    else "#000000"
                ),
                marker="o",
                s=240,
                linewidths=1.6,
                zorder=4,
            )
        plt.title(f"{title_prefix} with {name}")
        if name == "PCA":
            plt.xlabel(f"{name} 1 ({variance_ratios[0] * 100:.1f}%)")
            plt.ylabel(f"{name} 2 ({variance_ratios[1] * 100:.1f}%)")
        else:
            plt.xlabel(f"{name} 1")
            plt.ylabel(f"{name} 2")
        if name == "PCA" and pca_axis_limits is not None:
            plt.xlim(float(pca_axis_limits[0, 0]), float(pca_axis_limits[0, 1]))
            plt.ylim(float(pca_axis_limits[1, 0]), float(pca_axis_limits[1, 1]))
        plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_facecolor((0, 0, 0, 0))
        fig.patch.set_alpha(0)
        plt.tight_layout()
        static_plot_png = output_dir / f"static_{name}_plot.png"
        plt.savefig(
            static_plot_png,
            dpi=STATIC_PLOT_DPI,
            bbox_inches="tight",
            transparent=True,
        )
        static_plot_svg = static_plot_png.with_suffix(".svg")
        plt.savefig(
            static_plot_svg,
            dpi=STATIC_PLOT_DPI,
            bbox_inches="tight",
            transparent=True,
        )
        # Axisless variants
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.axis("off")
        axisless_png = static_plot_png.with_name(f"{static_plot_png.stem}_noaxes{static_plot_png.suffix}")
        plt.savefig(
            axisless_png,
            dpi=STATIC_PLOT_DPI,
            bbox_inches="tight",
            transparent=True,
        )
        axisless_svg = static_plot_svg.with_name(f"{static_plot_svg.stem}_noaxes{static_plot_svg.suffix}")
        plt.savefig(
            axisless_svg,
            dpi=STATIC_PLOT_DPI,
            bbox_inches="tight",
            transparent=True,
        )
        plt.close()
        print(
            f"Saved static {name} plot to {static_plot_png} and {static_plot_svg}"
        )


def run_local_linearity_probe(
    learned_embeddings,
    original_embeddings,
    output_dir: Path,
    *,
    neighbors: int = 15,
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
    ridge_alpha: float = 1e-3,
    enable_plots: bool = True,
    store_scores: bool = False,
    log_to_wandb: bool = False,
):
    """
    Evaluate how well the original embeddings can be reconstructed by a local
    linear map from the learned CEBRA embeddings.

    The probe fits a Ridge regression in the neighborhood of each sampled
    point (in the learned embedding space) to predict the corresponding
    original embedding vectors. We report the mean R^2 across neighborhoods as
    a measure of local linearity: higher scores indicate that the learned space
    is locally well-approximated by a linear map to the original space.
    """
    print("\nRunning local linearity probe...")

    learned = np.asarray(learned_embeddings, dtype=np.float32)
    original = np.asarray(original_embeddings, dtype=np.float32)
    if learned.shape[0] != original.shape[0]:
        raise ValueError(
            "learned_embeddings and original_embeddings must have the same number of rows."
        )
    if learned.ndim != 2 or original.ndim != 2:
        raise ValueError(
            "learned_embeddings and original_embeddings must be 2D arrays of shape (n_samples, n_features)."
        )
    n_samples = learned.shape[0]
    if n_samples < 3:
        raise ValueError("Local linearity probe requires at least 3 samples.")

    effective_neighbors = min(max(2, neighbors), n_samples - 1)
    neighbor_finder = NearestNeighbors(
        n_neighbors=min(effective_neighbors + 1, n_samples), metric="euclidean"
    )
    neighbor_finder.fit(learned)

    rng = np.random.default_rng(random_state)
    if sample_size is not None and sample_size < n_samples:
        eval_indices = rng.choice(n_samples, size=sample_size, replace=False)
    else:
        eval_indices = np.arange(n_samples)

    local_r2_scores: list[float] = []
    center_cosines: list[float] = []

    for idx in eval_indices:
        neighbors_idx = neighbor_finder.kneighbors(
            learned[idx][None, :], return_distance=False
        )[0]
        neighbors_idx = neighbors_idx[neighbors_idx != idx]
        if neighbors_idx.size > effective_neighbors:
            neighbors_idx = neighbors_idx[:effective_neighbors]
        if neighbors_idx.size < 2:
            continue

        X_local = learned[neighbors_idx]
        Y_local = original[neighbors_idx]

        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(X_local, Y_local)

        score = model.score(X_local, Y_local)
        if np.isfinite(score):
            local_r2_scores.append(float(score))

        pred_center = model.predict(learned[idx][None, :])[0]
        target_center = original[idx]
        denom = float(np.linalg.norm(pred_center) * np.linalg.norm(target_center))
        if denom > 0.0:
            center_cosines.append(float(np.dot(pred_center, target_center) / denom))

    if not local_r2_scores:
        raise ValueError(
            "Local linearity probe did not produce any scores. "
            "Check that the dataset has enough samples and neighbors."
        )

    summary = {
        "mean_r2": float(np.mean(local_r2_scores)),
        "median_r2": float(np.median(local_r2_scores)),
        "std_r2": float(np.std(local_r2_scores)),
        "min_r2": float(np.min(local_r2_scores)),
        "max_r2": float(np.max(local_r2_scores)),
        "num_evaluated": int(len(local_r2_scores)),
        "num_requested": int(len(eval_indices)),
        "neighbors": int(effective_neighbors),
        "sampled": bool(len(eval_indices) != n_samples),
    }
    if center_cosines:
        summary["mean_center_cosine"] = float(np.mean(center_cosines))
        summary["median_center_cosine"] = float(np.median(center_cosines))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "local_linearity_probe.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved local linearity summary to {summary_path}")

    if store_scores:
        scores_df = pd.DataFrame({"r2_score": local_r2_scores})
        if center_cosines:
            padded = center_cosines + [np.nan] * (len(local_r2_scores) - len(center_cosines))
            scores_df["center_cosine"] = padded[: len(local_r2_scores)]
        scores_path = output_dir / "local_linearity_scores.csv"
        scores_df.to_csv(scores_path, index=False)
        print(f"Stored per-sample local linearity scores to {scores_path}")

    if enable_plots:
        hist_path = output_dir / "local_linearity_r2_hist.png"
        plt.figure(figsize=(8, 5))
        sns.histplot(local_r2_scores, bins=30, kde=True)
        plt.xlabel("Local Ridge R^2")
        plt.ylabel("Count")
        plt.title("Local linearity probe (R^2 distribution)")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=STATIC_PLOT_DPI, bbox_inches="tight")
        plt.close()
        print(f"Saved local linearity histogram to {hist_path}")

        if center_cosines:
            cosine_path = output_dir / "local_linearity_cosine_hist.png"
            plt.figure(figsize=(8, 5))
            sns.histplot(center_cosines, bins=30, kde=True)
            plt.xlabel("Cosine(pred, target)")
            plt.ylabel("Count")
            plt.title("Center reconstruction cosine similarity")
            plt.tight_layout()
            plt.savefig(cosine_path, dpi=STATIC_PLOT_DPI, bbox_inches="tight")
            plt.close()
            print(f"Saved center reconstruction cosine histogram to {cosine_path}")

    if log_to_wandb and wandb.run is not None:
        payload = {
            "local_linearity_mean_r2": summary["mean_r2"],
            "local_linearity_median_r2": summary["median_r2"],
            "local_linearity_neighbors": effective_neighbors,
            "local_linearity_r2_distribution": wandb.Histogram(local_r2_scores),
        }
        if center_cosines:
            payload["local_linearity_mean_cosine"] = float(np.mean(center_cosines))
            payload["local_linearity_cosine_distribution"] = wandb.Histogram(center_cosines)
        wandb.log(payload)

    return summary, local_r2_scores


def run_knn_classification(
    train_embeddings,
    valid_embeddings,
    y_train,
    y_valid,
    label_map,
    output_dir: Path,
    knn_neighbors,
    enable_plots: bool = True,
    backend: str = "auto",
    use_gpu: bool | None = None,
    faiss_gpu_id: int = 0,
):
    """k-NN classification for discrete labels."""
    print("\nRunning k-NN Classification evaluation...")
    cuda_available = torch.cuda.is_available()
    if use_gpu is True and not cuda_available:
        raise RuntimeError("GPU execution requested for k-NN but CUDA is not available.")

    train_cpu = np.asarray(train_embeddings, dtype=np.float32)
    valid_cpu = np.asarray(valid_embeddings, dtype=np.float32)
    y_train_cpu = np.asarray(y_train)
    y_valid_cpu = np.asarray(y_valid)
    if y_train_cpu.ndim != 1:
        raise ValueError("y_train must be a 1D array for classification tasks.")
    y_train_cpu = y_train_cpu.astype(np.int64, copy=False)
    y_valid_cpu = y_valid_cpu.astype(np.int64, copy=False)

    backend_choice = (backend or "auto").lower()
    prefer_gpu = cuda_available if use_gpu is None else bool(use_gpu)
    faiss_use_gpu = False

    if backend_choice == "auto":
        use_cuml = prefer_gpu and _should_use_cuml()
        if use_cuml:
            selected_backend = "cuml"
        else:
            faiss_available, faiss_use_gpu = _resolve_faiss_backend(
                None if use_gpu is None else prefer_gpu,
                strict=False,
            )
            selected_backend = "faiss" if faiss_available else "sklearn"
    elif backend_choice == "cuml":
        if not _CUML_AVAILABLE:
            raise RuntimeError("cuML backend requested but cuML is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("cuML backend requested but no CUDA device is available.")
        if use_gpu is False:
            raise RuntimeError("cuML backend requires GPU execution (use_gpu=True).")
        selected_backend = "cuml"
    elif backend_choice == "faiss":
        _, faiss_use_gpu = _resolve_required_faiss_backend(
            prefer_gpu,
            missing_message="FAISS backend requested but faiss is not installed.",
        )
        selected_backend = "faiss"
    else:
        selected_backend = "sklearn"

    y_pred = None
    knn_cpu_model: KNeighborsClassifier | None = None
    knn_backend_printed = False

    if selected_backend == "cuml" and cuKNeighborsClassifier is not None:
        try:
            knn_gpu = cuKNeighborsClassifier(
                n_neighbors=knn_neighbors,
                weights="distance",
            )
            knn_gpu.fit(_to_gpu_array(train_cpu), _to_gpu_array(y_train_cpu))
            y_pred = _to_cpu_numpy(knn_gpu.predict(_to_gpu_array(valid_cpu)))
            y_pred = y_pred.astype(y_valid_cpu.dtype, copy=False)
            print("k-NN Classification: using cuML backend.")
            knn_backend_printed = True
        except Exception as err:  # pragma: no cover - GPU specific failure path
            print(f"cuML k-NN classification failed ({err}); falling back to scikit-learn.")
            selected_backend = "sklearn"

    if y_pred is None and selected_backend == "faiss":
        print(
            f"k-NN Classification: using FAISS {'GPU' if faiss_use_gpu else 'CPU'} backend."
        )
        knn_backend_printed = True
        distances, indices = _faiss_knn_search(
            train_cpu,
            valid_cpu,
            knn_neighbors,
            use_gpu=faiss_use_gpu,
            gpu_id=faiss_gpu_id,
        )
        distances = np.sqrt(np.maximum(np.asarray(distances, dtype=np.float64), 0.0))
        # Avoid division by zero
        weights = 1.0 / np.maximum(distances, 1e-12)
        neighbor_labels = y_train_cpu[indices.astype(np.int64, copy=False)]
        all_labels = np.array(sorted(label_map.keys()), dtype=np.int64)
        y_pred = _faiss_weighted_classification(neighbor_labels, weights, all_labels)

    if y_pred is None:
        if not knn_backend_printed:
            print("k-NN Classification: using scikit-learn backend.")
        knn_cpu_model = KNeighborsClassifier(n_neighbors=knn_neighbors, weights="distance")
        knn_cpu_model.fit(train_cpu, y_train_cpu)
        y_pred = knn_cpu_model.predict(valid_cpu)

    y_pred = np.asarray(y_pred)
    accuracy = accuracy_score(y_valid_cpu, y_pred)
    report = classification_report(
        y_valid_cpu,
        y_pred,
        target_names=list(label_map.values()),
        output_dict=True,
        zero_division=0,
    )
    print(f"k-NN Accuracy on Validation Set: {accuracy:.4f}")

    # --- Confusion Matrix ---
    if enable_plots:
        cm_plot_file = output_dir / "confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(10, 8))
        display_labels = list(label_map.values())
        if knn_cpu_model is not None:
            ConfusionMatrixDisplay.from_estimator(
                knn_cpu_model,
                valid_cpu,
                y_valid_cpu,
                display_labels=display_labels,
                cmap=plt.cm.Blues,
                ax=ax,
                xticks_rotation="vertical",
            )
        else:
            cm = confusion_matrix(y_valid_cpu, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical")
        ax.set_title(f"Confusion Matrix (k-NN={knn_neighbors})")
        plt.tight_layout()
        plt.savefig(cm_plot_file, dpi=STATIC_PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved confusion matrix to {cm_plot_file}")

    return accuracy, report


def run_knn_regression(
    train_embeddings,
    valid_embeddings,
    y_train,
    y_valid,
    output_dir: Path,
    knn_neighbors,
    backend: str = "auto",
    use_gpu: bool | None = None,
    faiss_gpu_id: int = 0,
):
    """k-NN regression for continuous labels (e.g., VAD)."""
    print("\nRunning k-NN Regression evaluation...")

    cuda_available = torch.cuda.is_available()
    if use_gpu is True and not cuda_available:
        raise RuntimeError("GPU execution requested for k-NN but CUDA is not available.")

    train_cpu = np.asarray(train_embeddings, dtype=np.float32)
    valid_cpu = np.asarray(valid_embeddings, dtype=np.float32)
    y_train_cpu = np.asarray(y_train, dtype=np.float32)
    y_valid_cpu = np.asarray(y_valid, dtype=np.float32)

    backend_choice = (backend or "auto").lower()
    prefer_gpu = cuda_available if use_gpu is None else bool(use_gpu)
    faiss_use_gpu = False

    if backend_choice == "auto":
        use_cuml = prefer_gpu and _should_use_cuml()
        if use_cuml:
            selected_backend = "cuml"
        else:
            faiss_available, faiss_use_gpu = _resolve_faiss_backend(
                None if use_gpu is None else prefer_gpu,
                strict=False,
            )
            selected_backend = "faiss" if faiss_available else "sklearn"
    elif backend_choice == "cuml":
        if not _CUML_AVAILABLE:
            raise RuntimeError("cuML backend requested but cuML is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("cuML backend requested but no CUDA device is available.")
        if use_gpu is False:
            raise RuntimeError("cuML backend requires GPU execution (use_gpu=True).")
        selected_backend = "cuml"
    elif backend_choice == "faiss":
        _, faiss_use_gpu = _resolve_required_faiss_backend(
            prefer_gpu,
            missing_message="FAISS backend requested but faiss is not installed.",
        )
        selected_backend = "faiss"
    else:
        selected_backend = "sklearn"

    y_pred = None

    if selected_backend == "cuml" and cuKNeighborsRegressor is not None:
        try:
            knn_gpu = cuKNeighborsRegressor(
                n_neighbors=knn_neighbors,
                weights="distance",
            )
            knn_gpu.fit(_to_gpu_array(train_cpu), _to_gpu_array(y_train_cpu))
            y_pred = _to_cpu_numpy(knn_gpu.predict(_to_gpu_array(valid_cpu)))
            print("k-NN Regression: using cuML backend.")
        except Exception as err:  # pragma: no cover - GPU specific failure path
            print(f"cuML k-NN regression failed ({err}); falling back to scikit-learn.")
            selected_backend = "sklearn"

    if y_pred is None and selected_backend == "faiss":
        print(
            f"k-NN Regression: using FAISS {'GPU' if faiss_use_gpu else 'CPU'} backend."
        )
        distances, indices = _faiss_knn_search(
            train_cpu,
            valid_cpu,
            knn_neighbors,
            use_gpu=faiss_use_gpu,
            gpu_id=faiss_gpu_id,
        )
        distances = np.sqrt(np.maximum(np.asarray(distances, dtype=np.float64), 0.0))
        weights = 1.0 / np.maximum(distances, 1e-12)
        y_train_matrix = y_train_cpu
        if y_train_matrix.ndim == 1:
            y_train_matrix = y_train_matrix[:, None]
        neighbor_targets = y_train_matrix[indices.astype(np.int64, copy=False)]
        preds_matrix = _faiss_weighted_regression(neighbor_targets, weights)
        y_pred = preds_matrix[:, 0] if y_train_cpu.ndim == 1 else preds_matrix

    if y_pred is None:
        print("k-NN Regression: using scikit-learn backend.")
        knn = KNeighborsRegressor(n_neighbors=knn_neighbors, weights="distance")
        knn.fit(train_cpu, y_train_cpu)
        y_pred = knn.predict(valid_cpu)

    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_valid_cpu, y_pred)
    r2 = r2_score(y_valid_cpu, y_pred)

    print(f"k-NN Regression MSE on Validation Set: {mse:.4f}")
    print(f"k-NN Regression R2 Score on Validation Set: {r2:.4f}")

    # 結果を辞書として保存
    report = {"mean_squared_error": mse, "r2_score": r2}
    report_path = output_dir / "regression_report.json"
    pd.Series(report).to_json(report_path, indent=4)

    return mse, r2


def run_consistency_check(
    X_train,
    y_train,
    X_valid,
    cfg: AppConfig,
    output_dir: Path,
    y_valid=None,

    dataset_embeddings=None,
    embeddings_list=None,
    labels_list=None,

    dataset_ids=None,
    enable_plots: bool = True,
    step: int | None = None,
    log_to_wandb: bool | None = None,
):

    print("\n--- Step 6: Running Consistency Check ---")
    check_cfg = cfg.consistency_check

    if log_to_wandb is None:
        log_to_wandb = wandb.run is not None

    # Between-datasets consistency
    if check_cfg.mode == "datasets":
        if dataset_embeddings is None or labels_list is None:
            raise ValueError(
                "dataset_embeddings and labels_list must be provided when mode='datasets'"
            )
        scores, pairs, ids_runs = consistency_score(
            embeddings=dataset_embeddings,
            labels=labels_list,
            dataset_ids=dataset_ids,
            between="datasets",
        )

        mean_score = scores.mean()
        if log_to_wandb and wandb.run is not None:
            wandb.log({"consistency_score_datasets": mean_score}, step=step)
        print(f"Mean consistency score (datasets): {mean_score:.4f}")

        if enable_plots:
            ax = plot_consistency(scores, pairs, ids_runs)
            plot_path = output_dir / "consistency_plot_datasets.png"
            ax.figure.savefig(plot_path, dpi=STATIC_PLOT_DPI, bbox_inches="tight")
            plt.close(ax.figure)
            if log_to_wandb and wandb.run is not None:
                wandb.save(str(plot_path))

        return mean_score, None

    num_runs = check_cfg.num_runs

    # Disable persistent DataLoader workers to prevent accumulation across runs
    original_persistent = cfg.cebra.persistent_workers
    cfg.cebra.persistent_workers = False

    model_paths = []
    for i in tqdm(range(num_runs), desc="Training models for consistency check"):
        arch = normalize_model_architecture(cfg.cebra.model_architecture)
        conditional_value = cfg.cebra.conditional
        if isinstance(conditional_value, str) and conditional_value.strip().lower() in {
            "none",
            "null",
            "",
        }:
            conditional_value = None
        model = cebra.CEBRA(
            model_architecture=arch,
            output_dimension=cfg.cebra.output_dim,
            max_iterations=cfg.cebra.max_iterations,
            batch_size=cfg.cebra.params.batch_size,
            learning_rate=cfg.cebra.params.learning_rate,
            conditional=conditional_value,
            device=cfg.device,
        )
        if y_train is None:
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pt",
            prefix=f"cebra_consistency_{os.getpid()}_{i}_",
        ) as tmp:
            tmp_file = Path(tmp.name)
        model.save(str(tmp_file))
        model_paths.append(tmp_file)
        del model
        gc.collect()
        clear_cuda_cache()

    train_embeddings = []
    valid_embeddings = []
    for tmp_file in tqdm(model_paths, desc="Transforming with saved models"):
        loaded_model = None
        try:
            loaded_model = cebra.CEBRA.load(str(tmp_file))
            train_embeddings.append(loaded_model.transform(X_train))
            valid_embeddings.append(loaded_model.transform(X_valid))
        finally:
            if loaded_model is not None:
                del loaded_model
            tmp_file.unlink(missing_ok=True)
            gc.collect()
            clear_cuda_cache()

    train_mean = valid_mean = None
    for name, embeddings in [("train", train_embeddings), ("valid", valid_embeddings)]:
        print(f"\nComputing consistency for {name} data...")
        scores, pairs, ids_runs = consistency_score(
            embeddings=embeddings, between="runs"
        )

        mean_score = scores.mean()
        if log_to_wandb and wandb.run is not None:
            wandb.log({f"consistency_score_{name}": mean_score}, step=step)
        print(f"Mean consistency score ({name}): {mean_score:.4f}")
        if name == "train":
            train_mean = mean_score
        else:
            valid_mean = mean_score

        if enable_plots:
            ax = plot_consistency(scores, pairs, ids_runs)
            plot_path = output_dir / f"consistency_plot_{name}.png"

            # Axesオブジェクト(ax)の親であるFigureオブジェクト(ax.figure)に対してsavefigを実行
            ax.figure.savefig(plot_path, dpi=STATIC_PLOT_DPI, bbox_inches="tight")

            # Figureを閉じる
            plt.close(ax.figure)
            if log_to_wandb and wandb.run is not None:
                wandb.save(str(plot_path))

    if (
        embeddings_list is not None
        and labels_list is not None
        and dataset_ids is not None
    ):
        print("\nComputing consistency across datasets...")
        scores, pairs, ids_datasets = consistency_score(
            embeddings=embeddings_list,
            labels=labels_list,
            dataset_ids=dataset_ids,
            between="datasets",
        )

        dataset_mean = scores.mean()
        if log_to_wandb and wandb.run is not None:
            wandb.log({"consistency_score_datasets": dataset_mean}, step=step)
        print(f"Mean consistency score (datasets): {dataset_mean:.4f}")

        if enable_plots:
            ax = plot_consistency(scores, pairs, ids_datasets)
            plot_path = output_dir / "consistency_plot_datasets.png"

            ax.figure.savefig(plot_path, dpi=STATIC_PLOT_DPI, bbox_inches="tight")
            plt.close(ax.figure)
            if log_to_wandb and wandb.run is not None:
                wandb.save(str(plot_path))

    # Restore original persistent_workers setting
    cfg.cebra.persistent_workers = original_persistent
    return train_mean, valid_mean
