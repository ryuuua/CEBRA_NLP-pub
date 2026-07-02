from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ..cebra_trainer import (
    get_cebra_output_dir,
    get_label_drift_output_dir,
    load_cebra_model,
    transform_cebra,
)
from ..config_schema import AppConfig
from ..data import load_and_prepare_dataset
from ..embeddings import (
    load_or_generate_embedding_collection,
    load_or_generate_embeddings,
)
from ..label_overlay import LabelOverlaySpec, load_or_generate_label_overlay_embeddings
from ..plotting import prepare_plot_labels
from ..results import (
    _compute_aligned_label_centroids,
    fit_pca_projection,
    project_with_pca_components,
)
from ..utils import apply_reproducibility
from .common import align_by_ids

TRAJECTORY_SAMPLE_ALPHA = 0.22
TRAJECTORY_SAMPLE_ALPHA_CLEAN = 0.12
TRAJECTORY_SAMPLE_MARKER_SIZE_2D = 8
TRAJECTORY_SAMPLE_MARKER_SIZE_3D = 5
TRAJECTORY_OVERLAY_LINE_ALPHA = 0.22
TRAJECTORY_OVERLAY_LINE_WIDTH = 1.0
TRAJECTORY_CENTROID_LINE_ALPHA = 0.10
TRAJECTORY_CENTROID_LINE_WIDTH = 0.85
TRAJECTORY_OVERLAY_MARKER_SIZE_2D = 12
TRAJECTORY_OVERLAY_MARKER_SIZE_3D = 8
TRAJECTORY_OVERLAY_MARKER = "o"
TRAJECTORY_OVERLAY_MARKER_ALPHA = 0.96
TRAJECTORY_CENTROID_MARKER_SIZE_2D = 12
TRAJECTORY_CENTROID_MARKER_SIZE_3D = 10
TRAJECTORY_CENTROID_MARKER_ALPHA = 0.34
TRAJECTORY_STATIC_FIGSIZE = (14.0, 12.0)
TRAJECTORY_ANIMATION_FIGSIZE = (14.0, 12.0)
TRAJECTORY_STATIC_DPI = 340
TRAJECTORY_ANIMATION_DPI = 150


def validate_trajectory_requirements(cfg: AppConfig) -> None:
    if not bool(getattr(cfg.trajectory_analysis, "enabled", False)):
        return
    if str(getattr(cfg.cebra, "conditional", "none")).lower() != "discrete":
        raise ValueError(
            "trajectory_analysis is only supported for discrete CEBRA runs."
        )
    if str(getattr(cfg.trajectory_analysis, "centroid_scope", "train")).lower() != "train":
        raise ValueError(
            "trajectory_analysis.centroid_scope only supports 'train' in v1."
        )
    render_dims = int(getattr(cfg.trajectory_analysis, "render_dims", 3))
    if render_dims not in {2, 3}:
        raise ValueError(
            "trajectory_analysis.render_dims only supports 2 or 3, got "
            f"{render_dims}."
        )
    if not (getattr(cfg.dataset, "label_map", None) or {}):
        raise ValueError(
            "trajectory_analysis requires dataset.label_map for discrete labels."
        )


def load_label_drift_checkpoint_records(
    trajectory_dir: Path,
) -> list[dict[str, object]]:
    trajectory_dir = Path(trajectory_dir)
    manifest_path = trajectory_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoints = manifest.get("checkpoints", [])
        if isinstance(checkpoints, list) and checkpoints:
            records: list[dict[str, object]] = []
            for item in checkpoints:
                if not isinstance(item, dict):
                    continue
                relative_path = item.get("relative_path")
                path_value = item.get("path")
                resolved_path = None
                if isinstance(relative_path, str) and relative_path:
                    candidate = Path(relative_path)
                    resolved_path = trajectory_dir / candidate
                    if not resolved_path.exists():
                        parts = candidate.parts
                        if parts and parts[0] == trajectory_dir.name:
                            resolved_path = trajectory_dir.joinpath(*parts[1:])
                elif isinstance(path_value, str) and path_value:
                    candidate = Path(path_value)
                    resolved_path = (
                        candidate
                        if candidate.is_absolute()
                        else trajectory_dir / candidate
                    )
                if resolved_path is None:
                    continue
                records.append(
                    {
                        "step": int(item.get("step", 0)),
                        "estimated_epoch": float(item.get("estimated_epoch", 0.0)),
                        "path": resolved_path,
                        "relative_path": str(resolved_path.relative_to(trajectory_dir)),
                    }
                )
            if records:
                return sorted(records, key=lambda item: int(item["step"]))

    checkpoint_dir = trajectory_dir / "checkpoints"
    checkpoint_paths = sorted(checkpoint_dir.glob("step_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No label drift checkpoints found under {checkpoint_dir}."
        )

    records = []
    for checkpoint_path in checkpoint_paths:
        stem = checkpoint_path.stem
        step_text = stem.split("_", 1)[1] if "_" in stem else "0"
        records.append(
            {
                "step": int(step_text),
                "estimated_epoch": 0.0,
                "path": checkpoint_path,
                "relative_path": str(checkpoint_path.relative_to(trajectory_dir)),
            }
        )
    return records


def _trajectory_filename(stem: str, render_dims: int, extension: str) -> str:
    if render_dims == 2:
        return f"{stem}.{extension}"
    return f"{stem}_{render_dims}d.{extension}"


def _projection_array_key(prefix: str, render_dims: int) -> str:
    return f"{prefix}_{render_dims}d"


def _compute_trajectory_axis_limits(
    render_dims: int,
    *trajectory_values: np.ndarray | None,
) -> np.ndarray:
    mins = np.full(render_dims, np.inf, dtype=np.float32)
    maxs = np.full(render_dims, -np.inf, dtype=np.float32)
    found = False
    for values in trajectory_values:
        if values is None:
            continue
        coords = np.asarray(values, dtype=np.float32)
        if coords.size == 0:
            continue
        coords = coords.reshape(-1, render_dims)
        finite_mask = np.all(np.isfinite(coords), axis=1)
        if not np.any(finite_mask):
            continue
        visible = coords[finite_mask]
        mins = np.minimum(mins, np.min(visible, axis=0))
        maxs = np.maximum(maxs, np.max(visible, axis=0))
        found = True
    if not found:
        return np.asarray([[-1.0, 1.0]] * render_dims, dtype=np.float32)

    span = np.maximum(maxs - mins, 1e-6)
    padding = span * 0.05
    return np.stack([mins - padding, maxs + padding], axis=1).astype(np.float32)


def _deterministic_display_indices(
    total_samples: int,
    limit: int | None,
) -> np.ndarray:
    if total_samples <= 0:
        return np.zeros(0, dtype=np.int64)
    if limit is None or limit <= 0 or total_samples <= limit:
        return np.arange(total_samples, dtype=np.int64)
    return np.unique(
        np.linspace(0, total_samples - 1, num=limit, dtype=np.int64)
    )


def _set_axis_limits(ax, axis_limits: np.ndarray, render_dims: int) -> None:
    ax.set_xlim(float(axis_limits[0, 0]), float(axis_limits[0, 1]))
    ax.set_ylim(float(axis_limits[1, 0]), float(axis_limits[1, 1]))
    if render_dims == 3:
        ax.set_zlim(float(axis_limits[2, 0]), float(axis_limits[2, 1]))


def _set_point_offsets(scatter, values: np.ndarray, render_dims: int) -> None:
    coords = np.asarray(values, dtype=np.float32)
    if render_dims == 2:
        if coords.size == 0:
            scatter.set_offsets(np.empty((0, 2), dtype=np.float32))
            return
        scatter.set_offsets(coords.reshape(1, 2))
        return

    if coords.size == 0:
        scatter._offsets3d = ([], [], [])  # type: ignore[attr-defined]
        return
    reshaped = coords.reshape(1, 3)
    scatter._offsets3d = (  # type: ignore[attr-defined]
        reshaped[:, 0],
        reshaped[:, 1],
        reshaped[:, 2],
    )


def _set_cloud_offsets(scatter, values: np.ndarray, render_dims: int) -> None:
    coords = np.asarray(values, dtype=np.float32)
    if render_dims == 2:
        if coords.size == 0:
            scatter.set_offsets(np.empty((0, 2), dtype=np.float32))
            return
        scatter.set_offsets(coords[:, :2])
        return

    if coords.size == 0:
        scatter._offsets3d = ([], [], [])  # type: ignore[attr-defined]
        return
    scatter._offsets3d = (  # type: ignore[attr-defined]
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
    )


def _set_line_data(line, values: np.ndarray, render_dims: int) -> None:
    coords = np.asarray(values, dtype=np.float32)
    if coords.size == 0:
        line.set_data([], [])
        if render_dims == 3:
            line.set_3d_properties([])
        return
    line.set_data(coords[:, 0], coords[:, 1])
    if render_dims == 3:
        line.set_3d_properties(coords[:, 2])


def _with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    return mcolors.to_rgba(color, alpha=max(0.0, min(1.0, float(alpha))))


def _trajectory_output_name(
    stem: str,
    render_dims: int,
    extension: str,
    *,
    variant_suffix: str | None = None,
) -> str:
    filename = _trajectory_filename(stem, render_dims, extension)
    if not variant_suffix:
        return filename
    base, _, ext = filename.rpartition(".")
    return f"{base}_{variant_suffix}.{ext}"


def _trajectory_annotation_text(*, show_centroids: bool) -> str:
    if show_centroids:
        return "Bright dot = label overlay, dim dot = train centroid"
    return "Bright dot = label overlay"


def _build_trajectory_legend_handles(
    overlay_spec: LabelOverlaySpec,
    palette: dict[str, str] | None,
    *,
    include_sample_cloud: bool,
    show_centroids: bool,
    show_trajectory_lines: bool,
    sample_alpha: float,
) -> list[Line2D]:
    handles: list[Line2D] = []
    if include_sample_cloud:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor="gray",
                markeredgecolor="gray",
                alpha=sample_alpha,
                label="Train samples (label-colored)",
            )
        )
    for label_index, label_name in enumerate(overlay_spec.label_names):
        color = (
            palette.get(label_name, f"C{label_index}")
            if palette is not None
            else f"C{label_index}"
        )
        linestyle = "-" if show_trajectory_lines else "None"
        marker = TRAJECTORY_OVERLAY_MARKER
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=TRAJECTORY_OVERLAY_LINE_WIDTH if show_trajectory_lines else 0.0,
                alpha=TRAJECTORY_OVERLAY_LINE_ALPHA if show_trajectory_lines else 1.0,
                linestyle=linestyle,
                marker=marker,
                markersize=7,
                markerfacecolor=_with_alpha(color, TRAJECTORY_OVERLAY_MARKER_ALPHA),
                markeredgecolor="none",
                markeredgewidth=0.0,
                label=label_name,
            )
        )
    if show_centroids:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                linewidth=TRAJECTORY_CENTROID_LINE_WIDTH if show_trajectory_lines else 0.0,
                alpha=TRAJECTORY_CENTROID_LINE_ALPHA if show_trajectory_lines else 1.0,
                linestyle="--" if show_trajectory_lines else "None",
                marker="o",
                markersize=5,
                markerfacecolor=_with_alpha("#555555", TRAJECTORY_CENTROID_MARKER_ALPHA),
                markeredgecolor="none",
                markeredgewidth=0.0,
                label="Centroid marker",
            )
        )
    return handles


def build_label_drift_metrics_frame(
    overlay_spec: LabelOverlaySpec,
    checkpoint_records: Sequence[dict[str, object]],
    overlay_cebra_trajectory: np.ndarray,
    centroid_cebra_trajectory: np.ndarray,
    overlay_pca_trajectory: np.ndarray,
    centroid_pca_trajectory: np.ndarray,
    sample_counts: np.ndarray,
) -> pd.DataFrame:
    overlay_cebra = np.asarray(overlay_cebra_trajectory, dtype=np.float32)
    centroid_cebra = np.asarray(centroid_cebra_trajectory, dtype=np.float32)
    overlay_pca = np.asarray(overlay_pca_trajectory, dtype=np.float32)
    centroid_pca = np.asarray(centroid_pca_trajectory, dtype=np.float32)
    counts = np.asarray(sample_counts, dtype=np.int64)
    pca_dims = int(overlay_pca.shape[-1])

    rows: list[dict[str, object]] = []
    for checkpoint_index, record in enumerate(checkpoint_records):
        for label_index, (overlay_id, label_id, label_name, overlay_text) in enumerate(
            zip(
                overlay_spec.ids.tolist(),
                overlay_spec.label_ids.tolist(),
                overlay_spec.label_names,
                overlay_spec.texts,
            )
        ):
            row: dict[str, object] = {
                "checkpoint_index": int(checkpoint_index),
                "step": int(record["step"]),
                "estimated_epoch": float(record["estimated_epoch"]),
                "checkpoint_file": str(record["relative_path"]),
                "overlay_id": overlay_id,
                "label_id": int(label_id),
                "label_name": label_name,
                "overlay_text": overlay_text,
                "sample_count": int(counts[label_index]),
            }

            delta_pca = (
                overlay_pca[checkpoint_index, label_index]
                - centroid_pca[checkpoint_index, label_index]
            )
            for dim_index in range(pca_dims):
                row[f"overlay_pca_dim_{dim_index + 1}"] = float(
                    overlay_pca[checkpoint_index, label_index, dim_index]
                )
                row[f"centroid_pca_dim_{dim_index + 1}"] = float(
                    centroid_pca[checkpoint_index, label_index, dim_index]
                )
                row[f"delta_pca_dim_{dim_index + 1}"] = float(delta_pca[dim_index])
            row["delta_pca_l2"] = float(np.linalg.norm(delta_pca))

            overlay_vec = overlay_cebra[checkpoint_index, label_index]
            centroid_vec = centroid_cebra[checkpoint_index, label_index]
            delta_cebra = overlay_vec - centroid_vec
            row["delta_cebra_l2"] = float(np.linalg.norm(delta_cebra))
            overlay_norm = float(np.linalg.norm(overlay_vec))
            centroid_norm = float(np.linalg.norm(centroid_vec))
            row["delta_cebra_cosine"] = float("nan")
            if overlay_norm > 0.0 and centroid_norm > 0.0:
                row["delta_cebra_cosine"] = float(
                    np.dot(overlay_vec, centroid_vec) / (overlay_norm * centroid_norm)
                )

            for dim_index, value in enumerate(overlay_vec, start=1):
                row[f"overlay_cebra_dim_{dim_index}"] = float(value)
            for dim_index, value in enumerate(centroid_vec, start=1):
                row[f"centroid_cebra_dim_{dim_index}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def _save_projected_step_artifact(
    output_dir: Path,
    *,
    record: dict[str, object],
    render_dims: int,
    sample_ids: np.ndarray,
    sample_label_names: np.ndarray,
    sample_projection: np.ndarray,
    centroid_projection: np.ndarray,
    overlay_projection: np.ndarray,
    overlay_spec: LabelOverlaySpec,
    sample_counts: np.ndarray,
) -> Path:
    projected_dir = output_dir / "projected_steps"
    projected_dir.mkdir(parents=True, exist_ok=True)
    path = projected_dir / f"step_{int(record['step']):06d}.npz"
    sample_projection = np.asarray(sample_projection, dtype=np.float32)
    centroid_projection = np.asarray(centroid_projection, dtype=np.float32)
    overlay_projection = np.asarray(overlay_projection, dtype=np.float32)
    np.savez(
        path,
        step=np.asarray([int(record["step"])], dtype=np.int64),
        estimated_epoch=np.asarray([float(record["estimated_epoch"])], dtype=np.float32),
        sample_ids=np.asarray(sample_ids, dtype=str),
        sample_label_names=np.asarray(sample_label_names, dtype=str),
        label_ids=np.asarray(overlay_spec.label_ids, dtype=np.int64),
        label_names=np.asarray(overlay_spec.label_names, dtype=str),
        overlay_ids=np.asarray(overlay_spec.ids, dtype=str),
        sample_counts=np.asarray(sample_counts, dtype=np.int64),
        sample_pca=sample_projection,
        centroid_pca=centroid_projection,
        label_pca=overlay_projection,
        **{
            _projection_array_key("sample_pca", render_dims): sample_projection,
            _projection_array_key("centroid_pca", render_dims): centroid_projection,
            _projection_array_key("label_pca", render_dims): overlay_projection,
        },
    )
    return path


def _save_label_drift_pca_panel(
    output_path: Path,
    *,
    overlay_spec: LabelOverlaySpec,
    overlay_pca_trajectory: np.ndarray,
    centroid_pca_trajectory: np.ndarray,
    sample_projection: np.ndarray,
    sample_label_names: np.ndarray,
    palette: dict[str, str] | None,
    explained_variance_ratio: np.ndarray,
    axis_limits: np.ndarray,
    render_dims: int,
    include_sample_cloud: bool,
    show_centroids: bool,
    show_trajectory_lines: bool,
    sample_alpha: float,
) -> Path:
    if render_dims == 3:
        fig = plt.figure(figsize=TRAJECTORY_STATIC_FIGSIZE)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=TRAJECTORY_STATIC_FIGSIZE)

    sample_labels = np.asarray(sample_label_names, dtype=str)
    if include_sample_cloud and sample_projection.size:
        for label_index, label_name in enumerate(overlay_spec.label_names):
            color = (
                palette.get(label_name, f"C{label_index}")
                if palette is not None
                else f"C{label_index}"
            )
            label_mask = sample_labels == str(label_name)
            if not np.any(label_mask):
                continue
            label_projection = sample_projection[label_mask]
            if render_dims == 3:
                ax.scatter(
                    label_projection[:, 0],
                    label_projection[:, 1],
                    label_projection[:, 2],
                    s=TRAJECTORY_SAMPLE_MARKER_SIZE_3D,
                    c=[color],
                    alpha=sample_alpha,
                    depthshade=False,
                    linewidths=0,
                )
            else:
                ax.scatter(
                    label_projection[:, 0],
                    label_projection[:, 1],
                    s=TRAJECTORY_SAMPLE_MARKER_SIZE_2D,
                    c=[color],
                    alpha=sample_alpha,
                    linewidths=0,
                )

    for label_index, label_name in enumerate(overlay_spec.label_names):
        color = (
            palette.get(label_name, f"C{label_index}")
            if palette is not None
            else f"C{label_index}"
        )
        overlay_coords = np.asarray(
            overlay_pca_trajectory[:, label_index, :],
            dtype=np.float32,
        )
        centroid_coords = np.asarray(
            centroid_pca_trajectory[:, label_index, :],
            dtype=np.float32,
        )
        overlay_mask = np.all(np.isfinite(overlay_coords), axis=1)
        centroid_mask = np.all(np.isfinite(centroid_coords), axis=1)
        if np.any(overlay_mask):
            overlay_visible = overlay_coords[overlay_mask]
            if render_dims == 3:
                if show_trajectory_lines:
                    ax.plot(
                        overlay_visible[:, 0],
                        overlay_visible[:, 1],
                        overlay_visible[:, 2],
                        color=_with_alpha(color, TRAJECTORY_OVERLAY_LINE_ALPHA),
                        linewidth=TRAJECTORY_OVERLAY_LINE_WIDTH,
                        label=label_name,
                    )
                ax.scatter(
                    [overlay_visible[-1, 0]],
                    [overlay_visible[-1, 1]],
                    [overlay_visible[-1, 2]],
                    marker=TRAJECTORY_OVERLAY_MARKER,
                    s=TRAJECTORY_OVERLAY_MARKER_SIZE_3D,
                    c=[_with_alpha(color, TRAJECTORY_OVERLAY_MARKER_ALPHA)],
                    edgecolors="none",
                    linewidths=0.0,
                    depthshade=False,
                )
                ax.text(
                    float(overlay_visible[-1, 0]),
                    float(overlay_visible[-1, 1]),
                    float(overlay_visible[-1, 2]),
                    label_name,
                    fontsize=9,
                )
            else:
                if show_trajectory_lines:
                    ax.plot(
                        overlay_visible[:, 0],
                        overlay_visible[:, 1],
                        color=_with_alpha(color, TRAJECTORY_OVERLAY_LINE_ALPHA),
                        linewidth=TRAJECTORY_OVERLAY_LINE_WIDTH,
                        label=label_name,
                    )
                ax.scatter(
                    [overlay_visible[-1, 0]],
                    [overlay_visible[-1, 1]],
                    marker=TRAJECTORY_OVERLAY_MARKER,
                    s=TRAJECTORY_OVERLAY_MARKER_SIZE_2D,
                    c=[_with_alpha(color, TRAJECTORY_OVERLAY_MARKER_ALPHA)],
                    edgecolors="none",
                    linewidths=0.0,
                    zorder=4,
                )
                ax.text(
                    float(overlay_visible[-1, 0]),
                    float(overlay_visible[-1, 1]),
                    label_name,
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    zorder=5,
                )
        if show_centroids and np.any(centroid_mask):
            centroid_visible = centroid_coords[centroid_mask]
            if render_dims == 3:
                if show_trajectory_lines:
                    ax.plot(
                        centroid_visible[:, 0],
                        centroid_visible[:, 1],
                        centroid_visible[:, 2],
                        color=_with_alpha(color, TRAJECTORY_CENTROID_LINE_ALPHA),
                        linewidth=TRAJECTORY_CENTROID_LINE_WIDTH,
                        linestyle="--",
                    )
                ax.scatter(
                    [centroid_visible[-1, 0]],
                    [centroid_visible[-1, 1]],
                    [centroid_visible[-1, 2]],
                    marker="o",
                    s=TRAJECTORY_CENTROID_MARKER_SIZE_3D,
                    c=[_with_alpha(color, TRAJECTORY_CENTROID_MARKER_ALPHA)],
                    edgecolors="none",
                    linewidths=0.0,
                    depthshade=False,
                )
            else:
                if show_trajectory_lines:
                    ax.plot(
                        centroid_visible[:, 0],
                        centroid_visible[:, 1],
                        color=_with_alpha(color, TRAJECTORY_CENTROID_LINE_ALPHA),
                        linewidth=TRAJECTORY_CENTROID_LINE_WIDTH,
                        linestyle="--",
                    )
                ax.scatter(
                    [centroid_visible[-1, 0]],
                    [centroid_visible[-1, 1]],
                    marker="o",
                    s=TRAJECTORY_CENTROID_MARKER_SIZE_2D,
                    c=[_with_alpha(color, TRAJECTORY_CENTROID_MARKER_ALPHA)],
                    edgecolors="none",
                    linewidths=0.0,
                    zorder=3,
                )

    ax.set_title("Step-wise label drift in final-train PCA basis")
    ax.set_xlabel(f"PCA 1 ({explained_variance_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PCA 2 ({explained_variance_ratio[1] * 100:.1f}%)")
    if render_dims == 3:
        ax.set_zlabel(f"PCA 3 ({explained_variance_ratio[2] * 100:.1f}%)")
        ax.view_init(elev=22, azim=38)
    _set_axis_limits(ax, axis_limits, render_dims)
    if render_dims == 3:
        ax.text2D(
            0.01,
            0.99,
            _trajectory_annotation_text(show_centroids=show_centroids),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    else:
        ax.text(
            0.01,
            0.99,
            _trajectory_annotation_text(show_centroids=show_centroids),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    ax.legend(
        handles=_build_trajectory_legend_handles(
            overlay_spec,
            palette,
            include_sample_cloud=include_sample_cloud and sample_projection.size > 0,
            show_centroids=show_centroids,
            show_trajectory_lines=show_trajectory_lines,
            sample_alpha=sample_alpha,
        ),
        title="Series",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=TRAJECTORY_STATIC_DPI,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)
    return output_path


def _save_label_drift_distance_panel(
    output_path: Path,
    *,
    metrics_frame: pd.DataFrame,
    overlay_spec: LabelOverlaySpec,
    palette: dict[str, str] | None,
) -> Path:
    metric_specs = [
        ("delta_cebra_l2", "CEBRA L2 distance"),
        ("delta_cebra_cosine", "CEBRA cosine similarity"),
        ("delta_pca_l2", "PCA L2 distance"),
    ]
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(12, 12), sharex=True)
    sorted_frame = metrics_frame.sort_values(["step", "label_id"]).copy()
    for axis, (metric_name, title) in zip(axes, metric_specs):
        for label_index, label_name in enumerate(overlay_spec.label_names):
            label_frame = sorted_frame[sorted_frame["label_name"] == label_name]
            if label_frame.empty:
                continue
            color = (
                palette.get(label_name, f"C{label_index}")
                if palette is not None
                else f"C{label_index}"
            )
            axis.plot(
                label_frame["step"].to_numpy(),
                label_frame[metric_name].to_numpy(),
                marker="o",
                linewidth=1.5,
                markersize=3,
                color=color,
                label=label_name,
            )
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    axes[0].legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[-1].set_xlabel("Training step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output_path


def _downsample_frame_indices(total_frames: int, max_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if max_frames <= 0 or total_frames <= max_frames:
        return np.arange(total_frames, dtype=np.int64)
    return np.unique(
        np.linspace(0, total_frames - 1, num=max_frames, dtype=np.int64)
    )


def _select_render_checkpoint_records(
    checkpoint_records: Sequence[dict[str, object]],
    checkpoint_stride: int,
) -> list[dict[str, object]]:
    records = list(checkpoint_records)
    if checkpoint_stride <= 1 or len(records) <= 1:
        return records
    selected = records[:: checkpoint_stride]
    if selected[-1] is not records[-1]:
        selected.append(records[-1])
    return selected


def _save_label_drift_animation(
    output_dir: Path,
    *,
    overlay_spec: LabelOverlaySpec,
    checkpoint_records: Sequence[dict[str, object]],
    projected_step_paths: Sequence[Path],
    overlay_pca_trajectory: np.ndarray,
    centroid_pca_trajectory: np.ndarray,
    sample_label_names: np.ndarray,
    palette: dict[str, str] | None,
    explained_variance_ratio: np.ndarray,
    axis_limits: np.ndarray,
    fps: int,
    max_frames: int,
    render_dims: int,
    include_sample_cloud: bool,
    show_centroids: bool,
    show_trajectory_lines: bool,
    sample_alpha: float,
    variant_suffix: str | None = None,
) -> dict[str, str]:
    frame_indices = _downsample_frame_indices(len(checkpoint_records), max_frames)
    if frame_indices.size == 0:
        return {}

    overlay_display = np.asarray(overlay_pca_trajectory[frame_indices], dtype=np.float32)
    centroid_display = np.asarray(
        centroid_pca_trajectory[frame_indices],
        dtype=np.float32,
    )
    records_display = [checkpoint_records[int(index)] for index in frame_indices.tolist()]
    projected_display_paths = [
        projected_step_paths[int(index)] for index in frame_indices.tolist()
    ]

    if render_dims == 3:
        fig = plt.figure(figsize=TRAJECTORY_ANIMATION_FIGSIZE)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=TRAJECTORY_ANIMATION_FIGSIZE)
    _set_axis_limits(ax, axis_limits, render_dims)
    ax.set_xlabel(f"PCA 1 ({explained_variance_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PCA 2 ({explained_variance_ratio[1] * 100:.1f}%)")
    if render_dims == 3:
        ax.set_zlabel(f"PCA 3 ({explained_variance_ratio[2] * 100:.1f}%)")
        ax.view_init(elev=22, azim=38)
    title = ax.set_title("Step-wise label drift in final-train PCA basis")
    if render_dims == 3:
        ax.text2D(
            0.01,
            0.99,
            _trajectory_annotation_text(show_centroids=show_centroids),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    else:
        ax.text(
            0.01,
            0.99,
            _trajectory_annotation_text(show_centroids=show_centroids),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

    sample_labels = np.asarray(sample_label_names, dtype=str)
    sample_clouds = []
    for label_index, label_name in enumerate(overlay_spec.label_names):
        color = (
            palette.get(label_name, f"C{label_index}")
            if palette is not None
            else f"C{label_index}"
        )
        if render_dims == 3:
            sample_cloud = ax.scatter(
                [],
                [],
                [],
                s=TRAJECTORY_SAMPLE_MARKER_SIZE_3D,
                c=[color],
                alpha=sample_alpha,
                depthshade=False,
                linewidths=0,
            )
        else:
            sample_cloud = ax.scatter(
                [],
                [],
                s=TRAJECTORY_SAMPLE_MARKER_SIZE_2D,
                c=[color],
                alpha=sample_alpha,
                linewidths=0,
            )
        sample_clouds.append(sample_cloud)
    overlay_lines = []
    overlay_points = []
    centroid_lines = []
    centroid_points = []
    for label_index, label_name in enumerate(overlay_spec.label_names):
        color = (
            palette.get(label_name, f"C{label_index}")
            if palette is not None
            else f"C{label_index}"
        )
        if render_dims == 3:
            overlay_line, = ax.plot(
                [],
                [],
                [],
                color=_with_alpha(color, TRAJECTORY_OVERLAY_LINE_ALPHA),
                linewidth=TRAJECTORY_OVERLAY_LINE_WIDTH,
            )
            overlay_point = ax.scatter(
                [],
                [],
                [],
                marker=TRAJECTORY_OVERLAY_MARKER,
                s=TRAJECTORY_OVERLAY_MARKER_SIZE_3D,
                c=[_with_alpha(color, TRAJECTORY_OVERLAY_MARKER_ALPHA)],
                edgecolors="none",
                linewidths=0.0,
                depthshade=False,
            )
            centroid_line, = ax.plot(
                [],
                [],
                [],
                color=_with_alpha(color, TRAJECTORY_CENTROID_LINE_ALPHA),
                linewidth=TRAJECTORY_CENTROID_LINE_WIDTH,
                linestyle="--",
            )
            centroid_point = ax.scatter(
                [],
                [],
                [],
                marker="o",
                s=TRAJECTORY_CENTROID_MARKER_SIZE_3D,
                c=[_with_alpha(color, TRAJECTORY_CENTROID_MARKER_ALPHA)],
                edgecolors="none",
                linewidths=0.0,
                depthshade=False,
            )
        else:
            overlay_line, = ax.plot(
                [],
                [],
                color=_with_alpha(color, TRAJECTORY_OVERLAY_LINE_ALPHA),
                linewidth=TRAJECTORY_OVERLAY_LINE_WIDTH,
            )
            overlay_point = ax.scatter(
                [],
                [],
                marker=TRAJECTORY_OVERLAY_MARKER,
                s=TRAJECTORY_OVERLAY_MARKER_SIZE_2D,
                c=[_with_alpha(color, TRAJECTORY_OVERLAY_MARKER_ALPHA)],
                edgecolors="none",
                linewidths=0.0,
                zorder=4,
            )
            centroid_line, = ax.plot(
                [],
                [],
                color=_with_alpha(color, TRAJECTORY_CENTROID_LINE_ALPHA),
                linewidth=TRAJECTORY_CENTROID_LINE_WIDTH,
                linestyle="--",
            )
            centroid_point = ax.scatter(
                [],
                [],
                marker="o",
                s=TRAJECTORY_CENTROID_MARKER_SIZE_2D,
                c=[_with_alpha(color, TRAJECTORY_CENTROID_MARKER_ALPHA)],
                edgecolors="none",
                linewidths=0.0,
                zorder=3,
            )
        overlay_lines.append(overlay_line)
        overlay_points.append(overlay_point)
        centroid_lines.append(centroid_line)
        centroid_points.append(centroid_point)

    ax.legend(
        handles=_build_trajectory_legend_handles(
            overlay_spec,
            palette,
            include_sample_cloud=include_sample_cloud,
            show_centroids=show_centroids,
            show_trajectory_lines=show_trajectory_lines,
            sample_alpha=sample_alpha,
        ),
        title="Series",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    fig.tight_layout()

    def _update(frame_index: int):
        record = records_display[frame_index]
        title.set_text(
            "Step-wise label drift in final-train PCA basis "
            f"(step={int(record['step'])}, epoch={float(record['estimated_epoch']):.2f})"
        )
        artists = [title]
        sample_projection = np.empty((0, render_dims), dtype=np.float32)
        if include_sample_cloud:
            with np.load(projected_display_paths[frame_index], allow_pickle=True) as payload:
                sample_projection = np.asarray(payload["sample_pca"], dtype=np.float32)
        for label_index, label_name in enumerate(overlay_spec.label_names):
            label_mask = sample_labels == str(label_name)
            label_projection = sample_projection[label_mask] if sample_projection.size else np.empty((0, render_dims), dtype=np.float32)
            _set_cloud_offsets(sample_clouds[label_index], label_projection, render_dims)
            artists.append(sample_clouds[label_index])
        for label_index in range(len(overlay_spec.label_names)):
            overlay_coords = overlay_display[: frame_index + 1, label_index, :]
            overlay_mask = np.all(np.isfinite(overlay_coords), axis=1)
            overlay_visible = overlay_coords[overlay_mask]
            if show_trajectory_lines:
                _set_line_data(overlay_lines[label_index], overlay_visible, render_dims)
            else:
                _set_line_data(
                    overlay_lines[label_index],
                    np.empty((0, render_dims), dtype=np.float32),
                    render_dims,
                )
            _set_point_offsets(
                overlay_points[label_index],
                overlay_visible[-1] if overlay_visible.size else np.empty((0, render_dims)),
                render_dims,
            )

            centroid_coords = centroid_display[: frame_index + 1, label_index, :]
            centroid_mask = np.all(np.isfinite(centroid_coords), axis=1)
            centroid_visible = centroid_coords[centroid_mask]
            if show_centroids:
                if show_trajectory_lines:
                    _set_line_data(
                        centroid_lines[label_index],
                        centroid_visible,
                        render_dims,
                    )
                else:
                    _set_line_data(
                        centroid_lines[label_index],
                        np.empty((0, render_dims), dtype=np.float32),
                        render_dims,
                    )
                _set_point_offsets(
                    centroid_points[label_index],
                    centroid_visible[-1]
                    if centroid_visible.size
                    else np.empty((0, render_dims)),
                    render_dims,
                )
            else:
                _set_line_data(
                    centroid_lines[label_index],
                    np.empty((0, render_dims), dtype=np.float32),
                    render_dims,
                )
                _set_point_offsets(
                    centroid_points[label_index],
                    np.empty((0, render_dims), dtype=np.float32),
                    render_dims,
                )
            artists.extend(
                [
                    overlay_lines[label_index],
                    overlay_points[label_index],
                ]
            )
            if show_centroids:
                artists.extend(
                    [
                        centroid_lines[label_index],
                        centroid_points[label_index],
                    ]
                )
        return artists

    animation = FuncAnimation(
        fig,
        _update,
        frames=len(records_display),
        interval=max(1, int(round(1000 / max(1, fps)))),
        blit=False,
    )

    outputs: dict[str, str] = {}
    gif_path = output_dir / _trajectory_output_name(
        "label_drift_pca",
        render_dims,
        "gif",
        variant_suffix=variant_suffix,
    )
    try:
        animation.save(
            gif_path,
            writer=PillowWriter(fps=max(1, fps)),
            dpi=TRAJECTORY_ANIMATION_DPI,
        )
        outputs["gif"] = gif_path.name
    except Exception as err:
        print(f"[WARN] Could not save label drift GIF: {err}")

    if shutil.which("ffmpeg") is not None:
        mp4_path = output_dir / _trajectory_output_name(
            "label_drift_pca",
            render_dims,
            "mp4",
            variant_suffix=variant_suffix,
        )
        try:
            animation.save(
                mp4_path,
                writer=FFMpegWriter(fps=max(1, fps)),
                dpi=TRAJECTORY_ANIMATION_DPI,
            )
            outputs["mp4"] = mp4_path.name
        except Exception as err:
            print(f"[WARN] Could not save label drift MP4: {err}")

    plt.close(fig)
    return outputs


def render_label_drift_trajectory(
    cfg: AppConfig,
    *,
    artifact_dir: Path,
    hydra_output_dir: Path | None = None,
) -> Path:
    if not bool(getattr(cfg.trajectory_analysis, "enabled", False)):
        raise ValueError(
            "visualize_trajectory requires trajectory_analysis.enabled=true."
        )
    validate_trajectory_requirements(cfg)
    apply_reproducibility(cfg)

    render_dims = int(getattr(cfg.trajectory_analysis, "render_dims", 3))
    include_sample_cloud = bool(
        getattr(cfg.trajectory_analysis, "include_sample_cloud", True)
    )
    show_centroids = bool(getattr(cfg.trajectory_analysis, "show_centroids", True))
    show_trajectory_lines = bool(
        getattr(cfg.trajectory_analysis, "show_trajectory_lines", True)
    )
    export_clean_variant = bool(
        getattr(cfg.trajectory_analysis, "export_clean_variant", False)
    )

    print("\n--- Trajectory Viz Stage: Loading dataset ---")
    texts, conditional_data, _time_indices, ids = load_and_prepare_dataset(cfg)
    ids = np.asarray(ids, dtype=str)
    conditional_data = np.asarray(conditional_data)

    print("\n--- Trajectory Viz Stage: Loading cached embeddings ---")
    if getattr(cfg, "embedding_collection", None) and cfg.embedding_collection.embeddings:
        X_vectors = load_or_generate_embedding_collection(
            cfg,
            texts,
            ids,
            labels=conditional_data,
            require_cache=cfg.stage.require_cache,
        )
    else:
        X_vectors = load_or_generate_embeddings(
            cfg,
            texts,
            ids,
            labels=conditional_data,
            require_cache=cfg.stage.require_cache,
        )

    train_ids_path = Path(artifact_dir) / "train_ids.npy"
    if not train_ids_path.exists():
        raise FileNotFoundError(
            f"train_ids.npy is required for trajectory centroid analysis, not found at {train_ids_path}."
        )
    train_ids = np.asarray(np.load(train_ids_path, allow_pickle=True), dtype=str)
    X_train = align_by_ids(ids, np.asarray(X_vectors, dtype=np.float32), train_ids)
    conditional_train = align_by_ids(ids, conditional_data, train_ids)
    labels_train, palette, _order = prepare_plot_labels(cfg, conditional_train)

    overlay_spec, overlay_input_embeddings = load_or_generate_label_overlay_embeddings(
        cfg,
        require_cache=cfg.stage.require_cache,
        force_enabled=True,
    )
    if overlay_spec is None or overlay_input_embeddings is None:
        raise ValueError(
            "trajectory_analysis requires label overlay inputs, but they could not be resolved."
        )

    trajectory_dir = get_label_drift_output_dir(Path(artifact_dir))
    checkpoint_records = load_label_drift_checkpoint_records(trajectory_dir)
    render_checkpoint_stride = int(
        getattr(cfg.trajectory_analysis, "render_checkpoint_stride", 1)
    )
    if render_checkpoint_stride <= 0:
        raise ValueError(
            "trajectory_analysis.render_checkpoint_stride must be >= 1, got "
            f"{render_checkpoint_stride}."
        )
    render_checkpoint_records = _select_render_checkpoint_records(
        checkpoint_records,
        render_checkpoint_stride,
    )

    final_model_path = Path(artifact_dir) / "cebra_model.pt"
    pca_basis_path = Path(checkpoint_records[-1]["path"])
    if final_model_path.exists() and int(checkpoint_records[-1]["step"]) < int(cfg.cebra.max_iterations):
        pca_basis_path = final_model_path

    final_train_cebra_embeddings = transform_cebra(
        load_cebra_model(pca_basis_path, cfg, X_train.shape[1]),
        X_train,
        cfg.device,
    )
    if final_train_cebra_embeddings.shape[1] < render_dims:
        raise ValueError(
            "trajectory_analysis.render_dims requires at least that many final CEBRA dimensions, "
            f"got render_dims={render_dims}, final_embedding_dim={final_train_cebra_embeddings.shape[1]}."
        )
    pca_projection = fit_pca_projection(
        final_train_cebra_embeddings,
        cfg=cfg,
        n_components=render_dims,
    )

    display_indices = (
        _deterministic_display_indices(
            final_train_cebra_embeddings.shape[0],
            getattr(cfg.pca_analysis, "plot_sample_limit", None),
        )
        if include_sample_cloud
        else np.zeros(0, dtype=np.int64)
    )
    display_sample_ids = (
        train_ids[display_indices]
        if include_sample_cloud
        else np.zeros(0, dtype=str)
    )
    display_sample_label_names = (
        np.asarray(labels_train, dtype=str)[display_indices]
        if include_sample_cloud
        else np.zeros(0, dtype=str)
    )
    np.save(trajectory_dir / "sample_display_ids.npy", np.asarray(display_sample_ids, dtype=str))

    overlay_cebra_frames = []
    centroid_cebra_frames = []
    overlay_pca_frames = []
    centroid_pca_frames = []
    projected_step_paths: list[Path] = []
    sample_counts = None
    axis_limit_candidates: list[np.ndarray] = []
    final_sample_projection = np.empty((0, render_dims), dtype=np.float32)

    for checkpoint_index, record in enumerate(render_checkpoint_records):
        checkpoint_path = Path(record["path"])
        model = load_cebra_model(checkpoint_path, cfg, X_train.shape[1])
        train_cebra_embeddings = transform_cebra(model, X_train, cfg.device).astype(np.float32)
        overlay_cebra_embeddings = transform_cebra(
            model,
            overlay_input_embeddings,
            cfg.device,
        ).astype(np.float32)
        centroid_cebra_embeddings, centroid_counts = _compute_aligned_label_centroids(
            train_cebra_embeddings,
            labels_train,
            overlay_spec.label_names,
        )
        if sample_counts is None:
            sample_counts = centroid_counts.astype(np.int64)

        overlay_pca_embeddings = project_with_pca_components(
            overlay_cebra_embeddings,
            pca_projection["mean"],
            pca_projection["components"],
        ).astype(np.float32)
        centroid_pca_embeddings = project_with_pca_components(
            centroid_cebra_embeddings,
            pca_projection["mean"],
            pca_projection["components"],
        ).astype(np.float32)
        sample_projection = (
            project_with_pca_components(
                train_cebra_embeddings[display_indices],
                pca_projection["mean"],
                pca_projection["components"],
            ).astype(np.float32)
            if include_sample_cloud and display_indices.size > 0
            else np.empty((0, render_dims), dtype=np.float32)
        )

        overlay_cebra_frames.append(overlay_cebra_embeddings)
        centroid_cebra_frames.append(centroid_cebra_embeddings.astype(np.float32))
        overlay_pca_frames.append(overlay_pca_embeddings)
        centroid_pca_frames.append(centroid_pca_embeddings)
        projected_step_paths.append(
            _save_projected_step_artifact(
                trajectory_dir,
                record=record,
                render_dims=render_dims,
                sample_ids=display_sample_ids,
                sample_label_names=display_sample_label_names,
                sample_projection=sample_projection,
                centroid_projection=centroid_pca_embeddings,
                overlay_projection=overlay_pca_embeddings,
                overlay_spec=overlay_spec,
                sample_counts=centroid_counts.astype(np.int64),
            )
        )
        axis_limit_candidates.extend(
            [sample_projection, overlay_pca_embeddings, centroid_pca_embeddings]
        )
        if checkpoint_index == len(render_checkpoint_records) - 1:
            final_sample_projection = sample_projection

    overlay_cebra_trajectory = np.stack(overlay_cebra_frames, axis=0)
    centroid_cebra_trajectory = np.stack(centroid_cebra_frames, axis=0)
    overlay_pca_trajectory = np.stack(overlay_pca_frames, axis=0)
    centroid_pca_trajectory = np.stack(centroid_pca_frames, axis=0)
    resolved_counts = (
        np.asarray(sample_counts, dtype=np.int64)
        if sample_counts is not None
        else np.zeros(len(overlay_spec.label_names), dtype=np.int64)
    )
    axis_limits = _compute_trajectory_axis_limits(
        render_dims,
        *axis_limit_candidates,
    )

    np.save(trajectory_dir / "label_overlay_cebra_trajectory.npy", overlay_cebra_trajectory)
    np.save(
        trajectory_dir / "label_centroid_cebra_trajectory.npy",
        centroid_cebra_trajectory,
    )
    np.save(trajectory_dir / "label_overlay_pca_trajectory.npy", overlay_pca_trajectory)
    np.save(
        trajectory_dir / "label_centroid_pca_trajectory.npy",
        centroid_pca_trajectory,
    )
    np.savez(
        trajectory_dir / "pca_model_final_train.npz",
        mean=pca_projection["mean"],
        components=pca_projection["components"],
        explained_variance_ratio=pca_projection["explained_variance_ratio"],
        fit_scope=np.asarray(["final_train"]),
        render_dims=np.asarray([render_dims], dtype=np.int64),
        axis_limits=axis_limits,
    )

    metrics_frame = build_label_drift_metrics_frame(
        overlay_spec,
        render_checkpoint_records,
        overlay_cebra_trajectory,
        centroid_cebra_trajectory,
        overlay_pca_trajectory,
        centroid_pca_trajectory,
        resolved_counts,
    )
    metrics_path = trajectory_dir / "label_drift_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    artifacts: dict[str, str] = {
        "pca_model": "pca_model_final_train.npz",
        "overlay_cebra_trajectory": "label_overlay_cebra_trajectory.npy",
        "centroid_cebra_trajectory": "label_centroid_cebra_trajectory.npy",
        "overlay_pca_trajectory": "label_overlay_pca_trajectory.npy",
        "centroid_pca_trajectory": "label_centroid_pca_trajectory.npy",
        "sample_display_ids": "sample_display_ids.npy",
        "projected_steps_dir": "projected_steps",
        "metrics_csv": metrics_path.name,
    }
    if bool(getattr(cfg.trajectory_analysis, "export_static_panels", True)):
        pca_plot_path = _save_label_drift_pca_panel(
            trajectory_dir
            / _trajectory_output_name("label_drift_pca", render_dims, "png"),
            overlay_spec=overlay_spec,
            overlay_pca_trajectory=overlay_pca_trajectory,
            centroid_pca_trajectory=centroid_pca_trajectory,
            sample_projection=final_sample_projection,
            sample_label_names=display_sample_label_names,
            palette=palette,
            explained_variance_ratio=pca_projection["explained_variance_ratio"],
            axis_limits=axis_limits,
            render_dims=render_dims,
            include_sample_cloud=include_sample_cloud,
            show_centroids=show_centroids,
            show_trajectory_lines=show_trajectory_lines,
            sample_alpha=TRAJECTORY_SAMPLE_ALPHA,
        )
        distance_plot_path = _save_label_drift_distance_panel(
            trajectory_dir / "label_drift_distance.png",
            metrics_frame=metrics_frame,
            overlay_spec=overlay_spec,
            palette=palette,
        )
        artifacts["pca_plot"] = pca_plot_path.name
        artifacts["distance_plot"] = distance_plot_path.name
        if export_clean_variant:
            pca_plot_clean_path = _save_label_drift_pca_panel(
                trajectory_dir
                / _trajectory_output_name(
                    "label_drift_pca",
                    render_dims,
                    "png",
                    variant_suffix="clean",
                ),
                overlay_spec=overlay_spec,
                overlay_pca_trajectory=overlay_pca_trajectory,
                centroid_pca_trajectory=centroid_pca_trajectory,
                sample_projection=final_sample_projection,
                sample_label_names=display_sample_label_names,
                palette=palette,
                explained_variance_ratio=pca_projection["explained_variance_ratio"],
                axis_limits=axis_limits,
                render_dims=render_dims,
                include_sample_cloud=include_sample_cloud,
                show_centroids=False,
                show_trajectory_lines=False,
                sample_alpha=TRAJECTORY_SAMPLE_ALPHA_CLEAN,
            )
            artifacts["pca_plot_clean"] = pca_plot_clean_path.name

    if bool(getattr(cfg.trajectory_analysis, "export_animation", True)):
        artifacts.update(
            _save_label_drift_animation(
                trajectory_dir,
                overlay_spec=overlay_spec,
                checkpoint_records=render_checkpoint_records,
                projected_step_paths=projected_step_paths,
                overlay_pca_trajectory=overlay_pca_trajectory,
                centroid_pca_trajectory=centroid_pca_trajectory,
                sample_label_names=display_sample_label_names,
                palette=palette,
                explained_variance_ratio=pca_projection["explained_variance_ratio"],
                axis_limits=axis_limits,
                fps=int(getattr(cfg.trajectory_analysis, "fps", 8)),
                max_frames=int(getattr(cfg.trajectory_analysis, "max_frames", 180)),
                render_dims=render_dims,
                include_sample_cloud=include_sample_cloud,
                show_centroids=show_centroids,
                show_trajectory_lines=show_trajectory_lines,
                sample_alpha=TRAJECTORY_SAMPLE_ALPHA,
            )
        )
        if export_clean_variant:
            clean_outputs = _save_label_drift_animation(
                trajectory_dir,
                overlay_spec=overlay_spec,
                checkpoint_records=render_checkpoint_records,
                projected_step_paths=projected_step_paths,
                overlay_pca_trajectory=overlay_pca_trajectory,
                centroid_pca_trajectory=centroid_pca_trajectory,
                sample_label_names=display_sample_label_names,
                palette=palette,
                explained_variance_ratio=pca_projection["explained_variance_ratio"],
                axis_limits=axis_limits,
                fps=int(getattr(cfg.trajectory_analysis, "fps", 8)),
                max_frames=int(getattr(cfg.trajectory_analysis, "max_frames", 180)),
                render_dims=render_dims,
                include_sample_cloud=include_sample_cloud,
                show_centroids=False,
                show_trajectory_lines=False,
                sample_alpha=TRAJECTORY_SAMPLE_ALPHA_CLEAN,
                variant_suffix="clean",
            )
            for key, value in clean_outputs.items():
                artifacts[f"{key}_clean"] = value

    manifest_path = trajectory_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "centroid_scope": "train",
            "render_dims": int(render_dims),
            "include_sample_cloud": bool(include_sample_cloud),
            "show_centroids": bool(show_centroids),
            "show_trajectory_lines": bool(show_trajectory_lines),
            "export_clean_variant": bool(export_clean_variant),
            "render_checkpoint_stride": int(render_checkpoint_stride),
            "source_checkpoint_count": int(len(checkpoint_records)),
            "num_checkpoints": int(len(render_checkpoint_records)),
            "num_labels": int(len(overlay_spec.label_names)),
            "num_display_samples": int(display_sample_ids.shape[0]),
            "num_train_samples": int(train_ids.shape[0]),
            "label_names": list(overlay_spec.label_names),
            "label_ids": [int(label_id) for label_id in overlay_spec.label_ids.tolist()],
            "sample_counts": [int(value) for value in resolved_counts.tolist()],
            "projected_steps": [
                {
                    "step": int(record["step"]),
                    "estimated_epoch": float(record["estimated_epoch"]),
                    "relative_path": str(path.relative_to(trajectory_dir)),
                }
                for record, path in zip(render_checkpoint_records, projected_step_paths)
            ],
            "pca_basis": {
                "fit_scope": "final_train",
                "path": "pca_model_final_train.npz",
                "source_model_path": str(pca_basis_path.name),
                "explained_variance_ratio": [
                    float(value) for value in pca_projection["explained_variance_ratio"].tolist()
                ],
            },
            "artifacts": artifacts,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if hydra_output_dir is not None:
        hydra_output_dir.mkdir(parents=True, exist_ok=True)
        (hydra_output_dir / "trajectory_artifact_dir.txt").write_text(
            str(trajectory_dir),
            encoding="utf-8",
        )

    print(f"Label drift trajectory artifacts written to {trajectory_dir}.")
    return trajectory_dir


def run(cfg: AppConfig, output_dir: Path, *, is_main_process: bool) -> Path | None:
    if not is_main_process:
        return None
    artifact_dir = get_cebra_output_dir(cfg)
    return render_label_drift_trajectory(
        cfg,
        artifact_dir=artifact_dir,
        hydra_output_dir=output_dir,
    )


__all__ = [
    "build_label_drift_metrics_frame",
    "load_label_drift_checkpoint_records",
    "render_label_drift_trajectory",
    "run",
    "validate_trajectory_requirements",
]
