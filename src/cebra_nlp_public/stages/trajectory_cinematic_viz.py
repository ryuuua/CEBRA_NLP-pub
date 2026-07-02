from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import moderngl
except Exception:
    moderngl = None

from .cinematic_gpu_renderer import (
    CameraFrame,
    CinematicGPUFrameRenderer,
    FrameSpec,
    LineBatch,
    ParticleBatch,
    RendererOptions,
    is_moderngl_available,
)
from ..cebra_trainer import get_cebra_output_dir, get_label_drift_output_dir
from ..config_schema import AppConfig
from ..utils import apply_reproducibility


ICY_BLUE = "#b6efff"
DEFAULT_COLOR = "#79c8ff"
NVIDIA_GREEN = "#76B900"
NVIDIA_WHITE = "#FFFFFF"
NVIDIA_DARK = "#1E1E1E"
NVIDIA_CINEMATIC_CYCLE = (
    "#E8FDFF",
    "#A4EEFF",
    "#68D6FF",
    "#5C93FF",
    "#75FFE0",
    NVIDIA_GREEN,
    "#93C7FF",
    "#51F1FF",
)
NVIDIA_LABEL_HINTS = {
    "anger": "#8CEBFF",
    "fear": "#7EA9FF",
    "joy": "#E8FDFF",
    "love": NVIDIA_GREEN,
    "sadness": "#4A82FF",
    "surprise": "#6CFFE6",
    "negative": "#71BBFF",
    "positive": "#E8FDFF",
}

LOOK_PRESET_PROFILES: dict[str, dict[str, Any]] = {
    "glass_wireframe": {
        "sample_layers": ((18.0, 0.24), (8.2, 0.42), (2.8, 1.00)),
        "overlay_layers": ((14.0, 0.22), (5.2, 0.34), (2.2, 1.00)),
        "centroid_layers": ((4.8, 0.05), (1.5, 0.26)),
        "sample_brighten": 0.18,
        "overlay_brighten": 0.22,
        "centroid_brighten": 0.04,
        "sample_cool_mix": 0.05,
        "overlay_cool_mix": 0.18,
        "centroid_cool_mix": 0.08,
        "overlay_trail_alpha": 0.17,
        "overlay_trail_width": 0.75,
        "centroid_trail_alpha": 0.04,
        "centroid_trail_width": 0.32,
        "guide_alpha": 0.42,
        "guide_text_alpha": 0.74,
        "guide_line_width": 1.20,
    },
    "balanced": {
        "sample_layers": ((13.0, 0.16), (4.8, 0.24), (2.1, 0.94)),
        "overlay_layers": ((14.0, 0.24), (5.0, 0.34), (2.3, 1.00)),
        "centroid_layers": ((5.0, 0.05), (1.6, 0.28)),
        "sample_brighten": 0.12,
        "overlay_brighten": 0.20,
        "centroid_brighten": 0.08,
        "sample_cool_mix": 0.10,
        "overlay_cool_mix": 0.16,
        "centroid_cool_mix": 0.08,
        "overlay_trail_alpha": 0.18,
        "overlay_trail_width": 0.82,
        "centroid_trail_alpha": 0.05,
        "centroid_trail_width": 0.35,
        "guide_alpha": 0.32,
        "guide_text_alpha": 0.56,
        "guide_line_width": 0.95,
    },
    "neon_trails": {
        "sample_layers": ((13.0, 0.18), (5.0, 0.26), (2.0, 0.98)),
        "overlay_layers": ((15.0, 0.30), (5.6, 0.46), (2.4, 1.00)),
        "centroid_layers": ((5.0, 0.06), (1.6, 0.30)),
        "sample_brighten": 0.18,
        "overlay_brighten": 0.32,
        "centroid_brighten": 0.10,
        "sample_cool_mix": 0.12,
        "overlay_cool_mix": 0.22,
        "centroid_cool_mix": 0.10,
        "overlay_trail_alpha": 0.20,
        "overlay_trail_width": 0.90,
        "centroid_trail_alpha": 0.06,
        "centroid_trail_width": 0.38,
        "guide_alpha": 0.34,
        "guide_text_alpha": 0.58,
        "guide_line_width": 1.00,
    },
}


def _with_alpha(color: str | tuple[float, float, float], alpha: float) -> tuple[float, float, float, float]:
    return mcolors.to_rgba(color, alpha=max(0.0, min(1.0, float(alpha))))


def _brighten(color: str | tuple[float, float, float], amount: float = 0.28) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float32)
    return tuple(np.clip(rgb + (1.0 - rgb) * float(amount), 0.0, 1.0).tolist())


def _mix_colors(
    left: str | tuple[float, float, float],
    right: str | tuple[float, float, float],
    amount: float,
) -> tuple[float, float, float]:
    left_rgb = np.asarray(mcolors.to_rgb(left), dtype=np.float32)
    right_rgb = np.asarray(mcolors.to_rgb(right), dtype=np.float32)
    blend = float(max(0.0, min(1.0, amount)))
    return tuple(np.clip(left_rgb * (1.0 - blend) + right_rgb * blend, 0.0, 1.0).tolist())


def _cool_tint(color: str | tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    return _mix_colors(color, ICY_BLUE, amount)


def _resolve_cinematic_palette(
    label_names: Sequence[str],
    palette: dict[str, str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    fallback_index = 0
    for label_name in label_names:
        key = str(label_name)
        lower_key = key.strip().lower()
        if lower_key in NVIDIA_LABEL_HINTS:
            resolved[key] = NVIDIA_LABEL_HINTS[lower_key]
            continue
        if key in palette and palette[key]:
            resolved[key] = mcolors.to_hex(
                _brighten(_cool_tint(str(palette[key]), 0.42), 0.12)
            )
            continue
        resolved[key] = NVIDIA_CINEMATIC_CYCLE[fallback_index % len(NVIDIA_CINEMATIC_CYCLE)]
        fallback_index += 1
    return resolved


def _master_variants(cfg: AppConfig) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    if _beauty_master_enabled(cfg):
        beauty_cfg = copy.deepcopy(cfg)
        beauty_cfg.cinematic_render.look_preset = "glass_wireframe"
        beauty_cfg.cinematic_render.camera_auto_zoom_out = True
        variants.append(
            {
                "key": "beauty_master",
                "stem": "label_drift_beauty_master",
                "cfg": beauty_cfg,
                "camera_mode": "auto_zoom_out",
                "role": "beauty",
            }
        )
    if bool(getattr(cfg.cinematic_render, "export_analysis_master", True)):
        analysis_cfg = copy.deepcopy(cfg)
        analysis_cfg.cinematic_render.look_preset = "balanced"
        analysis_cfg.cinematic_render.camera_auto_zoom_out = False
        analysis_cfg.cinematic_render.glow_gain = min(
            float(getattr(cfg.cinematic_render, "glow_gain", 0.9)),
            0.42,
        )
        analysis_cfg.cinematic_render.trail_length = min(
            int(getattr(cfg.cinematic_render, "trail_length", 36)),
            18,
        )
        variants.append(
            {
                "key": "analysis_master",
                "stem": "label_drift_analysis_master",
                "cfg": analysis_cfg,
                "camera_mode": "fixed_full",
                "role": "analysis",
            }
        )
    return variants


def _beauty_master_enabled(cfg: AppConfig) -> bool:
    return bool(
        getattr(cfg.cinematic_render, "export_beauty_master", True)
        or getattr(cfg.cinematic_render, "export_cinematic_master", False)
    )


def _master_role(master_key: str) -> str:
    return "beauty" if str(master_key) == "beauty_master" else "analysis"


def _master_trail_policy(cfg: AppConfig, master_key: str) -> str:
    if str(master_key) == "beauty_master":
        return str(getattr(cfg.cinematic_render, "beauty_trail_mode", "label_only")).strip().lower()
    return "overlay_centroid"


def _master_supersample_scale(cfg: AppConfig, master_key: str, kind: str) -> float:
    fallback = max(1.0, float(getattr(cfg.cinematic_render, "supersample_scale", 1.0)))
    if str(master_key) == "beauty_master":
        if kind == "poster":
            return max(
                1.0,
                float(
                    getattr(
                        cfg.cinematic_render,
                        "beauty_poster_supersample_scale",
                        fallback,
                    )
                ),
            )
        return max(
            1.0,
            float(
                getattr(
                    cfg.cinematic_render,
                    "beauty_video_supersample_scale",
                    fallback,
                )
            ),
        )
    if str(master_key) == "analysis_master":
        return max(
            1.0,
            float(getattr(cfg.cinematic_render, "analysis_supersample_scale", fallback)),
        )
    return fallback


def _master_depth_settings(cfg: AppConfig, master_key: str) -> tuple[float, float, float]:
    if str(master_key) == "beauty_master":
        return (
            0.24,
            float(getattr(cfg.cinematic_render, "beauty_depth_fog_strength", 0.20)),
            float(getattr(cfg.cinematic_render, "beauty_depth_fog_cool_mix", 0.18)),
        )
    beauty_strength = float(getattr(cfg.cinematic_render, "beauty_depth_fog_strength", 0.20))
    beauty_cool_mix = float(getattr(cfg.cinematic_render, "beauty_depth_fog_cool_mix", 0.18))
    return (0.08, min(0.08, beauty_strength * 0.35), min(0.06, beauty_cool_mix * 0.35))


def _master_particle_scale_multipliers(cfg: AppConfig, master_key: str) -> tuple[float, float]:
    if str(master_key) != "beauty_master":
        return (1.0, 1.0)
    return (
        max(0.2, float(getattr(cfg.cinematic_render, "beauty_particle_core_scale", 1.0))),
        max(0.5, float(getattr(cfg.cinematic_render, "beauty_particle_halo_scale", 1.0))),
    )


def _master_hold_frame_count(cfg: AppConfig, master_key: str) -> int:
    if str(master_key) != "beauty_master":
        return 0
    fps = max(1, int(getattr(cfg.cinematic_render, "fps", 18)))
    hold_seconds = max(0.0, float(getattr(cfg.cinematic_render, "beauty_hold_final_seconds", 1.0)))
    return int(round(fps * hold_seconds))


def _master_output_frame_indices(cfg: AppConfig, master_key: str, source_frame_count: int) -> np.ndarray:
    if source_frame_count <= 0:
        return np.zeros(0, dtype=np.int64)
    indices = np.arange(source_frame_count, dtype=np.int64)
    hold_frames = _master_hold_frame_count(cfg, master_key)
    if hold_frames <= 0:
        return indices
    hold = np.full(hold_frames, source_frame_count - 1, dtype=np.int64)
    return np.concatenate([indices, hold], axis=0)


def _master_camera_schedule(
    cfg: AppConfig,
    master_key: str,
    *,
    camera_mode: str,
    source_frame_count: int,
    output_frame_count: int,
) -> dict[str, Any]:
    return {
        "mode": str(camera_mode),
        "source_frame_count": int(source_frame_count),
        "output_frame_count": int(output_frame_count),
        "hold_final_seconds": (
            float(getattr(cfg.cinematic_render, "beauty_hold_final_seconds", 1.0))
            if str(master_key) == "beauty_master"
            else 0.0
        ),
    }


def _output_name(stem: str, render_dims: int, extension: str, *, preview: bool = False) -> str:
    suffix = "" if render_dims == 2 else f"_{render_dims}d"
    preview_suffix = "_preview" if preview else ""
    return f"{stem}{preview_suffix}{suffix}.{extension}"


def _downsample_frame_indices(total_frames: int, max_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if max_frames <= 0 or total_frames <= max_frames:
        return np.arange(total_frames, dtype=np.int64)
    return np.unique(np.linspace(0, total_frames - 1, num=max_frames, dtype=np.int64))


def _projection_key(prefix: str, render_dims: int) -> str:
    return f"{prefix}_{render_dims}d"


def _set_axis_limits(ax, axis_limits: np.ndarray) -> None:
    ax.set_xlim(float(axis_limits[0, 0]), float(axis_limits[0, 1]))
    ax.set_ylim(float(axis_limits[1, 0]), float(axis_limits[1, 1]))
    ax.set_zlim(float(axis_limits[2, 0]), float(axis_limits[2, 1]))


def _set_cloud_offsets(scatter, values: np.ndarray) -> None:
    coords = np.asarray(values, dtype=np.float32)
    if coords.size == 0:
        scatter._offsets3d = ([], [], [])  # type: ignore[attr-defined]
        return
    scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])  # type: ignore[attr-defined]


def _set_point_offsets(scatter, values: np.ndarray) -> None:
    coords = np.asarray(values, dtype=np.float32)
    if coords.size == 0:
        scatter._offsets3d = ([], [], [])  # type: ignore[attr-defined]
        return
    reshaped = coords.reshape(1, 3)
    scatter._offsets3d = (reshaped[:, 0], reshaped[:, 1], reshaped[:, 2])  # type: ignore[attr-defined]


def _set_line_data(line, values: np.ndarray) -> None:
    coords = np.asarray(values, dtype=np.float32)
    if coords.size == 0:
        line.set_data([], [])
        line.set_3d_properties([])
        return
    line.set_data(coords[:, 0], coords[:, 1])
    line.set_3d_properties(coords[:, 2])


def _style_cinematic_axis(ax, *, axis_limits: np.ndarray, background_color: str) -> None:
    ax.set_facecolor(background_color)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    _set_axis_limits(ax, axis_limits)
    try:
        ax.set_box_aspect(
            (
                float(axis_limits[0, 1] - axis_limits[0, 0]),
                float(axis_limits[1, 1] - axis_limits[1, 0]),
                float(axis_limits[2, 1] - axis_limits[2, 0]),
            )
        )
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
            axis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass
        try:
            axis.line.set_color((0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass
    ax.set_axis_off()


def _guide_color(profile: dict[str, Any]) -> tuple[float, float, float, float]:
    return _with_alpha(_cool_tint("#d9f7ff", 0.24), float(profile["guide_alpha"]))


def _guide_text_color(profile: dict[str, Any]) -> tuple[float, float, float, float]:
    return _with_alpha(_brighten(_cool_tint("#d9f7ff", 0.30), 0.04), float(profile["guide_text_alpha"]))


def _draw_corner_guides(ax, *, axis_limits: np.ndarray, profile: dict[str, Any]) -> None:
    mins = axis_limits[:, 0]
    spans = axis_limits[:, 1] - axis_limits[:, 0]
    origin = mins + spans * np.asarray([0.07, 0.09, 0.07], dtype=np.float32)
    lengths = spans * np.asarray([0.12, 0.12, 0.12], dtype=np.float32)
    color = _guide_color(profile)
    text_color = _guide_text_color(profile)
    line_width = float(profile["guide_line_width"])

    axes = (
        (np.asarray([lengths[0], 0.0, 0.0], dtype=np.float32), "X"),
        (np.asarray([0.0, lengths[1], 0.0], dtype=np.float32), "Y"),
        (np.asarray([0.0, 0.0, lengths[2]], dtype=np.float32), "Z"),
    )
    for delta, label in axes:
        tip = origin + delta
        ax.plot(
            [origin[0], tip[0]],
            [origin[1], tip[1]],
            [origin[2], tip[2]],
            color=color,
            linewidth=line_width,
        )
        text_offset = delta * 1.10
        text_pos = origin + text_offset
        ax.text(
            float(text_pos[0]),
            float(text_pos[1]),
            float(text_pos[2]),
            label,
            color=text_color,
            fontsize=9,
            ha="center",
            va="center",
        )


def _draw_thin_full_axes(ax, *, axis_limits: np.ndarray, profile: dict[str, Any]) -> None:
    mins = axis_limits[:, 0]
    maxs = axis_limits[:, 1]
    color = _guide_color(profile)
    text_color = _guide_text_color(profile)
    line_width = float(profile["guide_line_width"])

    axes = (
        (np.asarray([mins[0], mins[1], mins[2]], dtype=np.float32), np.asarray([maxs[0], mins[1], mins[2]], dtype=np.float32), "X"),
        (np.asarray([mins[0], mins[1], mins[2]], dtype=np.float32), np.asarray([mins[0], maxs[1], mins[2]], dtype=np.float32), "Y"),
        (np.asarray([mins[0], mins[1], mins[2]], dtype=np.float32), np.asarray([mins[0], mins[1], maxs[2]], dtype=np.float32), "Z"),
    )
    for start, end, label in axes:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=line_width,
        )
        ax.text(
            float(end[0]),
            float(end[1]),
            float(end[2]),
            label,
            color=text_color,
            fontsize=9,
            ha="center",
            va="center",
        )


def _draw_wireframe_box(ax, *, axis_limits: np.ndarray, profile: dict[str, Any]) -> None:
    mins = axis_limits[:, 0]
    maxs = axis_limits[:, 1]
    color = _guide_color(profile)
    line_width = max(0.6, float(profile["guide_line_width"]) * 0.9)
    vertices = np.asarray(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for start_idx, end_idx in edges:
        start = vertices[start_idx]
        end = vertices[end_idx]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=line_width,
        )


def _add_axis_guides(ax, *, axis_limits: np.ndarray, cfg: AppConfig, profile: dict[str, Any]) -> None:
    axis_style = str(getattr(cfg.cinematic_render, "axis_style", "corner_guides")).strip().lower()
    if axis_style == "corner_guides":
        return
    if axis_style == "thin_full_axes":
        _draw_thin_full_axes(ax, axis_limits=axis_limits, profile=profile)
        return
    if axis_style == "wireframe_box":
        _draw_wireframe_box(ax, axis_limits=axis_limits, profile=profile)
        return


def _load_projected_step_records(trajectory_dir: Path) -> list[dict[str, object]]:
    manifest_path = trajectory_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found under {trajectory_dir}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    projected_steps = manifest.get("projected_steps", [])
    records: list[dict[str, object]] = []
    for item in projected_steps:
        if not isinstance(item, dict):
            continue
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            continue
        path = trajectory_dir / relative_path
        if not path.exists():
            continue
        records.append(
            {
                "step": int(item.get("step", 0)),
                "estimated_epoch": float(item.get("estimated_epoch", 0.0)),
                "path": path,
                "relative_path": relative_path,
            }
        )
    if not records:
        raise FileNotFoundError(
            f"No projected step artifacts referenced by manifest under {trajectory_dir}."
        )
    return sorted(records, key=lambda item: int(item["step"]))


def _load_cinematic_frame_payloads(
    records: Sequence[dict[str, object]],
    *,
    render_dims: int,
) -> dict[str, object]:
    sample_key = _projection_key("sample_pca", render_dims)
    centroid_key = _projection_key("centroid_pca", render_dims)
    overlay_key = _projection_key("label_pca", render_dims)

    sample_frames: list[np.ndarray] = []
    centroid_frames: list[np.ndarray] = []
    overlay_frames: list[np.ndarray] = []
    sample_label_names: np.ndarray | None = None
    label_names: np.ndarray | None = None

    for record in records:
        with np.load(Path(record["path"]), allow_pickle=True) as payload:
            sample_projection = np.asarray(
                payload[sample_key] if sample_key in payload else payload["sample_pca"],
                dtype=np.float32,
            )
            centroid_projection = np.asarray(
                payload[centroid_key]
                if centroid_key in payload
                else payload["centroid_pca"],
                dtype=np.float32,
            )
            overlay_projection = np.asarray(
                payload[overlay_key] if overlay_key in payload else payload["label_pca"],
                dtype=np.float32,
            )
            sample_frames.append(sample_projection)
            centroid_frames.append(centroid_projection)
            overlay_frames.append(overlay_projection)
            if sample_label_names is None:
                sample_label_names = np.asarray(payload["sample_label_names"], dtype=str)
            if label_names is None:
                label_names = np.asarray(payload["label_names"], dtype=str)

    if sample_label_names is None or label_names is None:
        raise ValueError("Projected step artifacts are missing label metadata.")

    return {
        "sample_frames": np.stack(sample_frames, axis=0),
        "centroid_frames": np.stack(centroid_frames, axis=0),
        "overlay_frames": np.stack(overlay_frames, axis=0),
        "sample_label_names": sample_label_names,
        "label_names": label_names,
    }


def _camera_values(cfg: AppConfig, frame_index: int, num_frames: int) -> tuple[float, float]:
    t = 1.0 if num_frames <= 1 else float(frame_index) / float(num_frames - 1)
    azim = float(cfg.cinematic_render.camera_azim_start) + (
        float(cfg.cinematic_render.camera_azim_end)
        - float(cfg.cinematic_render.camera_azim_start)
    ) * t
    elev = float(cfg.cinematic_render.camera_elev) + float(
        cfg.cinematic_render.camera_elev_wobble
    ) * math.sin(t * math.pi)
    return elev, azim


def _camera_basis_vectors(elev: float, azim: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elev_rad = math.radians(float(elev))
    azim_rad = math.radians(float(azim))
    view_dir = np.asarray(
        [
            math.cos(elev_rad) * math.cos(azim_rad),
            math.cos(elev_rad) * math.sin(azim_rad),
            math.sin(elev_rad),
        ],
        dtype=np.float32,
    )
    view_dir = view_dir / max(np.linalg.norm(view_dir), 1e-8)
    forward = -view_dir
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    side = np.cross(forward, world_up)
    side_norm = float(np.linalg.norm(side))
    if side_norm <= 1e-8:
        side = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        side = side / side_norm
    true_up = np.cross(side, forward)
    true_up = true_up / max(np.linalg.norm(true_up), 1e-8)
    return view_dir.astype(np.float32), side.astype(np.float32), true_up.astype(np.float32)


def _default_camera_distance(cfg: AppConfig, axis_limits: np.ndarray) -> float:
    spans = (axis_limits[:, 1] - axis_limits[:, 0]).astype(np.float32)
    distance_scale = max(
        0.20,
        float(getattr(cfg.cinematic_render, "camera_distance_scale", 0.74)),
    )
    return float((np.linalg.norm(spans) * 0.78 + np.max(spans) * 0.46) * distance_scale)


def _frame_points_stack(
    sample_frames: np.ndarray,
    centroid_frames: np.ndarray,
    overlay_frames: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(sample_frames[frame_index], dtype=np.float32),
            np.asarray(centroid_frames[frame_index], dtype=np.float32),
            np.asarray(overlay_frames[frame_index], dtype=np.float32),
        ],
        axis=0,
    )


def _required_camera_distance(
    points: np.ndarray,
    *,
    center: np.ndarray,
    view_dir: np.ndarray,
    side: np.ndarray,
    true_up: np.ndarray,
    fov_y_degrees: float,
    aspect_ratio: float,
    margin: float,
) -> float:
    coords = np.asarray(points, dtype=np.float32)
    if coords.size == 0:
        return 1.0
    rel = coords - center.reshape(1, 3)
    horizontal = np.abs(rel @ side)
    vertical = np.abs(rel @ true_up)
    toward_camera = rel @ view_dir
    usable = max(0.05, min(0.98, float(margin)))
    tan_y = math.tan(math.radians(float(fov_y_degrees)) * 0.5) * usable
    tan_x = tan_y * max(1e-6, float(aspect_ratio))
    horizontal_req = horizontal / max(tan_x, 1e-6) + toward_camera
    vertical_req = vertical / max(tan_y, 1e-6) + toward_camera
    distance = max(float(np.max(horizontal_req)), float(np.max(vertical_req)), 1e-3)
    return distance


def _compute_camera_distance_schedule(
    cfg: AppConfig,
    *,
    sample_frames: np.ndarray,
    centroid_frames: np.ndarray,
    overlay_frames: np.ndarray,
    axis_limits: np.ndarray,
    render_resolution: tuple[int, int],
    mode: str,
) -> np.ndarray:
    frame_count = int(sample_frames.shape[0])
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float32)
    center = np.mean(axis_limits, axis=1).astype(np.float32)
    aspect_ratio = float(render_resolution[0]) / max(1.0, float(render_resolution[1]))
    fov_y = float(getattr(cfg.cinematic_render, "camera_fov_degrees", 23.0))
    margin = float(getattr(cfg.cinematic_render, "camera_zoom_margin", 0.88))
    base_distance = _default_camera_distance(cfg, axis_limits)
    required_distances: list[float] = []
    for frame_index in range(frame_count):
        elev, azim = _camera_values(cfg, frame_index, frame_count)
        view_dir, side, true_up = _camera_basis_vectors(elev, azim)
        required_distance = _required_camera_distance(
            _frame_points_stack(sample_frames, centroid_frames, overlay_frames, frame_index),
            center=center,
            view_dir=view_dir,
            side=side,
            true_up=true_up,
            fov_y_degrees=fov_y,
            aspect_ratio=aspect_ratio,
            margin=margin,
        )
        required_distances.append(max(base_distance, required_distance))
    required = np.asarray(required_distances, dtype=np.float32)
    if mode == "fixed_full":
        return np.full(frame_count, float(np.max(required)), dtype=np.float32)
    power = max(1.0, float(getattr(cfg.cinematic_render, "camera_zoom_curve_power", 1.8)))
    if frame_count == 1:
        return required
    start = float(required[0])
    end = float(np.max(required))
    t = np.linspace(0.0, 1.0, num=frame_count, dtype=np.float32)
    curve = start + (end - start) * np.power(t, power)
    return np.maximum.accumulate(np.maximum(required, curve)).astype(np.float32)


def _compute_reference_axis_directions(
    cfg: AppConfig,
    *,
    axis_limits: np.ndarray,
    render_resolution: tuple[int, int],
    frame_count: int,
    camera_distances: np.ndarray | None,
) -> list[tuple[str, np.ndarray]]:
    if frame_count <= 0:
        return []
    reference_index = frame_count - 1
    reference_distance = (
        float(camera_distances[reference_index])
        if camera_distances is not None and len(camera_distances) > reference_index
        else None
    )
    mvp_matrix, _projection = _orbit_camera_matrices(
        cfg,
        axis_limits=axis_limits,
        frame_index=reference_index,
        frame_count=frame_count,
        render_resolution=render_resolution,
        camera_distance=reference_distance,
    )
    center = np.mean(np.asarray(axis_limits, dtype=np.float32), axis=1)
    spans = np.asarray(axis_limits, dtype=np.float32)[:, 1] - np.asarray(axis_limits, dtype=np.float32)[:, 0]
    axis_scale = max(1e-6, float(np.max(spans)) * 0.22)
    world_points = [center]
    for axis_index in range(3):
        point = center.copy()
        point[axis_index] += axis_scale
        world_points.append(point)
    projected = _project_world_to_screen(
        np.stack(world_points, axis=0).astype(np.float32),
        mvp_matrix=mvp_matrix,
        render_resolution=render_resolution,
    )
    screen_center = projected[0]
    directions: list[tuple[str, np.ndarray]] = []
    for axis_label, point in zip(("PC1", "PC2", "PC3"), projected[1:], strict=False):
        direction = np.asarray(point - screen_center, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            continue
        directions.append((axis_label, direction / norm))
    return directions


def _look_profile(cfg: AppConfig) -> dict[str, Any]:
    preset = str(getattr(cfg.cinematic_render, "look_preset", "glass_wireframe")).strip().lower()
    return dict(LOOK_PRESET_PROFILES.get(preset, LOOK_PRESET_PROFILES["glass_wireframe"]))


def _label_render_colors(
    profile: dict[str, Any],
    resolved_palette: dict[str, str],
    label_name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    base_color = resolved_palette.get(str(label_name), DEFAULT_COLOR)
    sample_color = _brighten(
        _cool_tint(base_color, float(profile["sample_cool_mix"])),
        float(profile["sample_brighten"]),
    )
    overlay_color = _brighten(
        _cool_tint(base_color, float(profile["overlay_cool_mix"])),
        float(profile["overlay_brighten"]),
    )
    centroid_color = _brighten(
        _cool_tint(base_color, float(profile["centroid_cool_mix"])),
        float(profile["centroid_brighten"]),
    )
    return sample_color, overlay_color, centroid_color


def _target_resolution(cfg: AppConfig, kind: str) -> tuple[int, int]:
    if kind == "poster":
        width = int(getattr(cfg.cinematic_render, "poster_width", 0) or 0)
        height = int(getattr(cfg.cinematic_render, "poster_height", 0) or 0)
        dpi = max(1, int(getattr(cfg.cinematic_render, "static_dpi", 240)))
    elif kind == "gif":
        width = int(getattr(cfg.cinematic_render, "gif_width", 0) or 0)
        height = int(getattr(cfg.cinematic_render, "gif_height", 0) or 0)
        dpi = max(1, int(getattr(cfg.cinematic_render, "animation_dpi", 180)))
    else:
        width = int(getattr(cfg.cinematic_render, "video_width", 0) or 0)
        height = int(getattr(cfg.cinematic_render, "video_height", 0) or 0)
        dpi = max(1, int(getattr(cfg.cinematic_render, "animation_dpi", 180)))

    if width > 0 and height > 0:
        return width, height

    fallback_width = max(1, int(round(float(cfg.cinematic_render.figure_width) * dpi)))
    fallback_height = max(1, int(round(float(cfg.cinematic_render.figure_height) * dpi)))
    return fallback_width, fallback_height


def _render_resolution(cfg: AppConfig, kind: str, master_key: str | None = None) -> tuple[int, int]:
    width, height = _target_resolution(cfg, kind)
    supersample_scale = (
        _master_supersample_scale(cfg, master_key, kind)
        if master_key is not None
        else max(1.0, float(getattr(cfg.cinematic_render, "supersample_scale", 1.0)))
    )
    return (
        max(width, int(round(width * supersample_scale))),
        max(height, int(round(height * supersample_scale))),
    )


def _capture_canvas_image(
    fig,
    *,
    cfg: AppConfig,
    target_resolution: tuple[int, int],
    profile: dict[str, Any],
    axis_directions: list[tuple[str, np.ndarray]] | None,
    mvp_matrix: np.ndarray | None,
) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    image = Image.fromarray(rgba, mode="RGBA")
    image = _apply_glow(image, cfg=cfg)
    if image.size != target_resolution:
        image = image.resize(target_resolution, Image.Resampling.LANCZOS)
    return _overlay_axis_guides(
        image.convert("RGB"),
        cfg=cfg,
        profile=profile,
        mvp_matrix=mvp_matrix,
        axis_directions=axis_directions,
    )


def _apply_glow(image: Image.Image, *, cfg: AppConfig) -> Image.Image:
    glow_gain = max(0.0, float(getattr(cfg.cinematic_render, "glow_gain", 0.0)))
    if glow_gain <= 0.0:
        return image

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3:].astype(np.float32) / 255.0

    luminance = np.max(rgb, axis=2)
    threshold = 0.20
    bright_mask = np.clip((luminance - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    bright_rgb = np.clip(rgb * bright_mask[..., None], 0.0, 1.0)
    bright_image = Image.fromarray((bright_rgb * 255.0).astype(np.uint8), mode="RGB")

    blur_small_radius = max(0.0, float(getattr(cfg.cinematic_render, "glow_blur_small_px", 0.0)))
    blur_large_radius = max(0.0, float(getattr(cfg.cinematic_render, "glow_blur_large_px", 0.0)))

    glow_small = bright_image.filter(ImageFilter.GaussianBlur(radius=blur_small_radius))
    glow_large = bright_image.filter(ImageFilter.GaussianBlur(radius=blur_large_radius))
    glow_small_rgb = np.asarray(glow_small, dtype=np.float32) / 255.0
    glow_large_rgb = np.asarray(glow_large, dtype=np.float32) / 255.0

    composite_rgb = np.clip(
        rgb
        + glow_gain * (0.58 * glow_small_rgb + 1.00 * glow_large_rgb),
        0.0,
        1.0,
    )
    composite_rgba = np.concatenate([composite_rgb, alpha], axis=2)
    return Image.fromarray((composite_rgba * 255.0).astype(np.uint8), mode="RGBA")


def _resolve_render_backend(cfg: AppConfig) -> str:
    backend = str(getattr(cfg.cinematic_render, "render_backend", "auto")).strip().lower()
    if backend in {"auto", ""}:
        return "gpu_moderngl" if is_moderngl_available() else "cpu_matplotlib"
    if backend in {"gpu", "gpu_moderngl", "moderngl"}:
        if not is_moderngl_available():
            raise RuntimeError(
                "cinematic_render.render_backend=gpu_moderngl requires moderngl."
            )
        return "gpu_moderngl"
    if backend in {"cpu", "cpu_matplotlib", "matplotlib"}:
        return "cpu_matplotlib"
    raise ValueError(f"Unsupported cinematic_render.render_backend={backend!r}")


def _perspective_matrix(
    fov_y_degrees: float,
    aspect_ratio: float,
    near_plane: float,
    far_plane: float,
) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / max(aspect_ratio, 1e-6)
    matrix[1, 1] = f
    matrix[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
    matrix[2, 3] = (2.0 * far_plane * near_plane) / (near_plane - far_plane)
    matrix[3, 2] = -1.0
    return matrix


def _look_at_matrix(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-8:
        forward = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward = forward / forward_norm

    side = np.cross(forward, up)
    side_norm = np.linalg.norm(side)
    if side_norm <= 1e-8:
        side = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        side = side / side_norm

    true_up = np.cross(side, forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = side
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def _orbit_camera_matrices(
    cfg: AppConfig,
    *,
    axis_limits: np.ndarray,
    frame_index: int,
    frame_count: int,
    render_resolution: tuple[int, int],
    camera_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(axis_limits, axis=1).astype(np.float32)
    elev, azim = _camera_values(cfg, frame_index, frame_count)
    spans = (axis_limits[:, 1] - axis_limits[:, 0]).astype(np.float32)
    view_dir, _side, _true_up = _camera_basis_vectors(elev, azim)
    radius = float(camera_distance) if camera_distance is not None else _default_camera_distance(cfg, axis_limits)
    camera_offset = view_dir * radius
    eye = center + camera_offset
    target = center

    aspect_ratio = float(render_resolution[0]) / max(1.0, float(render_resolution[1]))
    near_plane = max(0.01, radius * 0.04)
    far_plane = radius * 4.0 + float(np.linalg.norm(spans))
    projection = _perspective_matrix(
        float(getattr(cfg.cinematic_render, "camera_fov_degrees", 23.0)),
        aspect_ratio,
        near_plane,
        far_plane,
    )
    view = _look_at_matrix(
        eye.astype(np.float32),
        target.astype(np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    return projection @ view, projection


def _project_world_to_screen(
    points: np.ndarray,
    *,
    mvp_matrix: np.ndarray,
    render_resolution: tuple[int, int],
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    homogeneous = np.concatenate(
        [np.asarray(points, dtype=np.float32), np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    clip = (mvp_matrix @ homogeneous.T).T
    w = np.clip(clip[:, 3:4], 1e-6, None)
    ndc = clip[:, :3] / w
    width, height = render_resolution
    screen_x = (ndc[:, 0] * 0.5 + 0.5) * float(width)
    screen_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(height)
    return np.stack([screen_x, screen_y], axis=1).astype(np.float32)


def _guide_geometry(
    axis_limits: np.ndarray,
    *,
    axis_style: str,
) -> tuple[list[np.ndarray], list[tuple[str, np.ndarray]]]:
    mins = axis_limits[:, 0]
    maxs = axis_limits[:, 1]
    spans = axis_limits[:, 1] - axis_limits[:, 0]

    if axis_style == "corner_guides":
        return [], []

    if axis_style == "thin_full_axes":
        origin = np.asarray([mins[0], mins[1], mins[2]], dtype=np.float32)
        endpoints = (
            (np.asarray([maxs[0], mins[1], mins[2]], dtype=np.float32), "X"),
            (np.asarray([mins[0], maxs[1], mins[2]], dtype=np.float32), "Y"),
            (np.asarray([mins[0], mins[1], maxs[2]], dtype=np.float32), "Z"),
        )
        lines = []
        labels = []
        for endpoint, label in endpoints:
            lines.append(np.stack([origin, endpoint], axis=0).astype(np.float32))
            labels.append((label, endpoint.astype(np.float32)))
        return lines, labels

    if axis_style == "wireframe_box":
        vertices = np.asarray(
            [
                [mins[0], mins[1], mins[2]],
                [maxs[0], mins[1], mins[2]],
                [mins[0], maxs[1], mins[2]],
                [maxs[0], maxs[1], mins[2]],
                [mins[0], mins[1], maxs[2]],
                [maxs[0], mins[1], maxs[2]],
                [mins[0], maxs[1], maxs[2]],
                [maxs[0], maxs[1], maxs[2]],
            ],
            dtype=np.float32,
        )
        edges = (
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        lines = [
            np.stack([vertices[start_idx], vertices[end_idx]], axis=0).astype(np.float32)
            for start_idx, end_idx in edges
        ]
        return lines, []

    return [], []


def _overlay_guide_labels(
    image: Image.Image,
    *,
    label_positions: list[tuple[str, np.ndarray]],
    mvp_matrix: np.ndarray,
    render_resolution: tuple[int, int],
    profile: dict[str, Any],
) -> Image.Image:
    if not label_positions:
        return image
    points = np.stack([position for _, position in label_positions], axis=0).astype(np.float32)
    screen_points = _project_world_to_screen(
        points,
        mvp_matrix=mvp_matrix,
        render_resolution=render_resolution,
    )
    draw = ImageDraw.Draw(image)
    text_rgba = _guide_text_color(profile)
    text_fill = tuple(int(round(channel * 255.0)) for channel in text_rgba)
    for (label, _world_position), screen_position in zip(label_positions, screen_points, strict=False):
        draw.text(
            (float(screen_position[0]) + 6.0, float(screen_position[1]) - 6.0),
            label,
            fill=text_fill,
        )
    return image


def _overlay_corner_guides_screen_space(
    image: Image.Image,
    *,
    profile: dict[str, Any],
    axis_directions: list[tuple[str, np.ndarray]] | None,
) -> Image.Image:
    width, height = image.size
    short_side = float(min(width, height))
    origin = np.asarray(
        [
            max(92.0, width * 0.145),
            min(height - 96.0, height * 0.815),
        ],
        dtype=np.float32,
    )
    length = max(88.0, short_side * 0.125)

    guide_rgba = _guide_color(profile)
    text_rgba = _guide_text_color(profile)
    halo_rgba = _with_alpha(_cool_tint(NVIDIA_WHITE, 0.20), guide_rgba[3] * 0.30)
    guide_fill = tuple(int(round(channel * 255.0)) for channel in guide_rgba)
    halo_fill = tuple(int(round(channel * 255.0)) for channel in halo_rgba)
    text_fill = tuple(int(round(channel * 255.0)) for channel in text_rgba)
    default_axes = (
        ("PC1", np.asarray([1.00, -0.18], dtype=np.float32)),
        ("PC2", np.asarray([0.42, -0.76], dtype=np.float32)),
        ("PC3", np.asarray([0.0, -1.00], dtype=np.float32)),
    )
    axes: list[tuple[str, np.ndarray]] = []
    if axis_directions:
        axes = [
            (
                str(label),
                np.asarray(direction, dtype=np.float32) / max(np.linalg.norm(direction), 1e-6),
            )
            for label, direction in axis_directions
        ]
    if not axes:
        axes = [(label, delta / max(np.linalg.norm(delta), 1e-6)) for label, delta in default_axes]

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    glow_width = max(4, int(round(short_side * 0.0064)))
    line_width = max(2, int(round(short_side * 0.0024)))
    origin_r_glow = max(4, int(round(short_side * 0.0046)))
    origin_r_core = max(2, int(round(short_side * 0.0022)))
    origin_glow_fill = tuple(
        int(round(channel * 255.0))
        for channel in _with_alpha(_cool_tint(NVIDIA_WHITE, 0.16), guide_rgba[3] * 0.42)
    )
    def _extend_to_edge(start: np.ndarray, direction: np.ndarray) -> np.ndarray:
        candidates: list[float] = []
        dx = float(direction[0])
        dy = float(direction[1])
        if abs(dx) > 1e-6:
            candidates.extend([(0.0 - float(start[0])) / dx, (float(width) - float(start[0])) / dx])
        if abs(dy) > 1e-6:
            candidates.extend([(0.0 - float(start[1])) / dy, (float(height) - float(start[1])) / dy])
        positive = [value for value in candidates if value > 0.0]
        scale = min(positive) if positive else length
        return start + direction * scale

    for label, unit_direction in axes:
        tip = _extend_to_edge(origin, unit_direction)
        draw.line(
            (float(origin[0]), float(origin[1]), float(tip[0]), float(tip[1])),
            fill=halo_fill,
            width=glow_width,
        )
        draw.line(
            (float(origin[0]), float(origin[1]), float(tip[0]), float(tip[1])),
            fill=guide_fill,
            width=line_width,
        )
        text_pos = origin + unit_direction * min(float(np.linalg.norm(tip - origin)) * 0.42, length * 1.30)
        text_pos = text_pos + np.asarray([8.0, -8.0], dtype=np.float32)
        draw.text((float(text_pos[0]), float(text_pos[1])), label, fill=text_fill)
    draw.ellipse(
        (
            float(origin[0] - origin_r_glow),
            float(origin[1] - origin_r_glow),
            float(origin[0] + origin_r_glow),
            float(origin[1] + origin_r_glow),
        ),
        fill=origin_glow_fill,
    )
    draw.ellipse(
        (
            float(origin[0] - origin_r_core),
            float(origin[1] - origin_r_core),
            float(origin[0] + origin_r_core),
            float(origin[1] + origin_r_core),
        ),
        fill=guide_fill,
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _overlay_axis_guides(
    image: Image.Image,
    *,
    cfg: AppConfig,
    profile: dict[str, Any],
    axis_directions: list[tuple[str, np.ndarray]] | None = None,
    label_positions: list[tuple[str, np.ndarray]] | None = None,
    mvp_matrix: np.ndarray | None = None,
    render_resolution: tuple[int, int] | None = None,
) -> Image.Image:
    axis_style = str(getattr(cfg.cinematic_render, "axis_style", "corner_guides")).strip().lower()
    if axis_style == "corner_guides":
        return _overlay_corner_guides_screen_space(
            image,
            profile=profile,
            axis_directions=axis_directions,
        )
    if label_positions and mvp_matrix is not None and render_resolution is not None:
        return _overlay_guide_labels(
            image,
            label_positions=label_positions,
            mvp_matrix=mvp_matrix,
            render_resolution=render_resolution,
            profile=profile,
        ).convert("RGB")
    return image.convert("RGB")


def _particle_batch_from_layers(
    positions: np.ndarray,
    *,
    color: tuple[float, float, float],
    layers: Sequence[tuple[float, float]],
    core_scale_multiplier: float,
    halo_scale_multiplier: float,
    point_size_multiplier: float = 1.0,
) -> ParticleBatch:
    layer_defs = tuple((float(size), float(alpha)) for size, alpha in layers)
    base_size = max(1.0, layer_defs[-1][0] * float(point_size_multiplier))
    max_size = max(layer[0] for layer in layer_defs)
    halo_scale = max(1.0, (max_size / max(base_size, 1e-6)) * float(halo_scale_multiplier))
    return ParticleBatch(
        positions=np.asarray(positions, dtype=np.float32),
        color=(*tuple(float(v) for v in color), 1.0),
        point_size_px=base_size,
        core_scale=max(0.2, float(core_scale_multiplier)),
        halo_scale=halo_scale,
        core_alpha=max(0.0, min(1.0, layer_defs[-1][1])),
        halo_alpha=max(0.0, min(1.0, layer_defs[0][1])),
        core_sharpness=3.9,
        halo_sharpness=1.35,
    )


def _build_gpu_frame_batches(
    *,
    cfg: AppConfig,
    master_key: str,
    profile: dict[str, Any],
    axis_limits: np.ndarray,
    palette: dict[str, str],
    label_names: np.ndarray,
    sample_masks: Sequence[np.ndarray],
    sample_frames: np.ndarray,
    centroid_frames: np.ndarray,
    overlay_frames: np.ndarray,
    frame_index: int,
) -> tuple[list[ParticleBatch], list[LineBatch], list[tuple[str, np.ndarray]]]:
    particle_batches: list[ParticleBatch] = []
    line_batches: list[LineBatch] = []
    axis_style = str(getattr(cfg.cinematic_render, "axis_style", "corner_guides")).strip().lower()
    guide_lines, guide_labels = _guide_geometry(axis_limits, axis_style=axis_style)
    if axis_style != "corner_guides":
        guide_color = _guide_color(profile)
        for positions in guide_lines:
            line_batches.append(
                LineBatch(
                    positions=np.asarray(positions, dtype=np.float32),
                    color=guide_color,
                    line_width_px=float(profile["guide_line_width"]),
                )
            )

    label_list = [str(item) for item in np.asarray(label_names, dtype=str).tolist()]
    resolved_palette = _resolve_cinematic_palette(label_list, dict(palette))
    sample_layers = tuple(profile["sample_layers"])
    overlay_layers = tuple(profile["overlay_layers"])
    centroid_layers = tuple(profile["centroid_layers"])
    trail_policy = _master_trail_policy(cfg, master_key)
    trail_length = max(1, int(getattr(cfg.cinematic_render, "trail_length", 36)))
    trail_start = max(0, frame_index + 1 - trail_length)
    core_scale_mult, halo_scale_mult = _master_particle_scale_multipliers(cfg, master_key)
    sample_point_multiplier = 1.18 if str(master_key) == "beauty_master" else 1.0
    overlay_point_multiplier = 1.05 if str(master_key) == "beauty_master" else 1.0
    centroid_point_multiplier = 0.88 if str(master_key) == "beauty_master" else 1.0

    sample_projection = np.asarray(sample_frames[frame_index], dtype=np.float32)
    for label_index, label_name in enumerate(label_list):
        sample_color, overlay_color, centroid_color = _label_render_colors(
            profile,
            resolved_palette,
            str(label_name),
        )
        label_projection = sample_projection[np.asarray(sample_masks[label_index], dtype=bool)]
        particle_batches.append(
            _particle_batch_from_layers(
                label_projection,
                color=sample_color,
                layers=sample_layers,
                core_scale_multiplier=core_scale_mult,
                halo_scale_multiplier=halo_scale_mult,
                point_size_multiplier=sample_point_multiplier,
            )
        )

        overlay_point = np.asarray(overlay_frames[frame_index, label_index, :], dtype=np.float32).reshape(1, 3)
        centroid_point = np.asarray(centroid_frames[frame_index, label_index, :], dtype=np.float32).reshape(1, 3)
        particle_batches.append(
            _particle_batch_from_layers(
                overlay_point,
                color=overlay_color,
                layers=overlay_layers,
                core_scale_multiplier=core_scale_mult,
                halo_scale_multiplier=halo_scale_mult,
                point_size_multiplier=overlay_point_multiplier,
            )
        )
        particle_batches.append(
            _particle_batch_from_layers(
                centroid_point,
                color=centroid_color,
                layers=centroid_layers,
                core_scale_multiplier=max(0.6, core_scale_mult * 0.85),
                halo_scale_multiplier=max(1.0, halo_scale_mult * 0.70),
                point_size_multiplier=centroid_point_multiplier,
            )
        )

        overlay_visible = np.asarray(overlay_frames[trail_start : frame_index + 1, label_index, :], dtype=np.float32)
        centroid_visible = np.asarray(centroid_frames[trail_start : frame_index + 1, label_index, :], dtype=np.float32)
        if trail_policy in {"label_only", "overlay_only", "overlay_centroid"}:
            line_batches.append(
                LineBatch(
                    positions=overlay_visible,
                    color=_with_alpha(overlay_color, float(profile["overlay_trail_alpha"])),
                    line_width_px=float(profile["overlay_trail_width"]),
                )
            )
        if trail_policy == "overlay_centroid":
            line_batches.append(
                LineBatch(
                    positions=centroid_visible,
                    color=_with_alpha(centroid_color, float(profile["centroid_trail_alpha"])),
                    line_width_px=float(profile["centroid_trail_width"]),
                )
            )

    return particle_batches, line_batches, guide_labels


class _ModernGLCinematicRenderer:
    def __init__(
        self,
        *,
        cfg: AppConfig,
        axis_limits: np.ndarray,
        palette: dict[str, str],
        label_names: np.ndarray,
        sample_label_names: np.ndarray,
        render_resolution: tuple[int, int],
        camera_distances: np.ndarray | None = None,
        axis_directions: list[tuple[str, np.ndarray]] | None = None,
    ) -> None:
        if moderngl is None:
            raise RuntimeError("moderngl is not available for GPU cinematic rendering.")

        self.cfg = cfg
        self.axis_limits = np.asarray(axis_limits, dtype=np.float32)
        self.label_names = np.asarray(label_names, dtype=str)
        self.sample_label_names = np.asarray(sample_label_names, dtype=str)
        self.palette = _resolve_cinematic_palette(self.label_names.tolist(), dict(palette))
        self.render_resolution = tuple(int(v) for v in render_resolution)
        self.camera_distances = (
            np.asarray(camera_distances, dtype=np.float32) if camera_distances is not None else None
        )
        self.axis_directions = list(axis_directions or [])
        self.profile = _look_profile(cfg)
        self.axis_style = str(getattr(cfg.cinematic_render, "axis_style", "corner_guides")).strip().lower()
        self.sample_masks = [
            self.sample_label_names == str(label_name) for label_name in self.label_names.tolist()
        ]

        self.ctx = moderngl.create_standalone_context(backend="egl", require=330)
        self.ctx.enable_only(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.color_texture = self.ctx.texture(self.render_resolution, 4)
        self.color_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_texture = self.ctx.depth_renderbuffer(self.render_resolution)
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_texture],
            depth_attachment=self.depth_texture,
        )

        self.point_program = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 u_mvp;
                uniform float u_point_size;
                in vec3 in_pos;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                    gl_PointSize = u_point_size;
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec4 u_color;
                uniform float u_sharpness;
                out vec4 f_color;
                void main() {
                    vec2 uv = gl_PointCoord * 2.0 - 1.0;
                    float radius_sq = dot(uv, uv);
                    if (radius_sq > 1.0) {
                        discard;
                    }
                    float falloff = pow(max(0.0, 1.0 - radius_sq), u_sharpness);
                    f_color = vec4(u_color.rgb, u_color.a * falloff);
                }
            """,
        )
        self.line_program = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 u_mvp;
                in vec3 in_pos;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec4 u_color;
                out vec4 f_color;
                void main() {
                    f_color = u_color;
                }
            """,
        )

    def close(self) -> None:
        try:
            self.fbo.release()
        except Exception:
            pass
        try:
            self.color_texture.release()
        except Exception:
            pass
        try:
            self.depth_texture.release()
        except Exception:
            pass
        try:
            self.point_program.release()
        except Exception:
            pass
        try:
            self.line_program.release()
        except Exception:
            pass
        try:
            self.ctx.release()
        except Exception:
            pass

    def _label_colors(self, label_name: str) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        base_color = self.palette.get(str(label_name), DEFAULT_COLOR)
        sample_color = _brighten(
            _cool_tint(base_color, float(self.profile["sample_cool_mix"])),
            float(self.profile["sample_brighten"]),
        )
        overlay_color = _brighten(
            _cool_tint(base_color, float(self.profile["overlay_cool_mix"])),
            float(self.profile["overlay_brighten"]),
        )
        centroid_color = _brighten(
            _cool_tint(base_color, float(self.profile["centroid_cool_mix"])),
            float(self.profile["centroid_brighten"]),
        )
        return sample_color, overlay_color, centroid_color

    def _draw_points(
        self,
        positions: np.ndarray,
        *,
        mvp_matrix: np.ndarray,
        color: tuple[float, float, float, float],
        point_size: float,
        sharpness: float,
    ) -> None:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.size == 0:
            return
        vbo = self.ctx.buffer(np.ascontiguousarray(positions, dtype=np.float32).tobytes())
        vao = self.ctx.simple_vertex_array(self.point_program, vbo, "in_pos")
        self.point_program["u_mvp"].write(np.asarray(mvp_matrix, dtype=np.float32).T.tobytes())
        self.point_program["u_point_size"].value = float(point_size)
        self.point_program["u_color"].value = tuple(float(v) for v in color)
        self.point_program["u_sharpness"].value = float(sharpness)
        vao.render(mode=moderngl.POINTS)
        vao.release()
        vbo.release()

    def _draw_line_strip(
        self,
        positions: np.ndarray,
        *,
        mvp_matrix: np.ndarray,
        color: tuple[float, float, float, float],
        line_width: float,
    ) -> None:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape[0] < 2:
            return
        vbo = self.ctx.buffer(np.ascontiguousarray(positions, dtype=np.float32).tobytes())
        vao = self.ctx.simple_vertex_array(self.line_program, vbo, "in_pos")
        self.line_program["u_mvp"].write(np.asarray(mvp_matrix, dtype=np.float32).T.tobytes())
        self.line_program["u_color"].value = tuple(float(v) for v in color)
        self.ctx.line_width = max(1.0, float(line_width))
        vao.render(mode=moderngl.LINE_STRIP)
        vao.release()
        vbo.release()

    def render_frame(
        self,
        *,
        sample_frames: np.ndarray,
        centroid_frames: np.ndarray,
        overlay_frames: np.ndarray,
        frame_index: int,
        target_resolution: tuple[int, int],
    ) -> Image.Image:
        mvp_matrix, _projection = _orbit_camera_matrices(
            self.cfg,
            axis_limits=self.axis_limits,
            frame_index=frame_index,
            frame_count=int(sample_frames.shape[0]),
            render_resolution=self.render_resolution,
            camera_distance=(
                float(self.camera_distances[frame_index])
                if self.camera_distances is not None and len(self.camera_distances) > frame_index
                else None
            ),
        )

        background = mcolors.to_rgba(str(self.cfg.cinematic_render.background_color))
        self.fbo.use()
        self.ctx.clear(*background)

        guide_lines, guide_labels = _guide_geometry(
            self.axis_limits,
            axis_style=self.axis_style,
        )
        guide_color = _guide_color(self.profile)
        for line_positions in guide_lines:
            self._draw_line_strip(
                line_positions,
                mvp_matrix=mvp_matrix,
                color=guide_color,
                line_width=float(self.profile["guide_line_width"]),
            )

        trail_length = max(1, int(getattr(self.cfg.cinematic_render, "trail_length", 36)))
        trail_start = max(0, frame_index + 1 - trail_length)
        sample_projection = np.asarray(sample_frames[frame_index], dtype=np.float32)

        sample_layers = tuple(self.profile["sample_layers"])
        overlay_layers = tuple(self.profile["overlay_layers"])
        centroid_layers = tuple(self.profile["centroid_layers"])

        for label_index, label_name in enumerate(self.label_names.tolist()):
            sample_color, overlay_color, centroid_color = self._label_colors(str(label_name))
            label_projection = sample_projection[self.sample_masks[label_index]]

            for layer_idx, (size, alpha) in enumerate(sample_layers):
                self._draw_points(
                    label_projection,
                    mvp_matrix=mvp_matrix,
                    color=_with_alpha(sample_color, float(alpha)),
                    point_size=float(size),
                    sharpness=1.5 + layer_idx * 2.0,
                )

            overlay_visible = overlay_frames[trail_start : frame_index + 1, label_index, :]
            centroid_visible = centroid_frames[trail_start : frame_index + 1, label_index, :]
            self._draw_line_strip(
                overlay_visible,
                mvp_matrix=mvp_matrix,
                color=_with_alpha(overlay_color, float(self.profile["overlay_trail_alpha"])),
                line_width=float(self.profile["overlay_trail_width"]),
            )
            self._draw_line_strip(
                centroid_visible,
                mvp_matrix=mvp_matrix,
                color=_with_alpha(centroid_color, float(self.profile["centroid_trail_alpha"])),
                line_width=float(self.profile["centroid_trail_width"]),
            )

            overlay_point = np.asarray(overlay_frames[frame_index, label_index, :], dtype=np.float32).reshape(1, 3)
            centroid_point = np.asarray(centroid_frames[frame_index, label_index, :], dtype=np.float32).reshape(1, 3)
            for layer_idx, (size, alpha) in enumerate(overlay_layers):
                self._draw_points(
                    overlay_point,
                    mvp_matrix=mvp_matrix,
                    color=_with_alpha(overlay_color, float(alpha)),
                    point_size=float(size),
                    sharpness=1.6 + layer_idx * 1.7,
                )
            for layer_idx, (size, alpha) in enumerate(centroid_layers):
                self._draw_points(
                    centroid_point,
                    mvp_matrix=mvp_matrix,
                    color=_with_alpha(centroid_color, float(alpha)),
                    point_size=float(size),
                    sharpness=1.8 + layer_idx * 1.8,
                )

        raw = self.fbo.read(components=4, alignment=1)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.render_resolution[1], self.render_resolution[0], 4)
        )[::-1, :, :]
        image = Image.fromarray(frame, mode="RGBA")
        image = _apply_glow(image, cfg=self.cfg)
        if image.size != tuple(int(v) for v in target_resolution):
            image = image.resize(tuple(int(v) for v in target_resolution), Image.Resampling.LANCZOS)
        return _overlay_axis_guides(
            image.convert("RGB"),
            cfg=self.cfg,
            profile=self.profile,
            axis_directions=self.axis_directions,
            label_positions=guide_labels,
            mvp_matrix=mvp_matrix,
            render_resolution=self.render_resolution,
        )


def _create_scene(
    *,
    cfg: AppConfig,
    axis_limits: np.ndarray,
    palette: dict[str, str],
    label_names: np.ndarray,
    sample_label_names: np.ndarray,
    render_resolution: tuple[int, int],
    axis_directions: list[tuple[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    dpi = max(1, int(getattr(cfg.cinematic_render, "animation_dpi", 180)))
    fig = plt.figure(
        figsize=(render_resolution[0] / dpi, render_resolution[1] / dpi),
        dpi=dpi,
        facecolor=str(cfg.cinematic_render.background_color),
    )
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")
    _style_cinematic_axis(
        ax,
        axis_limits=axis_limits,
        background_color=str(cfg.cinematic_render.background_color),
    )
    profile = _look_profile(cfg)
    _add_axis_guides(ax, axis_limits=axis_limits, cfg=cfg, profile=profile)

    try:
        ax.set_proj_type("persp", focal_length=1.15)
    except Exception:
        pass

    label_list = [str(item) for item in label_names.tolist()]
    resolved_palette = _resolve_cinematic_palette(label_list, palette)
    sample_labels = np.asarray(sample_label_names, dtype=str)
    sample_masks = [sample_labels == label for label in label_list]

    sample_cloud_layers: list[list[Any]] = []
    overlay_lines = []
    centroid_lines = []
    overlay_points: list[list[Any]] = []
    centroid_points: list[list[Any]] = []

    sample_layers = tuple(profile["sample_layers"])
    overlay_layers = tuple(profile["overlay_layers"])
    centroid_layers = tuple(profile["centroid_layers"])

    for label_name in label_list:
        sample_color, overlay_color, centroid_color = _label_render_colors(
            profile,
            resolved_palette,
            label_name,
        )

        sample_artists = []
        for size, alpha in sample_layers:
            sample_artists.append(
                ax.scatter(
                    [],
                    [],
                    [],
                    s=float(size),
                    c=[_with_alpha(sample_color, float(alpha))],
                    depthshade=False,
                    linewidths=0.0,
                )
            )
        sample_cloud_layers.append(sample_artists)

        overlay_line, = ax.plot(
            [],
            [],
            [],
            color=_with_alpha(overlay_color, float(profile["overlay_trail_alpha"])),
            linewidth=float(profile["overlay_trail_width"]),
        )
        centroid_line, = ax.plot(
            [],
            [],
            [],
            color=_with_alpha(centroid_color, float(profile["centroid_trail_alpha"])),
            linewidth=float(profile["centroid_trail_width"]),
        )
        overlay_lines.append(overlay_line)
        centroid_lines.append(centroid_line)

        overlay_layer_artists = []
        centroid_layer_artists = []
        for size, alpha in overlay_layers:
            overlay_layer_artists.append(
                ax.scatter(
                    [],
                    [],
                    [],
                    s=float(size),
                    c=[_with_alpha(overlay_color, float(alpha))],
                    depthshade=False,
                    linewidths=0.0,
                )
            )
        for size, alpha in centroid_layers:
            centroid_layer_artists.append(
                ax.scatter(
                    [],
                    [],
                    [],
                    s=float(size),
                    c=[_with_alpha(centroid_color, float(alpha))],
                    depthshade=False,
                    linewidths=0.0,
                )
            )
        overlay_points.append(overlay_layer_artists)
        centroid_points.append(centroid_layer_artists)

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    return {
        "fig": fig,
        "ax": ax,
        "profile": profile,
        "axis_limits": np.asarray(axis_limits, dtype=np.float32),
        "render_resolution": tuple(int(v) for v in render_resolution),
        "axis_directions": list(axis_directions or []),
        "trail_policy": _master_trail_policy(cfg, "analysis_master")
        if str(getattr(cfg.cinematic_render, "look_preset", "glass_wireframe")).strip().lower() == "balanced"
        else _master_trail_policy(cfg, "beauty_master"),
        "sample_masks": sample_masks,
        "label_list": label_list,
        "sample_cloud_layers": sample_cloud_layers,
        "overlay_lines": overlay_lines,
        "centroid_lines": centroid_lines,
        "overlay_points": overlay_points,
        "centroid_points": centroid_points,
        "current_mvp_matrix": None,
    }


def _update_scene(
    scene: dict[str, Any],
    *,
    cfg: AppConfig,
    sample_frames: np.ndarray,
    centroid_frames: np.ndarray,
    overlay_frames: np.ndarray,
    frame_index: int,
    camera_distances: np.ndarray | None = None,
) -> None:
    ax = scene["ax"]
    axis_limits = np.asarray(scene["axis_limits"], dtype=np.float32)
    render_resolution = tuple(int(v) for v in scene["render_resolution"])
    label_list = scene["label_list"]
    sample_masks = scene["sample_masks"]
    sample_cloud_layers = scene["sample_cloud_layers"]
    overlay_lines = scene["overlay_lines"]
    centroid_lines = scene["centroid_lines"]
    overlay_points = scene["overlay_points"]
    centroid_points = scene["centroid_points"]
    trail_policy = str(scene.get("trail_policy", "overlay_centroid")).strip().lower()

    frame_count = int(sample_frames.shape[0])
    elev, azim = _camera_values(cfg, frame_index, frame_count)
    ax.view_init(elev=elev, azim=azim)
    mvp_matrix, _projection = _orbit_camera_matrices(
        cfg,
        axis_limits=axis_limits,
        frame_index=frame_index,
        frame_count=frame_count,
        render_resolution=render_resolution,
        camera_distance=(
            float(camera_distances[frame_index])
            if camera_distances is not None and len(camera_distances) > frame_index
            else None
        ),
    )
    scene["current_mvp_matrix"] = mvp_matrix

    trail_length = max(1, int(getattr(cfg.cinematic_render, "trail_length", 36)))
    trail_start = max(0, frame_index + 1 - trail_length)

    sample_projection = sample_frames[frame_index]
    for label_index, _label_name in enumerate(label_list):
        label_projection = sample_projection[sample_masks[label_index]]
        for sample_layer in sample_cloud_layers[label_index]:
            _set_cloud_offsets(sample_layer, label_projection)

        overlay_visible = overlay_frames[trail_start : frame_index + 1, label_index, :]
        centroid_visible = centroid_frames[trail_start : frame_index + 1, label_index, :]
        if trail_policy in {"label_only", "overlay_only", "overlay_centroid"}:
            _set_line_data(overlay_lines[label_index], overlay_visible)
        else:
            _set_line_data(overlay_lines[label_index], np.zeros((0, 3), dtype=np.float32))
        if trail_policy == "overlay_centroid":
            _set_line_data(centroid_lines[label_index], centroid_visible)
        else:
            _set_line_data(centroid_lines[label_index], np.zeros((0, 3), dtype=np.float32))

        overlay_point = overlay_frames[frame_index, label_index, :]
        centroid_point = centroid_frames[frame_index, label_index, :]
        for overlay_artist in overlay_points[label_index]:
            _set_point_offsets(overlay_artist, overlay_point)
        for centroid_artist in centroid_points[label_index]:
            _set_point_offsets(centroid_artist, centroid_point)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffmpeg_supports_encoder(encoder_name: str) -> bool:
    if not _ffmpeg_available():
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return False
    return encoder_name in result.stdout


def _run_ffmpeg(args: list[str]) -> bool:
    try:
        subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except Exception as err:
        print(f"[WARN] ffmpeg command failed: {err}")
        return False


def _save_preview_gif_with_pillow(
    frame_paths: Sequence[Path],
    *,
    output_path: Path,
    gif_resolution: tuple[int, int],
    fps: int,
) -> bool:
    if not frame_paths:
        return False
    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as frame:
            frames.append(
                frame.convert("P", palette=Image.Palette.ADAPTIVE).resize(
                    gif_resolution,
                    Image.Resampling.LANCZOS,
                )
            )
    try:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, int(round(1000 / max(1, fps)))),
            loop=0,
            optimize=False,
            disposal=2,
        )
        return True
    except Exception as err:
        print(f"[WARN] Could not save preview GIF with Pillow: {err}")
        return False


def _probe_image_resolution(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return {"width": int(image.size[0]), "height": int(image.size[1])}


def _probe_video_resolution(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    if shutil.which("ffprobe") is None:
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        width_str, height_str = result.stdout.strip().split("x", maxsplit=1)
        return {"width": int(width_str), "height": int(height_str)}
    except Exception as err:
        print(f"[WARN] Could not probe video resolution for {path}: {err}")
        return None


def _save_cinematic_animation_gpu(
    output_dir: Path,
    *,
    cfg: AppConfig,
    master_key: str,
    stem: str,
    camera_mode: str,
    sample_frames: np.ndarray,
    centroid_frames: np.ndarray,
    overlay_frames: np.ndarray,
    sample_label_names: np.ndarray,
    label_names: np.ndarray,
    palette: dict[str, str],
    axis_limits: np.ndarray,
    render_dims: int,
) -> dict[str, Any]:
    if render_dims != 3:
        raise ValueError("Cinematic renderer currently supports only 3D trajectories.")
    source_frame_count = int(sample_frames.shape[0])
    if source_frame_count == 0:
        return {"artifacts": {}, "resolutions": {}, "video_encoder": "none"}

    video_resolution = _target_resolution(cfg, "video")
    gif_resolution = _target_resolution(cfg, "gif")
    fps = max(1, int(cfg.cinematic_render.fps))
    artifacts: dict[str, str] = {}
    resolutions: dict[str, dict[str, int]] = {}
    video_encoder = "none"
    source_camera_distances = _compute_camera_distance_schedule(
        cfg,
        sample_frames=sample_frames,
        centroid_frames=centroid_frames,
        overlay_frames=overlay_frames,
        axis_limits=axis_limits,
        render_resolution=video_resolution,
        mode=camera_mode,
    )
    axis_directions = _compute_reference_axis_directions(
        cfg,
        axis_limits=axis_limits,
        render_resolution=video_resolution,
        frame_count=source_frame_count,
        camera_distances=source_camera_distances,
    )
    output_source_indices = _master_output_frame_indices(cfg, master_key, source_frame_count)
    profile = _look_profile(cfg)
    sample_masks = [np.asarray(sample_label_names, dtype=str) == str(label_name) for label_name in label_names.tolist()]
    depth_cue_strength, depth_fog_strength, depth_fog_cool_mix = _master_depth_settings(cfg, master_key)

    renderer = CinematicGPUFrameRenderer(
        RendererOptions(
            output_size=video_resolution,
            supersample_scale=_master_supersample_scale(cfg, master_key, "video"),
            background_color=str(cfg.cinematic_render.background_color),
            depth_cue_strength=depth_cue_strength,
            depth_fog_strength=depth_fog_strength,
            depth_fog_cool_mix=depth_fog_cool_mix,
            depth_fog_color=_cool_tint(NVIDIA_WHITE, 0.36),
        )
    )

    try:
        with tempfile.TemporaryDirectory(prefix="cinematic_frames_", dir=str(output_dir)) as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            frame_paths: list[Path] = []
            for output_frame_index, source_frame_index in enumerate(output_source_indices.tolist()):
                mvp_matrix, _projection = _orbit_camera_matrices(
                    cfg,
                    axis_limits=axis_limits,
                    frame_index=source_frame_index,
                    frame_count=source_frame_count,
                    render_resolution=video_resolution,
                    camera_distance=(
                        float(source_camera_distances[source_frame_index])
                        if len(source_camera_distances) > source_frame_index
                        else None
                    ),
                )
                particle_batches, line_batches, guide_labels = _build_gpu_frame_batches(
                    cfg=cfg,
                    master_key=master_key,
                    profile=profile,
                    axis_limits=axis_limits,
                    palette=palette,
                    label_names=label_names,
                    sample_masks=sample_masks,
                    sample_frames=sample_frames,
                    centroid_frames=centroid_frames,
                    overlay_frames=overlay_frames,
                    frame_index=source_frame_index,
                )
                frame_image = renderer.render_frame(
                    FrameSpec(
                        camera=CameraFrame(mvp_matrix=mvp_matrix),
                        particle_batches=particle_batches,
                        line_batches=line_batches,
                        background_color=str(cfg.cinematic_render.background_color),
                        output_size=video_resolution,
                    )
                )
                frame_image = _overlay_axis_guides(
                    frame_image,
                    cfg=cfg,
                    profile=profile,
                    axis_directions=axis_directions,
                    label_positions=guide_labels,
                    mvp_matrix=mvp_matrix,
                    render_resolution=video_resolution,
                )
                frame_path = temp_dir / f"frame_{output_frame_index:06d}.png"
                frame_image.save(frame_path, format="PNG", compress_level=1)
                frame_paths.append(frame_path)

            if bool(getattr(cfg.cinematic_render, "export_animation", True)) and frame_paths:
                if _ffmpeg_available():
                    mp4_path = output_dir / _output_name(stem, render_dims, "mp4")
                    gpu_requested = bool(getattr(cfg.cinematic_render, "prefer_gpu_encode", True))
                    encoder_variants: list[tuple[str, list[str]]] = []
                    if gpu_requested and _ffmpeg_supports_encoder("h264_nvenc"):
                        encoder_variants.append(
                            (
                                "h264_nvenc",
                                [
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p5",
                                    "-cq",
                                    "18",
                                    "-pix_fmt",
                                    "yuv420p",
                                ],
                            )
                        )
                    encoder_variants.append(
                        (
                            "libx264",
                            [
                                "-c:v",
                                "libx264",
                                "-preset",
                                "slow",
                                "-crf",
                                "14",
                                "-pix_fmt",
                                "yuv420p",
                            ],
                        )
                    )
                    for encoder_name, encoder_args in encoder_variants:
                        mp4_command = [
                            "ffmpeg",
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-framerate",
                            str(fps),
                            "-i",
                            str(temp_dir / "frame_%06d.png"),
                            *encoder_args,
                            str(mp4_path),
                        ]
                        if _run_ffmpeg(mp4_command):
                            video_encoder = encoder_name
                            artifacts["mp4"] = mp4_path.name
                            resolutions["video"] = (
                                _probe_video_resolution(mp4_path)
                                or {"width": video_resolution[0], "height": video_resolution[1]}
                            )
                            break

                    preview_gif_path = output_dir / _output_name(stem, render_dims, "gif", preview=True)
                    gif_filter = (
                        f"fps={fps},scale={gif_resolution[0]}:{gif_resolution[1]}:flags=lanczos,"
                        "split[s0][s1];[s0]palettegen=stats_mode=single[p];"
                        "[s1][p]paletteuse=dither=sierra2_4a"
                    )
                    gif_command = [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-framerate",
                        str(fps),
                        "-i",
                        str(temp_dir / "frame_%06d.png"),
                        "-filter_complex",
                        gif_filter,
                        str(preview_gif_path),
                    ]
                    if _run_ffmpeg(gif_command):
                        artifacts["gif_preview"] = preview_gif_path.name
                        resolutions["preview_gif"] = (
                            _probe_image_resolution(preview_gif_path)
                            or {"width": gif_resolution[0], "height": gif_resolution[1]}
                        )
                else:
                    preview_gif_path = output_dir / _output_name(stem, render_dims, "gif", preview=True)
                    if _save_preview_gif_with_pillow(
                        frame_paths,
                        output_path=preview_gif_path,
                        gif_resolution=gif_resolution,
                        fps=fps,
                    ):
                        artifacts["gif_preview"] = preview_gif_path.name
                        resolutions["preview_gif"] = (
                            _probe_image_resolution(preview_gif_path)
                            or {"width": gif_resolution[0], "height": gif_resolution[1]}
                        )
    finally:
        renderer.close()

    return {
        "artifacts": artifacts,
        "resolutions": resolutions,
        "video_encoder": video_encoder,
    }


def _save_cinematic_animation(
    output_dir: Path,
    *,
    cfg: AppConfig,
    master_key: str,
    stem: str,
    camera_mode: str,
    sample_frames: np.ndarray,
    centroid_frames: np.ndarray,
    overlay_frames: np.ndarray,
    sample_label_names: np.ndarray,
    label_names: np.ndarray,
    palette: dict[str, str],
    axis_limits: np.ndarray,
    render_dims: int,
) -> dict[str, Any]:
    if render_dims != 3:
        raise ValueError("Cinematic renderer currently supports only 3D trajectories.")
    source_frame_count = int(sample_frames.shape[0])
    if source_frame_count == 0:
        return {"artifacts": {}, "resolutions": {}}

    video_resolution = _target_resolution(cfg, "video")
    gif_resolution = _target_resolution(cfg, "gif")
    render_resolution = _render_resolution(cfg, "video", master_key)
    source_camera_distances = _compute_camera_distance_schedule(
        cfg,
        sample_frames=sample_frames,
        centroid_frames=centroid_frames,
        overlay_frames=overlay_frames,
        axis_limits=axis_limits,
        render_resolution=render_resolution,
        mode=camera_mode,
    )
    axis_directions = _compute_reference_axis_directions(
        cfg,
        axis_limits=axis_limits,
        render_resolution=video_resolution,
        frame_count=source_frame_count,
        camera_distances=source_camera_distances,
    )
    output_source_indices = _master_output_frame_indices(cfg, master_key, source_frame_count)

    scene = _create_scene(
        cfg=cfg,
        axis_limits=axis_limits,
        palette=palette,
        label_names=label_names,
        sample_label_names=sample_label_names,
        render_resolution=render_resolution,
        axis_directions=axis_directions,
    )
    scene["trail_policy"] = _master_trail_policy(cfg, master_key)
    fig = scene["fig"]

    artifacts: dict[str, str] = {}
    resolutions: dict[str, dict[str, int]] = {}
    video_encoder = "libx264"
    fps = max(1, int(cfg.cinematic_render.fps))

    with tempfile.TemporaryDirectory(prefix="cinematic_frames_", dir=str(output_dir)) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        frame_paths: list[Path] = []
        for output_frame_index, source_frame_index in enumerate(output_source_indices.tolist()):
            _update_scene(
                scene,
                cfg=cfg,
                sample_frames=sample_frames,
                centroid_frames=centroid_frames,
                overlay_frames=overlay_frames,
                frame_index=source_frame_index,
                camera_distances=source_camera_distances,
            )
            frame_image = _capture_canvas_image(
                fig,
                cfg=cfg,
                target_resolution=video_resolution,
                profile=scene["profile"],
                axis_directions=scene["axis_directions"],
                mvp_matrix=scene["current_mvp_matrix"],
            )
            frame_path = temp_dir / f"frame_{output_frame_index:06d}.png"
            frame_image.save(frame_path, format="PNG", compress_level=1)
            frame_paths.append(frame_path)

        if bool(getattr(cfg.cinematic_render, "export_animation", True)) and frame_paths:
            if _ffmpeg_available():
                mp4_path = output_dir / _output_name(stem, render_dims, "mp4")
                gpu_requested = bool(getattr(cfg.cinematic_render, "prefer_gpu_encode", True))
                encoder_variants: list[tuple[str, list[str]]] = []
                if gpu_requested and _ffmpeg_supports_encoder("h264_nvenc"):
                    encoder_variants.append(
                        (
                            "h264_nvenc",
                            [
                                "-c:v",
                                "h264_nvenc",
                                "-preset",
                                "p5",
                                "-cq",
                                "18",
                                "-pix_fmt",
                                "yuv420p",
                            ],
                        )
                    )
                encoder_variants.append(
                    (
                        "libx264",
                        [
                            "-c:v",
                            "libx264",
                            "-preset",
                            "slow",
                            "-crf",
                            "14",
                            "-pix_fmt",
                            "yuv420p",
                        ],
                    )
                )

                for encoder_name, encoder_args in encoder_variants:
                    mp4_command = [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-framerate",
                        str(fps),
                        "-i",
                        str(temp_dir / "frame_%06d.png"),
                        *encoder_args,
                        str(mp4_path),
                    ]
                    if _run_ffmpeg(mp4_command):
                        video_encoder = encoder_name
                        artifacts["mp4"] = mp4_path.name
                        resolutions["video"] = (
                            _probe_video_resolution(mp4_path)
                            or {"width": video_resolution[0], "height": video_resolution[1]}
                        )
                        break

                preview_gif_path = output_dir / _output_name(stem, render_dims, "gif", preview=True)
                gif_filter = (
                    f"fps={fps},scale={gif_resolution[0]}:{gif_resolution[1]}:flags=lanczos,"
                    "split[s0][s1];[s0]palettegen=stats_mode=single[p];"
                    "[s1][p]paletteuse=dither=sierra2_4a"
                )
                gif_command = [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(temp_dir / "frame_%06d.png"),
                    "-filter_complex",
                    gif_filter,
                    str(preview_gif_path),
                ]
                if _run_ffmpeg(gif_command):
                    artifacts["gif_preview"] = preview_gif_path.name
                    resolutions["preview_gif"] = (
                        _probe_image_resolution(preview_gif_path)
                        or {"width": gif_resolution[0], "height": gif_resolution[1]}
                    )
            else:
                preview_gif_path = output_dir / _output_name(stem, render_dims, "gif", preview=True)
                if _save_preview_gif_with_pillow(
                    frame_paths,
                    output_path=preview_gif_path,
                    gif_resolution=gif_resolution,
                    fps=fps,
                ):
                    artifacts["gif_preview"] = preview_gif_path.name
                    resolutions["preview_gif"] = (
                        _probe_image_resolution(preview_gif_path)
                        or {"width": gif_resolution[0], "height": gif_resolution[1]}
                    )

        plt.close(fig)

        return {
            "artifacts": artifacts,
            "resolutions": resolutions,
            "video_encoder": video_encoder,
        }


def _poster_frame_index(frame_count: int) -> int:
    if frame_count <= 1:
        return 0
    return max(0, min(frame_count - 1, int(round((frame_count - 1) * 0.90))))


def _render_master_variant(
    *,
    trajectory_dir: Path,
    cfg: AppConfig,
    master_key: str,
    master_role: str,
    stem: str,
    camera_mode: str,
    render_backend: str,
    sample_frames: np.ndarray,
    centroid_frames: np.ndarray,
    overlay_frames: np.ndarray,
    sample_label_names: np.ndarray,
    label_names: np.ndarray,
    palette: dict[str, str],
    axis_limits: np.ndarray,
    render_dims: int,
) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    resolutions: dict[str, dict[str, int]] = {}
    video_encoder = "none"
    poster_resolution = _target_resolution(cfg, "poster")
    poster_render_resolution = _render_resolution(cfg, "poster", master_key)
    source_frame_count = int(sample_frames.shape[0])
    poster_camera_distances = _compute_camera_distance_schedule(
        cfg,
        sample_frames=sample_frames,
        centroid_frames=centroid_frames,
        overlay_frames=overlay_frames,
        axis_limits=axis_limits,
        render_resolution=poster_render_resolution,
        mode=camera_mode,
    )
    poster_axis_directions = _compute_reference_axis_directions(
        cfg,
        axis_limits=axis_limits,
        render_resolution=poster_resolution,
        frame_count=source_frame_count,
        camera_distances=poster_camera_distances,
    )

    if bool(getattr(cfg.cinematic_render, "export_poster", True)):
        poster_index = _poster_frame_index(source_frame_count)
        poster_path = trajectory_dir / _output_name(stem, render_dims, "png")
        if render_backend == "gpu_moderngl":
            profile = _look_profile(cfg)
            sample_masks = [
                np.asarray(sample_label_names, dtype=str) == str(label_name)
                for label_name in label_names.tolist()
            ]
            depth_cue_strength, depth_fog_strength, depth_fog_cool_mix = _master_depth_settings(
                cfg,
                master_key,
            )
            poster_renderer = CinematicGPUFrameRenderer(
                RendererOptions(
                    output_size=poster_resolution,
                    supersample_scale=_master_supersample_scale(cfg, master_key, "poster"),
                    background_color=str(cfg.cinematic_render.background_color),
                    depth_cue_strength=depth_cue_strength,
                    depth_fog_strength=depth_fog_strength,
                    depth_fog_cool_mix=depth_fog_cool_mix,
                    depth_fog_color=_cool_tint(NVIDIA_WHITE, 0.36),
                )
            )
            try:
                mvp_matrix, _projection = _orbit_camera_matrices(
                    cfg,
                    axis_limits=axis_limits,
                    frame_index=poster_index,
                    frame_count=source_frame_count,
                    render_resolution=poster_resolution,
                    camera_distance=(
                        float(poster_camera_distances[poster_index])
                        if len(poster_camera_distances) > poster_index
                        else None
                    ),
                )
                particle_batches, line_batches, guide_labels = _build_gpu_frame_batches(
                    cfg=cfg,
                    master_key=master_key,
                    profile=profile,
                    axis_limits=axis_limits,
                    palette=palette,
                    label_names=label_names,
                    sample_masks=sample_masks,
                    sample_frames=sample_frames,
                    centroid_frames=centroid_frames,
                    overlay_frames=overlay_frames,
                    frame_index=poster_index,
                )
                poster_image = poster_renderer.render_frame(
                    FrameSpec(
                        camera=CameraFrame(mvp_matrix=mvp_matrix),
                        particle_batches=particle_batches,
                        line_batches=line_batches,
                        background_color=str(cfg.cinematic_render.background_color),
                        output_size=poster_resolution,
                    )
                )
                poster_image = _overlay_axis_guides(
                    poster_image,
                    cfg=cfg,
                    profile=profile,
                    axis_directions=poster_axis_directions,
                    label_positions=guide_labels,
                    mvp_matrix=mvp_matrix,
                    render_resolution=poster_resolution,
                )
            finally:
                poster_renderer.close()
        else:
            poster_scene = _create_scene(
                cfg=cfg,
                axis_limits=axis_limits,
                palette=palette,
                label_names=label_names,
                sample_label_names=sample_label_names,
                render_resolution=poster_render_resolution,
                axis_directions=poster_axis_directions,
            )
            poster_scene["trail_policy"] = _master_trail_policy(cfg, master_key)
            _update_scene(
                poster_scene,
                cfg=cfg,
                sample_frames=sample_frames,
                centroid_frames=centroid_frames,
                overlay_frames=overlay_frames,
                frame_index=poster_index,
                camera_distances=poster_camera_distances,
            )
            poster_image = _capture_canvas_image(
                poster_scene["fig"],
                cfg=cfg,
                target_resolution=poster_resolution,
                profile=poster_scene["profile"],
                axis_directions=poster_scene["axis_directions"],
                mvp_matrix=poster_scene["current_mvp_matrix"],
            )
            plt.close(poster_scene["fig"])
        poster_image.save(poster_path, format="PNG", compress_level=1)
        artifacts["poster"] = poster_path.name
        resolutions["poster"] = (
            _probe_image_resolution(poster_path)
            or {"width": poster_resolution[0], "height": poster_resolution[1]}
        )

    if render_backend == "gpu_moderngl":
        animation_result = _save_cinematic_animation_gpu(
            trajectory_dir,
            cfg=cfg,
            master_key=master_key,
            stem=stem,
            camera_mode=camera_mode,
            sample_frames=sample_frames,
            centroid_frames=centroid_frames,
            overlay_frames=overlay_frames,
            sample_label_names=sample_label_names,
            label_names=label_names,
            palette=palette,
            axis_limits=axis_limits,
            render_dims=render_dims,
        )
    else:
        animation_result = _save_cinematic_animation(
            trajectory_dir,
            cfg=cfg,
            master_key=master_key,
            stem=stem,
            camera_mode=camera_mode,
            sample_frames=sample_frames,
            centroid_frames=centroid_frames,
            overlay_frames=overlay_frames,
            sample_label_names=sample_label_names,
            label_names=label_names,
            palette=palette,
            axis_limits=axis_limits,
            render_dims=render_dims,
        )
    artifacts.update(animation_result["artifacts"])
    resolutions.update(animation_result["resolutions"])
    video_encoder = str(animation_result.get("video_encoder", "libx264"))
    return {
        "artifacts": artifacts,
        "resolutions": resolutions,
        "video_encoder": video_encoder,
        "camera_mode": camera_mode,
        "role": master_role,
        "trail_policy": _master_trail_policy(cfg, master_key),
        "supersample_scale": _master_supersample_scale(cfg, master_key, "video"),
        "axis_direction_mode": "fixed_reference",
        "render_backend": render_backend,
        "camera_schedule": _master_camera_schedule(
            cfg,
            master_key,
            camera_mode=camera_mode,
            source_frame_count=source_frame_count,
            output_frame_count=len(_master_output_frame_indices(cfg, master_key, source_frame_count)),
        ),
        "look_preset": str(getattr(cfg.cinematic_render, "look_preset", "glass_wireframe")),
    }


def render_cinematic_trajectory(
    cfg: AppConfig,
    *,
    artifact_dir: Path,
    hydra_output_dir: Path | None = None,
) -> Path:
    if not bool(getattr(cfg.cinematic_render, "enabled", False)):
        raise ValueError(
            "visualize_trajectory_cinematic requires cinematic_render.enabled=true."
        )
    apply_reproducibility(cfg)

    trajectory_dir = get_label_drift_output_dir(Path(artifact_dir))
    records = _load_projected_step_records(trajectory_dir)
    if not records:
        raise FileNotFoundError(f"No projected step artifacts found under {trajectory_dir}.")

    render_dims = 3
    pca_model_path = trajectory_dir / "pca_model_final_train.npz"
    axis_limits = None
    pca_fit_scope = "unknown"
    if pca_model_path.exists():
        with np.load(pca_model_path, allow_pickle=True) as payload:
            if "render_dims" in payload:
                render_dims = int(payload["render_dims"][0])
            if "axis_limits" in payload:
                axis_limits = np.asarray(payload["axis_limits"], dtype=np.float32)
            if "fit_scope" in payload:
                pca_fit_scope = str(np.asarray(payload["fit_scope"]).reshape(-1)[0])
    if render_dims != 3:
        raise ValueError(
            f"trajectory cinematic renderer only supports 3D artifacts, got render_dims={render_dims}."
        )

    frame_indices = _downsample_frame_indices(
        len(records),
        int(getattr(cfg.cinematic_render, "max_frames", 240)),
    )
    records_display = [records[int(index)] for index in frame_indices.tolist()]
    payloads = _load_cinematic_frame_payloads(records_display, render_dims=render_dims)
    sample_frames = np.asarray(payloads["sample_frames"], dtype=np.float32)
    centroid_frames = np.asarray(payloads["centroid_frames"], dtype=np.float32)
    overlay_frames = np.asarray(payloads["overlay_frames"], dtype=np.float32)
    sample_label_names = np.asarray(payloads["sample_label_names"], dtype=str)
    label_names = np.asarray(payloads["label_names"], dtype=str)

    if axis_limits is None:
        all_values = np.concatenate(
            [
                sample_frames.reshape(-1, render_dims),
                centroid_frames.reshape(-1, render_dims),
                overlay_frames.reshape(-1, render_dims),
            ],
            axis=0,
        )
        mins = np.min(all_values, axis=0)
        maxs = np.max(all_values, axis=0)
        padding = np.maximum(maxs - mins, 1e-6) * 0.05
        axis_limits = np.stack([mins - padding, maxs + padding], axis=1).astype(np.float32)

    palette = dict(getattr(getattr(cfg.dataset, "visualization", None), "emotion_colors", {}) or {})
    render_backend = _resolve_render_backend(cfg)

    variant_reports: dict[str, dict[str, Any]] = {}
    for variant in _master_variants(cfg):
        variant_cfg = variant["cfg"]
        variant_key = str(variant["key"])
        variant_reports[variant_key] = _render_master_variant(
            trajectory_dir=trajectory_dir,
            cfg=variant_cfg,
            master_key=variant_key,
            master_role=str(variant.get("role", _master_role(variant_key))),
            stem=str(variant["stem"]),
            camera_mode=str(variant["camera_mode"]),
            render_backend=render_backend,
            sample_frames=sample_frames,
            centroid_frames=centroid_frames,
            overlay_frames=overlay_frames,
            sample_label_names=sample_label_names,
            label_names=label_names,
            palette=palette,
            axis_limits=axis_limits,
            render_dims=render_dims,
        )

    beauty_report = variant_reports.get("beauty_master", {})
    artifacts: dict[str, str] = {}
    if "artifacts" in beauty_report:
        beauty_artifacts = beauty_report["artifacts"]
        if "poster" in beauty_artifacts:
            artifacts["cinematic_poster"] = beauty_artifacts["poster"]
        if "mp4" in beauty_artifacts:
            artifacts["cinematic_mp4"] = beauty_artifacts["mp4"]
        if "gif_preview" in beauty_artifacts:
            artifacts["cinematic_gif_preview"] = beauty_artifacts["gif_preview"]
    resolutions = dict(beauty_report.get("resolutions", {}))
    video_encoder = str(beauty_report.get("video_encoder", "libx264"))

    manifest_path = trajectory_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cinematic_render"] = {
        "enabled": True,
        "render_dims": int(render_dims),
        "pca_fit_scope": pca_fit_scope,
        "num_frames": int(sample_frames.shape[0]),
        "fps": int(cfg.cinematic_render.fps),
        "max_frames": int(cfg.cinematic_render.max_frames),
        "trail_length": int(cfg.cinematic_render.trail_length),
        "look_preset": str(getattr(cfg.cinematic_render, "look_preset", "glass_wireframe")),
        "axis_style": str(getattr(cfg.cinematic_render, "axis_style", "corner_guides")),
        "axis_anchor_mode": (
            "screen_fixed"
            if str(getattr(cfg.cinematic_render, "axis_style", "corner_guides")).strip().lower()
            == "corner_guides"
            else "world_space"
        ),
        "axis_direction_mode": "fixed_reference",
        "poster_resolution": resolutions.get("poster"),
        "video_resolution": resolutions.get("video"),
        "preview_gif_resolution": resolutions.get("preview_gif"),
        "supersample_scale": float(getattr(cfg.cinematic_render, "supersample_scale", 1.0)),
        "beauty_supersample_scale": float(_master_supersample_scale(cfg, "beauty_master", "video")),
        "analysis_supersample_scale": float(_master_supersample_scale(cfg, "analysis_master", "video")),
        "video_encoder": video_encoder,
        "render_backend": render_backend,
        "artifacts": artifacts,
        "masters": {
            key: {
                "role": str(report.get("role", _master_role(key))),
                "camera_mode": str(report.get("camera_mode", "fixed_full")),
                "camera_schedule": report.get("camera_schedule", {}),
                "look_preset": str(report.get("look_preset", "glass_wireframe")),
                "trail_policy": str(report.get("trail_policy", "overlay_centroid")),
                "supersample_scale": float(report.get("supersample_scale", 1.0)),
                "axis_direction_mode": str(report.get("axis_direction_mode", "fixed_reference")),
                "render_backend": str(report.get("render_backend", render_backend)),
                "video_encoder": str(report.get("video_encoder", "none")),
                "artifacts": report.get("artifacts", {}),
                "resolutions": report.get("resolutions", {}),
            }
            for key, report in variant_reports.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if hydra_output_dir is not None:
        hydra_output_dir.mkdir(parents=True, exist_ok=True)
        (hydra_output_dir / "trajectory_cinematic_artifact_dir.txt").write_text(
            str(trajectory_dir),
            encoding="utf-8",
        )

    print(f"Cinematic trajectory artifacts written to {trajectory_dir}.")
    return trajectory_dir


def run(cfg: AppConfig, output_dir: Path, *, is_main_process: bool) -> Path | None:
    if not is_main_process:
        return None
    model_path = getattr(getattr(cfg, "stage", None), "model_path", None)
    artifact_dir = (
        Path(model_path)
        if model_path is not None and str(model_path).strip() != ""
        else get_cebra_output_dir(cfg)
    )
    return render_cinematic_trajectory(
        cfg,
        artifact_dir=artifact_dir,
        hydra_output_dir=output_dir,
    )


__all__ = ["render_cinematic_trajectory", "run"]
