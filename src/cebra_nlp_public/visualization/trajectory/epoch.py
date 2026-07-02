from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


DEFAULT_RENDER_OPTIONS: dict[str, object] = {
    "fps": 8,
    "max_frames": 180,
    "one_epoch_one_frame": False,
    "connect_segments": True,
    "rotate_camera": False,
    "camera_elev": 18.0,
    "camera_azim": 42.0,
    "trail_length": 10,
    "axis_padding": 0.08,
    "frame_width": 1920,
    "frame_height": 1080,
    "dpi": 180,
    "mp4_crf": 18,
    "mp4_preset": "slow",
}

_LEGACY_CEBRA_OPTION_NAMES = {
    "fps": "trajectory_fps",
    "max_frames": "trajectory_max_frames",
    "one_epoch_one_frame": "trajectory_one_epoch_one_frame",
    "connect_segments": "trajectory_connect_segments",
    "rotate_camera": "trajectory_rotate_camera",
    "camera_elev": "trajectory_camera_elev",
    "camera_azim": "trajectory_camera_azim",
    "trail_length": "trajectory_trail_length",
    "axis_padding": "trajectory_axis_padding",
    "frame_width": "trajectory_frame_width",
    "frame_height": "trajectory_frame_height",
    "dpi": "trajectory_dpi",
    "mp4_crf": "trajectory_mp4_crf",
    "mp4_preset": "trajectory_mp4_preset",
}


def get_epoch_trajectory_output_dir(artifact_dir: Path) -> Path:
    path = Path(artifact_dir) / "epoch_trajectory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_trajectory_seed(cfg: Any, fallback_seed: int = 0) -> int:
    reproducibility = getattr(cfg, "reproducibility", None)
    seed = getattr(reproducibility, "seed", None)
    if seed is not None:
        return int(seed)
    evaluation = getattr(cfg, "evaluation", None)
    seed = getattr(evaluation, "random_state", None)
    if seed is not None:
        return int(seed)
    return int(fallback_seed)


def select_trajectory_indices(
    total_samples: int,
    sample_size: int | None,
    *,
    seed: int,
) -> np.ndarray:
    total_samples = int(total_samples)
    if total_samples <= 0:
        return np.zeros(0, dtype=np.int64)
    if sample_size is None or int(sample_size) <= 0 or total_samples <= int(sample_size):
        return np.arange(total_samples, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(total_samples, size=int(sample_size), replace=False)
    return np.sort(indices.astype(np.int64))


def _coerce_options(options: dict[str, object]) -> dict[str, object]:
    int_keys = {
        "fps",
        "max_frames",
        "trail_length",
        "frame_width",
        "frame_height",
        "dpi",
        "mp4_crf",
    }
    float_keys = {"camera_elev", "camera_azim", "axis_padding"}
    bool_keys = {
        "one_epoch_one_frame",
        "connect_segments",
        "rotate_camera",
    }

    for key in int_keys:
        options[key] = int(options[key])
    for key in float_keys:
        options[key] = float(options[key])
    for key in bool_keys:
        options[key] = bool(options[key])
    options["mp4_preset"] = str(options["mp4_preset"])

    for key in ("fps", "max_frames", "frame_width", "frame_height", "dpi"):
        options[key] = max(1, int(options[key]))
    options["trail_length"] = max(0, int(options["trail_length"]))
    options["mp4_crf"] = max(0, int(options["mp4_crf"]))
    options["axis_padding"] = max(0.0, float(options["axis_padding"]))
    return options


def resolve_epoch_trajectory_render_options(
    cfg: Any | None = None,
    *,
    base_options: Mapping[str, object] | None = None,
    overrides: Mapping[str, object | None] | None = None,
) -> dict[str, object]:
    options = dict(DEFAULT_RENDER_OPTIONS)
    cebra_cfg = getattr(cfg, "cebra", None) if cfg is not None else None
    if cebra_cfg is not None:
        for option_name, legacy_name in _LEGACY_CEBRA_OPTION_NAMES.items():
            value = getattr(cebra_cfg, legacy_name, None)
            if value is not None:
                options[option_name] = value
    if base_options is not None:
        for key, value in base_options.items():
            if key in options and value is not None:
                options[key] = value
    if overrides is not None:
        for key, value in overrides.items():
            if key in options and value is not None:
                options[key] = value
    return _coerce_options(options)


def validate_epoch_trajectory_config(cfg: Any) -> None:
    cebra_cfg = getattr(cfg, "cebra", None)
    if cebra_cfg is None:
        return
    if int(getattr(cebra_cfg, "trajectory_every_n_epochs", 1)) <= 0:
        raise ValueError("cebra.trajectory_every_n_epochs must be > 0.")
    sample_size = getattr(cebra_cfg, "trajectory_sample_size", None)
    if sample_size is not None and int(sample_size) <= 0:
        raise ValueError("cebra.trajectory_sample_size must be > 0 when set.")


def load_epoch_trajectory_manifest(trajectory_dir: Path) -> dict[str, Any]:
    trajectory_dir = Path(trajectory_dir)
    manifest_path = trajectory_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found in trajectory directory: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots = manifest.get("snapshots", [])
    if not isinstance(snapshots, list):
        raise ValueError("epoch trajectory manifest field `snapshots` must be a list.")

    records: list[dict[str, Any]] = []
    for item in snapshots:
        if not isinstance(item, dict):
            continue
        relative_path = item.get("relative_path")
        path_value = item.get("path")
        if isinstance(relative_path, str) and relative_path:
            path = trajectory_dir / relative_path
        elif isinstance(path_value, str) and path_value:
            candidate = Path(path_value)
            path = candidate if candidate.is_absolute() else trajectory_dir / candidate
        else:
            continue
        records.append(
            {
                **item,
                "epoch": int(item.get("epoch", 0)),
                "step": int(item.get("step", 0)),
                "path": str(path),
                "relative_path": str(path.relative_to(trajectory_dir)),
            }
        )

    if not records:
        snapshot_dir = trajectory_dir / "snapshots"
        for path in sorted(snapshot_dir.glob("epoch_*.npy")):
            epoch_text = path.stem.split("_", 1)[1] if "_" in path.stem else "0"
            records.append(
                {
                    "epoch": int(epoch_text),
                    "step": 0,
                    "path": str(path),
                    "relative_path": str(path.relative_to(trajectory_dir)),
                }
            )
    if not records:
        raise FileNotFoundError(f"No epoch trajectory snapshots found in {trajectory_dir}.")

    for record in records:
        if not Path(record["path"]).exists():
            raise FileNotFoundError(f"Epoch trajectory snapshot not found: {record['path']}")

    sample_indices_path = trajectory_dir / "sample_indices.npy"
    if not sample_indices_path.exists():
        raise FileNotFoundError(f"Sample indices file not found: {sample_indices_path}")
    labels_path = trajectory_dir / "sample_labels.npy"
    sample_labels = (
        np.asarray(np.load(labels_path, allow_pickle=True), dtype=str)
        if labels_path.exists()
        else np.asarray(["sample"] * len(np.load(sample_indices_path)), dtype=str)
    )
    sample_ids_path = trajectory_dir / "sample_ids.npy"
    sample_ids = (
        np.asarray(np.load(sample_ids_path, allow_pickle=True), dtype=str)
        if sample_ids_path.exists()
        else np.asarray(np.load(sample_indices_path), dtype=str)
    )

    manifest["snapshots"] = sorted(records, key=lambda item: (int(item["epoch"]), int(item["step"])))
    manifest["sample_indices"] = np.asarray(np.load(sample_indices_path), dtype=np.int64)
    manifest["sample_labels"] = sample_labels
    manifest["sample_ids"] = sample_ids
    return manifest


def _fit_pca3d(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D final snapshot, got shape {values.shape}.")
    mean = values.mean(axis=0)
    centered = values - mean.reshape(1, -1)
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:3].astype(np.float32)
    if components.shape[0] < 3:
        pad = np.zeros((3 - components.shape[0], values.shape[1]), dtype=np.float32)
        components = np.vstack([components, pad])

    if values.shape[0] > 1:
        variances = (singular_values**2) / float(values.shape[0] - 1)
    else:
        variances = np.zeros_like(singular_values, dtype=np.float32)
    total = float(np.sum(variances))
    ratio = variances[:3] / total if total > 0.0 else np.zeros(3, dtype=np.float32)
    ratio = np.asarray(ratio, dtype=np.float32)
    if ratio.shape[0] < 3:
        ratio = np.pad(ratio, (0, 3 - ratio.shape[0]))
    return {"mean": mean.astype(np.float32), "components": components, "ratio": ratio}


def _project_snapshots(snapshots: list[np.ndarray], pca: dict[str, np.ndarray]) -> np.ndarray:
    projected = []
    mean = pca["mean"].reshape(1, -1)
    components = pca["components"]
    for snapshot in snapshots:
        snapshot = np.asarray(snapshot, dtype=np.float32)
        if snapshot.ndim != 2:
            raise ValueError(f"Expected 2D snapshot arrays, got shape {snapshot.shape}.")
        projected.append((snapshot - mean) @ components.T)
    return np.stack(projected, axis=0).astype(np.float32)


def _select_frame_indices(total_frames: int, max_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if max_frames <= 0 or total_frames <= max_frames:
        return np.arange(total_frames, dtype=np.int64)
    return np.unique(
        np.linspace(0, total_frames - 1, num=max_frames, dtype=np.int64)
    )


def _axis_limits(values: np.ndarray, padding_fraction: float) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    finite = flattened[np.all(np.isfinite(flattened), axis=1)]
    if finite.size == 0:
        return np.asarray([[-1.0, 1.0]] * 3, dtype=np.float32)
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    padding = span * float(padding_fraction)
    return np.stack([mins - padding, maxs + padding], axis=1).astype(np.float32)


def _label_colors(labels: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    unique_labels = sorted({str(label) for label in labels.tolist()})
    cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))
    return {
        label: mcolors.to_rgba(cmap(index))
        for index, label in enumerate(unique_labels)
    }


def _render_frame(
    path: Path,
    *,
    trajectory: np.ndarray,
    frame_index: int,
    epoch: int,
    step: int,
    sample_labels: np.ndarray,
    axis_limits: np.ndarray,
    options: Mapping[str, object],
) -> None:
    dpi = int(options["dpi"])
    figsize = (
        float(options["frame_width"]) / float(dpi),
        float(options["frame_height"]) / float(dpi),
    )
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    colors = _label_colors(sample_labels)
    current = trajectory[frame_index]
    for label_name in sorted(colors):
        mask = sample_labels == label_name
        if not np.any(mask):
            continue
        coords = current[mask]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=18,
            c=[colors[label_name]],
            depthshade=False,
            linewidths=0,
            label=label_name,
        )

    if bool(options["connect_segments"]):
        trail_length = int(options["trail_length"])
        start = 0 if trail_length <= 0 else max(0, frame_index - trail_length)
        for sample_index in range(current.shape[0]):
            coords = trajectory[start : frame_index + 1, sample_index, :]
            if coords.shape[0] < 2:
                continue
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                color="#7f7f7f",
                alpha=0.25,
                linewidth=0.8,
            )

    ax.set_xlim(float(axis_limits[0, 0]), float(axis_limits[0, 1]))
    ax.set_ylim(float(axis_limits[1, 0]), float(axis_limits[1, 1]))
    ax.set_zlim(float(axis_limits[2, 0]), float(axis_limits[2, 1]))
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    azim = float(options["camera_azim"])
    if bool(options["rotate_camera"]):
        azim += 360.0 * (float(frame_index) / float(max(1, trajectory.shape[0] - 1)))
    ax.view_init(elev=float(options["camera_elev"]), azim=azim)
    ax.set_title(f"Epoch {epoch} / step {step}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Label")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _render_gif_with_pillow(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    if not frame_paths:
        raise ValueError("No frame paths provided.")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on runtime image stack
        raise RuntimeError("Pillow is required to render GIF output.") from exc

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    try:
        duration_ms = int(round(1000.0 / max(1, int(fps))))
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
    finally:
        for image in images:
            image.close()


def _render_mp4_with_ffmpeg(
    frame_paths: list[Path],
    output_path: Path,
    *,
    fps: int,
    crf: int,
    preset: str,
) -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return "ffmpeg executable not found"
    if not frame_paths:
        return "No frame paths provided"
    frame_dir = frame_paths[0].parent
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frame_dir / "%05d.png"),
        "-c:v",
        "libx264",
        "-crf",
        str(int(crf)),
        "-preset",
        str(preset),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return result.stderr.strip() or f"ffmpeg exited with {result.returncode}"
    return None


def render_saved_epoch_trajectory(
    trajectory_dir: Path,
    *,
    cfg: Any | None = None,
    fps: int | None = None,
    max_frames: int | None = None,
    one_epoch_one_frame: bool | None = None,
    connect_segments: bool | None = None,
    rotate_camera: bool | None = None,
    camera_elev: float | None = None,
    camera_azim: float | None = None,
    trail_length: int | None = None,
    axis_padding: float | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    dpi: int | None = None,
    mp4_crf: int | None = None,
    mp4_preset: str | None = None,
) -> dict[str, str]:
    trajectory_dir = Path(trajectory_dir)
    overrides = {
        "fps": fps,
        "max_frames": max_frames,
        "one_epoch_one_frame": one_epoch_one_frame,
        "connect_segments": connect_segments,
        "rotate_camera": rotate_camera,
        "camera_elev": camera_elev,
        "camera_azim": camera_azim,
        "trail_length": trail_length,
        "axis_padding": axis_padding,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "dpi": dpi,
        "mp4_crf": mp4_crf,
        "mp4_preset": mp4_preset,
    }
    options = resolve_epoch_trajectory_render_options(cfg, overrides=overrides)
    manifest = load_epoch_trajectory_manifest(trajectory_dir)
    records = list(manifest["snapshots"])
    snapshots = [np.asarray(np.load(record["path"]), dtype=np.float32) for record in records]
    first_shape = snapshots[0].shape
    if any(snapshot.shape != first_shape for snapshot in snapshots):
        raise ValueError("Snapshot shapes differ across epochs.")
    pca = _fit_pca3d(snapshots[-1])
    trajectory = _project_snapshots(snapshots, pca)
    np.save(trajectory_dir / "trajectory_pca3d.npy", trajectory)
    with (trajectory_dir / "trajectory_pca_explained_variance.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "explained_variance_ratio"])
        for component, ratio in enumerate(pca["ratio"], start=1):
            writer.writerow([component, float(ratio)])

    frame_indices = _select_frame_indices(trajectory.shape[0], int(options["max_frames"]))
    sample_labels = np.asarray(manifest["sample_labels"], dtype=str)
    if sample_labels.shape[0] != trajectory.shape[1]:
        sample_labels = np.asarray(["sample"] * trajectory.shape[1], dtype=str)
    limits = _axis_limits(trajectory, float(options["axis_padding"]))

    gif_error = None
    mp4_error = None
    gif_path = trajectory_dir / "trajectory_pca3d.gif"
    mp4_path = trajectory_dir / "trajectory_pca3d.mp4"
    with tempfile.TemporaryDirectory(prefix="epoch_trajectory_frames_") as temp_name:
        temp_dir = Path(temp_name)
        frame_paths: list[Path] = []
        for output_index, source_index in enumerate(frame_indices.tolist()):
            frame_path = temp_dir / f"{output_index:05d}.png"
            record = records[int(source_index)]
            _render_frame(
                frame_path,
                trajectory=trajectory,
                frame_index=int(source_index),
                epoch=int(record["epoch"]),
                step=int(record["step"]),
                sample_labels=sample_labels,
                axis_limits=limits,
                options=options,
            )
            frame_paths.append(frame_path)

        try:
            _render_gif_with_pillow(frame_paths, gif_path, int(options["fps"]))
        except Exception as exc:  # pragma: no cover - failure is reported for diagnostics
            gif_error = str(exc)

        if gif_error is None:
            mp4_error = _render_mp4_with_ffmpeg(
                frame_paths,
                mp4_path,
                fps=int(options["fps"]),
                crf=int(options["mp4_crf"]),
                preset=str(options["mp4_preset"]),
            )

    artifacts: dict[str, str] = {
        "pca": "trajectory_pca3d.npy",
        "explained_variance": "trajectory_pca_explained_variance.csv",
    }
    outputs: dict[str, str] = {}
    if gif_error is None and gif_path.exists():
        artifacts["gif"] = gif_path.name
        outputs["gif"] = gif_path.name
    if mp4_error is None and mp4_path.exists():
        artifacts["mp4"] = mp4_path.name
        outputs["mp4"] = mp4_path.name

    report = {
        "num_snapshots": int(len(records)),
        "selected_epochs": [int(records[int(index)]["epoch"]) for index in frame_indices.tolist()],
        "selected_steps": [int(records[int(index)]["step"]) for index in frame_indices.tolist()],
        "pca_basis_epoch": int(records[-1]["epoch"]),
        "pca_basis_step": int(records[-1]["step"]),
        "render_options": options,
        "artifacts": artifacts,
        "gif_error": gif_error,
        "mp4_error": mp4_error,
    }
    (trajectory_dir / "trajectory_render_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return outputs


__all__ = [
    "get_epoch_trajectory_output_dir",
    "load_epoch_trajectory_manifest",
    "render_saved_epoch_trajectory",
    "resolve_epoch_trajectory_render_options",
    "resolve_trajectory_seed",
    "select_trajectory_indices",
    "validate_epoch_trajectory_config",
]
