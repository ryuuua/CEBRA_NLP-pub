from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _video_size(path: Path) -> tuple[int, int] | None:
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width_str, height_str = result.stdout.strip().split("x", maxsplit=1)
    return int(width_str), int(height_str)


def _write_projected_step(
    path: Path,
    *,
    sample_projection: np.ndarray,
    centroid_projection: np.ndarray,
    overlay_projection: np.ndarray,
    sample_label_names: np.ndarray,
    label_names: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        sample_pca=sample_projection.astype(np.float32),
        centroid_pca=centroid_projection.astype(np.float32),
        label_pca=overlay_projection.astype(np.float32),
        sample_pca_3d=sample_projection.astype(np.float32),
        centroid_pca_3d=centroid_projection.astype(np.float32),
        label_pca_3d=overlay_projection.astype(np.float32),
        sample_label_names=np.asarray(sample_label_names, dtype=str),
        label_names=np.asarray(label_names, dtype=str),
        sample_ids=np.asarray(["a", "b", "c", "d"], dtype=str),
        label_ids=np.asarray([0, 1], dtype=np.int64),
        overlay_ids=np.asarray(["label::imdb::0", "label::imdb::1"], dtype=str),
        sample_counts=np.asarray([2, 2], dtype=np.int64),
    )


def _assert_dual_master_manifest(
    updated_manifest: dict[str, object],
    *,
    beauty_poster_name: str,
    beauty_gif_name: str,
    beauty_mp4_name: str | None,
    analysis_poster_name: str,
    analysis_gif_name: str,
    analysis_mp4_name: str | None,
    expected_backend: str,
) -> None:
    cinematic_render = updated_manifest["cinematic_render"]
    assert cinematic_render["enabled"] is True
    assert cinematic_render["num_frames"] == 3
    assert cinematic_render["pca_fit_scope"] == "final_train"
    assert cinematic_render["axis_anchor_mode"] == "screen_fixed"
    assert cinematic_render["axis_direction_mode"] == "fixed_reference"
    assert cinematic_render["masters"]
    assert "beauty_master" in cinematic_render["masters"]
    assert "analysis_master" in cinematic_render["masters"]

    assert cinematic_render["artifacts"]["cinematic_poster"] == beauty_poster_name
    assert cinematic_render["artifacts"]["cinematic_gif_preview"] == beauty_gif_name
    if beauty_mp4_name is not None:
        assert cinematic_render["artifacts"]["cinematic_mp4"] == beauty_mp4_name

    beauty_master = cinematic_render["masters"]["beauty_master"]
    analysis_master = cinematic_render["masters"]["analysis_master"]

    assert beauty_master["artifacts"]["poster"] == beauty_poster_name
    assert beauty_master["artifacts"]["gif_preview"] == beauty_gif_name
    if beauty_mp4_name is not None:
        assert beauty_master["artifacts"]["mp4"] == beauty_mp4_name
    assert beauty_master["camera_mode"] == "auto_zoom_out"
    assert beauty_master["axis_direction_mode"] == "fixed_reference"
    assert beauty_master["supersample_scale"] == 4.0
    assert beauty_master.get("trail_policy", beauty_master.get("trail_mode")) == "label_only"
    assert beauty_master["render_backend"] == expected_backend

    assert analysis_master["artifacts"]["poster"] == analysis_poster_name
    assert analysis_master["artifacts"]["gif_preview"] == analysis_gif_name
    if analysis_mp4_name is not None:
        assert analysis_master["artifacts"]["mp4"] == analysis_mp4_name
    assert analysis_master["camera_mode"] == "fixed_full"
    assert analysis_master["axis_direction_mode"] == "fixed_reference"
    assert analysis_master["supersample_scale"] == 1.25
    assert analysis_master["render_backend"] == expected_backend


def test_cinematic_trajectory_cli_renders_from_projected_steps(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    artifact_dir = model_root / "imdb__sentence_bert__12345678" / "cinematic-test"
    trajectory_dir = artifact_dir / "label_drift_trajectory"
    projected_dir = trajectory_dir / "projected_steps"
    projected_dir.mkdir(parents=True, exist_ok=True)

    sample_label_names = np.asarray(["negative", "negative", "positive", "positive"], dtype=str)
    label_names = np.asarray(["negative", "positive"], dtype=str)
    frames = []
    for step in range(3):
        sample_projection = np.asarray(
            [
                [-1.0 + 0.15 * step, -0.2, 0.1],
                [-0.8 + 0.15 * step, -0.1, 0.05],
                [0.8 - 0.1 * step, 0.25, -0.1],
                [1.0 - 0.1 * step, 0.2, -0.05],
            ],
            dtype=np.float32,
        )
        centroid_projection = np.asarray(
            [
                [-0.9 + 0.15 * step, -0.15, 0.08],
                [0.9 - 0.1 * step, 0.22, -0.08],
            ],
            dtype=np.float32,
        )
        overlay_projection = np.asarray(
            [
                [-1.05 + 0.15 * step, -0.05, 0.15],
                [1.05 - 0.1 * step, 0.35, -0.12],
            ],
            dtype=np.float32,
        )
        path = projected_dir / f"step_{step:06d}.npz"
        _write_projected_step(
            path,
            sample_projection=sample_projection,
            centroid_projection=centroid_projection,
            overlay_projection=overlay_projection,
            sample_label_names=sample_label_names,
            label_names=label_names,
        )
        frames.append(
            {
                "step": step,
                "estimated_epoch": float(step),
                "relative_path": str(path.relative_to(trajectory_dir)),
            }
        )

    axis_limits = np.asarray([[-1.4, 1.4], [-0.5, 0.5], [-0.4, 0.4]], dtype=np.float32)
    np.savez(
        trajectory_dir / "pca_model_final_train.npz",
        mean=np.zeros(3, dtype=np.float32),
        components=np.eye(3, dtype=np.float32),
        explained_variance_ratio=np.asarray([0.5, 0.3, 0.2], dtype=np.float32),
        fit_scope=np.asarray(["final_train"]),
        render_dims=np.asarray([3], dtype=np.int64),
        axis_limits=axis_limits,
    )

    manifest = {
        "render_dims": 3,
        "label_names": label_names.tolist(),
        "projected_steps": frames,
    }
    (trajectory_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    _run(
        [
            sys.executable,
            "visualize_trajectory_cinematic.py",
            "dataset=imdb",
            "embedding=sentence_bert",
            "cebra=offset1-model",
            "cebra.max_iterations=2",
            "cebra.output_dim=3",
            "consistency_check.enabled=false",
            "evaluation.enable_plots=false",
            "evaluation.local_linearity_probe.enabled=false",
                "device=cpu",
                f"paths.model_dir={model_root}",
                "stage.run_tag=cinematic-test",
                f"stage.model_path={artifact_dir}",
            "cinematic_render.enabled=true",
            "cinematic_render.render_backend=cpu_matplotlib",
            "cinematic_render.max_frames=3",
            "cinematic_render.fps=4",
            "cinematic_render.look_preset=glass_wireframe",
            "cinematic_render.axis_style=corner_guides",
            "cinematic_render.poster_width=640",
            "cinematic_render.poster_height=360",
            "cinematic_render.video_width=640",
            "cinematic_render.video_height=360",
            "cinematic_render.gif_width=320",
            "cinematic_render.gif_height=180",
            "cinematic_render.static_dpi=80",
            "cinematic_render.animation_dpi=80",
            "cinematic_render.supersample_scale=1.0",
            "cinematic_render.glow_blur_small_px=2.0",
            "cinematic_render.glow_blur_large_px=5.0",
            "cinematic_render.glow_gain=0.7",
            f"hydra.run.dir={tmp_path / 'cinematic'}",
        ],
        env=env,
    )

    beauty_poster_path = trajectory_dir / "label_drift_beauty_master_3d.png"
    beauty_gif_path = trajectory_dir / "label_drift_beauty_master_preview_3d.gif"
    beauty_mp4_path = trajectory_dir / "label_drift_beauty_master_3d.mp4"
    analysis_poster_path = trajectory_dir / "label_drift_analysis_master_3d.png"
    analysis_gif_path = trajectory_dir / "label_drift_analysis_master_preview_3d.gif"
    analysis_mp4_path = trajectory_dir / "label_drift_analysis_master_3d.mp4"
    hydra_artifact_path = tmp_path / "cinematic" / "trajectory_cinematic_artifact_dir.txt"
    assert beauty_poster_path.exists()
    assert beauty_gif_path.exists()
    assert analysis_poster_path.exists()
    assert analysis_gif_path.exists()
    assert _image_size(beauty_poster_path) == (640, 360)
    assert _image_size(beauty_gif_path) == (320, 180)
    assert _image_size(analysis_poster_path) == (640, 360)
    assert _image_size(analysis_gif_path) == (320, 180)
    assert hydra_artifact_path.read_text(encoding="utf-8").strip() == str(trajectory_dir)

    if shutil.which("ffmpeg") is not None:
        assert beauty_mp4_path.exists()
        assert analysis_mp4_path.exists()
        assert _video_size(beauty_mp4_path) == (640, 360)
        assert _video_size(analysis_mp4_path) == (640, 360)

    updated_manifest = json.loads((trajectory_dir / "manifest.json").read_text(encoding="utf-8"))
    assert updated_manifest["cinematic_render"]["enabled"] is True
    assert updated_manifest["cinematic_render"]["num_frames"] == 3
    assert updated_manifest["cinematic_render"]["axis_style"] == "corner_guides"
    assert updated_manifest["cinematic_render"]["axis_anchor_mode"] == "screen_fixed"
    assert updated_manifest["cinematic_render"]["axis_direction_mode"] == "fixed_reference"
    assert updated_manifest["cinematic_render"]["pca_fit_scope"] == "final_train"
    assert updated_manifest["cinematic_render"]["render_backend"] == "cpu_matplotlib"
    assert updated_manifest["cinematic_render"]["poster_resolution"] == {
        "width": 640,
        "height": 360,
    }
    assert updated_manifest["cinematic_render"]["preview_gif_resolution"] == {
        "width": 320,
        "height": 180,
    }
    assert updated_manifest["cinematic_render"]["beauty_supersample_scale"] == 4.0
    assert updated_manifest["cinematic_render"]["analysis_supersample_scale"] == 1.25
    _assert_dual_master_manifest(
        updated_manifest,
        beauty_poster_name=beauty_poster_path.name,
        beauty_gif_name=beauty_gif_path.name,
        beauty_mp4_name=beauty_mp4_path.name if shutil.which("ffmpeg") is not None else None,
        analysis_poster_name=analysis_poster_path.name,
        analysis_gif_name=analysis_gif_path.name,
        analysis_mp4_name=analysis_mp4_path.name if shutil.which("ffmpeg") is not None else None,
        expected_backend="cpu_matplotlib",
    )
    if shutil.which("ffmpeg") is not None:
        assert updated_manifest["cinematic_render"]["video_resolution"] == {
            "width": 640,
            "height": 360,
        }


@pytest.mark.skip(reason="GPU cinematic smoke is outside the public CPU tutorial validation path.")
def test_cinematic_trajectory_cli_prefers_gpu_backend_when_available(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    artifact_dir = model_root / "imdb__sentence_bert__gpu12345" / "cinematic-gpu-test"
    trajectory_dir = artifact_dir / "label_drift_trajectory"
    projected_dir = trajectory_dir / "projected_steps"
    projected_dir.mkdir(parents=True, exist_ok=True)

    sample_label_names = np.asarray(["negative", "negative", "positive", "positive"], dtype=str)
    label_names = np.asarray(["negative", "positive"], dtype=str)
    frames = []
    for step in range(3):
        sample_projection = np.asarray(
            [
                [-1.0 + 0.15 * step, -0.2, 0.1],
                [-0.8 + 0.15 * step, -0.1, 0.05],
                [0.8 - 0.1 * step, 0.25, -0.1],
                [1.0 - 0.1 * step, 0.2, -0.05],
            ],
            dtype=np.float32,
        )
        centroid_projection = np.asarray(
            [
                [-0.9 + 0.15 * step, -0.15, 0.08],
                [0.9 - 0.1 * step, 0.22, -0.08],
            ],
            dtype=np.float32,
        )
        overlay_projection = np.asarray(
            [
                [-1.05 + 0.15 * step, -0.05, 0.15],
                [1.05 - 0.1 * step, 0.35, -0.12],
            ],
            dtype=np.float32,
        )
        path = projected_dir / f"step_{step:06d}.npz"
        _write_projected_step(
            path,
            sample_projection=sample_projection,
            centroid_projection=centroid_projection,
            overlay_projection=overlay_projection,
            sample_label_names=sample_label_names,
            label_names=label_names,
        )
        frames.append(
            {
                "step": step,
                "estimated_epoch": float(step),
                "relative_path": str(path.relative_to(trajectory_dir)),
            }
        )

    axis_limits = np.asarray([[-1.4, 1.4], [-0.5, 0.5], [-0.4, 0.4]], dtype=np.float32)
    np.savez(
        trajectory_dir / "pca_model_final_train.npz",
        mean=np.zeros(3, dtype=np.float32),
        components=np.eye(3, dtype=np.float32),
        explained_variance_ratio=np.asarray([0.5, 0.3, 0.2], dtype=np.float32),
        fit_scope=np.asarray(["final_train"]),
        render_dims=np.asarray([3], dtype=np.int64),
        axis_limits=axis_limits,
    )
    (trajectory_dir / "manifest.json").write_text(
        json.dumps(
            {"render_dims": 3, "label_names": label_names.tolist(), "projected_steps": frames},
            indent=2,
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    _run(
        [
            sys.executable,
            "visualize_trajectory_cinematic.py",
            "dataset=imdb",
            "embedding=sentence_bert",
            "cebra=offset1-model",
            "cebra.max_iterations=2",
            "cebra.output_dim=3",
            "consistency_check.enabled=false",
            "evaluation.enable_plots=false",
            "evaluation.local_linearity_probe.enabled=false",
            "device=cpu",
            f"paths.model_dir={model_root}",
            "stage.run_tag=cinematic-gpu-test",
            f"stage.model_path={artifact_dir}",
            "cinematic_render.enabled=true",
            "cinematic_render.render_backend=auto",
            "cinematic_render.max_frames=3",
            "cinematic_render.fps=4",
            "cinematic_render.poster_width=640",
            "cinematic_render.poster_height=360",
            "cinematic_render.video_width=640",
            "cinematic_render.video_height=360",
            "cinematic_render.gif_width=320",
            "cinematic_render.gif_height=180",
            "cinematic_render.static_dpi=80",
            "cinematic_render.animation_dpi=80",
            "cinematic_render.supersample_scale=1.0",
            "cinematic_render.glow_blur_small_px=2.0",
            "cinematic_render.glow_blur_large_px=5.0",
            "cinematic_render.glow_gain=0.7",
            f"hydra.run.dir={tmp_path / 'cinematic_gpu'}",
        ],
        env=env,
    )

    beauty_poster_path = trajectory_dir / "label_drift_beauty_master_3d.png"
    beauty_gif_path = trajectory_dir / "label_drift_beauty_master_preview_3d.gif"
    beauty_mp4_path = trajectory_dir / "label_drift_beauty_master_3d.mp4"
    analysis_poster_path = trajectory_dir / "label_drift_analysis_master_3d.png"
    analysis_gif_path = trajectory_dir / "label_drift_analysis_master_preview_3d.gif"
    analysis_mp4_path = trajectory_dir / "label_drift_analysis_master_3d.mp4"
    assert beauty_poster_path.exists()
    assert beauty_gif_path.exists()
    assert analysis_poster_path.exists()
    assert analysis_gif_path.exists()
    assert _image_size(beauty_poster_path) == (640, 360)
    assert _image_size(beauty_gif_path) == (320, 180)
    assert _image_size(analysis_poster_path) == (640, 360)
    assert _image_size(analysis_gif_path) == (320, 180)
    if shutil.which("ffmpeg") is not None:
        assert beauty_mp4_path.exists()
        assert analysis_mp4_path.exists()
        assert _video_size(beauty_mp4_path) == (640, 360)
        assert _video_size(analysis_mp4_path) == (640, 360)

    updated_manifest = json.loads((trajectory_dir / "manifest.json").read_text(encoding="utf-8"))
    assert updated_manifest["cinematic_render"]["render_backend"] == "gpu_moderngl"
    assert updated_manifest["cinematic_render"]["axis_anchor_mode"] == "screen_fixed"
    assert updated_manifest["cinematic_render"]["axis_direction_mode"] == "fixed_reference"
    assert updated_manifest["cinematic_render"]["pca_fit_scope"] == "final_train"
    assert updated_manifest["cinematic_render"]["beauty_supersample_scale"] == 4.0
    assert updated_manifest["cinematic_render"]["analysis_supersample_scale"] == 1.25
    _assert_dual_master_manifest(
        updated_manifest,
        beauty_poster_name=beauty_poster_path.name,
        beauty_gif_name=beauty_gif_path.name,
        beauty_mp4_name=beauty_mp4_path.name if shutil.which("ffmpeg") is not None else None,
        analysis_poster_name=analysis_poster_path.name,
        analysis_gif_name=analysis_gif_path.name,
        analysis_mp4_name=analysis_mp4_path.name if shutil.which("ffmpeg") is not None else None,
        expected_backend="gpu_moderngl",
    )
    assert updated_manifest["cinematic_render"]["poster_resolution"] == {
        "width": 640,
        "height": 360,
    }
