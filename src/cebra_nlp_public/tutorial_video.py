from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "examples" / "tiny_sentiment.csv"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _common_overrides(workdir: Path, dataset_csv: Path) -> list[str]:
    return [
        "hpt=default",
        "dataset=tiny_sentiment",
        f"dataset.data_files={dataset_csv}",
        "dataset.shuffle=false",
        "embedding=sentence_bert",
        "embedding.batch_size=8",
        "cebra=offset1-model",
        "cebra.max_iterations=4",
        "cebra.output_dim=3",
        "cebra.num_workers=0",
        "cebra.pin_memory=false",
        "cebra.persistent_workers=false",
        "cebra.params.batch_size=4",
        "cebra.params.learning_rate=0.001",
        "consistency_check.enabled=false",
        "evaluation.enable_plots=false",
        "evaluation.local_linearity_probe.enabled=false",
        "evaluation.knn_backend=sklearn",
        "wandb.project=",
        "device=cpu",
        f"paths.embedding_cache_dir={workdir / 'cache' / 'embeddings'}",
        f"paths.model_dir={workdir / 'models'}",
        "stage.run_tag=tutorial_video",
        "label_overlay.enabled=false",
        "trajectory_analysis.enabled=true",
        "trajectory_analysis.checkpoint_every_n_steps=1",
        "trajectory_analysis.render_dims=3",
        "trajectory_analysis.include_sample_cloud=true",
        "trajectory_analysis.export_animation=true",
        "trajectory_analysis.export_static_panels=true",
        "trajectory_analysis.export_clean_variant=false",
        "trajectory_analysis.max_frames=6",
        "trajectory_analysis.fps=4",
        "pca_analysis.plot_sample_limit=6",
    ]


def _cinematic_overrides() -> list[str]:
    return [
        "cinematic_render.enabled=true",
        "cinematic_render.render_backend=cpu_matplotlib",
        "cinematic_render.max_frames=4",
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
        "cinematic_render.beauty_video_supersample_scale=1.0",
        "cinematic_render.beauty_poster_supersample_scale=1.0",
        "cinematic_render.analysis_supersample_scale=1.0",
        "cinematic_render.glow_blur_small_px=2.0",
        "cinematic_render.glow_blur_large_px=5.0",
        "cinematic_render.glow_gain=0.7",
        "cinematic_render.prefer_gpu_encode=false",
    ]


def _find_artifact_dir(workdir: Path) -> Path:
    matches = sorted((workdir / "models").glob("**/tutorial_video/label_drift_trajectory"))
    if not matches:
        raise FileNotFoundError(
            f"No label drift trajectory artifact was produced under {workdir / 'models'}."
        )
    return matches[0].parent


def _assert_outputs(trajectory_dir: Path) -> None:
    expected = [
        trajectory_dir / "label_drift_beauty_master_3d.png",
        trajectory_dir / "label_drift_beauty_master_preview_3d.gif",
        trajectory_dir / "label_drift_beauty_master_3d.mp4",
        trajectory_dir / "manifest.json",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Tutorial did not produce expected artifacts:\n{joined}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the public tiny CSV -> CEBRA trajectory -> video tutorial."
    )
    parser.add_argument("--workdir", type=Path, default=Path("runs/tutorial_video"))
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--force", action="store_true", help="Remove workdir before running.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workdir = args.workdir.resolve()
    dataset_csv = args.dataset_csv.resolve()
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
    if workdir.exists() and args.force:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    common = _common_overrides(workdir, dataset_csv)
    _run(
        [
            sys.executable,
            "cache_embeddings.py",
            *common,
            f"hydra.run.dir={workdir / 'runs' / 'cache'}",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    _run(
        [
            sys.executable,
            "train_cebra.py",
            *common,
            f"hydra.run.dir={workdir / 'runs' / 'train'}",
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    artifact_dir = _find_artifact_dir(workdir)
    _run(
        [
            sys.executable,
            "visualize_trajectory_cinematic.py",
            *common,
            *_cinematic_overrides(),
            f"stage.model_path={artifact_dir}",
            f"hydra.run.dir={workdir / 'runs' / 'cinematic'}",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    trajectory_dir = artifact_dir / "label_drift_trajectory"
    _assert_outputs(trajectory_dir)
    print(f"Tutorial artifacts: {trajectory_dir}")
    print(f"MP4: {trajectory_dir / 'label_drift_beauty_master_3d.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
