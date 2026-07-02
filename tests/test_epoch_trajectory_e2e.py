from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_csv(path: Path) -> None:
    rows = [
        ("I feel happy and hopeful today.", 1),
        ("This is terrible and disappointing.", 0),
        ("What a wonderful and bright morning.", 1),
        ("I am frustrated by this broken workflow.", 0),
        ("The result looks great and clean.", 1),
        ("This output is messy and confusing.", 0),
        ("Everything is calm and under control.", 1),
        ("I am upset and annoyed right now.", 0),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def test_epoch_trajectory_train_and_cli_rerender(tmp_path: Path) -> None:
    workdir = tmp_path / "epoch_trajectory_e2e"
    run_dir = workdir / "runs"
    cache_store = workdir / "embedding_cache"
    model_store = workdir / "models"
    csv_path = workdir / "data" / "tiny_imdb.csv"
    _write_tiny_csv(csv_path)

    overrides = [
        "hpt=default",
        "dataset=imdb",
        "dataset.source=csv",
        f"+dataset.data_files={csv_path}",
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
        "cebra.save_epoch_trajectory=true",
        "cebra.trajectory_every_n_epochs=1",
        "cebra.trajectory_sample_size=3",
        "cebra.trajectory_fps=2",
        "cebra.trajectory_max_frames=6",
        "cebra.trajectory_frame_width=320",
        "cebra.trajectory_frame_height=240",
        "cebra.trajectory_dpi=80",
        "consistency_check.enabled=false",
        "evaluation.enable_plots=false",
        "evaluation.local_linearity_probe.enabled=false",
        "evaluation.knn_backend=sklearn",
        "wandb.project=",
        "device=cpu",
        f"paths.embedding_cache_dir={cache_store}",
        f"paths.model_dir={model_store}",
        "label_overlay.enabled=false",
        "trajectory_analysis.enabled=false",
    ]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    _run(
        [
            sys.executable,
            "cache_embeddings.py",
            *overrides,
            f"hydra.run.dir={run_dir / 'cache'}",
        ],
        env=env,
    )
    _run(
        [
            sys.executable,
            "train_cebra.py",
            *overrides,
            f"hydra.run.dir={run_dir / 'train'}",
        ],
        env=env,
    )

    trajectory_dirs = sorted(model_store.glob("**/epoch_trajectory"))
    assert trajectory_dirs
    trajectory_dir = trajectory_dirs[0]
    snapshot_paths = sorted((trajectory_dir / "snapshots").glob("epoch_*.npy"))
    assert [path.name for path in snapshot_paths] == [
        "epoch_000000.npy",
        "epoch_000001.npy",
        "epoch_000002.npy",
    ]
    assert (trajectory_dir / "sample_indices.npy").exists()
    assert (trajectory_dir / "sample_ids.npy").exists()
    assert (trajectory_dir / "sample_labels.npy").exists()
    assert (trajectory_dir / "trajectory_pca3d.npy").exists()
    assert (trajectory_dir / "trajectory_pca3d.gif").exists()
    assert (trajectory_dir / "trajectory_pca_explained_variance.csv").exists()
    assert (trajectory_dir / "trajectory_render_report.json").exists()

    sample_indices = np.load(trajectory_dir / "sample_indices.npy")
    sample_ids = np.asarray(np.load(trajectory_dir / "sample_ids.npy", allow_pickle=True), dtype=str)
    sample_labels = np.asarray(np.load(trajectory_dir / "sample_labels.npy", allow_pickle=True), dtype=str)
    assert sample_indices.shape == (3,)
    assert sample_ids.shape == (3,)
    assert sample_labels.shape == (3,)

    manifest = json.loads((trajectory_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_size"] == 3
    assert manifest["num_samples"] == 3
    assert [item["epoch"] for item in manifest["snapshots"]] == [0, 1, 2]
    assert [item["step"] for item in manifest["snapshots"]] == [0, 2, 4]
    assert manifest["render_report"]["artifacts"]["gif"] == "trajectory_pca3d.gif"

    (trajectory_dir / "trajectory_pca3d.gif").unlink()
    _run(
        [
            sys.executable,
            "tools/render_epoch_trajectory.py",
            str(trajectory_dir),
            "--fps",
            "2",
            "--max-frames",
            "2",
            "--frame-width",
            "320",
            "--frame-height",
            "240",
            "--dpi",
            "80",
        ],
        env=env,
    )
    assert (trajectory_dir / "trajectory_pca3d.gif").exists()
