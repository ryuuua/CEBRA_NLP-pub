from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_csv(path: Path) -> int:
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
    return len(rows)


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def test_label_drift_trajectory_train_and_rerender(tmp_path: Path) -> None:
    workdir = tmp_path / "trajectory_e2e"
    data_dir = workdir / "data"
    run_dir = workdir / "runs"
    cache_store = workdir / "embedding_cache"
    model_store = workdir / "models"
    csv_path = data_dir / "tiny_imdb.csv"
    _write_tiny_csv(csv_path)

    common_overrides = [
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
        "consistency_check.enabled=false",
        "evaluation.enable_plots=false",
        "evaluation.local_linearity_probe.enabled=false",
        "evaluation.knn_backend=sklearn",
        "wandb.project=",
        "device=cpu",
        f"paths.embedding_cache_dir={cache_store}",
        f"paths.model_dir={model_store}",
        "label_overlay.enabled=false",
        "trajectory_analysis.enabled=true",
        "trajectory_analysis.checkpoint_every_n_steps=1",
        "trajectory_analysis.render_dims=3",
        "trajectory_analysis.include_sample_cloud=true",
        "trajectory_analysis.export_animation=true",
        "trajectory_analysis.export_static_panels=true",
        "trajectory_analysis.export_clean_variant=true",
        "trajectory_analysis.max_frames=8",
        "pca_analysis.plot_sample_limit=3",
    ]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    _run(
        [
            sys.executable,
            "cache_embeddings.py",
            *common_overrides,
            f"hydra.run.dir={run_dir / 'cache'}",
        ],
        env=env,
    )
    _run(
        [
            sys.executable,
            "train_cebra.py",
            *common_overrides,
            f"hydra.run.dir={run_dir / 'train'}",
        ],
        env=env,
    )

    cache_manifest = run_dir / "cache" / "label_overlay_manifest.csv"
    assert cache_manifest.exists()

    trajectory_dirs = sorted(model_store.glob("**/label_drift_trajectory"))
    assert trajectory_dirs
    trajectory_dir = trajectory_dirs[0]
    manifest_path = trajectory_dir / "manifest.json"
    pca_model_path = trajectory_dir / "pca_model_final_train.npz"
    metrics_path = trajectory_dir / "label_drift_metrics.csv"
    pca_plot_path = trajectory_dir / "label_drift_pca_3d.png"
    pca_plot_clean_path = trajectory_dir / "label_drift_pca_3d_clean.png"
    gif_path = trajectory_dir / "label_drift_pca_3d.gif"
    gif_clean_path = trajectory_dir / "label_drift_pca_3d_clean.gif"
    distance_plot_path = trajectory_dir / "label_drift_distance.png"
    overlay_path = trajectory_dir / "label_overlay_cebra_trajectory.npy"
    centroid_path = trajectory_dir / "label_centroid_cebra_trajectory.npy"
    projected_steps_dir = trajectory_dir / "projected_steps"
    sample_ids_path = trajectory_dir / "sample_display_ids.npy"
    assert manifest_path.exists()
    assert pca_model_path.exists()
    assert metrics_path.exists()
    assert pca_plot_path.exists()
    assert pca_plot_clean_path.exists()
    assert gif_path.exists()
    assert gif_clean_path.exists()
    assert distance_plot_path.exists()
    assert overlay_path.exists()
    assert centroid_path.exists()
    assert projected_steps_dir.exists()
    assert sample_ids_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["step"] for item in manifest["checkpoints"]] == [0, 1, 2, 3, 4]
    assert manifest["centroid_scope"] == "train"
    assert manifest["pca_basis"]["fit_scope"] == "final_train"
    assert manifest["render_dims"] == 3
    assert manifest["include_sample_cloud"] is True
    assert manifest["export_clean_variant"] is True
    assert manifest["num_display_samples"] == 3

    overlay_trajectory = np.load(overlay_path)
    centroid_trajectory = np.load(centroid_path)
    assert overlay_trajectory.shape == (5, 2, 3)
    assert centroid_trajectory.shape == (5, 2, 3)

    pca_model = np.load(pca_model_path)
    assert pca_model["fit_scope"][0] == "final_train"
    assert int(pca_model["render_dims"][0]) == 3
    assert pca_model["components"].shape == (3, 3)

    metrics_rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8")))
    assert len(metrics_rows) == 10
    assert {
        "step",
        "label_name",
        "delta_cebra_l2",
        "delta_cebra_cosine",
        "delta_pca_l2",
        "overlay_pca_dim_3",
        "centroid_pca_dim_3",
    }.issubset(metrics_rows[0].keys())

    train_ids_path = sorted(model_store.glob("**/train_ids.npy"))
    assert train_ids_path
    train_ids = np.asarray(np.load(train_ids_path[0], allow_pickle=True), dtype=str)
    assert not any(item.startswith("label::") for item in train_ids.tolist())

    sample_ids = np.asarray(np.load(sample_ids_path, allow_pickle=True), dtype=str)
    assert sample_ids.shape == (3,)
    assert set(sample_ids.tolist()).issubset(set(train_ids.tolist()))

    projected_step_paths = sorted(projected_steps_dir.glob("step_*.npz"))
    assert [path.stem for path in projected_step_paths] == [
        "step_000000",
        "step_000001",
        "step_000002",
        "step_000003",
        "step_000004",
    ]
    with np.load(projected_step_paths[0], allow_pickle=True) as payload:
        assert payload["sample_pca"].shape == (3, 3)
        assert payload["centroid_pca"].shape == (2, 3)
        assert payload["label_pca"].shape == (2, 3)
        assert payload["sample_pca_3d"].shape == (3, 3)
        assert payload["centroid_pca_3d"].shape == (2, 3)
        assert payload["label_pca_3d"].shape == (2, 3)
        assert payload["sample_ids"].shape == (3,)

    pca_plot_path.unlink()
    pca_plot_clean_path.unlink()
    gif_path.unlink()
    gif_clean_path.unlink()
    distance_plot_path.unlink()

    _run(
        [
            sys.executable,
            "visualize_trajectory.py",
            *common_overrides,
            f"hydra.run.dir={run_dir / 'trajectory_viz'}",
        ],
        env=env,
    )

    assert pca_plot_path.exists()
    assert pca_plot_clean_path.exists()
    assert gif_path.exists()
    assert gif_clean_path.exists()
    assert distance_plot_path.exists()
