from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cebra_nlp_public.config_runtime import to_typed_app_config
from cebra_nlp_public.visualization.trajectory.epoch import (
    load_epoch_trajectory_manifest,
    render_saved_epoch_trajectory,
    resolve_epoch_trajectory_render_options,
    select_trajectory_indices,
)


def _compose_train(overrides: list[str] | None = None):
    conf_dir = REPO_ROOT / "conf"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.2"):
        return compose(config_name="train", overrides=overrides or [])


def test_cebra_epoch_trajectory_defaults_match_main_contract() -> None:
    cfg = to_typed_app_config(_compose_train(["hpt=default"]))

    assert cfg.cebra.save_epoch_trajectory is False
    assert cfg.cebra.trajectory_every_n_epochs == 1
    assert cfg.cebra.trajectory_sample_size == 1000
    assert cfg.cebra.trajectory_fps == 8
    assert cfg.cebra.trajectory_max_frames == 180
    assert cfg.cebra.trajectory_one_epoch_one_frame is False
    assert cfg.cebra.trajectory_connect_segments is True
    assert cfg.cebra.trajectory_rotate_camera is False
    assert cfg.cebra.trajectory_camera_elev == 18.0
    assert cfg.cebra.trajectory_camera_azim == 42.0
    assert cfg.cebra.trajectory_trail_length == 10
    assert cfg.cebra.trajectory_axis_padding == 0.08
    assert cfg.cebra.trajectory_frame_width == 1920
    assert cfg.cebra.trajectory_frame_height == 1080
    assert cfg.cebra.trajectory_dpi == 180
    assert cfg.cebra.trajectory_mp4_crf == 18
    assert cfg.cebra.trajectory_mp4_preset == "slow"


def test_select_trajectory_indices_is_deterministic_sorted_and_bounded() -> None:
    first = select_trajectory_indices(10, 4, seed=7)
    second = select_trajectory_indices(10, 4, seed=7)

    assert first.tolist() == second.tolist()
    assert first.tolist() == sorted(first.tolist())
    assert first.shape == (4,)
    assert np.all(first >= 0)
    assert np.all(first < 10)
    assert select_trajectory_indices(3, 10, seed=7).tolist() == [0, 1, 2]
    assert select_trajectory_indices(3, None, seed=7).tolist() == [0, 1, 2]


def test_resolve_epoch_trajectory_render_options_reads_legacy_cebra_fields() -> None:
    cfg = SimpleNamespace(
        cebra=SimpleNamespace(
            trajectory_fps=0,
            trajectory_max_frames=-5,
            trajectory_one_epoch_one_frame=True,
            trajectory_connect_segments=False,
            trajectory_rotate_camera=True,
            trajectory_camera_elev=12.5,
            trajectory_camera_azim=24.0,
            trajectory_trail_length=3,
            trajectory_axis_padding=0.2,
            trajectory_frame_width=0,
            trajectory_frame_height=-1,
            trajectory_dpi=0,
            trajectory_mp4_crf=21,
            trajectory_mp4_preset="medium",
        )
    )

    options = resolve_epoch_trajectory_render_options(
        cfg,
        overrides={"fps": 6, "frame_width": 320},
    )

    assert options["fps"] == 6
    assert options["max_frames"] == 1
    assert options["one_epoch_one_frame"] is True
    assert options["connect_segments"] is False
    assert options["rotate_camera"] is True
    assert options["camera_elev"] == 12.5
    assert options["camera_azim"] == 24.0
    assert options["trail_length"] == 3
    assert options["axis_padding"] == 0.2
    assert options["frame_width"] == 320
    assert options["frame_height"] == 1
    assert options["dpi"] == 1
    assert options["mp4_crf"] == 21
    assert options["mp4_preset"] == "medium"


def test_load_epoch_trajectory_manifest_resolves_snapshot_paths(tmp_path: Path) -> None:
    trajectory_dir = tmp_path / "epoch_trajectory"
    snapshot_dir = trajectory_dir / "snapshots"
    snapshot_dir.mkdir(parents=True)
    np.save(snapshot_dir / "epoch_000000.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(snapshot_dir / "epoch_000001.npy", np.ones((2, 3), dtype=np.float32))
    np.save(trajectory_dir / "sample_indices.npy", np.asarray([0, 2], dtype=np.int64))
    np.save(trajectory_dir / "sample_labels.npy", np.asarray(["a", "b"], dtype=str))
    (trajectory_dir / "manifest.json").write_text(
        json.dumps(
            {
                "snapshots": [
                    {
                        "epoch": 0,
                        "step": 0,
                        "relative_path": "snapshots/epoch_000000.npy",
                    },
                    {
                        "epoch": 1,
                        "step": 4,
                        "relative_path": "snapshots/epoch_000001.npy",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_epoch_trajectory_manifest(trajectory_dir)

    assert [record["epoch"] for record in manifest["snapshots"]] == [0, 1]
    assert [record["step"] for record in manifest["snapshots"]] == [0, 4]
    assert all(Path(record["path"]).exists() for record in manifest["snapshots"])
    assert manifest["sample_indices"].tolist() == [0, 2]
    assert manifest["sample_labels"].tolist() == ["a", "b"]


def test_render_saved_epoch_trajectory_writes_pca_gif_and_report(tmp_path: Path) -> None:
    trajectory_dir = tmp_path / "epoch_trajectory"
    snapshot_dir = trajectory_dir / "snapshots"
    snapshot_dir.mkdir(parents=True)
    snapshots = [
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [0.2, 0.0, 0.0],
                [1.2, 0.1, 0.0],
                [0.0, 1.1, 0.2],
                [0.1, 0.0, 1.2],
            ],
            dtype=np.float32,
        ),
    ]
    manifest_records = []
    for epoch, values in enumerate(snapshots):
        path = snapshot_dir / f"epoch_{epoch:06d}.npy"
        np.save(path, values)
        manifest_records.append(
            {"epoch": epoch, "step": epoch * 4, "relative_path": f"snapshots/{path.name}"}
        )
    np.save(trajectory_dir / "sample_indices.npy", np.arange(4, dtype=np.int64))
    np.save(trajectory_dir / "sample_ids.npy", np.asarray(["s0", "s1", "s2", "s3"], dtype=str))
    np.save(trajectory_dir / "sample_labels.npy", np.asarray(["neg", "pos", "neg", "pos"], dtype=str))
    (trajectory_dir / "manifest.json").write_text(
        json.dumps({"snapshots": manifest_records}),
        encoding="utf-8",
    )

    outputs = render_saved_epoch_trajectory(
        trajectory_dir,
        fps=2,
        max_frames=2,
        frame_width=320,
        frame_height=240,
        dpi=80,
    )

    assert outputs["gif"].endswith("trajectory_pca3d.gif")
    assert (trajectory_dir / outputs["gif"]).exists()
    assert (trajectory_dir / "trajectory_pca3d.npy").exists()
    assert (trajectory_dir / "trajectory_pca_explained_variance.csv").exists()
    report_path = trajectory_dir / "trajectory_render_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["num_snapshots"] == 2
    assert report["selected_epochs"] == [0, 1]
    assert report["artifacts"]["gif"] == "trajectory_pca3d.gif"
    trajectory = np.load(trajectory_dir / "trajectory_pca3d.npy")
    assert trajectory.shape == (2, 4, 3)
