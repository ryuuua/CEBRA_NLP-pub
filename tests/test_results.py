import numpy as np
import wandb
import os
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.results import (
    run_consistency_check,
    run_knn_classification,
    save_static_2d_plots,
)
from src.config_schema import (
    AppConfig,
    PathsConfig,
    DatasetConfig,
    EmbeddingConfig,
    CEBRAConfig,
    EvaluationConfig,
    WandBConfig,
    ConsistencyCheckConfig,
    HyperParamTuningConfig,
    VisualizationConfig,
    DDPConfig,
    ReproducibilityConfig,
)

def make_config() -> AppConfig:
    cfg = AppConfig(
        paths=PathsConfig(embedding_cache_dir=""),
        dataset=DatasetConfig(
            name="dummy",
            hf_path="",
            text_column="text",
            label_column="label",
            label_map={0: "a", 1: "b"},
            visualization=VisualizationConfig(emotion_colors={}, emotion_order=[]),
        ),
        embedding=EmbeddingConfig(name="dummy", type="dummy", model_name="dummy"),
        cebra=CEBRAConfig(
            output_dim=2,
            max_iterations=1,
            conditional="discrete",
            params={"batch_size": 4},
        ),
        evaluation=EvaluationConfig(test_size=0.2, random_state=0, knn_neighbors=1),
        wandb=WandBConfig(project="", run_name="", entity=None),
        consistency_check=ConsistencyCheckConfig(enabled=True, num_runs=2),
        hpt=HyperParamTuningConfig(),
        ddp=DDPConfig(world_size=1, rank=0, local_rank=0),

        reproducibility=ReproducibilityConfig(seed=0, deterministic=False),

        device="cpu",
    )
    return cfg


def test_consistency_check_runs_without_value_error(tmp_path):
    cfg = make_config()
    X_train = np.random.rand(8, 5).astype(np.float32)
    X_valid = np.random.rand(4, 5).astype(np.float32)
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    os.environ["WANDB_MODE"] = "offline"
    with wandb.init(project="test-exp", dir=str(tmp_path)):
        try:
            run_consistency_check(X_train, y_train, X_valid, cfg, tmp_path)
        except ValueError as exc:
            pytest.fail(f"Consistency check raised ValueError: {exc}")


def test_consistency_check_across_datasets(tmp_path):
    cfg = make_config()
    X_train = np.random.rand(8, 5).astype(np.float32)
    X_valid = np.random.rand(4, 5).astype(np.float32)
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    embeddings_list = [np.random.rand(120, 3), np.random.rand(120, 3)]
    labels_list = [np.linspace(0, 1, 120), np.linspace(0, 1, 120)]
    dataset_ids = ["d1", "d2"]

    os.environ["WANDB_MODE"] = "offline"
    with wandb.init(project="test-exp", dir=str(tmp_path)):
        try:
            run_consistency_check(
                X_train,
                y_train,
                X_valid,
                cfg,
                tmp_path,
                embeddings_list=embeddings_list,
                labels_list=labels_list,
                dataset_ids=dataset_ids,
                enable_plots=False,
            )
        except ValueError as exc:
            pytest.fail(
                f"Consistency check across datasets raised ValueError: {exc}"
            )


def test_consistency_check_datasets_mode_uses_labels_argument(tmp_path, monkeypatch):
    cfg = make_config()
    cfg.consistency_check.mode = "datasets"

    X_train = np.random.rand(4, 3).astype(np.float32)
    X_valid = np.random.rand(2, 3).astype(np.float32)
    y_train = np.array([0, 1, 0, 1])

    dataset_embeddings = [np.random.rand(5, 3), np.random.rand(5, 3)]
    labels_list = [np.arange(5), np.arange(5)]
    dataset_ids_list = ["d1", "d2"]
    expected_scores = np.array([0.3, 0.7])

    def fake_consistency_score(*, embeddings, labels, dataset_ids, between):
        assert embeddings is dataset_embeddings
        assert labels is labels_list
        assert dataset_ids is dataset_ids_list
        assert between == "datasets"
        return expected_scores, [("d1", "d2")], ["run0"]

    monkeypatch.setattr("src.results.consistency_score", fake_consistency_score)

    mean_score, _ = run_consistency_check(
        X_train,
        y_train,
        X_valid,
        cfg,
        tmp_path,
        dataset_embeddings=dataset_embeddings,
        labels_list=labels_list,
        dataset_ids=dataset_ids_list,
        enable_plots=False,
    )

    assert mean_score == pytest.approx(expected_scores.mean())


def test_knn_classification_skips_plots(tmp_path):
    train_embeddings = np.random.rand(8, 2)
    valid_embeddings = np.random.rand(4, 2)
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_valid = np.array([0, 1, 0, 1])
    label_map = {0: "a", 1: "b"}

    run_knn_classification(
        train_embeddings,
        valid_embeddings,
        y_train,
        y_valid,
        label_map,
        tmp_path,
        knn_neighbors=1,
        enable_plots=False,
    )

    assert not (tmp_path / "confusion_matrix.png").exists()


def test_save_static_2d_plots_logs_variance_ratios(tmp_path):
    os.environ["WANDB_MODE"] = "offline"

    embeddings = np.random.rand(20, 5)
    text_labels = ["label"] * embeddings.shape[0]
    palette = {"label": "#000000"}

    with wandb.init(project="test-exp", dir=str(tmp_path)) as run:
        save_static_2d_plots(
            embeddings=embeddings,
            text_labels=text_labels,
            palette=palette,
            title_prefix="Test",
            output_dir=tmp_path,
            hue_order=["label"],
        )
        summary = dict(run.summary)

    assert "pca_variance_ratio_dim1" in summary
    assert "pca_variance_ratio_dim2" in summary

