import numpy as np
import wandb
import os
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.results import run_consistency_check, run_knn_classification
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

