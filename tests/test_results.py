import numpy as np
import mlflow
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.results import run_consistency_check
from src.config_schema import (
    AppConfig,
    PathsConfig,
    DatasetConfig,
    EmbeddingConfig,
    CEBRAConfig,
    EvaluationConfig,
    MLflowConfig,
    ConsistencyCheckConfig,
    VisualizationConfig,
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
        mlflow=MLflowConfig(experiment_name="", run_name=""),
        consistency_check=ConsistencyCheckConfig(enabled=True, num_runs=2),
    )
    cfg.device = "cpu"
    return cfg


def test_consistency_check_runs_without_value_error(tmp_path):
    cfg = make_config()
    X_train = np.random.rand(8, 5).astype(np.float32)
    X_valid = np.random.rand(4, 5).astype(np.float32)
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    mlflow.set_tracking_uri(tmp_path.as_posix())
    mlflow.set_experiment("test-exp")
    with mlflow.start_run():
        try:
            run_consistency_check(X_train, y_train, X_valid, cfg, tmp_path)
        except ValueError as exc:
            pytest.fail(f"Consistency check raised ValueError: {exc}")

