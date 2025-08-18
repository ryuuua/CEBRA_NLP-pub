import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cebra_trainer import train_cebra
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
    DDPConfig,
)


def make_config(batch_size: int) -> AppConfig:
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
            params={"batch_size": batch_size},
        ),
        evaluation=EvaluationConfig(test_size=0.2, random_state=0, knn_neighbors=1),
        mlflow=MLflowConfig(experiment_name="", run_name=""),
        consistency_check=ConsistencyCheckConfig(),
        ddp=DDPConfig(world_size=1, rank=0, local_rank=0),
    )
    cfg.device = "cpu"
    return cfg


def test_train_one_step_no_type_error():
    cfg = make_config(batch_size=8)
    X = np.random.rand(8, 5).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    try:
        train_cebra(X, y, cfg, Path("."))
    except TypeError as exc:  # pragma: no cover - ensure failure message
        assert False, f"TypeError raised: {exc}"
