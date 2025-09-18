import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import load_and_prepare_dataset
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


def make_kaggle_config() -> AppConfig:
    cfg = AppConfig(
        paths=PathsConfig(embedding_cache_dir=""),
        dataset=DatasetConfig(
            name="kaggle-dummy",
            hf_path="",
            text_column="text",
            label_column="label",
            label_map={0: "a", 1: "b"},
            visualization=VisualizationConfig(emotion_colors={}, emotion_order=[]),
            source="kaggle",
            kaggle_handle="kashnitsky/hierarchical-text-classification",

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
        consistency_check=ConsistencyCheckConfig(enabled=False, num_runs=1),
        hpt=HyperParamTuningConfig(),
        ddp=DDPConfig(world_size=1, rank=0, local_rank=0),
        reproducibility=ReproducibilityConfig(seed=0),
    )
    cfg.device = "cpu"
    return cfg


def make_kaggle_multilabel_config() -> AppConfig:
    cfg = AppConfig(
        paths=PathsConfig(embedding_cache_dir=""),
        dataset=DatasetConfig(
            name="kaggle-multilabel",
            hf_path="",
            text_column="text",
            label_column="labels",
            label_map={0: "a", 1: "b", 2: "c"},
            visualization=VisualizationConfig(emotion_colors={}, emotion_order=[]),
            source="kaggle",
            kaggle_handle="dummy/multilabel",
            multi_label=True,
            label_delimiter="|",

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
        consistency_check=ConsistencyCheckConfig(enabled=False, num_runs=1),
        hpt=HyperParamTuningConfig(),
        ddp=DDPConfig(world_size=1, rank=0, local_rank=0),
        reproducibility=ReproducibilityConfig(seed=0),
    )
    cfg.device = "cpu"
    return cfg


def test_kaggle_loading(monkeypatch, tmp_path):
    df = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    def fake_download(handle):
        assert handle == "kashnitsky/hierarchical-text-classification"
        return str(tmp_path)

    monkeypatch.setattr("src.data.kagglehub.dataset_download", fake_download)

    cfg = make_kaggle_config()
    texts, labels, time_indices, _ = load_and_prepare_dataset(cfg)

    assert texts == ["hello", "world"]
    assert labels.tolist() == [0, 1]
    assert np.array_equal(time_indices, np.arange(2))


def test_kaggle_multilabel(monkeypatch, tmp_path):
    df = pd.DataFrame({"text": ["hello", "world"], "labels": ["a|b", "b"]})
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    def fake_download(handle):
        assert handle == "dummy/multilabel"
        return str(tmp_path)

    monkeypatch.setattr("src.data.kagglehub.dataset_download", fake_download)

    cfg = make_kaggle_multilabel_config()
    texts, labels, _, _ = load_and_prepare_dataset(cfg)

    assert texts == ["hello", "world"]
    assert labels.shape == (2, 3)
    assert np.array_equal(labels, np.array([[1, 1, 0], [0, 1, 0]]))

def test_kaggle_multilabel_encoding(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "text": ["t1", "t2"],
            "label": [np.array([1, 0, 1]), np.array([0, 1, 0])],
        }
    )
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("dummy")

    def fake_download(handle):
        assert handle == "kashnitsky/hierarchical-text-classification"
        return str(tmp_path)

    def fake_read_csv(path):
        assert path == str(csv_path)
        return df

    monkeypatch.setattr("src.data.kagglehub.dataset_download", fake_download)
    monkeypatch.setattr("src.data.pd.read_csv", fake_read_csv)
    monkeypatch.setattr(pd.Series, "isin", lambda self, values: pd.Series([True] * len(self)))

    cfg = make_kaggle_config()
    cfg.dataset.label_map = {0: "a", 1: "b", 2: "c"}

    texts, labels, _, _ = load_and_prepare_dataset(cfg)

    assert texts == ["t1", "t2"]
    labels_matrix = np.stack(labels)
    assert labels_matrix.shape == (2, 3)
    assert np.array_equal(labels_matrix, np.array([[1, 0, 1], [0, 1, 0]]))
    assert set(np.unique(labels_matrix)).issubset({0, 1})


