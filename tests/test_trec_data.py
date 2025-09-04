
import sys
from pathlib import Path
import pandas as pd

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
import sys

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
)



def make_trec_config() -> AppConfig:
    cfg = AppConfig(
        paths=PathsConfig(embedding_cache_dir=""),
        dataset=DatasetConfig(
            name="trec",
            hf_path="trec",
            text_column="text",
            label_column="label",
            label_map={0: "desc", 1: "ent"},
            visualization=VisualizationConfig(emotion_colors={}, emotion_order=[]),
            source="hf",
            splits=["train"],
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
    )
    cfg.device = "cpu"
    return cfg


def test_trec_label_alignment(monkeypatch):
    fake_dataset = {"text": ["q1", "q2"], "label": [0, 1]}

    def fake_load_dataset(path, split=None):
        assert path == "trec"
        assert split == "train"
        return fake_dataset

    monkeypatch.setattr("src.data.load_dataset", fake_load_dataset)

    cfg = make_trec_config()
    texts, labels, time_indices, _ = load_and_prepare_dataset(cfg)

    assert texts == ["q1", "q2"]
    assert labels.tolist() == [0, 1]
    assert len(texts) == len(labels) == len(time_indices)
    mapped = [cfg.dataset.label_map[int(l)] for l in labels]
    assert mapped == ["desc", "ent"]

def _make_app_cfg() -> AppConfig:
    with initialize(version_base="1.2", config_path="../conf"):
        cfg = compose(config_name="config", overrides=["dataset=trec"])
    dataset_cfg = OmegaConf.merge(OmegaConf.structured(DatasetConfig), cfg.dataset)
    return AppConfig(
        paths=PathsConfig(embedding_cache_dir=""),
        dataset=dataset_cfg,
        embedding=EmbeddingConfig(name="dummy", type="dummy", model_name="dummy"),
        cebra=CEBRAConfig(output_dim=2, max_iterations=1, conditional="discrete", params={}),
        evaluation=EvaluationConfig(test_size=0.2, random_state=0, knn_neighbors=1),
        wandb=WandBConfig(project="", run_name="", entity=None),
        consistency_check=ConsistencyCheckConfig(),
        hpt=HyperParamTuningConfig(),
        ddp=DDPConfig(world_size=1, rank=0, local_rank=0),
        device="cpu",
    )


def test_trec_loading(monkeypatch):
    cfg = _make_app_cfg()

    def fake_load_dataset(path, split=None):
        assert path == "trec"
        data = {
            "train": [
                {"text": "What is AI?", "label-coarse": 0},
                {"text": "Who invented AI?", "label-coarse": 3},
            ],
            "test": [
                {"text": "When was AI invented?", "label-coarse": 5},
            ],
        }
        if split is None:
            return data
        return data[split]

    monkeypatch.setattr("src.data.load_dataset", fake_load_dataset)

    texts, labels, time_indices, ids = load_and_prepare_dataset(cfg)

    assert texts == ["What is AI?", "Who invented AI?", "When was AI invented?"]
    assert labels.tolist() == [0, 3, 5]
    assert np.array_equal(time_indices, np.arange(3))
    assert np.array_equal(ids, np.arange(3))
