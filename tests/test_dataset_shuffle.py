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
    ReproducibilityConfig,
)


def _make_app_cfg(shuffle: bool) -> AppConfig:
    overrides = [
        "dataset=imdb",
        f"dataset.shuffle={'true' if shuffle else 'false'}",
        "evaluation.random_state=0",
    ]
    with initialize(version_base="1.2", config_path="../conf"):
        cfg = compose(config_name="config", overrides=overrides)
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
        reproducibility=ReproducibilityConfig(seed=0),
        device="cpu",
    )


def fake_load_dataset(path, split=None):
    data = {
        "train": [{"text": "a", "label": 0}, {"text": "b", "label": 1}],
        "test": [{"text": "c", "label": 0}, {"text": "d", "label": 1}],
    }
    return data[split]


def test_dataset_shuffle(monkeypatch):
    monkeypatch.setattr("src.data.load_dataset", fake_load_dataset)
    cfg_unshuffled = _make_app_cfg(False)
    texts_unshuffled, labels_unshuffled, _, ids_unshuffled = load_and_prepare_dataset(cfg_unshuffled)

    cfg_shuffled = _make_app_cfg(True)
    texts_shuffled, labels_shuffled, time_indices, ids_shuffled = load_and_prepare_dataset(cfg_shuffled)
    assert texts_shuffled == ["c", "d", "b", "a"]

    assert texts_unshuffled == ["a", "b", "c", "d"]
    assert texts_shuffled != texts_unshuffled

    # time indices should always be sequential after shuffling
    assert np.array_equal(time_indices, np.arange(len(time_indices)))

    # ids should retain their original labels after shuffling
    original_mapping = dict(zip(ids_unshuffled, labels_unshuffled))
    shuffled_mapping = dict(zip(ids_shuffled, labels_shuffled))
    assert original_mapping == shuffled_mapping
