from hydra import compose, initialize
from omegaconf import OmegaConf
from pathlib import Path
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


def _make_app_cfg() -> AppConfig:
    with initialize(version_base="1.2", config_path="../conf"):
        cfg = compose(config_name="config", overrides=["dataset=imdb"])
    dataset_cfg = OmegaConf.merge(OmegaConf.structured(DatasetConfig), cfg.dataset)
    app_cfg = AppConfig(
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
    return app_cfg


def test_imdb_unlabeled_filter(monkeypatch):
    cfg = _make_app_cfg()

    def fake_load_dataset(path, split=None):
        data = {
            "train": [{"text": "a", "label": 0}, {"text": "b", "label": 1}],
            "test": [{"text": "c", "label": -1}, {"text": "d", "label": 1}],
        }
        if split is None:
            return data
        return data[split]

    monkeypatch.setattr("src.data.load_dataset", fake_load_dataset)

    texts, labels, _, _ = load_and_prepare_dataset(cfg)

    valid_labels = set(cfg.dataset.label_map.keys())
    assert set(labels.tolist()).issubset(valid_labels)
    assert len(labels) == 3
