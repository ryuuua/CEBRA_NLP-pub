import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import load_and_prepare_dataset
import src.embeddings as embeddings
from src.utils import get_embedding_cache_path, load_text_embedding, save_text_embedding
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
from omegaconf import OmegaConf


def make_cfg(tmp_path, seed):
    cfg = AppConfig(
        paths=PathsConfig(embedding_cache_dir=str(tmp_path)),
        dataset=DatasetConfig(
            name="dummy",
            text_column="text",
            label_column="label",
            label_map={0: "a", 1: "b"},
            visualization=VisualizationConfig(emotion_colors={}, emotion_order=[]),
            hf_path="dummy",
            source="hf",
            splits=["train", "test"],
            shuffle=True,
            shuffle_seed=seed,
        ),
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
    return OmegaConf.structured(cfg)


def fake_load_dataset(path, split=None):
    data = {
        "train": [
            {"id": 0, "text": "a", "label": 0},
            {"id": 1, "text": "b", "label": 1},
        ],
        "test": [
            {"id": 2, "text": "c", "label": 0},
            {"id": 3, "text": "d", "label": 1},
        ],
    }
    return data[split] if split is not None else data


def fake_get_embeddings(texts, cfg):
    fake_get_embeddings.calls += 1
    return np.arange(len(texts)).reshape(len(texts), 1) + fake_get_embeddings.calls


def run_once(cfg):
    texts, _, _, ids = load_and_prepare_dataset(cfg)
    cache_path = get_embedding_cache_path(cfg)
    cache = load_text_embedding(cache_path)
    seed = cfg.dataset.shuffle_seed
    if cache is not None and cache[2] == seed:
        cached_ids, cached_emb, _ = cache
        id_to_idx = {str(i): idx for idx, i in enumerate(cached_ids)}
        embeddings_arr = np.stack([cached_emb[id_to_idx[str(i)]] for i in ids])
    else:
        embeddings_arr = embeddings.get_embeddings(texts, cfg)
        save_text_embedding(ids, embeddings_arr, seed, cache_path)
    return embeddings_arr


def test_cache_reused(monkeypatch, tmp_path):
    monkeypatch.setattr("src.data.load_dataset", fake_load_dataset)
    fake_get_embeddings.calls = 0
    monkeypatch.setattr("src.embeddings.get_embeddings", fake_get_embeddings)

    cfg = make_cfg(tmp_path, seed=1)
    emb1 = run_once(cfg)
    assert fake_get_embeddings.calls == 1

    cfg2 = make_cfg(tmp_path, seed=1)
    emb2 = run_once(cfg2)
    assert fake_get_embeddings.calls == 1
    assert np.array_equal(emb1, emb2)


def test_cache_miss_on_seed_change(monkeypatch, tmp_path):
    monkeypatch.setattr("src.data.load_dataset", fake_load_dataset)
    fake_get_embeddings.calls = 0
    monkeypatch.setattr("src.embeddings.get_embeddings", fake_get_embeddings)

    cfg = make_cfg(tmp_path, seed=1)
    run_once(cfg)
    assert fake_get_embeddings.calls == 1

    cfg2 = make_cfg(tmp_path, seed=2)
    run_once(cfg2)
    assert fake_get_embeddings.calls == 2
