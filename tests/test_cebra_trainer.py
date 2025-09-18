import numpy as np
from pathlib import Path
import sys
import torch
import pytest
from cebra.distributions.discrete import DiscreteUniform

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cebra_trainer import train_cebra, normalize_model_architecture, transform_cebra
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


def make_config(batch_size: int, criterion: str = "infonce", conditional: str = "discrete") -> AppConfig:
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
            conditional=conditional,
            criterion=criterion,
            params={"batch_size": batch_size},
        ),
        evaluation=EvaluationConfig(test_size=0.2, random_state=0, knn_neighbors=1),
        wandb=WandBConfig(project="", run_name="", entity=None),
        consistency_check=ConsistencyCheckConfig(),
        hpt=HyperParamTuningConfig(),
        ddp=DDPConfig(world_size=1, rank=0, local_rank=0),
        reproducibility=ReproducibilityConfig(seed=0),
        device="cpu",
    )
    return cfg


def test_normalize_model_architecture_valid(monkeypatch):
    import cebra

    monkeypatch.setattr(cebra.models, "get_options", lambda: ["offset0-model"])
    assert normalize_model_architecture("Offset0-Model") == "offset0-model"


def test_normalize_model_architecture_invalid(monkeypatch):
    import cebra

    monkeypatch.setattr(cebra.models, "get_options", lambda: ["offset0-model"])
    with pytest.raises(ValueError):
        normalize_model_architecture("unknown-model")


def test_transform_cebra_tuple_output():
    class TupleModel(torch.nn.Module):
        def forward(self, x):
            return x + 1, x - 1

    model = TupleModel()
    X = np.random.rand(2, 3).astype(np.float32)
    embeddings = transform_cebra(model, X, "cpu")
    assert isinstance(embeddings, np.ndarray)
    np.testing.assert_allclose(embeddings, X + 1)


def test_train_one_step_no_type_error():
    cfg = make_config(batch_size=8)
    X = np.random.rand(8, 5).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    try:
        train_cebra(X, y, cfg, Path("."))
    except TypeError as exc:  # pragma: no cover - ensure failure message
        assert False, f"TypeError raised: {exc}"


def test_train_infomse_loss():
    cfg = make_config(batch_size=4, criterion="infomse")
    X = np.random.rand(4, 5).astype(np.float32)
    y = np.array([0, 0, 1, 1])
    train_cebra(X, y, cfg, Path("."))


def test_invalid_criterion():
    cfg = make_config(batch_size=4, criterion="not_real")
    X = np.random.rand(4, 5).astype(np.float32)
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        train_cebra(X, y, cfg, Path("."))


def test_distribution_sampling_respects_labels():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    dist = DiscreteUniform(labels, device="cpu")
    batch_size = 4
    anchor_idx = dist.sample_prior(batch_size)
    pos_idx = dist.sample_conditional(labels[anchor_idx])
    same_mask = pos_idx == anchor_idx
    while torch.any(same_mask):
        resample = dist.sample_conditional(labels[anchor_idx[same_mask]])
        pos_idx[same_mask] = resample
        same_mask = pos_idx == anchor_idx

    neg_idx = dist.sample_prior(batch_size)
    neg_mask = labels[neg_idx] != labels[anchor_idx]
    while not torch.all(neg_mask):
        resample = dist.sample_prior((~neg_mask).sum())
        neg_idx[~neg_mask] = resample
        neg_mask = labels[neg_idx] != labels[anchor_idx]

    assert torch.all(labels[anchor_idx] == labels[pos_idx])
    assert torch.all(labels[anchor_idx] != labels[neg_idx])
