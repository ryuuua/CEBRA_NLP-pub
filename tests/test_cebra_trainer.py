import numpy as np
from pathlib import Path
import sys
import torch
import pytest
from cebra.distributions.discrete import DiscreteUniform

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cebra_trainer import train_cebra, normalize_model_architecture
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


def make_config(batch_size: int, loss: str = "infonce") -> AppConfig:
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
            conditional="none" if loss == "mse" else "discrete",
            params={"batch_size": batch_size, "loss": loss},
        ),
        evaluation=EvaluationConfig(test_size=0.2, random_state=0, knn_neighbors=1),
        wandb=WandBConfig(project="", run_name="", entity=None),
        consistency_check=ConsistencyCheckConfig(),
        hpt=HyperParamTuningConfig(),
        ddp=DDPConfig(world_size=1, rank=0, local_rank=0),
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


def test_train_one_step_no_type_error():
    cfg = make_config(batch_size=8)
    X = np.random.rand(8, 5).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    try:
        train_cebra(X, y, cfg, Path("."))
    except TypeError as exc:  # pragma: no cover - ensure failure message
        assert False, f"TypeError raised: {exc}"


def test_train_mse_loss():
    cfg = make_config(batch_size=4, loss="mse")
    X = np.random.rand(4, 5).astype(np.float32)
    y = np.random.rand(4, cfg.cebra.output_dim).astype(np.float32)
    train_cebra(X, y, cfg, Path("."))


def test_mse_integer_labels_auto_one_hot():
    cfg = make_config(batch_size=4, loss="mse")
    cfg.cebra.output_dim = 3
    X = np.random.rand(4, 5).astype(np.float32)
    y = np.array([0, 1, 2, 1])
    train_cebra(X, y, cfg, Path("."))


def test_mse_integer_labels_output_dim_mismatch():
    cfg = make_config(batch_size=4, loss="mse")
    X = np.random.rand(4, 5).astype(np.float32)
    y = np.array([0, 1, 2, 1])
    train_cebra(X, y, cfg, Path("."))
    assert cfg.cebra.output_dim == 3


def test_classifier_model_tuple_output(monkeypatch):
    cfg = make_config(batch_size=2, loss="mse")
    X = np.random.rand(2, 3).astype(np.float32)
    y = np.random.rand(2, cfg.cebra.output_dim).astype(np.float32)

    class DummyModel(torch.nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.linear = torch.nn.Linear(in_dim, cfg.cebra.output_dim)
            self.classifier = None

        def set_output_num(self, n):
            self.classifier = torch.nn.Linear(cfg.cebra.output_dim, n)

        def forward(self, x):
            emb = self.linear(x)
            pred = self.classifier(emb) if self.classifier is not None else None
            return emb, pred

    def dummy_build_model(cfg_, num_neurons):
        return DummyModel(num_neurons)

    monkeypatch.setattr("src.cebra_trainer._build_model", dummy_build_model)

    model = train_cebra(X, y, cfg, Path("."))
    assert getattr(model, "classifier") is not None

    with pytest.raises(ValueError):
        train_cebra(X, None, cfg, Path("."))


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
