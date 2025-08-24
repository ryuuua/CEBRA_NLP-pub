import numpy as np
from pathlib import Path
import sys
import torch
import pytest

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
        mlflow=MLflowConfig(experiment_name="", run_name=""),
        consistency_check=ConsistencyCheckConfig(),
        hpt=HyperParamTuningConfig(),
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


def test_train_mse_loss():
    cfg = make_config(batch_size=4, loss="mse")
    X = np.random.rand(4, 5).astype(np.float32)
    y = np.random.rand(4, cfg.cebra.output_dim).astype(np.float32)
    train_cebra(X, y, cfg, Path("."))


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


def _vectorized_sample(labels, rand_pos, rand_neg):
    same_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_mask.fill_diagonal_(False)
    diff_mask = ~same_mask
    diff_mask.fill_diagonal_(False)
    same_counts = same_mask.sum(dim=1)
    diff_counts = diff_mask.sum(dim=1)
    pos_choice = (rand_pos * same_counts).floor().long()
    neg_choice = (rand_neg * diff_counts).floor().long()
    same_cumsum = same_mask.cumsum(dim=1) - 1
    diff_cumsum = diff_mask.cumsum(dim=1) - 1
    same_cumsum[~same_mask] = -1
    diff_cumsum[~diff_mask] = -1
    pos_idx = (same_cumsum == pos_choice.unsqueeze(1)).float().argmax(dim=1)
    neg_idx = (diff_cumsum == neg_choice.unsqueeze(1)).float().argmax(dim=1)
    return pos_idx.long(), neg_idx.long()


def _loop_sample(labels, rand_pos, rand_neg):
    pos_idx = torch.empty(labels.size(0), dtype=torch.long)
    neg_idx = torch.empty_like(pos_idx)
    for i in range(labels.size(0)):
        same = (labels == labels[i]).nonzero().view(-1)
        same = same[same != i]
        diff = (labels != labels[i]).nonzero().view(-1)
        pos_choice = int((rand_pos[i] * same.numel()).floor())
        neg_choice = int((rand_neg[i] * diff.numel()).floor())
        pos_idx[i] = same[pos_choice]
        neg_idx[i] = diff[neg_choice]
    return pos_idx, neg_idx


def test_vectorized_sampling_matches_loop():
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    rand_pos = torch.rand(labels.size(0))
    rand_neg = torch.rand(labels.size(0))
    pos_v, neg_v = _vectorized_sample(labels, rand_pos, rand_neg)
    pos_l, neg_l = _loop_sample(labels, rand_pos, rand_neg)
    assert torch.equal(pos_v, pos_l)
    assert torch.equal(neg_v, neg_l)
