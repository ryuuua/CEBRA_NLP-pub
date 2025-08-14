import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from src.config_schema import AppConfig

def get_cebra_config_hash(cfg):
    import json, hashlib
    relevant_cfg = {
        'dataset': cfg.dataset.name,
        'embedding': cfg.embedding.name,
        'cebra': dict(cfg.cebra)
    }
    hash_str = json.dumps(relevant_cfg, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:8]

def get_cebra_output_dir(cfg, base="model_outputs"):
    h = get_cebra_config_hash(cfg)
    path = Path(base) / f"{cfg.dataset.name}__{cfg.embedding.name}__{h}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_cebra_model(model, output_dir):
    import torch
    path = output_dir / "cebra_model.pt"
    torch.save(model.state_dict(), path)
    return path

def save_cebra_embeddings(embeddings, output_dir):
    path = output_dir / "cebra_embeddings.npy"
    np.save(path, embeddings)
    return path

def _build_model(cfg: AppConfig, num_neurons: int):
    import cebra

    name = getattr(cfg.cebra, "model_architecture", "offset0-model").lower()
    registry = {
        "offset0-model": cebra.models.Offset0Model,
        "offset5-model": getattr(cebra.models, "Offset5Model", cebra.models.Offset0Model),
        "offset10-model": getattr(cebra.models, "Offset10Model", cebra.models.Offset0Model),
    }
    ModelClass = registry.get(name)
    if ModelClass is None:
        raise ValueError(f"Unsupported model_architecture: {name}")
    return ModelClass(
        num_neurons=num_neurons,
        num_units=cfg.cebra.params.get("num_units", 512),
        num_output=cfg.cebra.output_dim,
    ).to(cfg.device)


def load_cebra_model(model_path, cfg: AppConfig, input_dimension: int):
    import torch

    model = _build_model(cfg, input_dimension)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()
    return model

def transform_cebra(model, X, device):
    import torch
    was_training = model.training
    model.eval()
    with torch.no_grad():
        embeddings = (
            model(torch.as_tensor(X, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
    if was_training:
        model.train()
    return embeddings


def train_cebra(X_vectors, labels, cfg: AppConfig, output_dir):
    """Train CEBRA using its native PyTorch API.

    Parameters
    ----------
    cfg : AppConfig
        Training configuration. ``cfg.cebra.max_iterations`` denotes the
        maximum number of gradient steps (i.e., batches) to execute. The
        loop stops once this limit is reached, matching the scikit-learn
        ``max_iter`` semantics.
    """

    import torch, cebra
    from torch.utils.data import DataLoader, TensorDataset

    try:
        from cebra.criterions import InfoNCE
    except Exception:  # pragma: no cover - fallback for older versions
        from cebra.models import InfoNCE

    tensors = [torch.as_tensor(X_vectors, dtype=torch.float32)]
    if labels is not None:
        dtype = torch.long if cfg.cebra.conditional == "discrete" else torch.float32
        tensors.append(torch.as_tensor(labels, dtype=dtype))
    dataset = TensorDataset(*tensors)
    loader = DataLoader(
        dataset,
        batch_size=cfg.cebra.params.get("batch_size", 512),
        shuffle=True,
    )

    model = _build_model(cfg, X_vectors.shape[1])

    try:
        criterion = InfoNCE(model.get_offset())
    except TypeError:  # pragma: no cover - for fallback implementation
        criterion = InfoNCE()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.cebra.params.get("learning_rate", 1e-3),
    )

    steps = 0
    while steps < cfg.cebra.max_iterations:
        for batch in loader:
            if labels is not None:
                batch_x, batch_y = batch
                try:
                    loss = criterion(model(batch_x.to(cfg.device)), batch_y.to(cfg.device))
                except TypeError:  # pragma: no cover
                    emb = model(batch_x.to(cfg.device))
                    loss = criterion(emb, emb, emb)
            else:
                (batch_x,) = batch
                try:
                    loss = criterion(model(batch_x.to(cfg.device)))
                except TypeError:  # pragma: no cover
                    emb = model(batch_x.to(cfg.device))
                    loss = criterion(emb, emb, emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps += 1
            if steps >= cfg.cebra.max_iterations:
                break

    return model
