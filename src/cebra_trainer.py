import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from src.config_schema import AppConfig
from tqdm.auto import tqdm
from collections import deque
import mlflow

def get_cebra_config_hash(cfg):
    import json, hashlib

    relevant_cfg = {
        "dataset": cfg.dataset.name,
        "embedding": cfg.embedding.name,
        "cebra": dict(cfg.cebra),
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
    import cebra, re

    name = getattr(cfg.cebra, "model_architecture", "offset0-model").lower()

    registry = {
        "offset0-model": cebra.models.Offset0Model,
        "offset5-model": getattr(
            cebra.models, "Offset5Model", cebra.models.Offset0Model
        ),
        "offset10-model": getattr(
            cebra.models, "Offset10Model", cebra.models.Offset0Model
        ),
    }

    ModelClass = registry.get(name)
    if ModelClass is None:
        # Attempt to dynamically resolve the model class from its name
        parts = [p for p in re.split(r"[-_]", name) if p]
        class_name = "".join(part.capitalize() for part in parts)
        ModelClass = getattr(cebra.models, class_name, None)

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
            model(torch.as_tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
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
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    import inspect

    loss_type = cfg.cebra.params.get("loss", "infonce").lower()

    if X_vectors is None:
        raise ValueError("Embeddings `X_vectors` must not be None")
    X_vectors = np.asarray(X_vectors)
    if X_vectors.ndim != 2:
        raise ValueError(
            f"`X_vectors` must be 2D (n_samples, n_features), got shape {X_vectors.shape}"
        )
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] != X_vectors.shape[0]:
            raise ValueError(
                "`labels` must have the same number of samples as `X_vectors`"
            )
    elif loss_type == "mse" or cfg.cebra.conditional != "none":
        raise ValueError("`labels` are required for the selected training configuration")

    from cebra.models.criterions import FixedCosineInfoNCE as InfoNCE

    tensors = [torch.as_tensor(X_vectors, dtype=torch.float32)]
    if labels is not None:
        dtype = torch.long if cfg.cebra.conditional == "discrete" else torch.float32
        tensors.append(torch.as_tensor(labels, dtype=dtype))
    dataset = TensorDataset(*tensors)
    sampler = DistributedSampler(
        dataset, num_replicas=cfg.ddp.world_size, rank=cfg.ddp.rank
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.cebra.params.get("batch_size", 512),
        sampler=sampler,
        num_workers=cfg.cebra.num_workers,
        pin_memory=cfg.cebra.pin_memory,
        persistent_workers=cfg.cebra.persistent_workers if cfg.cebra.num_workers > 0 else False,
        prefetch_factor=cfg.cebra.prefetch_factor if cfg.cebra.num_workers > 0 else None,
    )

    model = _build_model(cfg, X_vectors.shape[1])
    if cfg.ddp.world_size > 1 and torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.ddp.local_rank]
        )

    if loss_type == "mse":
        criterion = torch.nn.MSELoss()
    else:
        params = inspect.signature(InfoNCE).parameters
        if "offset" in params:
            criterion = InfoNCE(model.get_offset())
        else:
            criterion = InfoNCE()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.cebra.params.get("learning_rate", 1e-3),
    )

    steps = 0
    skipped = 0
    ma = deque(maxlen=50)

    with tqdm(total=cfg.cebra.max_iterations, desc="CEBRA Training") as pbar:
        while steps < cfg.cebra.max_iterations:
            for batch in loader:
                if loss_type == "mse":
                    batch_x, batch_y = batch
                    embeddings = model(batch_x.to(cfg.device, non_blocking=True))
                    loss = criterion(
                        embeddings, batch_y.to(cfg.device, non_blocking=True)
                    )
                else:
                    if labels is None:
                        (batch_x,) = batch
                        embeddings = model(batch_x.to(cfg.device, non_blocking=True))
                        if embeddings is None:
                            raise ValueError("Model returned no embeddings")
                        loss = criterion(embeddings)
                    else:
                        batch_x, batch_y = batch
                        embeddings = model(batch_x.to(cfg.device, non_blocking=True))
                        if embeddings is None:
                            raise ValueError("Model returned no embeddings")
                        if batch_y is None:
                            raise ValueError("Labels are missing for supervised training")
                        if embeddings.shape[0] != batch_y.shape[0]:
                            raise ValueError(
                                "Embedding batch size does not match label batch size"
                            )
                        labels_device = batch_y.to(cfg.device, non_blocking=True)
                        unique, counts = torch.unique(labels_device, return_counts=True)
                        if unique.numel() < 2 or torch.any(counts < 2):
                            skipped += 1
                            if mlflow.active_run():
                                mlflow.log_metric("skipped_batches", skipped, step=steps)
                            continue
    
                        batch_size = labels_device.shape[0]
                        # Precompute label-wise masks to avoid per-sample loops
                        same_mask = labels_device.unsqueeze(0) == labels_device.unsqueeze(1)
                        same_mask.fill_diagonal_(False)
                        diff_mask = ~same_mask
                        diff_mask.fill_diagonal_(False)
    
                        # Random choices for positive/negative samples
                        rand_pos = torch.rand(batch_size, device=labels_device.device)
                        rand_neg = torch.rand(batch_size, device=labels_device.device)
    
                        same_counts = same_mask.sum(dim=1)
                        diff_counts = diff_mask.sum(dim=1)
                        if torch.any(same_counts == 0) or torch.any(diff_counts == 0):
                            skipped += 1
                            if mlflow.active_run():
                                mlflow.log_metric("skipped_batches", skipped, step=steps)
                            continue
                        pos_choice = (rand_pos * same_counts).floor().long()
                        neg_choice = (rand_neg * diff_counts).floor().long()
    
                        same_cumsum = same_mask.cumsum(dim=1) - 1
                        diff_cumsum = diff_mask.cumsum(dim=1) - 1
                        same_cumsum[~same_mask] = -1
                        diff_cumsum[~diff_mask] = -1
    
                        pos_indices = (
                            (same_cumsum == pos_choice.unsqueeze(1)).float().argmax(dim=1)
                        )
                        neg_indices = (
                            (diff_cumsum == neg_choice.unsqueeze(1)).float().argmax(dim=1)
                        )
    
                        pos_embeddings = embeddings[pos_indices]
                        neg_embeddings = embeddings[neg_indices]
                        loss_tuple = criterion(embeddings, pos_embeddings, neg_embeddings)
                        loss = (
                            loss_tuple[0]
                            if isinstance(loss_tuple, tuple)
                            else loss_tuple
                        )
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if mlflow.active_run():
                        mlflow.log_metric("loss", loss.item(), step=steps)
    
                steps += 1
                pbar.update(1)
                if steps >= cfg.cebra.max_iterations:
                    break

    if mlflow.active_run():
        mlflow.log_metric("total_skipped", skipped)

    # Explicitly shut down DataLoader workers to avoid process accumulation
    if cfg.cebra.num_workers > 0:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
        del loader
        import gc
        gc.collect()
    return model
