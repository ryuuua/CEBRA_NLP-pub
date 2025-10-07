import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from src.config_schema import AppConfig
from tqdm.auto import tqdm
from collections import deque
import wandb
from cebra.distributions.discrete import DiscreteUniform, DiscreteEmpirical

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


def normalize_model_architecture(name: str) -> str:
    """Normalize and validate a model architecture name.

    Parameters
    ----------
    name: str
        Requested model architecture name.

    Returns
    -------
    str
        Normalized architecture name that is available in the cebra registry.
    """

    import cebra

    normalized = name.lower()
    if normalized not in cebra.models.get_options():
        raise ValueError(f"Unsupported model_architecture: {name}")
    return normalized


def _build_model(cfg: AppConfig, num_neurons: int):
    import cebra

    normalized = normalize_model_architecture(
        getattr(cfg.cebra, "model_architecture", "offset0-model")
    )

    return cebra.models.init(
        normalized,
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
        output = model(torch.as_tensor(X, dtype=torch.float32).to(device))
        if isinstance(output, tuple):
            output = output[0]
        embeddings = output.cpu().numpy()
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

    cfg.cebra.conditional = cfg.cebra.conditional.lower()
    loss_type = cfg.cebra.criterion.lower()

    reproducibility = getattr(cfg, "reproducibility", None)
    deterministic = bool(getattr(reproducibility, "deterministic", False))
    seed_value = getattr(reproducibility, "seed", None)
    if seed_value is None and getattr(cfg, "evaluation", None) is not None:
        seed_value = getattr(cfg.evaluation, "random_state", None)

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
    elif cfg.cebra.conditional != "none":
        raise ValueError("`labels` are required for the selected training configuration")
    from cebra.models import criterions as cebra_criterions

    X_tensor = torch.as_tensor(X_vectors, dtype=torch.float32)
    label_tensor = None
    dist = None
    if labels is not None:
        dtype = torch.long if cfg.cebra.conditional == "discrete" else torch.float32
        label_tensor = torch.as_tensor(labels, dtype=dtype)
        if cfg.cebra.conditional == "discrete":
            X_tensor = X_tensor.to(cfg.device)
            label_tensor = label_tensor.to(cfg.device)
            if cfg.cebra.params.get("prior", "uniform") == "uniform":
                dist = DiscreteUniform(label_tensor, device=cfg.device)
            else:
                dist = DiscreteEmpirical(label_tensor, device=cfg.device)

    loader = None
    sampler = None
    data_generator = None
    if dist is None:
        dataset = TensorDataset(X_tensor)
        sampler_kwargs = dict(
            dataset=dataset,
            num_replicas=cfg.ddp.world_size,
            rank=cfg.ddp.rank,
        )
        if deterministic and seed_value is not None:
            sampler_kwargs["seed"] = seed_value
        sampler = DistributedSampler(**sampler_kwargs)

        loader_kwargs = dict(
            dataset=dataset,
            batch_size=cfg.cebra.params.get("batch_size", 512),
            sampler=sampler,
            num_workers=cfg.cebra.num_workers,
            # Use pinned memory only when running on CUDA to avoid warnings on MPS
            pin_memory=cfg.cebra.pin_memory and cfg.device.startswith("cuda"),
            persistent_workers=cfg.cebra.persistent_workers if cfg.cebra.num_workers > 0 else False,
            prefetch_factor=cfg.cebra.prefetch_factor if cfg.cebra.num_workers > 0 else None,
        )

        if deterministic:
            generator_device = "cuda" if str(cfg.device).startswith("cuda") else "cpu"
            data_generator = torch.Generator(device=generator_device)
            data_generator.manual_seed(int(seed_value) if seed_value is not None else 0)
            loader_kwargs["generator"] = data_generator

        loader = DataLoader(**loader_kwargs)

    model = _build_model(cfg, X_vectors.shape[1])

    # Some CEBRA models expose a classifier head that needs to be configured
    # with the number of output classes.  This is indicated by the presence of
    # a ``set_output_num`` method.  When such a model is requested we infer the
    # number of classes from the provided labels and initialize the classifier.
    if hasattr(model, "set_output_num"):
        if labels is None:
            raise ValueError(
                "Classifier model requested but `labels` are missing"
            )
        if labels.ndim == 1:
            num_classes = int(labels.max()) + 1
        else:
            num_classes = labels.shape[1]
        model.set_output_num(num_classes)
        model = model.to(cfg.device)
        if getattr(model, "classifier", None) is not None:
            model.classifier = model.classifier.to(cfg.device)

    if cfg.ddp.world_size > 1 and torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.ddp.local_rank]
        )

    criterion_map = {
        "infonce": cebra_criterions.InfoNCE,
        "infomse": cebra_criterions.InfoMSE,
        "fixedcosine": cebra_criterions.FixedCosineInfoNCE,
        "fixedeuclidean": cebra_criterions.FixedEuclideanInfoNCE,
        "learnablecosine": cebra_criterions.LearnableCosineInfoNCE,
        "learnableeuclidean": cebra_criterions.LearnableEuclideanInfoNCE,
        "nce": cebra_criterions.NCE,
    }
    if loss_type not in criterion_map:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    Criterion = criterion_map[loss_type]
    criterion_kwargs = {
        k: v for k, v in cfg.cebra.params.items() if k in inspect.signature(Criterion).parameters
    }
    criterion = Criterion(**criterion_kwargs)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.cebra.params.get("learning_rate", 1e-3),
    )

    steps = 0
    skipped = 0
    ma = deque(maxlen=50)

    if dist is not None:
        batch_size = cfg.cebra.params.get("batch_size", 512)
        with tqdm(total=cfg.cebra.max_iterations, desc="CEBRA Training") as pbar:
            while steps < cfg.cebra.max_iterations:
                anchor_idx = dist.sample_prior(batch_size)
                pos_idx = dist.sample_conditional(label_tensor[anchor_idx])
                same_mask = pos_idx == anchor_idx
                while torch.any(same_mask):
                    resample = dist.sample_conditional(label_tensor[anchor_idx[same_mask]])
                    pos_idx[same_mask] = resample
                    same_mask = pos_idx == anchor_idx

                neg_idx = dist.sample_prior(batch_size)
                neg_mask = label_tensor[neg_idx] != label_tensor[anchor_idx]
                while not torch.all(neg_mask):
                    missing = int((~neg_mask).sum().item())
                    resample = dist.sample_prior(missing)
                    neg_idx[~neg_mask] = resample
                    neg_mask = label_tensor[neg_idx] != label_tensor[anchor_idx]

                anchor_x = X_tensor[anchor_idx]
                pos_x = X_tensor[pos_idx]
                neg_x = X_tensor[neg_idx]

                anchor_emb = model(anchor_x)
                if isinstance(anchor_emb, tuple):
                    anchor_emb = anchor_emb[0]
                pos_emb = model(pos_x)
                if isinstance(pos_emb, tuple):
                    pos_emb = pos_emb[0]
                neg_emb = model(neg_x)
                if isinstance(neg_emb, tuple):
                    neg_emb = neg_emb[0]
                if anchor_emb is None or pos_emb is None or neg_emb is None:
                    raise ValueError("Model returned no embeddings")
                loss_tuple = criterion(anchor_emb, pos_emb, neg_emb)
                loss = loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if wandb.run is not None:
                    wandb.log({"loss": loss.item()}, step=steps)

                steps += 1
                pbar.update(1)
    else:
        epoch = 0
        with tqdm(total=cfg.cebra.max_iterations, desc="CEBRA Training") as pbar:
            while steps < cfg.cebra.max_iterations:
                if sampler is not None:
                    sampler.set_epoch(epoch)
                if deterministic and data_generator is not None:
                    base_seed = int(seed_value) if seed_value is not None else 0
                    data_generator.manual_seed(base_seed + epoch)
                for batch in loader:
                    (batch_x,) = batch
                    embeddings = model(batch_x.to(cfg.device, non_blocking=True))
                    if isinstance(embeddings, tuple):
                        embeddings = embeddings[0]
                    if embeddings is None:
                        raise ValueError("Model returned no embeddings")
                    loss = criterion(embeddings)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if wandb.run is not None:
                        wandb.log({"loss": loss.item()}, step=steps)

                    steps += 1
                    pbar.update(1)
                    if steps >= cfg.cebra.max_iterations:
                        break
                epoch += 1

    if wandb.run is not None:
        wandb.log({"total_skipped": skipped})

    # Explicitly shut down DataLoader workers to avoid process accumulation
    if loader is not None and cfg.cebra.num_workers > 0:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
        del loader
        import gc
        gc.collect()
    return model
