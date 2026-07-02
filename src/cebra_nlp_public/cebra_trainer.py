import numpy as np
from pathlib import Path
from dataclasses import asdict
from .config_schema import AppConfig
from tqdm.auto import tqdm
from .optional_wandb import wandb
from cebra.distributions.discrete import DiscreteUniform, DiscreteEmpirical


def _embedding_signature(cfg: AppConfig) -> str:
    collection_cfg = getattr(cfg, "embedding_collection", None)
    if collection_cfg is not None and collection_cfg.embeddings:
        child_names = [child.name for child in collection_cfg.embeddings]
        children = "+".join(child_names)
        return f"{collection_cfg.name}__{children}"
    return cfg.embedding.name


def get_cebra_config_hash(cfg):
    import hashlib
    import json

    cebra_cfg = cfg.cebra
    relevant_cfg = {
        "dataset": cfg.dataset.name,
        "embedding": _embedding_signature(cfg),
        "cebra": asdict(cebra_cfg),
    }
    hash_str = json.dumps(relevant_cfg, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:8]


def get_cebra_output_dir(cfg: AppConfig, base: str | None = None) -> Path:
    h = get_cebra_config_hash(cfg)
    emb_sig = _embedding_signature(cfg)
    base_dir = base or getattr(getattr(cfg, "paths", None), "model_dir", "model_outputs")
    path = Path(base_dir) / f"{cfg.dataset.name}__{emb_sig}__{h}"
    run_tag = getattr(getattr(cfg, "stage", None), "run_tag", None)
    if run_tag:
        path = path / str(run_tag)
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


def get_label_drift_output_dir(artifact_dir: Path) -> Path:
    path = Path(artifact_dir) / "label_drift_trajectory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_label_drift_checkpoint_dir(artifact_dir: Path) -> Path:
    path = get_label_drift_output_dir(artifact_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_trajectory_checkpoint_steps(
    max_iterations: int,
    checkpoint_every_n_steps: int,
    *,
    save_initial: bool,
    save_final: bool,
) -> list[int]:
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be >= 0, got {max_iterations}.")
    if checkpoint_every_n_steps <= 0:
        raise ValueError(
            "trajectory_analysis.checkpoint_every_n_steps must be > 0, got "
            f"{checkpoint_every_n_steps}."
        )

    checkpoint_steps: set[int] = set()
    if save_initial:
        checkpoint_steps.add(0)
    checkpoint_steps.update(
        range(
            checkpoint_every_n_steps,
            max_iterations + 1,
            checkpoint_every_n_steps,
        )
    )
    if save_final:
        checkpoint_steps.add(int(max_iterations))
    return sorted(int(step) for step in checkpoint_steps)


def save_label_drift_checkpoint(
    model,
    optimizer,
    checkpoint_dir: Path,
    *,
    step: int,
    estimated_epoch: float,
    config_hash: str,
) -> Path:
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{int(step):06d}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": int(step),
            "estimated_epoch": float(estimated_epoch),
            "config_hash": str(config_hash),
        },
        path,
    )
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
        num_units=cfg.cebra.num_units,
        num_output=cfg.cebra.output_dim,
    ).to(cfg.device)


def load_cebra_model(model_path, cfg: AppConfig, input_dimension: int):
    import torch

    model = _build_model(cfg, input_dimension)
    raw_state = torch.load(model_path, map_location=cfg.device)

    def _maybe_strip_module(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        if not has_module_prefix:
            return state_dict
        return {k[len("module.") :]: v for k, v in state_dict.items()}

    def _extract_model_state(state):
        if not isinstance(state, dict):
            return state
        if "model_state" in state:
            return _maybe_strip_module(state["model_state"])
        if "state_dict" in state:
            return _maybe_strip_module(state["state_dict"])
        return _maybe_strip_module(state)

    try:
        model.load_state_dict(_extract_model_state(raw_state))
    except RuntimeError:
        candidate_state = _extract_model_state(raw_state)
        model.load_state_dict(candidate_state)

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


def train_cebra(X_vectors, labels, cfg: AppConfig, output_dir, sample_ids=None):
    """Train CEBRA using its native PyTorch API.

    Parameters
    ----------
    cfg : AppConfig
        Training configuration. ``cfg.cebra.max_iterations`` denotes the
        maximum number of gradient steps (i.e., batches) to execute. The
        loop stops once this limit is reached, matching the scikit-learn
        ``max_iter`` semantics.
    """

    import torch
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    import inspect

    conditional = cfg.cebra.conditional.lower()
    cfg.cebra.conditional = conditional
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
    elif conditional != "none":
        raise ValueError("`labels` are required for the selected training configuration")

    track_epoch_trajectory = bool(getattr(cfg.cebra, "save_epoch_trajectory", False))
    if track_epoch_trajectory:
        from .visualization.trajectory.epoch import validate_epoch_trajectory_config

        validate_epoch_trajectory_config(cfg)

    from cebra.models import criterions as cebra_criterions

    X_tensor = torch.as_tensor(X_vectors, dtype=torch.float32)
    label_tensor = None
    dist = None
    if labels is not None:
        dtype = torch.long if conditional == "discrete" else torch.float32
        label_tensor = torch.as_tensor(labels, dtype=dtype)
        if conditional == "discrete":
            X_tensor = X_tensor.to(cfg.device)
            label_tensor = label_tensor.to(cfg.device)
            if cfg.cebra.params.prior == "uniform":
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
            batch_size=cfg.cebra.params.batch_size,
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
    params_dict = asdict(cfg.cebra.params)
    criterion_kwargs = {
        k: v
        for k, v in params_dict.items()
        if v is not None and k in inspect.signature(Criterion).parameters
    }
    criterion = Criterion(**criterion_kwargs)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.cebra.params.learning_rate,
    )

    steps = 0
    skipped = 0
    is_main_process = int(getattr(cfg.ddp, "rank", 0)) == 0
    trajectory_cfg = getattr(cfg, "trajectory_analysis", None)
    track_label_drift = bool(getattr(trajectory_cfg, "enabled", False))
    checkpoint_steps: set[int] = set()
    saved_checkpoint_steps: set[int] = set()
    checkpoint_records: list[dict[str, object]] = []
    checkpoint_dir: Path | None = None
    config_hash = get_cebra_config_hash(cfg)
    epoch_trajectory_records: list[dict[str, object]] = []
    epoch_trajectory_dir: Path | None = None
    epoch_trajectory_snapshot_dir: Path | None = None
    epoch_trajectory_indices: np.ndarray | None = None
    epoch_trajectory_inputs: np.ndarray | None = None
    saved_epoch_trajectory_epochs: set[int] = set()
    trajectory_every_n_epochs = int(
        getattr(cfg.cebra, "trajectory_every_n_epochs", 1)
    )

    def _unwrap_embeddings(output):
        if isinstance(output, tuple):
            output = output[0]
        if output is None:
            raise ValueError("Model returned no embeddings")
        return output

    if track_label_drift and is_main_process:
        checkpoint_steps = set(
            build_trajectory_checkpoint_steps(
                int(cfg.cebra.max_iterations),
                int(getattr(trajectory_cfg, "checkpoint_every_n_steps", 100)),
                save_initial=bool(getattr(trajectory_cfg, "save_initial_checkpoint", True)),
                save_final=bool(getattr(trajectory_cfg, "save_final_checkpoint", True)),
            )
        )
        checkpoint_dir = get_label_drift_checkpoint_dir(Path(output_dir))
        trajectory_output_dir = get_label_drift_output_dir(Path(output_dir))
    else:
        trajectory_output_dir = None

    if track_epoch_trajectory and is_main_process:
        from .visualization.trajectory.epoch import (
            get_epoch_trajectory_output_dir,
            resolve_trajectory_seed,
            select_trajectory_indices,
        )

        epoch_trajectory_dir = get_epoch_trajectory_output_dir(Path(output_dir))
        epoch_trajectory_snapshot_dir = epoch_trajectory_dir / "snapshots"
        epoch_trajectory_snapshot_dir.mkdir(parents=True, exist_ok=True)
        epoch_trajectory_indices = select_trajectory_indices(
            X_vectors.shape[0],
            getattr(cfg.cebra, "trajectory_sample_size", 1000),
            seed=resolve_trajectory_seed(cfg),
        )
        epoch_trajectory_inputs = X_vectors[epoch_trajectory_indices].astype(np.float32)
        np.save(epoch_trajectory_dir / "sample_indices.npy", epoch_trajectory_indices)
        if sample_ids is not None:
            sample_ids_array = np.asarray(sample_ids, dtype=str)
            if sample_ids_array.shape[0] != X_vectors.shape[0]:
                raise ValueError("`sample_ids` must have the same number of samples as `X_vectors`")
            selected_ids = sample_ids_array[epoch_trajectory_indices]
        else:
            selected_ids = np.asarray(
                [f"sample_{int(index)}" for index in epoch_trajectory_indices],
                dtype=str,
            )
        np.save(epoch_trajectory_dir / "sample_ids.npy", selected_ids)

        if labels is None:
            selected_labels = np.asarray(["sample"] * epoch_trajectory_indices.shape[0], dtype=str)
        else:
            raw_selected_labels = np.asarray(labels)[epoch_trajectory_indices]
            label_map = getattr(getattr(cfg, "dataset", None), "label_map", {}) or {}
            selected_labels = []
            for value in raw_selected_labels.tolist():
                if isinstance(value, (list, tuple)):
                    selected_labels.append(",".join(str(item) for item in value))
                    continue
                try:
                    selected_labels.append(str(label_map.get(int(value), value)))
                except (TypeError, ValueError):
                    selected_labels.append(str(value))
            selected_labels = np.asarray(selected_labels, dtype=str)
        np.save(epoch_trajectory_dir / "sample_labels.npy", selected_labels)

    def _capture_label_drift_checkpoint(
        current_step: int,
        *,
        estimated_steps_per_epoch: int,
    ) -> None:
        if (
            not track_label_drift
            or not is_main_process
            or checkpoint_dir is None
            or current_step not in checkpoint_steps
            or current_step in saved_checkpoint_steps
        ):
            return
        estimated_epoch = float(current_step) / float(max(1, estimated_steps_per_epoch))
        checkpoint_path = save_label_drift_checkpoint(
            model,
            optimizer,
            checkpoint_dir,
            step=current_step,
            estimated_epoch=estimated_epoch,
            config_hash=config_hash,
        )
        checkpoint_records.append(
            {
                "step": int(current_step),
                "estimated_epoch": float(estimated_epoch),
                "path": str(checkpoint_path),
                "relative_path": (
                    str(checkpoint_path.relative_to(trajectory_output_dir))
                    if trajectory_output_dir is not None
                    else checkpoint_path.name
                ),
            }
        )
        saved_checkpoint_steps.add(int(current_step))

    def _capture_epoch_trajectory_snapshot(epoch_num: int, current_step: int) -> None:
        if (
            not track_epoch_trajectory
            or not is_main_process
            or epoch_trajectory_dir is None
            or epoch_trajectory_snapshot_dir is None
            or epoch_trajectory_inputs is None
            or int(epoch_num) in saved_epoch_trajectory_epochs
        ):
            return
        snapshot = transform_cebra(model, epoch_trajectory_inputs, cfg.device)
        snapshot_path = epoch_trajectory_snapshot_dir / f"epoch_{int(epoch_num):06d}.npy"
        np.save(snapshot_path, np.asarray(snapshot, dtype=np.float32))
        epoch_trajectory_records.append(
            {
                "epoch": int(epoch_num),
                "step": int(current_step),
                "relative_path": str(snapshot_path.relative_to(epoch_trajectory_dir)),
            }
        )
        saved_epoch_trajectory_epochs.add(int(epoch_num))

    if dist is not None:
        batch_size = cfg.cebra.params.batch_size
        estimated_steps_per_epoch = max(
            1,
            int(np.ceil(X_vectors.shape[0] / max(1, int(batch_size)))),
        )
        _capture_label_drift_checkpoint(
            0,
            estimated_steps_per_epoch=estimated_steps_per_epoch,
        )
        _capture_epoch_trajectory_snapshot(0, 0)
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

                anchor_emb = _unwrap_embeddings(model(anchor_x))
                pos_emb = _unwrap_embeddings(model(pos_x))
                neg_emb = _unwrap_embeddings(model(neg_x))
                loss_tuple = criterion(anchor_emb, pos_emb, neg_emb)
                loss = loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if wandb.run is not None:
                    wandb.log({"loss": loss.item()}, step=steps)

                steps += 1
                pbar.update(1)
                _capture_label_drift_checkpoint(
                    steps,
                    estimated_steps_per_epoch=estimated_steps_per_epoch,
                )
                if steps % estimated_steps_per_epoch == 0:
                    completed_epoch = int(steps // estimated_steps_per_epoch)
                    if completed_epoch % trajectory_every_n_epochs == 0:
                        _capture_epoch_trajectory_snapshot(completed_epoch, steps)
    else:
        epoch = 0
        estimated_steps_per_epoch = max(1, len(loader) if loader is not None else 1)
        _capture_label_drift_checkpoint(
            0,
            estimated_steps_per_epoch=estimated_steps_per_epoch,
        )
        _capture_epoch_trajectory_snapshot(0, 0)
        with tqdm(total=cfg.cebra.max_iterations, desc="CEBRA Training") as pbar:
            while steps < cfg.cebra.max_iterations:
                if sampler is not None:
                    sampler.set_epoch(epoch)
                if deterministic and data_generator is not None:
                    base_seed = int(seed_value) if seed_value is not None else 0
                    data_generator.manual_seed(base_seed + epoch)
                for batch in loader:
                    (batch_x,) = batch
                    embeddings = _unwrap_embeddings(
                        model(batch_x.to(cfg.device, non_blocking=True))
                    )
                    loss = criterion(embeddings)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if wandb.run is not None:
                        wandb.log({"loss": loss.item()}, step=steps)

                    steps += 1
                    pbar.update(1)
                    _capture_label_drift_checkpoint(
                        steps,
                        estimated_steps_per_epoch=estimated_steps_per_epoch,
                    )
                    if steps % estimated_steps_per_epoch == 0:
                        completed_epoch = int(steps // estimated_steps_per_epoch)
                        if completed_epoch % trajectory_every_n_epochs == 0:
                            _capture_epoch_trajectory_snapshot(completed_epoch, steps)
                    if steps >= cfg.cebra.max_iterations:
                        break
                epoch += 1

    if track_epoch_trajectory and is_main_process:
        import json

        final_epoch = int(np.ceil(float(steps) / float(max(1, estimated_steps_per_epoch))))
        _capture_epoch_trajectory_snapshot(final_epoch, steps)
        manifest_path = epoch_trajectory_dir / "manifest.json" if epoch_trajectory_dir is not None else None
        if manifest_path is not None:
            manifest = {
                "config_hash": config_hash,
                "max_iterations": int(cfg.cebra.max_iterations),
                "estimated_steps_per_epoch": int(estimated_steps_per_epoch),
                "trajectory_every_n_epochs": int(trajectory_every_n_epochs),
                "sample_size": int(epoch_trajectory_indices.shape[0])
                if epoch_trajectory_indices is not None
                else 0,
                "num_samples": int(epoch_trajectory_indices.shape[0])
                if epoch_trajectory_indices is not None
                else 0,
                "snapshots": epoch_trajectory_records,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            from .visualization.trajectory.epoch import render_saved_epoch_trajectory

            render_saved_epoch_trajectory(epoch_trajectory_dir, cfg=cfg)
            report_path = epoch_trajectory_dir / "trajectory_render_report.json"
            if report_path.exists():
                manifest["render_report"] = json.loads(report_path.read_text(encoding="utf-8"))
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if track_label_drift and is_main_process:
        _capture_label_drift_checkpoint(
            steps,
            estimated_steps_per_epoch=max(1, estimated_steps_per_epoch),
        )
        manifest_path = get_label_drift_output_dir(Path(output_dir)) / "manifest.json"
        import json

        manifest_path.write_text(
            json.dumps(
                {
                    "config_hash": config_hash,
                    "checkpoint_every_n_steps": int(
                        getattr(trajectory_cfg, "checkpoint_every_n_steps", 100)
                    ),
                    "save_initial_checkpoint": bool(
                        getattr(trajectory_cfg, "save_initial_checkpoint", True)
                    ),
                    "save_final_checkpoint": bool(
                        getattr(trajectory_cfg, "save_final_checkpoint", True)
                    ),
                    "max_iterations": int(cfg.cebra.max_iterations),
                    "checkpoints": checkpoint_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

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
