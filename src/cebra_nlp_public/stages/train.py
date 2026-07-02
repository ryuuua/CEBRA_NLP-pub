from pathlib import Path
from copy import deepcopy

import numpy as np
from ..optional_wandb import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

from ..cebra_trainer import (
    get_cebra_output_dir,
    normalize_model_architecture,
    save_cebra_embeddings,
    save_cebra_model,
    train_cebra,
    transform_cebra,
)
from ..config_schema import AppConfig, EmbeddingConfig
from ..config_runtime import app_config_to_dict
from ..data import load_and_prepare_dataset
from ..embeddings import (
    get_last_cache_usage,
    get_embeddings,
    load_or_generate_embedding_collection,
    load_or_generate_embeddings,
)
from ..results import run_consistency_check, run_knn_classification, run_knn_regression
from ..utils import apply_reproducibility
from .common import split_with_ids
from .trajectory_viz import render_label_drift_trajectory, validate_trajectory_requirements


def _init_wandb(cfg: AppConfig, output_dir: Path, is_main_process: bool):
    if not is_main_process:
        return None
    project = getattr(cfg.wandb, "project", None) if cfg.wandb else None
    if not project:
        return None
    requested_run_name = str(getattr(cfg.wandb, "run_name", "") or "").strip()
    use_requested_name = requested_run_name and requested_run_name != "default_run"
    run_name = requested_run_name if use_requested_name else HydraConfig.get().job.name
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        config=app_config_to_dict(cfg),
    )
    if run is not None:
        run_id = run.id
        print(f"W&B Run Name: {run_name}, Run ID: {run_id}")
        (output_dir / "wandb_run_id.txt").write_text(run_id)
    return run


def _record_consistency_metrics(
    metrics: dict[str, object],
    *,
    train_mean: float | None,
    valid_mean: float | None,
    cfg: AppConfig,
) -> None:
    if train_mean is not None:
        metrics["consistency_score_train"] = float(train_mean)
    if valid_mean is not None:
        metrics["consistency_score_valid"] = float(valid_mean)

    knn_acc = metrics.get("knn_accuracy")
    should_compute_composite = (
        isinstance(knn_acc, (int, float))
        and valid_mean is not None
        and cfg.cebra.conditional == "discrete"
    )
    if should_compute_composite:
        metrics["composite_knn_x_consistency_valid"] = float(knn_acc) * float(valid_mean)


def run(cfg: AppConfig, output_dir: Path, *, is_main_process: bool) -> Path:
    """Train a CEBRA model and save artifacts."""
    apply_reproducibility(cfg)
    cfg.cebra.conditional = cfg.cebra.conditional.lower()
    validate_trajectory_requirements(cfg)

    def _transform_batched(model, X: np.ndarray, *, batch_size: int) -> np.ndarray:
        import torch

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {X.shape}")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        was_training = model.training
        model.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                chunk = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32).to(
                    cfg.device
                )
                out = model(chunk)
                if isinstance(out, tuple):
                    out = out[0]
                batches.append(out.detach().cpu())
        if was_training:
            model.train()
        return torch.cat(batches, dim=0).numpy()

    metrics: dict[str, object] | None = None
    try:
        _init_wandb(cfg, output_dir, is_main_process)

        arch = normalize_model_architecture(cfg.cebra.model_architecture)
        cfg.cebra.model_architecture = arch
        if is_main_process and wandb.run is not None:
            wandb.config.update(
                {
                    "cebra_output_dim": cfg.cebra.output_dim,
                    "cebra_max_iterations": cfg.cebra.max_iterations,
                    "cebra_conditional": cfg.cebra.conditional,
                    "cebra_model_architecture": arch,
                }
            )

        print("\n--- Train Stage: Loading dataset ---")
        texts, conditional_data, time_indices, ids = load_and_prepare_dataset(cfg)

        print("\n--- Train Stage: Loading cached embeddings ---")
        if getattr(cfg, "embedding_collection", None) and cfg.embedding_collection.embeddings:
            X_vectors = load_or_generate_embedding_collection(
                cfg,
                texts,
                ids,
                labels=conditional_data,
                require_cache=cfg.stage.require_cache,
            )
        else:
            X_vectors = load_or_generate_embeddings(
                cfg,
                texts,
                ids,
                labels=conditional_data,
                require_cache=cfg.stage.require_cache,
            )
        cache_usage = get_last_cache_usage()
        if is_main_process and wandb.run is not None and cache_usage is not None:
            wandb.config.update(
                {
                    "dataset_name": cache_usage["dataset_name"],
                    "embedding_name": cache_usage["embedding_name"],
                    "embedding_model_name": cache_usage["embedding_model_name"],
                    "embedding_cache_reused": cache_usage["cache_hit"],
                    "embedding_cache_path": cache_usage["cache_path"],
                    "embedding_cache_requested_path": cache_usage["requested_cache_path"],
                    "embedding_cache_dataset_key": cache_usage["dataset_key"],
                    "embedding_cache_variant_tag": cache_usage["variant_tag"],
                    "embedding_cache_registry_key": cache_usage["registry_key"],
                    "embedding_cache_rulebook_id": cache_usage["rulebook_id"],
                },
                allow_val_change=True,
            )
            wandb.summary["embedding_cache_reused"] = cache_usage["cache_hit"]
            wandb.summary["embedding_cache_path"] = cache_usage["cache_path"]

        (
            X_train,
            X_valid,
            conditional_train,
            conditional_valid,
            _time_train,
            _time_valid,
            ids_train,
            ids_valid,
        ) = split_with_ids(X_vectors, conditional_data, time_indices, ids, cfg)

        labels_for_training = (
            None if cfg.cebra.conditional == "none" else conditional_train
        )

        artifact_dir = get_cebra_output_dir(cfg)
        print("\n--- Train Stage: Training CEBRA ---")
        cebra_model = train_cebra(
            X_train,
            labels_for_training,
            cfg,
            artifact_dir,
            sample_ids=ids_train,
        )

        model_path = save_cebra_model(cebra_model, artifact_dir)
        (artifact_dir / "config_resolved.yaml").write_text(
            yaml.safe_dump(app_config_to_dict(cfg), sort_keys=False, allow_unicode=True)
        )

        save_splits = bool(cfg.stage.save_splits) or bool(
            getattr(cfg.trajectory_analysis, "enabled", False)
        )
        if save_splits:
            np.save(artifact_dir / "train_ids.npy", np.asarray(ids_train, dtype=str))
            np.save(artifact_dir / "valid_ids.npy", np.asarray(ids_valid, dtype=str))

        if cfg.cebra.save_embeddings:
            cebra_embeddings_full = transform_cebra(cebra_model, X_vectors, cfg.device)
            emb_path = save_cebra_embeddings(cebra_embeddings_full, artifact_dir)
            np.save(artifact_dir / "cebra_embedding_ids.npy", np.asarray(ids, dtype=str))
            if is_main_process and wandb.run is not None:
                emb_artifact = wandb.Artifact(name=emb_path.stem, type="embeddings")
                emb_artifact.add_file(str(emb_path))
                wandb.log_artifact(emb_artifact)

        if is_main_process and wandb.run is not None:
            model_artifact = wandb.Artifact(name=model_path.stem, type="model")
            model_artifact.add_file(str(model_path))
            wandb.log_artifact(model_artifact)

        if is_main_process and getattr(cfg.stage, "evaluate_after_train", False):
            print("\n--- Train Stage: Quick evaluation (k-NN on validation) ---")
            batch_size = int(cfg.cebra.params.batch_size)
            batch_size = max(16, min(batch_size, 8192))

            cebra_train_embeddings = _transform_batched(
                cebra_model, X_train, batch_size=batch_size
            )
            cebra_valid_embeddings = _transform_batched(
                cebra_model, X_valid, batch_size=batch_size
            )

            metrics = {
                "cebra_output_dim": int(cfg.cebra.output_dim),
                "cebra_max_iterations": int(cfg.cebra.max_iterations),
                "cebra_conditional": str(cfg.cebra.conditional),
                "cebra_model_architecture": str(cfg.cebra.model_architecture),
                "cebra_batch_size": int(cfg.cebra.params.batch_size or 0),
                "cebra_learning_rate": float(cfg.cebra.params.learning_rate or 0.0),
            }

            if cfg.cebra.conditional == "discrete":
                label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}
                accuracy, report = run_knn_classification(
                    train_embeddings=cebra_train_embeddings,
                    valid_embeddings=cebra_valid_embeddings,
                    y_train=conditional_train,
                    y_valid=conditional_valid,
                    label_map=label_map,
                    output_dir=output_dir,
                    knn_neighbors=cfg.evaluation.knn_neighbors,
                    enable_plots=cfg.evaluation.enable_plots,
                    backend=cfg.evaluation.knn_backend,
                    faiss_gpu_id=cfg.evaluation.faiss_gpu_id,
                )
                metrics.update(
                    {
                        "objective": "knn_accuracy",
                        "objective_value": float(accuracy),
                        "knn_accuracy": float(accuracy),
                        "classification_report": report,
                    }
                )
                if wandb.run is not None:
                    wandb.log({"knn_accuracy": float(accuracy)})
            else:
                mse, r2 = run_knn_regression(
                    train_embeddings=cebra_train_embeddings,
                    valid_embeddings=cebra_valid_embeddings,
                    y_train=conditional_train,
                    y_valid=conditional_valid,
                    output_dir=output_dir,
                    knn_neighbors=cfg.evaluation.knn_neighbors,
                    backend=cfg.evaluation.knn_backend,
                    faiss_gpu_id=cfg.evaluation.faiss_gpu_id,
                )
                metrics.update(
                    {
                        "objective": "knn_regression_r2",
                        "objective_value": float(r2),
                        "knn_regression_mse": float(mse),
                        "knn_regression_r2": float(r2),
                    }
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "knn_regression_mse": float(mse),
                            "knn_regression_r2": float(r2),
                        }
                    )

        if cfg.consistency_check.enabled:
            print("\n--- Train Stage: Running Consistency Check ---")
            if cfg.consistency_check.mode == "datasets":
                embeddings_list = []
                embedding_dir = Path(get_original_cwd()) / "conf" / "embedding"
                for emb_name in cfg.consistency_check.dataset_ids:
                    emb_path = embedding_dir / f"{emb_name}.yaml"
                    emb_dict = yaml.safe_load(emb_path.read_text())
                    if not isinstance(emb_dict, dict):
                        raise ValueError(
                            f"Expected mapping config in {emb_path}, got {type(emb_dict)!r}"
                        )
                    tmp_cfg = deepcopy(cfg)
                    tmp_cfg.embedding = EmbeddingConfig(**emb_dict)
                    embeddings_list.append(get_embeddings(texts, tmp_cfg))

                labels_list = [conditional_data for _ in embeddings_list]
                dataset_mean, _ = run_consistency_check(
                    None,
                    None,
                    None,
                    cfg,
                    output_dir,
                    dataset_embeddings=embeddings_list,
                    labels_list=labels_list,
                    dataset_ids=cfg.consistency_check.dataset_ids,
                    enable_plots=cfg.evaluation.enable_plots,
                    log_to_wandb=is_main_process,
                )
                if metrics is not None and dataset_mean is not None:
                    metrics["consistency_score_datasets"] = float(dataset_mean)
            else:
                train_mean, valid_mean = run_consistency_check(
                    X_train,
                    labels_for_training,
                    X_valid,
                    cfg,
                    output_dir,
                    enable_plots=cfg.evaluation.enable_plots,
                    log_to_wandb=is_main_process,
                )
                if metrics is not None:
                    _record_consistency_metrics(
                        metrics,
                        train_mean=train_mean,
                        valid_mean=valid_mean,
                        cfg=cfg,
                    )

        if (
            is_main_process
            and bool(getattr(cfg.trajectory_analysis, "enabled", False))
            and bool(getattr(cfg.trajectory_analysis, "render_after_train", True))
        ):
            print("\n--- Train Stage: Rendering label drift trajectory ---")
            render_label_drift_trajectory(
                cfg,
                artifact_dir=artifact_dir,
                hydra_output_dir=output_dir,
            )

        if metrics is not None:
            import json

            metrics_json = json.dumps(metrics, indent=2, sort_keys=True)
            (output_dir / "metrics.json").write_text(metrics_json)
            (artifact_dir / "metrics.json").write_text(metrics_json)

        print(f"Train stage complete. Model artifacts in {artifact_dir}.")
        return artifact_dir
    finally:
        if is_main_process and wandb.run is not None:
            wandb.finish()
