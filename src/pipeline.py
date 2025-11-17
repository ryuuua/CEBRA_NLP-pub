from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from src.config_schema import AppConfig, EmbeddingConfig
from src.data import load_and_prepare_dataset
from src.embeddings import get_embeddings, load_or_generate_embeddings
from src.cebra_trainer import (
    normalize_model_architecture,
    save_cebra_embeddings,
    save_cebra_model,
    train_cebra,
    transform_cebra,
)
from src.results import run_consistency_check, run_knn_classification, run_knn_regression
from src.plotting import (
    prepare_plot_labels,
    render_discrete_visualizations,
    render_continuous_visualizations,
)
from src.utils import normalize_binary_labels


def run_pipeline(cfg: AppConfig, *, log_to_wandb: bool, is_main_process: bool) -> None:
    """Execute the shared training, evaluation, and visualization pipeline."""
    conditional_type = cfg.cebra.conditional
    output_dir = Path(HydraConfig.get().run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arch = normalize_model_architecture(cfg.cebra.model_architecture)
    cfg.cebra.model_architecture = arch

    wandb_enabled = log_to_wandb and wandb.run is not None
    if wandb_enabled:
        wandb.config.update(
            {
                "cebra_output_dim": cfg.cebra.output_dim,
                "cebra_max_iterations": cfg.cebra.max_iterations,
                "cebra_conditional": cfg.cebra.conditional,
                "cebra_model_architecture": arch,
            }
        )

    print("\n--- Step 1: Loading dataset ---")
    texts, conditional_data, _time_indices, ids = load_and_prepare_dataset(cfg)

    print("\n--- Step 2: Generating text embeddings ---")
    X_vectors = load_or_generate_embeddings(cfg, texts, ids)

    print("\n--- Step 3: Splitting data ---")
    X_train, X_valid, conditional_train, conditional_valid = train_test_split(
        X_vectors,
        conditional_data,
        test_size=cfg.evaluation.test_size,
        random_state=cfg.evaluation.random_state,
        stratify=(conditional_data if conditional_type == "discrete" else None),
    )

    print("\n--- Step 4: Training CEBRA model ---")
    labels_for_training = None if conditional_type == "none" else conditional_train
    cebra_model = train_cebra(X_train, labels_for_training, cfg, output_dir)
    model_path = save_cebra_model(cebra_model, output_dir)

    if wandb_enabled:
        model_artifact = wandb.Artifact(name=model_path.stem, type="model")
        model_artifact.add_file(str(model_path))
        wandb.log_artifact(model_artifact)

    print("\n--- Step 5: Transforming data with trained CEBRA model ---")
    cebra_embeddings_full = transform_cebra(cebra_model, X_vectors, cfg.device)
    if cfg.cebra.save_embeddings:
        emb_path = save_cebra_embeddings(cebra_embeddings_full, output_dir)
        if wandb_enabled:
            emb_artifact = wandb.Artifact(name=emb_path.stem, type="embeddings")
            emb_artifact.add_file(str(emb_path))
            wandb.log_artifact(emb_artifact)
    cebra_train_embeddings = transform_cebra(cebra_model, X_train, cfg.device)
    cebra_valid_embeddings = transform_cebra(cebra_model, X_valid, cfg.device)

    print("\n--- Step 6: Visualization and Evaluation ---")
    if conditional_type == "discrete":
        print("Running discrete evaluation and visualization...")
        label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}
        conditional_data = normalize_binary_labels(np.asarray(conditional_data)).tolist()
        text_labels_full = [label_map[int(l)] for l in conditional_data]
        palette = OmegaConf.to_container(
            cfg.dataset.visualization.emotion_colors, resolve=True
        )
        order = OmegaConf.to_container(
            cfg.dataset.visualization.emotion_order, resolve=True
        )

        def _wandb_hook(paths: Iterable[Path]) -> None:
            if not wandb_enabled:
                return
            for path in paths:
                name = path.stem if path.suffix == ".html" else "cebra-static-plots"
                artifact = wandb.Artifact(name=name, type="evaluation")
                artifact.add_file(str(path))
                wandb.log_artifact(artifact)

        render_discrete_visualizations(
            cfg,
            cebra_embeddings_full,
            text_labels_full,
            output_dir,
            wandb_hook=_wandb_hook if wandb_enabled else None,
        )

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
        if wandb_enabled:
            wandb.log({"knn_accuracy": accuracy})
        report_path = output_dir / "classification_report.json"
        pd.Series(report).to_json(report_path, indent=4)
        if wandb_enabled:
            report_artifact = wandb.Artifact(name=report_path.stem, type="evaluation")
            report_artifact.add_file(str(report_path))
            wandb.log_artifact(report_artifact)

    elif conditional_type == "none":
        print("Running None evaluation and visualization...")
        valence_scores = conditional_data[:, 0]
        def _wandb_continuous(paths: Iterable[Path]) -> None:
            if not wandb_enabled:
                return
            for path in paths:
                artifact = wandb.Artifact(name=path.stem, type="evaluation")
                artifact.add_file(str(path))
                wandb.log_artifact(artifact)

        render_continuous_visualizations(
            cfg,
            cebra_embeddings_full,
            valence_scores,
            output_dir,
            wandb_hook=_wandb_continuous if wandb_enabled else None,
        )

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
        if wandb_enabled:
            wandb.log({"knn_regression_mse": mse, "knn_regression_r2": r2})

    if cfg.consistency_check.enabled:
        print("\n--- Step 7: Running Consistency Check ---")
        if cfg.consistency_check.mode == "datasets":
            embeddings_list: List[np.ndarray] = []
            original_cwd = get_original_cwd()
            embedding_dir = Path(original_cwd) / "conf" / "embedding"
            for emb_name in cfg.consistency_check.dataset_ids:
                emb_path = embedding_dir / f"{emb_name}.yaml"
                emb_conf = OmegaConf.load(emb_path)
                emb_dict = OmegaConf.to_container(emb_conf, resolve=True)
                tmp_cfg = deepcopy(cfg)
                tmp_cfg.embedding = EmbeddingConfig(**emb_dict)
                embeddings_list.append(get_embeddings(texts, tmp_cfg))

            labels_list = [conditional_data for _ in embeddings_list]
            run_consistency_check(
                None,
                None,
                None,
                cfg,
                output_dir,
                dataset_embeddings=embeddings_list,
                labels=labels_list,
                dataset_ids=cfg.consistency_check.dataset_ids,
                enable_plots=cfg.evaluation.enable_plots,
                log_to_wandb=is_main_process,
            )
        else:
            run_consistency_check(
                X_train,
                labels_for_training,
                X_valid,
                cfg,
                output_dir,
                enable_plots=cfg.evaluation.enable_plots,
                log_to_wandb=is_main_process,
            )

    print("\n--- Pipeline Complete ---")

