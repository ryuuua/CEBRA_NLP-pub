from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from ..optional_wandb import wandb
from hydra.core.hydra_config import HydraConfig

from ..cebra_trainer import (
    get_cebra_output_dir,
    load_cebra_model,
    normalize_model_architecture,
    transform_cebra,
)
from ..config_schema import AppConfig
from ..config_runtime import app_config_to_dict
from ..data import load_and_prepare_dataset
from ..embeddings import (
    load_or_generate_embedding_collection,
    load_or_generate_embeddings,
)
from ..label_overlay import load_or_generate_label_overlay_embeddings
from ..plotting import prepare_plot_labels
from ..results import (
    export_pca_projection_artifacts,
    run_knn_classification,
    run_knn_regression,
    run_local_linearity_probe,
    save_interactive_plot,
    save_static_2d_plots,
)
from ..utils import apply_reproducibility
from .common import align_by_ids, split_with_ids


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


def _indices_from_ids(all_ids: Sequence, target_ids: Sequence) -> np.ndarray:
    id_to_index = {str(item): idx for idx, item in enumerate(all_ids)}
    return np.asarray([id_to_index[str(item)] for item in target_ids], dtype=int)


def _log_wandb_artifact_if_exists(
    path: Path,
    *,
    is_main_process: bool,
    artifact_type: str,
) -> None:
    if not is_main_process or wandb.run is None or not path.exists():
        return
    artifact = wandb.Artifact(name=path.stem, type=artifact_type)
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)


def run(cfg: AppConfig, output_dir: Path, *, is_main_process: bool) -> None:
    """Load a trained CEBRA model and run visualization + evaluation."""
    apply_reproducibility(cfg)
    cfg.cebra.conditional = cfg.cebra.conditional.lower()

    try:
        _init_wandb(cfg, output_dir, is_main_process)

        arch = normalize_model_architecture(cfg.cebra.model_architecture)
        cfg.cebra.model_architecture = arch

        print("\n--- Viz Stage: Loading dataset ---")
        texts, conditional_data, time_indices, ids = load_and_prepare_dataset(cfg)
        conditional_data = np.asarray(conditional_data)
        time_indices = np.asarray(time_indices)

        print("\n--- Viz Stage: Loading cached embeddings ---")
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

        artifact_dir = get_cebra_output_dir(cfg)
        if cfg.stage.model_path:
            model_path = Path(cfg.stage.model_path)
        else:
            model_path = artifact_dir / "cebra_model.pt"
        if cfg.stage.require_model and not model_path.exists():
            raise FileNotFoundError(f"CEBRA model not found at {model_path}")

        cebra_model = load_cebra_model(model_path, cfg, X_vectors.shape[1])

        train_ids_path = artifact_dir / "train_ids.npy"
        valid_ids_path = artifact_dir / "valid_ids.npy"
        if (
            cfg.stage.use_saved_splits
            and train_ids_path.exists()
            and valid_ids_path.exists()
        ):
            ids_train = np.asarray(np.load(train_ids_path, allow_pickle=True), dtype=str)
            ids_valid = np.asarray(np.load(valid_ids_path, allow_pickle=True), dtype=str)
            idx_train = _indices_from_ids(ids, ids_train)
            idx_valid = _indices_from_ids(ids, ids_valid)
            _X_train = X_vectors[idx_train]
            _X_valid = X_vectors[idx_valid]
            conditional_train = conditional_data[idx_train]
            conditional_valid = conditional_data[idx_valid]
        else:
            (
                _X_train,
                _X_valid,
                conditional_train,
                conditional_valid,
                _time_train,
                _time_valid,
                ids_train,
                ids_valid,
            ) = split_with_ids(X_vectors, conditional_data, time_indices, ids, cfg)
            idx_train = _indices_from_ids(ids, ids_train)
            idx_valid = _indices_from_ids(ids, ids_valid)

        saved_embeddings_path = artifact_dir / "cebra_embeddings.npy"
        saved_ids_path = artifact_dir / "cebra_embedding_ids.npy"
        if (
            cfg.stage.use_saved_embeddings
            and saved_embeddings_path.exists()
            and saved_ids_path.exists()
        ):
            saved_embeddings = np.load(saved_embeddings_path)
            saved_ids = np.asarray(np.load(saved_ids_path, allow_pickle=True), dtype=str)
            cebra_embeddings_full = align_by_ids(saved_ids, saved_embeddings, ids)
        else:
            cebra_embeddings_full = transform_cebra(cebra_model, X_vectors, cfg.device)

        cebra_train_embeddings = cebra_embeddings_full[idx_train]
        cebra_valid_embeddings = cebra_embeddings_full[idx_valid]

        labels_full, palette, order = prepare_plot_labels(cfg, conditional_data)
        labels_train = [labels_full[i] for i in idx_train]
        labels_valid = [labels_full[i] for i in idx_valid]

        overlay_spec = None
        overlay_cebra_embeddings = None
        if cfg.cebra.conditional == "discrete":
            overlay_spec, overlay_input_embeddings = load_or_generate_label_overlay_embeddings(
                cfg,
                require_cache=cfg.stage.require_cache,
            )
            if overlay_input_embeddings is not None:
                overlay_cebra_embeddings = transform_cebra(
                    cebra_model,
                    overlay_input_embeddings,
                    cfg.device,
                )

        show_overlay_in_cebra = bool(getattr(cfg.label_overlay, "show_in_cebra_space", True))
        show_overlay_in_pca = bool(getattr(cfg.label_overlay, "show_in_pca", True))
        show_centroids_in_pca = bool(
            getattr(cfg.label_overlay, "show_centroids_in_pca", True)
        )
        include_split_overlay_views = bool(
            getattr(cfg.label_overlay, "include_split_views", True)
        )
        share_full_pca_basis = bool(
            getattr(cfg.pca_analysis, "share_full_basis_across_views", True)
        )
        overlay_spec_full = (
            overlay_spec
            if overlay_spec is not None
            and overlay_cebra_embeddings is not None
            and show_overlay_in_cebra
            else None
        )
        overlay_embeddings_full = (
            overlay_cebra_embeddings if overlay_spec_full is not None else None
        )
        overlay_spec_split = (
            overlay_spec_full if include_split_overlay_views else None
        )
        overlay_embeddings_split = (
            overlay_embeddings_full if include_split_overlay_views else None
        )

        full_pca_projection = None
        train_pca_projection = None
        valid_pca_projection = None
        train_viz_dir = output_dir / "visualizations_train"
        val_viz_dir = output_dir / "visualizations_valid"
        pca_plot_spec = (
            overlay_spec
            if overlay_spec is not None and (show_overlay_in_pca or show_centroids_in_pca)
            else None
        )
        split_pca_plot_spec = pca_plot_spec if include_split_overlay_views else None
        pca_overlay_embeddings = (
            overlay_cebra_embeddings
            if overlay_spec is not None
            and overlay_cebra_embeddings is not None
            and show_overlay_in_pca
            else None
        )
        split_pca_overlay_embeddings = (
            pca_overlay_embeddings if include_split_overlay_views else None
        )
        if cfg.evaluation.enable_plots or pca_plot_spec is not None:
            full_pca_projection = export_pca_projection_artifacts(
                cebra_embeddings_full,
                output_dir,
                scope_name="full",
                cfg=cfg,
                projected_text_labels=labels_full if pca_plot_spec is not None else None,
                overlay_spec=pca_plot_spec,
                overlay_embeddings=pca_overlay_embeddings,
            )
            train_viz_dir.mkdir(parents=True, exist_ok=True)
            val_viz_dir.mkdir(parents=True, exist_ok=True)
            train_fit_embeddings = (
                cebra_embeddings_full if share_full_pca_basis else cebra_train_embeddings
            )
            valid_fit_embeddings = (
                cebra_embeddings_full if share_full_pca_basis else cebra_valid_embeddings
            )
            train_projected_embeddings = (
                cebra_train_embeddings if share_full_pca_basis else None
            )
            valid_projected_embeddings = (
                cebra_valid_embeddings if share_full_pca_basis else None
            )
            shared_fit_scope = "full" if share_full_pca_basis else None
            shared_axis_limits = (
                full_pca_projection.get("axis_limits")
                if share_full_pca_basis and full_pca_projection is not None
                else None
            )
            train_pca_projection = export_pca_projection_artifacts(
                train_fit_embeddings,
                train_viz_dir,
                scope_name="train",
                cfg=cfg,
                projected_embeddings=train_projected_embeddings,
                projected_text_labels=(
                    labels_train if split_pca_plot_spec is not None else None
                ),
                fit_scope_name=shared_fit_scope,
                axis_limits=shared_axis_limits,
                overlay_spec=split_pca_plot_spec,
                overlay_embeddings=split_pca_overlay_embeddings,
            )
            valid_pca_projection = export_pca_projection_artifacts(
                valid_fit_embeddings,
                val_viz_dir,
                scope_name="valid",
                cfg=cfg,
                projected_embeddings=valid_projected_embeddings,
                projected_text_labels=(
                    labels_valid if split_pca_plot_spec is not None else None
                ),
                fit_scope_name=shared_fit_scope,
                axis_limits=shared_axis_limits,
                overlay_spec=split_pca_plot_spec,
                overlay_embeddings=split_pca_overlay_embeddings,
            )

        if cfg.cebra.conditional == "discrete":
            label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}

            if cfg.evaluation.enable_plots:
                interactive_path = output_dir / "cebra_interactive_discrete.html"
                save_interactive_plot(
                    cebra_embeddings_full,
                    labels_full,
                    cfg.cebra.output_dim,
                    palette,
                    "Interactive CEBRA (Discrete)",
                    interactive_path,
                    overlay_spec=overlay_spec_full,
                    overlay_embeddings=overlay_embeddings_full,
                )
                if is_main_process and wandb.run is not None and interactive_path.exists():
                    vis_artifact = wandb.Artifact(
                        name=interactive_path.stem, type="evaluation"
                    )
                    vis_artifact.add_file(str(interactive_path))
                    wandb.log_artifact(vis_artifact)

                save_static_2d_plots(
                    cebra_embeddings_full,
                    labels_full,
                    palette,
                    "CEBRA Embeddings (Discrete)",
                    output_dir,
                    order,
                    cfg=cfg,
                    scope_name="full",
                    pca_projection=full_pca_projection,
                    overlay_spec=pca_plot_spec if full_pca_projection is not None else None,
                )
                if is_main_process and wandb.run is not None:
                    static_artifact = wandb.Artifact(
                        "cebra-static-plots", type="evaluation"
                    )
                    static_artifact.add_file(str(output_dir / "static_PCA_plot.png"))
                    static_artifact.add_file(str(output_dir / "static_UMAP_plot.png"))
                    wandb.log_artifact(static_artifact)

                train_viz_dir = output_dir / "visualizations_train"
                train_viz_dir.mkdir(parents=True, exist_ok=True)
                interactive_train = train_viz_dir / "cebra_interactive_discrete_train.html"
                save_interactive_plot(
                    cebra_train_embeddings,
                    labels_train,
                    cfg.cebra.output_dim,
                    palette,
                    "Interactive CEBRA (Discrete, train only)",
                    interactive_train,
                    overlay_spec=overlay_spec_split,
                    overlay_embeddings=overlay_embeddings_split,
                )
                save_static_2d_plots(
                    cebra_train_embeddings,
                    labels_train,
                    palette,
                    "CEBRA Embeddings (Discrete, train only)",
                    train_viz_dir,
                    order,
                    cfg=cfg,
                    log_to_wandb=False,
                    scope_name="train",
                    pca_projection=train_pca_projection,
                    overlay_spec=(
                        split_pca_plot_spec if train_pca_projection is not None else None
                    ),
                )

                val_viz_dir.mkdir(parents=True, exist_ok=True)
                interactive_valid = val_viz_dir / "cebra_interactive_discrete_valid.html"
                save_interactive_plot(
                    cebra_valid_embeddings,
                    labels_valid,
                    cfg.cebra.output_dim,
                    palette,
                    "Interactive CEBRA (Discrete, validation only)",
                    interactive_valid,
                    overlay_spec=overlay_spec_split,
                    overlay_embeddings=overlay_embeddings_split,
                )
                save_static_2d_plots(
                    cebra_valid_embeddings,
                    labels_valid,
                    palette,
                    "CEBRA Embeddings (Discrete, validation only)",
                    val_viz_dir,
                    order,
                    cfg=cfg,
                    log_to_wandb=False,
                    scope_name="valid",
                    pca_projection=valid_pca_projection,
                    overlay_spec=(
                        split_pca_plot_spec if valid_pca_projection is not None else None
                    ),
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
            if is_main_process and wandb.run is not None:
                wandb.log({"knn_accuracy": accuracy})
            report_path = output_dir / "classification_report.json"
            pd.Series(report).to_json(report_path, indent=4)
            if is_main_process and wandb.run is not None:
                report_artifact = wandb.Artifact(
                    name=report_path.stem, type="evaluation"
                )
                report_artifact.add_file(str(report_path))
                wandb.log_artifact(report_artifact)

        elif cfg.cebra.conditional == "none":
            if cfg.evaluation.enable_plots:
                interactive_path = output_dir / "None.html"
                save_interactive_plot(
                    embeddings=cebra_embeddings_full,
                    text_labels=labels_full,
                    output_dim=cfg.cebra.output_dim,
                    palette=None,
                    title="Interactive CEBRA (None - Colored by Valence)",
                    output_path=interactive_path,
                )
                _log_wandb_artifact_if_exists(
                    interactive_path,
                    is_main_process=is_main_process,
                    artifact_type="evaluation",
                )

                train_viz_dir = output_dir / "visualizations_train"
                train_viz_dir.mkdir(parents=True, exist_ok=True)
                interactive_train = train_viz_dir / "None_train.html"
                save_interactive_plot(
                    embeddings=cebra_train_embeddings,
                    text_labels=[labels_full[i] for i in idx_train],
                    output_dim=cfg.cebra.output_dim,
                    palette=None,
                    title="Interactive CEBRA (None - Train, Colored by Valence)",
                    output_path=interactive_train,
                )

                val_viz_dir = output_dir / "visualizations_valid"
                val_viz_dir.mkdir(parents=True, exist_ok=True)
                interactive_valid = val_viz_dir / "None_valid.html"
                save_interactive_plot(
                    embeddings=cebra_valid_embeddings,
                    text_labels=[labels_full[i] for i in idx_valid],
                    output_dim=cfg.cebra.output_dim,
                    palette=None,
                    title="Interactive CEBRA (None - Validation, Colored by Valence)",
                    output_path=interactive_valid,
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
            if is_main_process and wandb.run is not None:
                wandb.log({"knn_regression_mse": mse, "knn_regression_r2": r2})

        probe_cfg = getattr(cfg.evaluation, "local_linearity_probe", None)
        if probe_cfg is not None and getattr(probe_cfg, "enabled", False):
            print("\n--- Running Local Linearity Probe ---")
            run_local_linearity_probe(
                learned_embeddings=cebra_embeddings_full,
                original_embeddings=X_vectors,
                output_dir=output_dir,
                neighbors=probe_cfg.neighbors,
                sample_size=probe_cfg.sample_size,
                random_state=probe_cfg.random_state,
                ridge_alpha=probe_cfg.ridge_alpha,
                enable_plots=cfg.evaluation.enable_plots,
                store_scores=getattr(probe_cfg, "store_scores", False),
                log_to_wandb=is_main_process,
            )

        print("\n--- Visualization Stage Complete ---")
    finally:
        if is_main_process and wandb.run is not None:
            wandb.finish()
