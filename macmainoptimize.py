import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import random
import wandb
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from src.config_schema import AppConfig
from hydra.core.hydra_config import HydraConfig
from src.data import load_and_prepare_dataset
from src.utils import get_embedding_cache_path, save_text_embedding, load_text_embedding
from src.embeddings import get_embeddings
from sklearn.model_selection import train_test_split
from src.results import (
    save_interactive_plot,
    run_knn_classification,
    run_knn_regression,
    run_consistency_check,
)
from dotenv import load_dotenv
import os
from cebra.integrations.sklearn.metrics import goodness_of_fit_score
import cebra
from src.cebra_trainer import normalize_model_architecture

load_dotenv()

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: AppConfig) -> None:
    default_cfg = OmegaConf.structured(AppConfig)
    cfg = OmegaConf.merge(default_cfg, cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.ddp.world_size = 1
    cfg.ddp.rank = 0
    cfg.ddp.local_rank = 0

    cfg.device = "mps" if torch.backends.mps.is_available() else "cpu"
    random.seed(cfg.evaluation.random_state)
    np.random.seed(cfg.evaluation.random_state)
    torch.manual_seed(cfg.evaluation.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.evaluation.random_state)

    output_dir = Path(HydraConfig.get().run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = HydraConfig.get().job.name

    # 1. Load Dataset
    print("\n--- Step 1: Loading dataset ---")
    if cfg.cebra.conditional == 'None':
        dataset_cfg = cfg.dataset
        dataset = load_dataset(
            path=dataset_cfg.hf_path,
            data_files=dataset_cfg.data_files
        )
        df = pd.concat([pd.DataFrame(dataset[s]) for s in dataset.keys()], ignore_index=True)
        df = df.dropna(subset=[dataset_cfg.text_column, 'V', 'A', 'D']).reset_index(drop=True)
        vad_columns = ['V', 'A', 'D']
        df_vad = df[vad_columns]
        conditional_data = df_vad.to_numpy(dtype=np.float32)
        texts = df[dataset_cfg.text_column].astype(str).tolist()
        time_indices = np.arange(len(texts))
    else:
        texts, conditional_data, time_indices = load_and_prepare_dataset(cfg)

    # 2. Get Text Embeddings
    print("\n--- Step 2: Generating text embeddings ---")
    embedding_cache_path = get_embedding_cache_path(cfg)
    X_vectors = load_text_embedding(embedding_cache_path)
    if X_vectors is None:
        X_vectors = get_embeddings(texts, cfg)
        save_text_embedding(X_vectors, embedding_cache_path)

    # Data Splitting
    print("\n--- Step 3: Splitting data ---")
    X_train, X_valid, conditional_train, conditional_valid, time_train, time_valid = train_test_split(
        X_vectors, conditional_data, time_indices,
        test_size=cfg.evaluation.test_size,
        random_state=cfg.evaluation.random_state,
        stratify=(conditional_data if cfg.cebra.conditional == 'discrete' else None)
    )

    labels_for_training = None if cfg.cebra.conditional == 'None' else conditional_train

    results_records = []
    step_counter = 0
    # Loop over hyperparameters and dimensions
    for dim in cfg.hpt.output_dims:
        for batch_size in cfg.hpt.batch_sizes:
            for lr in cfg.hpt.learning_rates:
                nested_run_name = f"dim_{dim}_bs{batch_size}_lr{lr}"
                with wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=nested_run_name, group=run_name):
                    print(f"\n=== Running pipeline for output_dim={dim}, batch_size={batch_size}, lr={lr} ===")
                    cfg.cebra.output_dim = dim
                    cfg.cebra.params["batch_size"] = batch_size
                    cfg.cebra.params["learning_rate"] = lr
                    dim_output_dir = output_dir / f"dim_{dim}_bs{batch_size}_lr{lr}"
                    dim_output_dir.mkdir(parents=True, exist_ok=True)
                    wandb.config.update({"output_dim": dim, "batch_size": batch_size, "learning_rate": lr})

                        # Train CEBRA model
                        print("\n--- Step 4: Training CEBRA model ---")
                        arch = normalize_model_architecture(cfg.cebra.model_architecture)
                        cebra_model = cebra.CEBRA(
                            model_architecture=arch,
                            output_dimension=dim,
                            max_iterations=cfg.cebra.max_iterations,
                            batch_size=batch_size,
                            learning_rate=lr,
                            conditional=None if cfg.cebra.conditional == 'None' else cfg.cebra.conditional,
                            device=cfg.device,
                        )
                        if labels_for_training is None:
                            cebra_model.fit(X_train)
                        else:
                            cebra_model.fit(X_train, labels_for_training)
                        model_path = dim_output_dir / "cebra_model.pt"
                        cebra_model.save(str(model_path))
                        model_artifact = wandb.Artifact(name=model_path.stem, type="model")
                        model_artifact.add_file(str(model_path))
                        wandb.log_artifact(model_artifact)

                        # Transform Data
                        print("\n--- Step 5: Transforming data ---")
                        cebra_embeddings_full = cebra_model.transform(X_vectors)
                        cebra_train_embeddings = cebra_model.transform(X_train)
                        cebra_valid_embeddings = cebra_model.transform(X_valid)

                        # Goodness of Fit
                        if cfg.cebra.conditional == 'None':
                            gof = goodness_of_fit_score(cebra_model, X_valid)
                        else:
                            gof = goodness_of_fit_score(cebra_model, X_valid, conditional_valid)
                        print(f"Goodness of fit (bits): {gof:.4f}")
                        wandb.log({"goodness_of_fit_bits": gof}, step=step_counter)

                        # Visualization & Evaluation
                        print("\n--- Step 6: Visualization and Evaluation ---")
                        if cfg.cebra.conditional == 'discrete':
                            label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}
                            text_labels_full = [label_map[l] for l in conditional_data]
                            palette = OmegaConf.to_container(cfg.dataset.visualization.emotion_colors, resolve=True)
                            order = OmegaConf.to_container(cfg.dataset.visualization.emotion_order, resolve=True)
                            interactive_path = dim_output_dir / "cebra_interactive_discrete.html"
                            save_interactive_plot(
                                cebra_embeddings_full,
                                text_labels_full,
                                dim,
                                palette,
                                "Interactive CEBRA (Discrete)",
                                interactive_path,
                            )
                            if interactive_path.exists():
                                vis_artifact = wandb.Artifact(
                                    name=interactive_path.stem, type="evaluation"
                                )
                                vis_artifact.add_file(str(interactive_path))
                                wandb.log_artifact(vis_artifact)
                            accuracy, report = run_knn_classification(
                                train_embeddings=cebra_train_embeddings,
                                valid_embeddings=cebra_valid_embeddings,
                                y_train=conditional_train,
                                y_valid=conditional_valid,
                                label_map=label_map,
                                output_dir=dim_output_dir,
                                knn_neighbors=cfg.evaluation.knn_neighbors,
                            )
                            wandb.log({"knn_accuracy": accuracy}, step=step_counter)
                            report_path = dim_output_dir / f"classification_report_dim_{dim}_bs{batch_size}_lr{lr}.json"
                            pd.Series(report).to_json(report_path, indent=4)
                            report_artifact = wandb.Artifact(name=report_path.stem, type="evaluation")
                            report_artifact.add_file(str(report_path))
                            wandb.log_artifact(report_artifact)
                        elif cfg.cebra.conditional == 'None':
                            valence_scores = conditional_data[:, 0]
                            interactive_path = dim_output_dir / "None.html"
                            save_interactive_plot(
                                embeddings=cebra_embeddings_full,
                                text_labels=valence_scores,
                                output_dim=dim,
                                palette=None,
                                title="Interactive CEBRA (None - Colored by Valence)",
                                output_path=interactive_path,
                            )
                            if interactive_path.exists():
                                vis_artifact = wandb.Artifact(
                                    name=interactive_path.stem, type="evaluation"
                                )
                                vis_artifact.add_file(str(interactive_path))
                                wandb.log_artifact(vis_artifact)
                            mse, r2 = run_knn_regression(
                                train_embeddings=cebra_train_embeddings,
                                valid_embeddings=cebra_valid_embeddings,
                                y_train=conditional_train,
                                y_valid=conditional_valid,
                                output_dir=dim_output_dir,
                                knn_neighbors=cfg.evaluation.knn_neighbors,
                            )
                            wandb.log({"knn_regression_mse": mse, "knn_regression_r2": r2}, step=step_counter)

                        train_consistency = valid_consistency = None
                        # Consistency Check
                        if cfg.consistency_check.enabled:
                            train_consistency, valid_consistency = run_consistency_check(
                                X_train,
                                labels_for_training,
                                X_valid,
                                cfg,
                                dim_output_dir,
                                step=step_counter,
                            )

                        results_records.append(
                            {
                                "output_dim": dim,
                                "batch_size": batch_size,
                                "learning_rate": lr,
                                "gof": gof,
                                "train_consistency": train_consistency,
                                "valid_consistency": valid_consistency,
                            }
                        )
                        step_counter += 1

        if results_records:
            results_df = pd.DataFrame(results_records)
            results_path = output_dir / "hyperparameter_search_results.csv"
            results_df.to_csv(results_path, index=False)
            results_artifact = wandb.Artifact(name=results_path.stem, type="evaluation")
            results_artifact.add_file(str(results_path))
            wandb.log_artifact(results_artifact)
            best_row = results_df.sort_values("gof", ascending=False).iloc[0]
            print(
                f"Best GoF found for dim={best_row.output_dim}, batch_size={best_row.batch_size}, lr={best_row.learning_rate}: {best_row.gof:.4f}"
            )
            with wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=f"{run_name}_best", group=run_name):
                wandb.config.update({
                    "best_dim": int(best_row.output_dim),
                    "best_batch_size": int(best_row.batch_size),
                    "best_learning_rate": float(best_row.learning_rate),
                })
                wandb.log({"best_gof": float(best_row.gof)})

        print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main()
