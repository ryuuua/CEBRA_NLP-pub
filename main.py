import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import wandb
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from src.config_schema import AppConfig
from hydra.core.hydra_config import HydraConfig
from src.data import load_and_prepare_dataset
from src.utils import get_embedding_cache_path, save_text_embedding, load_text_embedding
from src.embeddings import get_embeddings
from src.cebra_trainer import (
    train_cebra,
    save_cebra_model,
    transform_cebra,
    normalize_model_architecture,
)
from sklearn.model_selection import train_test_split
from src.results import (save_interactive_plot, save_static_2d_plots,
                         run_knn_classification, run_knn_regression,
                         run_consistency_check)
from dotenv import load_dotenv
import os
import torch.distributed as dist

# .envファイルから環境変数を読み込む
load_dotenv()

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: AppConfig) -> None:
    default_cfg = OmegaConf.structured(AppConfig)
    cfg = OmegaConf.merge(default_cfg, cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.cebra.conditional = cfg.cebra.conditional.lower()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.ddp.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.ddp.rank = int(os.environ.get("RANK", 0))
    cfg.ddp.local_rank = local_rank
    is_main_process = cfg.ddp.rank == 0
    if cfg.ddp.world_size > 1:
        if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
            print(
                "Warning: WORLD_SIZE > 1 but RANK or LOCAL_RANK not set. "
                "Distributed training may be misconfigured."
            )
        else:
            dist.init_process_group(
                backend="nccl", rank=cfg.ddp.rank, world_size=cfg.ddp.world_size
            )
            torch.cuda.set_device(local_rank)
            if torch.cuda.is_available():
                cfg.device = f"cuda:{local_rank}"
    else:
        if torch.cuda.is_available():
            cfg.device = "cuda"
    output_dir = Path(HydraConfig.get().run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:

        # Initialize Weights & Biases
        run_name = HydraConfig.get().job.name
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        run_id = run.id
        print(f"W&B Run Name: {run_name}, Run ID: {run_id}")
        (output_dir / "wandb_run_id.txt").write_text(run_id)

        arch = normalize_model_architecture(cfg.cebra.model_architecture)
        cfg.cebra.model_architecture = arch
        wandb.config.update(
            {
                "cebra_output_dim": cfg.cebra.output_dim,
                "cebra_max_iterations": cfg.cebra.max_iterations,
                "cebra_conditional": cfg.cebra.conditional,
                "cebra_model_architecture": arch,
            }
        )

        # --- 1. Load Dataset ---
        print("\n--- Step 1: Loading dataset ---")
        # 'conditional_data' という変数名に統一
        if cfg.cebra.conditional == 'none':
            dataset_cfg = cfg.dataset
            # データセットのソースに応じて読み込み
            source = dataset_cfg.source
            if source == "hf":
                dataset = load_dataset(dataset_cfg.hf_path)
            elif source == "csv":
                dataset = load_dataset("csv", data_files=dataset_cfg.data_files)
            elif source == "kaggle":
                kaggle_path = Path(cfg.paths.kaggle_data_dir) / dataset_cfg.data_files
                dataset = load_dataset("csv", data_files=str(kaggle_path))
            else:
                raise ValueError(f"Unsupported dataset source: {source}")
            df = pd.concat([pd.DataFrame(dataset[s]) for s in dataset.keys()], ignore_index=True)

            # --- VAD列を直接使用 ---
            df = df.dropna(subset=[dataset_cfg.text_column, 'V', 'A', 'D']).reset_index(drop=True)

            vad_columns = ['V', 'A', 'D']
            df_vad = df[vad_columns]
            conditional_data = df_vad.to_numpy(dtype=np.float32)

            texts = df[dataset_cfg.text_column].astype(str).tolist()
            time_indices = np.arange(len(texts))
        else:
            texts, conditional_data, time_indices = load_and_prepare_dataset(cfg)

        # --- 2. Get Text Embeddings ---
        print("\n--- Step 2: Generating text embeddings ---")
        # (このセクションは変更なし)
        embedding_cache_path = get_embedding_cache_path(cfg)
        X_vectors = load_text_embedding(embedding_cache_path)
        if X_vectors is None:
            X_vectors = get_embeddings(texts, cfg)
            save_text_embedding(X_vectors, embedding_cache_path)

        # --- Data Splitting ---
        print("\n--- Step 3: Splitting data ---")
        X_train, X_valid, conditional_train, conditional_valid, time_train, time_valid = train_test_split(
            X_vectors, conditional_data, time_indices,
            test_size=cfg.evaluation.test_size,
            random_state=cfg.evaluation.random_state,
            stratify=(conditional_data if cfg.cebra.conditional == 'discrete' else None)
        )

        # --- 4. Train CEBRA ---
        print("\n--- Step 4: Training CEBRA model ---")

        labels_for_training = (
            None if cfg.cebra.conditional == "none" else conditional_train
        )
        cebra_model = train_cebra(X_train, labels_for_training, cfg, output_dir)
        model_path = save_cebra_model(cebra_model, output_dir)


        model_artifact = wandb.Artifact(name=model_path.stem, type="model")
        model_artifact.add_file(str(model_path))
        wandb.log_artifact(model_artifact)
        
        # --- 5. Transform Data ---
        print("\n--- Step 5: Transforming data with trained CEBRA model ---")
        cebra_embeddings_full = transform_cebra(cebra_model, X_vectors, cfg.device)
        cebra_train_embeddings = transform_cebra(cebra_model, X_train, cfg.device)
        cebra_valid_embeddings = transform_cebra(cebra_model, X_valid, cfg.device)

        # --- 6. Visualization & Evaluation ---
        print("\n--- Step 6: Visualization and Evaluation ---")
    
        # ★★★ ここからが具体的な分岐ロジック ★★★
        if cfg.cebra.conditional == 'discrete':
            # [DISCRETE CASE]
            print("Running discrete evaluation and visualization...")
            label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}
            text_labels_full = [label_map[l] for l in conditional_data]
            palette = OmegaConf.to_container(cfg.dataset.visualization.emotion_colors, resolve=True)
            order = OmegaConf.to_container(cfg.dataset.visualization.emotion_order, resolve=True)

            # 可視化
            if cfg.evaluation.enable_plots:
                interactive_path = output_dir / "cebra_interactive_discrete.html"
                save_interactive_plot(
                    cebra_embeddings_full,
                    text_labels_full,
                    cfg.cebra.output_dim,
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
                save_static_2d_plots(cebra_embeddings_full, text_labels_full, palette, "CEBRA Embeddings (Discrete)", output_dir, order)

                static_artifact = wandb.Artifact("cebra-static-plots", type="evaluation")
                static_artifact.add_file(str(output_dir / "static_PCA_plot.png"))
                static_artifact.add_file(str(output_dir / "static_UMAP_plot.png"))
                wandb.log_artifact(static_artifact)

            # 評価
            accuracy, report = run_knn_classification(
                train_embeddings=cebra_train_embeddings, valid_embeddings=cebra_valid_embeddings,
                y_train=conditional_train, y_valid=conditional_valid,
                label_map=label_map, output_dir=output_dir, knn_neighbors=cfg.evaluation.knn_neighbors,
                enable_plots=cfg.evaluation.enable_plots
            )
            wandb.log({"knn_accuracy": accuracy})
            report_path = output_dir / "classification_report.json"
            pd.Series(report).to_json(report_path, indent=4)
            report_artifact = wandb.Artifact(name=report_path.stem, type="evaluation")
            report_artifact.add_file(str(report_path))
            wandb.log_artifact(report_artifact)

        elif cfg.cebra.conditional == 'none':
            # [None CASE]
            print("Running None evaluation and visualization...")
        
            # 可視化 (Valenceスコアで色付け)
            # conditional_dataはVADのNumpy配列
            valence_scores = conditional_data[:, 0]
            if cfg.evaluation.enable_plots:
                interactive_path = output_dir / "None.html"
                save_interactive_plot(
                    embeddings=cebra_embeddings_full, text_labels=valence_scores,
                    output_dim=cfg.cebra.output_dim, palette=None, # 連続値なのでplotlyが自動でカラースケールを適用
                    title="Interactive CEBRA (None - Colored by Valence)",
                    output_path=interactive_path
                )
                if interactive_path.exists():
                    vis_artifact = wandb.Artifact(
                        name=interactive_path.stem, type="evaluation"
                    )
                    vis_artifact.add_file(str(interactive_path))
                    wandb.log_artifact(vis_artifact)
            # 注意: 連続値の場合、カテゴリ別の静的プロットはそのままでは適用できない

            # 評価
            mse, r2 = run_knn_regression(
                train_embeddings=cebra_train_embeddings, valid_embeddings=cebra_valid_embeddings,
                y_train=conditional_train, y_valid=conditional_valid,
                output_dir=output_dir, knn_neighbors=cfg.evaluation.knn_neighbors
            )
            wandb.log({"knn_regression_mse": mse, "knn_regression_r2": r2})

        # --- 7. Consistency Check ---
        if cfg.consistency_check.enabled:
            print("\n--- Step 7: Running Consistency Check ---")
            run_consistency_check(
                X_train,
                labels_for_training,
                X_valid,
                cfg,
                output_dir,
                enable_plots=cfg.evaluation.enable_plots,
            )
    finally:
        if wandb.run is not None:
            wandb.finish()
        if cfg.ddp.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main()
