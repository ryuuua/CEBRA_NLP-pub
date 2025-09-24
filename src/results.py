# results.py
import matplotlib

matplotlib.use("Agg")  # Use Agg backend for headless environments


import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import os
import tempfile
import wandb
from tqdm import tqdm
from .config_schema import AppConfig
from cebra.integrations.sklearn.metrics import consistency_score
from cebra import plot_consistency
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import gc
import cebra
from .cebra_trainer import normalize_model_architecture

# train_test_splitはこのファイルで使われていないため削除


def clear_cuda_cache() -> None:
    """Clear the CUDA cache if running on a GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_interactive_plot(
    embeddings, text_labels, output_dim, palette, title, output_path: Path
):
    """Saves a 2D or 3D interactive plot as an HTML file and a static SVG image."""
    print(
        f"\nGenerating interactive visualization for {output_dim}-dimensional output..."
    )
    if not (output_dim == 2 or output_dim == 3):
        print(
            f"Skipping interactive plot: output_dim is {output_dim}, but must be 2 or 3."
        )
        return

    plot_df = pd.DataFrame(embeddings[:, :output_dim])
    plot_df.columns = [f"Dim {i+1}" for i in range(output_dim)]
    plot_df["label"] = text_labels

    if output_dim == 2:
        fig = px.scatter(
            plot_df,
            x="Dim 1",
            y="Dim 2",
            color="label",
            hover_name="label",
            title=title,
            color_discrete_map=palette,
        )
    else:  # output_dim == 3
        fig = px.scatter_3d(
            plot_df,
            x="Dim 1",
            y="Dim 2",
            z="Dim 3",
            color="label",
            hover_name="label",
            title=title,
            color_discrete_map=palette,
        )

    fig.update_traces(marker=dict(size=2, opacity=0.6))

    # Adjust layout and camera for 3D plots
    if output_dim == 3:
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.7, y=1.7, z=0.5),
        )
        fig.update_layout(scene_camera=camera, margin=dict(l=0, r=0, b=0, t=40))
    else:
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    # Save interactive HTML
    fig.write_html(str(output_path))
    print(f"Saved interactive {output_dim}D plot to {output_path}")

    # Save static SVG
    svg_path = output_path.with_suffix(".svg")
    try:
        fig.write_image(str(svg_path), width=1200, height=900)
        print(f"Saved static SVG image to {svg_path}")
    except Exception as e:
        print(f"\n--- SVG Export Warning ---")
        print(f"Could not save SVG image. Error: {e}")
        print(
            "Please ensure the 'kaleido' package is installed (`pip install kaleido`)"
        )
        print("--------------------------")


def save_static_2d_plots(
    embeddings,
    text_labels,
    palette,
    title_prefix,
    output_dir: Path,
    hue_order: list,
    cfg: Optional[AppConfig] = None,
):
    """Generates and saves 2D static plots using PCA and UMAP."""
    print("Generating static 2D scatter plots using PCA and UMAP...")

    pca_model = PCA(n_components=2)
    reproducibility = getattr(cfg, "reproducibility", None) if cfg is not None else None
    deterministic = bool(getattr(reproducibility, "deterministic", False))
    umap_seed = None
    if deterministic:
        umap_seed = getattr(reproducibility, "seed", None)
        if umap_seed is None and cfg is not None:
            eval_cfg = getattr(cfg, "evaluation", None)
            if eval_cfg is not None:
                umap_seed = getattr(eval_cfg, "random_state", None)

    umap_kwargs = dict(n_components=2, n_neighbors=15, min_dist=0.1, n_jobs=1 if deterministic else -1)
    if deterministic and umap_seed is not None:
        umap_kwargs["random_state"] = umap_seed

    umap_model = umap.UMAP(**umap_kwargs)
    X_pca = pca_model.fit_transform(embeddings)
    variance_ratios = pca_model.explained_variance_ratio_
    print(
        "PCA explained variance ratios:",
        ", ".join(f"{ratio * 100:.2f}%" for ratio in variance_ratios),
    )
    X_umap = umap_model.fit_transform(embeddings)

    for X_reduced, name in [(X_pca, "PCA"), (X_umap, "UMAP")]:
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            hue=text_labels,
            palette=palette,
            s=10,
            hue_order=hue_order,
        )
        plt.title(f"{title_prefix} with {name}")
        if name == "PCA":
            plt.xlabel(f"{name} 1 ({variance_ratios[0] * 100:.1f}%)")
            plt.ylabel(f"{name} 2 ({variance_ratios[1] * 100:.1f}%)")
        else:
            plt.xlabel(f"{name} 1")
            plt.ylabel(f"{name} 2")
        plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        static_plot_file = output_dir / f"static_{name}_plot.png"
        plt.savefig(static_plot_file)
        plt.close()
        print(f"Saved static {name} plot to {static_plot_file}")


def run_knn_classification(
    train_embeddings,
    valid_embeddings,
    y_train,
    y_valid,
    label_map,
    output_dir: Path,
    knn_neighbors,

    enable_plots: bool = True,

):
    """k-NN classification for discrete labels."""
    print("\nRunning k-NN Classification evaluation...")
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors, weights="distance")
    knn.fit(train_embeddings, y_train)
    y_pred = knn.predict(valid_embeddings)

    accuracy = accuracy_score(y_valid, y_pred)
    report = classification_report(
        y_valid,
        y_pred,
        target_names=list(label_map.values()),
        output_dict=True,
        zero_division=0,
    )

    print(f"k-NN Accuracy on Validation Set: {accuracy:.4f}")

    # --- Confusion Matrix ---
    if enable_plots:
        cm_plot_file = output_dir / "confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_estimator(
            knn,
            valid_embeddings,
            y_valid,
            display_labels=list(label_map.values()),
            cmap=plt.cm.Blues,
            ax=ax,
            xticks_rotation="vertical",
        )
        ax.set_title(f"Confusion Matrix (k-NN={knn_neighbors})")
        plt.tight_layout()
        plt.savefig(cm_plot_file)
        plt.close(fig)
        print(f"Saved confusion matrix to {cm_plot_file}")

    return accuracy, report


def run_knn_regression(
    train_embeddings,
    valid_embeddings,
    y_train,
    y_valid,
    output_dir: Path,
    knn_neighbors,
):
    """k-NN regression for continuous labels (e.g., VAD)."""
    print("\nRunning k-NN Regression evaluation...")

    knn = KNeighborsRegressor(n_neighbors=knn_neighbors, weights="distance")
    knn.fit(train_embeddings, y_train)
    y_pred = knn.predict(valid_embeddings)

    mse = mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)

    print(f"k-NN Regression MSE on Validation Set: {mse:.4f}")
    print(f"k-NN Regression R2 Score on Validation Set: {r2:.4f}")

    # 結果を辞書として保存
    report = {"mean_squared_error": mse, "r2_score": r2}
    report_path = output_dir / "regression_report.json"
    pd.Series(report).to_json(report_path, indent=4)

    return mse, r2


def run_consistency_check(
    X_train,
    y_train,
    X_valid,
    cfg: AppConfig,
    output_dir: Path,
    y_valid=None,

    dataset_embeddings=None,
    embeddings_list=None,
    labels_list=None,

    dataset_ids=None,
    enable_plots: bool = True,
    step: int | None = None,
    log_to_wandb: bool | None = None,
):

    print("\n--- Step 6: Running Consistency Check ---")
    check_cfg = cfg.consistency_check

    if log_to_wandb is None:
        log_to_wandb = wandb.run is not None

    # Between-datasets consistency
    if check_cfg.mode == "datasets":
        if dataset_embeddings is None or labels_list is None:
            raise ValueError(
                "dataset_embeddings and labels_list must be provided when mode='datasets'"
            )
        scores, pairs, ids_runs = consistency_score(
            embeddings=dataset_embeddings,
            labels=labels_list,
            dataset_ids=dataset_ids,
            between="datasets",
        )

        mean_score = scores.mean()
        if log_to_wandb:
            wandb.log({"consistency_score_datasets": mean_score}, step=step)
        print(f"Mean consistency score (datasets): {mean_score:.4f}")

        if enable_plots:
            ax = plot_consistency(scores, pairs, ids_runs)
            plot_path = output_dir / "consistency_plot_datasets.png"
            ax.figure.savefig(plot_path)
            plt.close(ax.figure)
            if log_to_wandb:
                wandb.save(str(plot_path))

        return mean_score, None

    num_runs = check_cfg.num_runs

    # Disable persistent DataLoader workers to prevent accumulation across runs
    original_persistent = cfg.cebra.persistent_workers
    cfg.cebra.persistent_workers = False

    model_paths = []
    for i in tqdm(range(num_runs), desc="Training models for consistency check"):
        arch = normalize_model_architecture(cfg.cebra.model_architecture)
        model = cebra.CEBRA(
            model_architecture=arch,
            output_dimension=cfg.cebra.output_dim,
            max_iterations=cfg.cebra.max_iterations,
            batch_size=cfg.cebra.params.get("batch_size", 512),
            learning_rate=cfg.cebra.params.get("learning_rate", 1e-3),
            conditional=(
                None if cfg.cebra.conditional == "None" else cfg.cebra.conditional
            ),
            device=cfg.device,
        )
        if y_train is None:
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pt",
            prefix=f"cebra_consistency_{os.getpid()}_{i}_",
        ) as tmp:
            tmp_file = Path(tmp.name)
        model.save(str(tmp_file))
        model_paths.append(tmp_file)
        del model
        gc.collect()
        clear_cuda_cache()

    train_embeddings = []
    valid_embeddings = []
    for tmp_file in tqdm(model_paths, desc="Transforming with saved models"):
        loaded_model = None
        try:
            loaded_model = cebra.CEBRA.load(str(tmp_file))
            train_embeddings.append(loaded_model.transform(X_train))
            valid_embeddings.append(loaded_model.transform(X_valid))
        finally:
            if loaded_model is not None:
                del loaded_model
            tmp_file.unlink(missing_ok=True)
            gc.collect()
            clear_cuda_cache()

    train_mean = valid_mean = None
    for name, embeddings in [("train", train_embeddings), ("valid", valid_embeddings)]:
        print(f"\nComputing consistency for {name} data...")
        scores, pairs, ids_runs = consistency_score(
            embeddings=embeddings, between="runs"
        )

        mean_score = scores.mean()
        if log_to_wandb:
            wandb.log({f"consistency_score_{name}": mean_score}, step=step)
        print(f"Mean consistency score ({name}): {mean_score:.4f}")
        if name == "train":
            train_mean = mean_score
        else:
            valid_mean = mean_score

        if enable_plots:
            ax = plot_consistency(scores, pairs, ids_runs)
            plot_path = output_dir / f"consistency_plot_{name}.png"

            # Axesオブジェクト(ax)の親であるFigureオブジェクト(ax.figure)に対してsavefigを実行
            ax.figure.savefig(plot_path)

            # Figureを閉じる
            plt.close(ax.figure)
            if log_to_wandb:
                wandb.save(str(plot_path))

    if (
        embeddings_list is not None
        and labels_list is not None
        and dataset_ids is not None
    ):
        print("\nComputing consistency across datasets...")
        scores, pairs, ids_datasets = consistency_score(
            embeddings=embeddings_list,
            labels=labels_list,
            dataset_ids=dataset_ids,
            between="datasets",
        )

        dataset_mean = scores.mean()
        if log_to_wandb:
            wandb.log({"consistency_score_datasets": dataset_mean}, step=step)
        print(f"Mean consistency score (datasets): {dataset_mean:.4f}")

        if enable_plots:
            ax = plot_consistency(scores, pairs, ids_datasets)
            plot_path = output_dir / "consistency_plot_datasets.png"

            ax.figure.savefig(plot_path)
            plt.close(ax.figure)
            if log_to_wandb:
                wandb.save(str(plot_path))

    # Restore original persistent_workers setting
    cfg.cebra.persistent_workers = original_persistent
    return train_mean, valid_mean
