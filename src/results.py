# results.py
import matplotlib
matplotlib.use('Agg') # ← この2行をファイルの先頭に追加


import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import tempfile
import mlflow
from tqdm import tqdm
from .config_schema import AppConfig
from cebra.integrations.sklearn.metrics import consistency_score
from cebra import plot_consistency
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import gc
from .cebra_trainer import train_cebra, transform_cebra, load_cebra_model
# train_test_splitはこのファイルで使われていないため削除


def save_interactive_plot(embeddings, text_labels, output_dim, palette, title, output_path: Path):
    """Saves a 2D or 3D interactive plot as an HTML file and a static SVG image."""
    print(f"\nGenerating interactive visualization for {output_dim}-dimensional output...")
    if not (output_dim == 2 or output_dim == 3):
        print(f"Skipping interactive plot: output_dim is {output_dim}, but must be 2 or 3.")
        return

    plot_df = pd.DataFrame(embeddings[:, :output_dim])
    plot_df.columns = [f'Dim {i+1}' for i in range(output_dim)]
    plot_df['label'] = text_labels

    if output_dim == 2:
        fig = px.scatter(
            plot_df, x='Dim 1', y='Dim 2', color='label',
            hover_name='label', title=title, color_discrete_map=palette
        )
    else:  # output_dim == 3
        fig = px.scatter_3d(
            plot_df, x='Dim 1', y='Dim 2', z='Dim 3', color='label',
            hover_name='label', title=title, color_discrete_map=palette
        )

    fig.update_traces(marker=dict(size=2, opacity=0.6))

    # Adjust layout and camera for 3D plots
    if output_dim == 3:
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.7, y=1.7, z=0.5)
        )
        fig.update_layout(
            scene_camera=camera,
            margin=dict(l=0, r=0, b=0, t=40)
        )
    else:
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    # Save interactive HTML
    fig.write_html(str(output_path))
    print(f"Saved interactive {output_dim}D plot to {output_path}")

    # Save static SVG
    svg_path = output_path.with_suffix('.svg')
    try:
        fig.write_image(str(svg_path), width=1200, height=900)
        print(f"Saved static SVG image to {svg_path}")
    except Exception as e:
        print(f"\n--- SVG Export Warning ---")
        print(f"Could not save SVG image. Error: {e}")
        print("Please ensure the 'kaleido' package is installed (`pip install kaleido`)")
        print("--------------------------")

def save_static_2d_plots(embeddings, text_labels, palette, title_prefix, output_dir: Path, hue_order: list):
    """Generates and saves 2D static plots using PCA and UMAP."""
    print(f"Generating static 2D scatter plots using PCA and UMAP...")
    
    pca_model = PCA(n_components=2)
    #umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    # random_stateを削除し、n_jobs=-1（利用可能な全コアを使用）を追加
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_jobs=-1)
    X_pca = pca_model.fit_transform(embeddings)
    X_umap = umap_model.fit_transform(embeddings)

    for X_reduced, name in [(X_pca, "PCA"), (X_umap, "UMAP")]:
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=text_labels, palette=palette, s=10, hue_order=hue_order)
        #sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=text_labels, palette=palette, s=10)
        plt.title(f'{title_prefix} with {name}')
        plt.xlabel(f'{name} 1')
        plt.ylabel(f'{name} 2')
        plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        static_plot_file = output_dir / f"static_{name}_plot.png"
        plt.savefig(static_plot_file)
        plt.close()
        print(f"Saved static {name} plot to {static_plot_file}")

def run_knn_classification(train_embeddings, valid_embeddings, y_train, y_valid,
                           label_map, output_dir: Path, knn_neighbors):
    """k-NN classification for discrete labels."""
    print("\nRunning k-NN Classification evaluation...")
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors, weights='distance')
    knn.fit(train_embeddings, y_train)
    y_pred = knn.predict(valid_embeddings)
    
    accuracy = accuracy_score(y_valid, y_pred)
    report = classification_report(y_valid, y_pred, target_names=list(label_map.values()), output_dict=True, zero_division=0)
    
    print(f"k-NN Accuracy on Validation Set: {accuracy:.4f}")
    
    # --- Confusion Matrix ---
    cm_plot_file = output_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_estimator(
        knn, valid_embeddings, y_valid, display_labels=list(label_map.values()),
        cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical'
    )
    ax.set_title(f'Confusion Matrix (k-NN={knn_neighbors})')
    plt.tight_layout()
    plt.savefig(cm_plot_file)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_plot_file}")
    
    return accuracy, report

def run_knn_regression(train_embeddings, valid_embeddings, y_train, y_valid,
                       output_dir: Path, knn_neighbors):
    """k-NN regression for continuous labels (e.g., VAD)."""
    print("\nRunning k-NN Regression evaluation...")
    
    knn = KNeighborsRegressor(n_neighbors=knn_neighbors, weights='distance')
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
):

    print("\n--- Step 6: Running Consistency Check ---")
    check_cfg = cfg.consistency_check
    num_runs = check_cfg.num_runs

    # Disable persistent DataLoader workers to prevent accumulation across runs
    original_persistent = cfg.cebra.persistent_workers
    cfg.cebra.persistent_workers = False

    model_paths = []
    for i in tqdm(range(num_runs), desc="Training models for consistency check"):
        model = train_cebra(X_train, y_train, cfg, output_dir)
        tmp_file = Path(tempfile.gettempdir(), f"cebra_consistency_{i}.pt")
        torch.save(model.state_dict(), tmp_file)
        model_paths.append(tmp_file)
        # Release resources from this training run
        del model
        gc.collect()
        torch.cuda.empty_cache()

    train_embeddings = []
    valid_embeddings = []
    for tmp_file in tqdm(model_paths, desc="Transforming with saved models"):
        loaded_model = load_cebra_model(tmp_file, cfg, X_train.shape[1])
        train_embeddings.append(transform_cebra(loaded_model, X_train, cfg.device))
        valid_embeddings.append(transform_cebra(loaded_model, X_valid, cfg.device))
        tmp_file.unlink()
        del loaded_model
        gc.collect()
        torch.cuda.empty_cache()

    for name, embeddings in [("train", train_embeddings), ("valid", valid_embeddings)]:
        print(f"\nComputing consistency for {name} data...")
        scores, pairs, ids_runs = consistency_score(embeddings=embeddings, between="runs")
        
        mean_score = scores.mean()
        mlflow.log_metric(f"consistency_score_{name}", mean_score)
        print(f"Mean consistency score ({name}): {mean_score:.4f}")

        ax = plot_consistency(scores, pairs, ids_runs)
        plot_path = output_dir / f"consistency_plot_{name}.png"
        
        # Axesオブジェクト(ax)の親であるFigureオブジェクト(ax.figure)に対してsavefigを実行
        ax.figure.savefig(plot_path)
        
        # Figureを閉じる
        plt.close(ax.figure)
        mlflow.log_artifact(str(plot_path), "plots")

    # Restore original persistent_workers setting
    cfg.cebra.persistent_workers = original_persistent
