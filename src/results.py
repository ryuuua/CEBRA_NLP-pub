# results.py
import matplotlib

matplotlib.use("Agg")  # Use Agg backend for headless environments


import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
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

_CUML_AVAILABLE = False
try:
    import cupy as cp
    from cuml.decomposition import PCA as cuPCA
    from cuml.manifold import UMAP as cuUMAP
    from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
    from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor

    _CUML_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError):
    cp = None  # type: ignore[assignment]
    cuPCA = cuUMAP = cuKNeighborsClassifier = cuKNeighborsRegressor = None

_FAISS_AVAILABLE = False
_FAISS_GPU_AVAILABLE = False
try:
    import faiss  # type: ignore[assignment]

    _FAISS_AVAILABLE = True
    _FAISS_GPU_AVAILABLE = hasattr(faiss, "StandardGpuResources")
except (ImportError, ModuleNotFoundError, OSError):
    # OSError can be raised when GPU builds are present but incompatible with the
    # installed CUDA runtime. Treat this the same as faiss being unavailable so the
    # rest of the pipeline can gracefully fall back to scikit-learn implementations.
    faiss = None  # type: ignore[assignment]


def clear_cuda_cache() -> None:
    """Clear the CUDA cache if running on a GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_env_flag(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _should_use_cuml(cfg: Optional[AppConfig] = None, override: Optional[bool] = None) -> bool:
    """Decide whether to use cuML-backed implementations."""
    if not _CUML_AVAILABLE:
        return False
    if override is not None:
        return override

    if _read_env_flag("CEBRA_DISABLE_CUML") is True:
        return False
    if _read_env_flag("CEBRA_FORCE_CUML") is True:
        return True

    if cfg is not None:
        device = getattr(cfg, "device", "")
        if isinstance(device, str) and device.lower().startswith("cuda"):
            return True

    return torch.cuda.is_available()


def _to_gpu_array(array):
    if cp is None:  # pragma: no cover - defensive guard
        raise RuntimeError("cuML requested but CuPy is not available.")
    if isinstance(array, cp.ndarray):
        return array
    return cp.asarray(array)


def _to_cpu_numpy(array):
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def _resolve_faiss_backend(use_gpu: bool | None, *, strict: bool = False) -> tuple[bool, bool]:
    """Return (is_available, use_gpu) for FAISS based on requested policy."""
    if not _FAISS_AVAILABLE:
        return False, False
    gpu_possible = bool(_FAISS_GPU_AVAILABLE and torch.cuda.is_available())
    if use_gpu is True:
        if not gpu_possible:
            if strict:
                raise RuntimeError(
                    "FAISS GPU backend requested but no CUDA-enabled FAISS build is available."
                )
            return True, False
        return True, True
    if use_gpu is False:
        return True, False
    # use_gpu is None: pick GPU when possible
    return True, gpu_possible


def _faiss_knn_search(
    train_matrix: np.ndarray,
    query_matrix: np.ndarray,
    k: int,
    *,
    use_gpu: bool,
    gpu_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not _FAISS_AVAILABLE:
        raise RuntimeError("FAISS backend requested but `faiss` is not installed.")

    train32 = np.asarray(train_matrix, dtype=np.float32, order="C")
    query32 = np.asarray(query_matrix, dtype=np.float32, order="C")
    if train32.shape[1] != query32.shape[1]:
        raise ValueError(
            "Training and query embeddings must have the same dimensionality for FAISS."
        )

    index = faiss.IndexFlatL2(train32.shape[1])  # type: ignore[attr-defined]
    if use_gpu:
        if not _FAISS_GPU_AVAILABLE:
            raise RuntimeError(
                "FAISS GPU backend requested but `faiss-gpu` is not available."
            )
        res = faiss.StandardGpuResources()  # type: ignore[attr-defined]
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)  # type: ignore[attr-defined]

    index.add(train32)
    distances, indices = index.search(query32, k)
    return distances, indices


def _faiss_weighted_classification(
    neighbor_labels: np.ndarray,
    weights: np.ndarray,
    all_labels: np.ndarray,
) -> np.ndarray:
    """Compute weighted majority votes given neighbor labels and weights."""
    label_to_pos = {int(label): idx for idx, label in enumerate(all_labels)}
    num_queries, _ = neighbor_labels.shape
    scores = np.zeros((num_queries, len(all_labels)), dtype=np.float64)

    for row in range(num_queries):
        label_indices = [label_to_pos[int(lbl)] for lbl in neighbor_labels[row]]
        np.add.at(scores[row], label_indices, weights[row])

    predicted_indices = scores.argmax(axis=1)
    return all_labels[predicted_indices]


def _faiss_weighted_regression(
    neighbor_targets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return weighted average of neighbor targets."""
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum == 0.0, 1e-12, weight_sum)
    weighted_sum = np.sum(neighbor_targets * weights[..., None], axis=1)
    return weighted_sum / weight_sum


def _resolve_knn_backend(
    backend: str,
    use_gpu: bool | None,
) -> tuple[str, bool]:
    """Resolve k-NN backend selection logic. Returns (selected_backend, faiss_use_gpu)."""
    backend_choice = (backend or "auto").lower()
    prefer_gpu = torch.cuda.is_available() if use_gpu is None else bool(use_gpu)
    faiss_use_gpu = False

    if backend_choice == "auto":
        use_cuml = prefer_gpu and _should_use_cuml()
        if use_cuml:
            return "cuml", False
        faiss_available, faiss_use_gpu = _resolve_faiss_backend(
            None if use_gpu is None else prefer_gpu,
            strict=False,
        )
        return ("faiss" if faiss_available else "sklearn"), faiss_use_gpu
    elif backend_choice == "cuml":
        if _read_env_flag("CEBRA_DISABLE_CUML") is True:
            raise RuntimeError("cuML backend is disabled via CEBRA_DISABLE_CUML.")
        if not _CUML_AVAILABLE:
            raise RuntimeError("cuML backend requested but cuML is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("cuML backend requested but no CUDA device is available.")
        if use_gpu is False:
            raise RuntimeError("cuML backend requires GPU execution (use_gpu=True).")
        return "cuml", False
    elif backend_choice == "faiss":
        faiss_available, faiss_use_gpu = _resolve_faiss_backend(
            prefer_gpu,
            strict=True,
        )
        if not faiss_available:
            raise RuntimeError("FAISS backend requested but faiss is not installed.")
        return "faiss", faiss_use_gpu
    else:
        return "sklearn", False


def _resolve_umap_seed(cfg: Optional[AppConfig]) -> Optional[int]:
    """Resolve UMAP random seed from configuration."""
    if cfg is None:
        return None
    reproducibility = getattr(cfg, "reproducibility", None)
    if reproducibility is None:
        return None
    deterministic = bool(getattr(reproducibility, "deterministic", False))
    if not deterministic:
        return None
    seed = getattr(reproducibility, "seed", None)
    if seed is not None:
        return seed
    eval_cfg = getattr(cfg, "evaluation", None)
    if eval_cfg is not None:
        return getattr(eval_cfg, "random_state", None)
    return None


def _compute_faiss_weights(distances: np.ndarray) -> np.ndarray:
    """Compute inverse distance weights for FAISS k-NN, avoiding division by zero."""
    distances = np.sqrt(np.maximum(np.asarray(distances, dtype=np.float64), 0.0))
    return 1.0 / np.maximum(distances, 1e-12)


def save_interactive_plot(
    embeddings, text_labels, output_dim, palette, title, output_path: Path
):
    """Saves a 2D or 3D interactive plot as an HTML file and a static SVG image."""
    print(
        f"\nGenerating interactive visualization for {output_dim}-dimensional output..."
    )
    if output_dim not in (2, 3):
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
    log_to_wandb: Optional[bool] = None,
):
    """Generates and saves 2D static plots using PCA and UMAP."""
    print("Generating static 2D scatter plots using PCA and UMAP...")

    embeddings_np = np.asarray(embeddings)
    use_gpu_backend = _should_use_cuml(cfg)
    embeddings_gpu = None
    if use_gpu_backend:
        try:
            embeddings_gpu = _to_gpu_array(embeddings_np)
        except Exception as err:  # pragma: no cover - GPU initialisation specific
            print(
                f"cuML backend requested but moving embeddings to GPU failed ({err}); "
                "falling back to CPU implementations."
            )
            use_gpu_backend = False

    deterministic = cfg is not None and bool(
        getattr(getattr(cfg, "reproducibility", None), "deterministic", False)
    )
    umap_seed = _resolve_umap_seed(cfg)

    umap_base_kwargs = dict(n_components=2, n_neighbors=15, min_dist=0.1)
    if deterministic and umap_seed is not None:
        umap_base_kwargs["random_state"] = umap_seed

    X_pca = None
    variance_ratios = None
    if use_gpu_backend and cuPCA is not None and embeddings_gpu is not None:
        try:
            pca_gpu = cuPCA(n_components=2)
            X_pca = _to_cpu_numpy(pca_gpu.fit_transform(embeddings_gpu))
            variance_ratios = _to_cpu_numpy(pca_gpu.explained_variance_ratio_)
            print("PCA: using cuML implementation.")
        except Exception as err:  # pragma: no cover - GPU specific failure path
            print(f"cuML PCA failed ({err}); reverting to scikit-learn PCA.")

    if X_pca is None or variance_ratios is None:
        pca_model = PCA(n_components=2)
        X_pca = pca_model.fit_transform(embeddings_np)
        variance_ratios = pca_model.explained_variance_ratio_

    X_umap = None
    if use_gpu_backend and cuUMAP is not None and embeddings_gpu is not None:
        try:
            umap_gpu = cuUMAP(**umap_base_kwargs)
            X_umap = _to_cpu_numpy(umap_gpu.fit_transform(embeddings_gpu))
            print("UMAP: using cuML implementation.")
        except Exception as err:  # pragma: no cover - GPU specific failure path
            print("cuML UMAP failed "
                  f"({err}); reverting to umap-learn CPU implementation.")

    if X_umap is None:
        cpu_umap_kwargs = dict(umap_base_kwargs)
        cpu_umap_kwargs["n_jobs"] = 1 if deterministic else -1
        umap_model = umap.UMAP(**cpu_umap_kwargs)
        X_umap = umap_model.fit_transform(embeddings_np)

    print(
        "PCA explained variance ratios:",
        ", ".join(f"{ratio * 100:.2f}%" for ratio in variance_ratios),
    )

    if log_to_wandb is None:
        log_to_wandb = wandb.run is not None

    if log_to_wandb:
        wandb.log(
            {
                f"pca_variance_ratio_dim{i + 1}": float(ratio)
                for i, ratio in enumerate(variance_ratios)
            }
        )

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
    backend: str = "auto",
    use_gpu: bool | None = None,
    faiss_gpu_id: int = 0,
):
    """k-NN classification for discrete labels."""
    print("\nRunning k-NN Classification evaluation...")
    if use_gpu is True and not torch.cuda.is_available():
        raise RuntimeError("GPU execution requested for k-NN but CUDA is not available.")

    train_cpu = np.asarray(train_embeddings, dtype=np.float32)
    valid_cpu = np.asarray(valid_embeddings, dtype=np.float32)
    y_train_cpu = np.asarray(y_train)
    y_valid_cpu = np.asarray(y_valid)
    if y_train_cpu.ndim != 1:
        raise ValueError("y_train must be a 1D array for classification tasks.")
    y_train_cpu = y_train_cpu.astype(np.int64, copy=False)
    y_valid_cpu = y_valid_cpu.astype(np.int64, copy=False)

    selected_backend, faiss_use_gpu = _resolve_knn_backend(backend, use_gpu)

    y_pred = None
    knn_cpu_model: KNeighborsClassifier | None = None
    knn_backend_printed = False

    if selected_backend == "cuml" and cuKNeighborsClassifier is not None:
        try:
            knn_gpu = cuKNeighborsClassifier(
                n_neighbors=knn_neighbors,
                weights="distance",
            )
            knn_gpu.fit(_to_gpu_array(train_cpu), _to_gpu_array(y_train_cpu))
            y_pred = _to_cpu_numpy(knn_gpu.predict(_to_gpu_array(valid_cpu)))
            y_pred = y_pred.astype(y_valid_cpu.dtype, copy=False)
            print("k-NN Classification: using cuML backend.")
            knn_backend_printed = True
        except Exception as err:  # pragma: no cover - GPU specific failure path
            print(f"cuML k-NN classification failed ({err}); falling back to scikit-learn.")
            selected_backend = "sklearn"

    if y_pred is None and selected_backend == "faiss":
        print(
            f"k-NN Classification: using FAISS {'GPU' if faiss_use_gpu else 'CPU'} backend."
        )
        knn_backend_printed = True
        distances, indices = _faiss_knn_search(
            train_cpu,
            valid_cpu,
            knn_neighbors,
            use_gpu=faiss_use_gpu,
            gpu_id=faiss_gpu_id,
        )
        weights = _compute_faiss_weights(distances)
        neighbor_labels = y_train_cpu[indices.astype(np.int64, copy=False)]
        all_labels = np.array(sorted(label_map.keys()), dtype=np.int64)
        y_pred = _faiss_weighted_classification(neighbor_labels, weights, all_labels)

    if y_pred is None:
        if not knn_backend_printed:
            print("k-NN Classification: using scikit-learn backend.")
        knn_cpu_model = KNeighborsClassifier(n_neighbors=knn_neighbors, weights="distance")
        knn_cpu_model.fit(train_cpu, y_train_cpu)
        y_pred = knn_cpu_model.predict(valid_cpu)

    y_pred = np.asarray(y_pred)
    accuracy = accuracy_score(y_valid_cpu, y_pred)
    report = classification_report(
        y_valid_cpu,
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
        display_labels = list(label_map.values())
        if knn_cpu_model is not None:
            ConfusionMatrixDisplay.from_estimator(
                knn_cpu_model,
                valid_cpu,
                y_valid_cpu,
                display_labels=display_labels,
                cmap=plt.cm.Blues,
                ax=ax,
                xticks_rotation="vertical",
            )
        else:
            cm = confusion_matrix(y_valid_cpu, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical")
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
    backend: str = "auto",
    use_gpu: bool | None = None,
    faiss_gpu_id: int = 0,
):
    """k-NN regression for continuous labels (e.g., VAD)."""
    print("\nRunning k-NN Regression evaluation...")

    if use_gpu is True and not torch.cuda.is_available():
        raise RuntimeError("GPU execution requested for k-NN but CUDA is not available.")

    train_cpu = np.asarray(train_embeddings, dtype=np.float32)
    valid_cpu = np.asarray(valid_embeddings, dtype=np.float32)
    y_train_cpu = np.asarray(y_train, dtype=np.float32)
    y_valid_cpu = np.asarray(y_valid, dtype=np.float32)

    selected_backend, faiss_use_gpu = _resolve_knn_backend(backend, use_gpu)

    y_pred = None

    if selected_backend == "cuml" and cuKNeighborsRegressor is not None:
        try:
            knn_gpu = cuKNeighborsRegressor(
                n_neighbors=knn_neighbors,
                weights="distance",
            )
            knn_gpu.fit(_to_gpu_array(train_cpu), _to_gpu_array(y_train_cpu))
            y_pred = _to_cpu_numpy(knn_gpu.predict(_to_gpu_array(valid_cpu)))
            print("k-NN Regression: using cuML backend.")
        except Exception as err:  # pragma: no cover - GPU specific failure path
            print(f"cuML k-NN regression failed ({err}); falling back to scikit-learn.")
            selected_backend = "sklearn"

    if y_pred is None and selected_backend == "faiss":
        print(
            f"k-NN Regression: using FAISS {'GPU' if faiss_use_gpu else 'CPU'} backend."
        )
        distances, indices = _faiss_knn_search(
            train_cpu,
            valid_cpu,
            knn_neighbors,
            use_gpu=faiss_use_gpu,
            gpu_id=faiss_gpu_id,
        )
        weights = _compute_faiss_weights(distances)
        y_train_matrix = y_train_cpu
        if y_train_matrix.ndim == 1:
            y_train_matrix = y_train_matrix[:, None]
        neighbor_targets = y_train_matrix[indices.astype(np.int64, copy=False)]
        preds_matrix = _faiss_weighted_regression(neighbor_targets, weights)
        y_pred = preds_matrix[:, 0] if y_train_cpu.ndim == 1 else preds_matrix

    if y_pred is None:
        print("k-NN Regression: using scikit-learn backend.")
        knn = KNeighborsRegressor(n_neighbors=knn_neighbors, weights="distance")
        knn.fit(train_cpu, y_train_cpu)
        y_pred = knn.predict(valid_cpu)

    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_valid_cpu, y_pred)
    r2 = r2_score(y_valid_cpu, y_pred)

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
