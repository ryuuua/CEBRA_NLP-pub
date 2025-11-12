import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

import wandb

from src.config_schema import AppConfig
from src.data import load_and_prepare_dataset
from src.embeddings import get_embeddings
from src.cebra_trainer import load_cebra_model, transform_cebra, normalize_model_architecture
from src.results import save_static_2d_plots, save_interactive_plot


def _resolve_device(cfg) -> str:
    import torch

    device = getattr(cfg, "device", None)
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_cfg_from_wandb_run(run: "wandb.apis.public.Run"):
    # Pull run config and merge into structured AppConfig to keep shape/types
    run_cfg_dict = dict(run.config)
    default_cfg = OmegaConf.structured(AppConfig)
    cfg = OmegaConf.merge(default_cfg, OmegaConf.create(run_cfg_dict))
    OmegaConf.set_struct(cfg, False)
    # Normalize model architecture naming (defensive)
    cfg.cebra.model_architecture = normalize_model_architecture(
        getattr(cfg.cebra, "model_architecture", "offset0-model")
    )
    # Device resolution
    cfg.device = _resolve_device(cfg)
    return cfg


def _download_model_from_artifact(
    entity: str,
    project: str,
    artifact_spec: str,
    output_dir: Path,
) -> Path:
    """Download a model artifact like "model_name:latest".

    Returns path to the downloaded .pt file.
    """
    api = wandb.Api()
    artifact_path = f"{entity}/{project}/{artifact_spec}"
    art = api.artifact(artifact_path)
    target_dir = output_dir / art.name.replace(":", "_")
    target_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path(art.download(root=str(target_dir)))
    # Heuristics: pick first .pt file
    candidates = list(art_dir.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .pt files found in artifact {artifact_path} (downloaded to {art_dir})."
        )
    return candidates[0]


def _download_model_from_run(
    entity: str,
    project: str,
    run_id: str,
    output_dir: Path,
) -> Tuple[Path, "wandb.apis.public.Run"]:
    """Download the newest 'model' artifact logged in the run.

    Returns (model_path, run_obj).
    """
    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}"
    run = api.run(run_path)

    # Prefer logged artifacts of type 'model'
    model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not model_artifacts:
        # Fallback: check if a run file named cebra_model.pt was saved directly
        for f in run.files():
            if f.name.endswith(".pt"):
                target_dir = output_dir / f"run_{run_id}_files"
                target_dir.mkdir(parents=True, exist_ok=True)
                local_path = Path(f.download(root=str(target_dir)))
                return local_path, run
        raise FileNotFoundError(
            "No model artifact (type='model') or .pt file found in the specified run."
        )

    # Sort artifacts by creation time, pick latest
    model_artifacts.sort(key=lambda a: a.created_at or 0, reverse=True)
    latest = model_artifacts[0]
    target_dir = output_dir / latest.name.replace(":", "_")
    target_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path(latest.download(root=str(target_dir)))
    candidates = list(art_dir.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .pt files found in model artifact {latest.name} (downloaded to {art_dir})."
        )
    return candidates[0], run


def _compute_pca_report(embeddings: np.ndarray, max_components: Optional[int] = None) -> pd.DataFrame:
    n_features = embeddings.shape[1]
    n_components = n_features if max_components is None else min(n_features, int(max_components))
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    ratios = pca.explained_variance_ratio_
    ev = pca.explained_variance_
    cum = np.cumsum(ratios)
    df = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1, dtype=int),
            "explained_variance": ev,
            "explained_variance_ratio": ratios,
            "cumulative_ratio": cum,
        }
    )
    return df


def _prepare_labels_text(
    conditional_mode: str,
    conditional_data: np.ndarray,
    cfg,
) -> Tuple[List, Optional[dict], Optional[List]]:
    """Return (text_labels, palette, order) for plotting.

    For continuous labels (mode 'none'), returns numeric labels and (None, None).
    """
    if conditional_mode == "discrete":
        # Handle {-1,1} -> {0,1}
        data = conditional_data
        if set(np.unique(data).tolist()) == {-1, 1}:
            data = np.array([0 if x == -1 else 1 for x in data], dtype=int)

        label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}
        text_labels = [label_map[int(i)] for i in data]

        palette = OmegaConf.to_container(
            cfg.dataset.visualization.emotion_colors, resolve=True
        )
        order = OmegaConf.to_container(
            cfg.dataset.visualization.emotion_order, resolve=True
        )
        return text_labels, palette, order
    else:
        # Use Valence for coloring if available
        if conditional_data.ndim == 2 and conditional_data.shape[1] >= 1:
            valence = conditional_data[:, 0].tolist()
            return valence, None, None
        return conditional_data.tolist(), None, None


def analyze(
    entity: str,
    project: str,
    run_id: Optional[str] = None,
    artifact: Optional[str] = None,
    outdir: Path = Path("wandb_analysis"),
    pca_max_components: Optional[int] = None,
    make_interactive: bool = False,
):
    outdir.mkdir(parents=True, exist_ok=True)

    if artifact:
        model_path = _download_model_from_artifact(entity, project, artifact, outdir)
        # Need the run to reconstruct cfg; require run_id in this case
        if not run_id:
            raise ValueError("--run-id is required when using --artifact to reconstruct the config.")
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
    else:
        if not run_id:
            raise ValueError("Provide either --artifact or --run-id (or both).")
        model_path, run = _download_model_from_run(entity, project, run_id, outdir)

    cfg = _build_cfg_from_wandb_run(run)

    # Load dataset and base embeddings
    texts, conditional_data, _time_idx, _ids = load_and_prepare_dataset(cfg)
    X_vectors = get_embeddings(texts, cfg)

    # Load the CEBRA model and transform
    try:
        # Try high-level CEBRA loader first (if the file was saved by CEBRA().save)
        import cebra

        loaded = cebra.CEBRA.load(str(model_path))
        cebra_embeddings_full = loaded.transform(X_vectors)
    except Exception:
        # Fall back to our state-dict based loader
        model = load_cebra_model(model_path, cfg, input_dimension=X_vectors.shape[1])
        cebra_embeddings_full = transform_cebra(model, X_vectors, cfg.device)

    # Prepare labels and palettes
    conditional_mode = getattr(cfg.cebra, "conditional", "none").lower()
    text_labels, palette, order = _prepare_labels_text(conditional_mode, np.asarray(conditional_data), cfg)

    # Plots + PCA report
    title_prefix = f"CEBRA Embeddings ({conditional_mode.capitalize()})"
    if conditional_mode == "discrete":
        save_static_2d_plots(
            cebra_embeddings_full,
            text_labels,
            palette,
            title_prefix,
            outdir,
            order,
            cfg=cfg,
            log_to_wandb=False,
        )
    else:
        # Simple PCA scatter for continuous values (no palette)
        pca2 = PCA(n_components=2)
        X_pca2 = pca2.fit_transform(cebra_embeddings_full)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 10))
        sc = plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=text_labels, cmap="viridis", s=10)
        plt.colorbar(sc, label="Valence")
        plt.title(f"{title_prefix} with PCA (colored by Valence)")
        plt.xlabel(f"PCA 1 ({pca2.explained_variance_ratio_[0] * 100:.1f}%)")
        plt.ylabel(f"PCA 2 ({pca2.explained_variance_ratio_[1] * 100:.1f}%)")
        plt.tight_layout()
        plt.savefig(outdir / "static_PCA_plot.png")
        plt.close()

    if make_interactive and cfg.cebra.output_dim in (2, 3):
        html_path = outdir / f"interactive_{conditional_mode}.html"
        save_interactive_plot(
            cebra_embeddings_full,
            text_labels,
            cfg.cebra.output_dim,
            palette,
            f"Interactive CEBRA ({conditional_mode})",
            html_path,
        )

    # PCA explained variance report (many components)
    pca_report = _compute_pca_report(cebra_embeddings_full, max_components=pca_max_components)
    pca_csv = outdir / "pca_explained_variance.csv"
    pca_json = outdir / "pca_explained_variance.json"
    pca_report.to_csv(pca_csv, index=False)
    pca_report.to_json(pca_json, orient="records", indent=2)

    print(f"Saved: {pca_csv}")
    print(f"Saved: {pca_json}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download a CEBRA model from W&B, reconstruct it, transform the dataset "
            "embeddings, and produce plots + PCA explained-variance report."
        )
    )
    parser.add_argument("--entity", required=True, help="W&B entity (username or team)")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument(
        "--run-id",
        help="W&B run ID to use for config reconstruction and model lookup",
    )
    parser.add_argument(
        "--artifact",
        help="Optional artifact spec like 'cebra_model:latest'. If provided, --run-id is still required to reconstruct the config.",
    )
    parser.add_argument(
        "--outdir",
        default="wandb_analysis",
        help="Output directory for artifacts and plots",
    )
    parser.add_argument(
        "--pca-max-components",
        type=int,
        default=None,
        help="Max PCA components in the explained-variance report (defaults to all output dims)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Also save an interactive 2D/3D HTML plot when output_dim is 2 or 3",
    )

    args = parser.parse_args()
    outdir = Path(args.outdir)
    analyze(
        entity=args.entity,
        project=args.project,
        run_id=args.run_id,
        artifact=args.artifact,
        outdir=outdir,
        pca_max_components=args.pca_max_components,
        make_interactive=bool(args.interactive),
    )


if __name__ == "__main__":
    main()

