import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Optional, Sequence, Tuple

from src.config_schema import AppConfig
from src.data import load_and_prepare_dataset
from src.embeddings import load_or_generate_embeddings
from src.plotting import prepare_plot_labels
from src.results import save_interactive_plot
from src.utils import apply_reproducibility


def _resolve_rng_seed(cfg: AppConfig) -> Optional[int]:
    repro = getattr(cfg, "reproducibility", None)
    dataset = getattr(cfg, "dataset", None)
    evaluation = getattr(cfg, "evaluation", None)
    for candidate in (
        getattr(repro, "seed", None),
        getattr(dataset, "shuffle_seed", None),
        getattr(evaluation, "random_state", None),
    ):
        if candidate is not None:
            return int(candidate)
    return None


def _fit_pca(embeddings: np.ndarray, cfg: AppConfig) -> Tuple[np.ndarray, np.ndarray]:
    analysis_cfg = cfg.pca_analysis
    max_possible = min(embeddings.shape[0], embeddings.shape[1])
    if max_possible == 0:
        raise ValueError("Embeddings are empty; cannot fit PCA.")
    requested = analysis_cfg.max_components
    n_components = max_possible if requested is None else min(requested, max_possible)
    if n_components < 2:
        print(
            "[WARN] PCA has fewer than 2 components; 2D/3D projections will be limited."
        )
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    return transformed, explained


def _determine_report_length(
    explained: np.ndarray, cfg: AppConfig, min_required: int
) -> int:
    threshold = getattr(cfg.pca_analysis, "residual_variance_threshold", None)
    floor = getattr(cfg.pca_analysis, "component_variance_floor", None)
    cumulative = np.cumsum(explained)
    residual = np.maximum(0.0, 1.0 - cumulative)

    if threshold is None or floor is None:
        return len(explained)

    for idx, ratio in enumerate(explained):
        if residual[idx] <= threshold and ratio < floor:
            return max(min_required, idx + 1)

    return len(explained)


def _build_variance_report(
    explained: np.ndarray, cfg: AppConfig, min_required: int
) -> pd.DataFrame:
    cumulative = np.cumsum(explained)
    residual = np.maximum(0.0, 1.0 - cumulative)
    report_length = _determine_report_length(explained, cfg, min_required)
    indices = np.arange(report_length)
    data = {
        "component": indices + 1,
        "explained_variance_ratio": explained[:report_length],
        "cumulative_explained_variance": cumulative[:report_length],
        "residual_variance": residual[:report_length],
    }
    return pd.DataFrame(data)


def _maybe_subsample(
    embeddings: np.ndarray,
    labels: Sequence,
    limit: Optional[int],
    seed: Optional[int],
) -> Tuple[np.ndarray, Sequence]:
    if limit is None or embeddings.shape[0] <= limit:
        return embeddings, labels
    rng = np.random.default_rng(seed)
    indices = rng.choice(embeddings.shape[0], size=limit, replace=False)
    subset_embeddings = embeddings[indices]
    subset_labels = [labels[i] for i in indices]
    return subset_embeddings, subset_labels


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def run(cfg: AppConfig) -> None:
    default_cfg = OmegaConf.structured(AppConfig)
    cfg = OmegaConf.merge(default_cfg, cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.cebra.conditional = cfg.cebra.conditional.lower()
    apply_reproducibility(cfg)

    output_dir = Path(HydraConfig.get().run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "pca_analysis"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Step 1: Loading dataset ---")
    texts, conditional_data, _time_indices, ids = load_and_prepare_dataset(cfg)

    print("\n--- Step 2: Loading embeddings (with cache) ---")
    embeddings = load_or_generate_embeddings(cfg, texts, ids)

    print("\n--- Step 3: Preparing labels for plotting ---")
    labels, palette, _order = prepare_plot_labels(cfg, conditional_data)

    print("\n--- Step 4: Running PCA on base embeddings ---")
    pca_scores, explained = _fit_pca(embeddings, cfg)
    requested_plot_dims = getattr(cfg.pca_analysis, "min_components_for_plots", 3)
    min_plot_dims = min(requested_plot_dims, pca_scores.shape[1])
    variance_report = _build_variance_report(explained, cfg, min_plot_dims)
    residual_threshold = cfg.pca_analysis.residual_variance_threshold * 100
    print(
        "Explained variance ratios (first components):",
        ", ".join(f"{ratio * 100:.2f}%" for ratio in explained[: min_plot_dims or 1]),
    )
    print(
        f"Report covers {len(variance_report)} components "
        f"(target residual variance ≤ {residual_threshold:.2f}%)."
    )

    scores_path = viz_dir / "pca_scores.npy"
    np.save(scores_path, pca_scores)
    report_path = viz_dir / "explained_variance.csv"
    variance_report.to_csv(report_path, index=False)
    print(f"Saved PCA scores to {scores_path}")
    print(f"Saved explained variance report to {report_path}")

    plot_limit = cfg.pca_analysis.plot_sample_limit
    rng_seed = _resolve_rng_seed(cfg)
    plot_embeddings, plot_labels = _maybe_subsample(
        pca_scores, labels, plot_limit, rng_seed
    )

    title_base = f"{cfg.dataset.name} · Embedding PCA"
    interactive_2d = viz_dir / "pca_embeddings_2d.html"
    interactive_3d = viz_dir / "pca_embeddings_3d.html"

    if plot_embeddings.shape[1] >= 2:
        save_interactive_plot(
            embeddings=plot_embeddings,
            text_labels=plot_labels,
            output_dim=2,
            palette=palette,
            title=f"{title_base} (2D)",
            output_path=interactive_2d,
        )
    else:
        print("[WARN] PCA produced fewer than 2 components; skipping 2D plot.")

    if plot_embeddings.shape[1] >= 3:
        save_interactive_plot(
            embeddings=plot_embeddings,
            text_labels=plot_labels,
            output_dim=3,
            palette=palette,
            title=f"{title_base} (3D)",
            output_path=interactive_3d,
        )
    else:
        print("[WARN] PCA produced fewer than 3 components; skipping 3D plot.")

    print("\n--- PCA analysis complete ---")


if __name__ == "__main__":
    run()
