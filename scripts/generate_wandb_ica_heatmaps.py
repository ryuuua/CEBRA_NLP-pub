import argparse
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

try:
    # Optional: used for kurtosis (non-Gaussianity) summary
    from scipy.stats import kurtosis  # type: ignore
except Exception:
    kurtosis = None  # scipy not available; we'll skip kurtosis table
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import generate_wandb_visualizations as viz_scripts  # noqa: E402
from src.utils import apply_reproducibility  # noqa: E402


def _determine_random_state(cfg, user_override: Optional[int]) -> Optional[int]:
    if user_override is not None:
        return user_override
    cfg_seeds: Sequence[Optional[int]] = (
        getattr(cfg.dataset, "shuffle_seed", None),
        getattr(getattr(cfg, "evaluation", None), "random_state", None),
    )
    for candidate in cfg_seeds:
        if candidate is not None:
            return candidate
    return None


def _apply_ica(
    embeddings: np.ndarray, *, n_components: Optional[int], random_state: Optional[int], max_iter: int
) -> Tuple[np.ndarray, FastICA]:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}.")

    max_components = min(embeddings.shape[0], embeddings.shape[1])
    if max_components <= 0:
        raise ValueError("Embeddings array is empty; cannot apply ICA.")

    if n_components is None:
        requested = max_components
    else:
        requested = n_components
    resolved_components = max(1, min(requested, max_components))

    if resolved_components < requested:
        print(
            f"[INFO] Requested {requested} components but only "
            f"{max_components} are feasible; using {resolved_components}."
        )

    ica = FastICA(
        n_components=resolved_components,
        random_state=random_state,
        max_iter=max_iter,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        transformed = ica.fit_transform(embeddings)
    return transformed, ica


def _figure_size(n_rows: int, n_cols: int) -> Tuple[float, float]:
    # Scale dimensions to keep large matrices legible without producing huge files.
    width = max(6.0, min(1.2 * n_cols, 18.0))
    height = max(4.0, min(0.2 * n_rows + 2.0, 18.0))
    return width, height


def _save_component_correlation_heatmap(
    transformed: np.ndarray,
    component_names: List[str],
    output_path: Path,
) -> None:
    if transformed.shape[1] <= 1:
        return
    corr = np.corrcoef(transformed, rowvar=False)
    width, height = _figure_size(len(component_names), len(component_names))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        center=0.0,
        cbar=True,
        xticklabels=component_names,
        yticklabels=component_names,
        square=True,
    )
    ax.set_title("Pairwise correlation between ICA components")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Wrote component correlation heatmap to {output_path}")


def _save_component_histograms(
    transformed: np.ndarray,
    component_names: List[str],
    output_path: Path,
    bins: int = 60,
) -> None:
    n_comp = transformed.shape[1]
    if n_comp == 0:
        return
    # Make a roughly square grid
    n_cols = int(np.ceil(np.sqrt(n_comp)))
    n_rows = int(np.ceil(n_comp / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.4))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        if i < n_comp:
            ax.hist(transformed[:, i], bins=bins)
            ax.set_title(component_names[i], fontsize=9)
            ax.set_yticks([])
        else:
            ax.axis("off")
    fig.suptitle("ICA component histograms (non-Gaussianity check)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Wrote component histograms to {output_path}")


def _save_kurtosis_table(
    transformed: np.ndarray,
    component_names: List[str],
    output_path: Path,
) -> None:
    # Save a simple JSON mapping: component -> excess kurtosis (Fisher)
    if kurtosis is None:
        print("[INFO] scipy not available; skipping kurtosis table.")
        return
    try:
        vals = kurtosis(transformed, axis=0, fisher=True, bias=False)
    except Exception as e:
        print(f"[WARN] Failed to compute kurtosis: {e}")
        return
    data = {name: float(v) for name, v in zip(component_names, vals)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"kurtosis_excess": data}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote kurtosis table to {output_path}")


def _save_sample_heatmap(
    transformed: np.ndarray,
    component_names: List[str],
    output_path: Path,
    *,
    max_samples: Optional[int],
) -> None:
    sample_count = transformed.shape[0]
    if sample_count == 0:
        print("[WARN] ICA produced zero samples; skipping sample heatmap.")
        return
    limit = sample_count if max_samples is None else min(sample_count, max_samples)
    heatmap_data = transformed[:limit, :]

    width, height = _figure_size(limit, len(component_names))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        cbar=True,
        xticklabels=component_names,
        yticklabels=False,
    )
    ax.set_xlabel("Independent components")
    if limit == sample_count:
        ax.set_ylabel("Samples (ordered as in dataset)")
    else:
        ax.set_ylabel(f"Samples (first {limit} rows)")
    ax.set_title("ICA-transformed embeddings")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Wrote sample heatmap to {output_path}")


def _save_label_heatmap(
    transformed: np.ndarray,
    labels: Sequence,
    order: Sequence,
    output_path: Path,
) -> None:
    if not labels:
        return
    # Accept strings or numbers; otherwise skip
    if not isinstance(labels[0], (str, int, np.integer)):
        return

    label_order = list(order) if order else sorted(set(labels))
    aggregated: List[np.ndarray] = []
    valid_labels: List[str] = []
    for label in label_order:
        mask = [idx for idx, target in enumerate(labels) if target == label]
        if not mask:
            continue
        aggregated.append(np.mean(transformed[mask, :], axis=0))
        valid_labels.append(str(label))

    if not aggregated:
        return

    heatmap_data = np.vstack(aggregated)
    component_names = [f"IC{i+1}" for i in range(heatmap_data.shape[1])]
    width, height = _figure_size(len(valid_labels), len(component_names))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="viridis",
        cbar=True,
        xticklabels=component_names,
        yticklabels=valid_labels,
    )
    ax.set_xlabel("Independent components")
    ax.set_ylabel("Label")
    ax.set_title("Mean ICA components per label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Wrote label heatmap to {output_path}")


def _generate_ica_visualizations(
    run_dir: Path,
    run_id: str,
    *,
    components: Optional[int],
    max_samples: Optional[int],
    random_state: Optional[int],
    max_iter: int,
) -> None:
    print(f"\nProcessing run {run_id} at {run_dir}")
    cfg = viz_scripts._load_cfg(run_dir)
    apply_reproducibility(cfg)

    texts, conditional_data, _time_indices, ids = viz_scripts.load_and_prepare_dataset(cfg)

    base_embeddings: Optional[np.ndarray] = None
    embeddings_path = run_dir / "cebra_embeddings.npy"
    if not embeddings_path.exists():
        base_embeddings = viz_scripts._prepare_base_embeddings(cfg, ids, texts)

    cebra_embeddings = viz_scripts._load_cebra_embeddings(run_dir, cfg, base_embeddings)
    if cebra_embeddings is None:
        return

    labels, _palette, order = viz_scripts._prepare_labels(cfg, conditional_data)
    resolved_state = _determine_random_state(cfg, random_state)
    ica_embeddings, ica_model = _apply_ica(
        cebra_embeddings, n_components=components, random_state=resolved_state, max_iter=max_iter
    )
    component_names = [f"IC{i+1}" for i in range(ica_embeddings.shape[1])]

    viz_dir = run_dir / "Linear_ICA"
    viz_dir.mkdir(parents=True, exist_ok=True)

    sample_heatmap_path = viz_dir / "cebra_ica_samples.png"
    _save_sample_heatmap(
        ica_embeddings,
        component_names,
        sample_heatmap_path,
        max_samples=max_samples,
    )

    label_heatmap_path = viz_dir / "cebra_ica_labels.png"
    _save_label_heatmap(
        ica_embeddings,
        labels,
        order,
        label_heatmap_path,
    )

    # Additional diagnostics/visualizations
    corr_path = viz_dir / "cebra_ica_component_corr.png"
    _save_component_correlation_heatmap(ica_embeddings, component_names, corr_path)

    hist_path = viz_dir / "cebra_ica_component_hist.png"
    _save_component_histograms(ica_embeddings, component_names, hist_path)

    kurtosis_path = viz_dir / "cebra_ica_kurtosis.json"
    _save_kurtosis_table(ica_embeddings, component_names, kurtosis_path)

    if hasattr(ica_model, "mixing_") and ica_model.mixing_ is not None:
        mixing_path = viz_dir / "cebra_ica_mixing.npy"
        np.save(mixing_path, ica_model.mixing_)
        print(f"[INFO] Saved ICA mixing matrix to {mixing_path}")

    components_path = viz_dir / "cebra_ica_components.npy"
    np.save(components_path, ica_model.components_)
    signals_path = viz_dir / "cebra_ica_embeddings.npy"
    np.save(signals_path, ica_embeddings)
    print(f"[INFO] Saved ICA signals to {signals_path}")
    print(f"[INFO] Saved ICA components to {components_path}")


def _drive(
    run_ids: Iterable[str],
    results_root: Path,
    *,
    components: Optional[int],
    max_samples: Optional[int],
    random_state: Optional[int],
    max_iter: int,
) -> None:
    for run_id in run_ids:
        matches = viz_scripts._find_run_dirs(results_root, run_id)
        if not matches:
            print(f"[WARN] No results directory found for run ID {run_id}")
            continue
        for run_dir in matches:
            _generate_ica_visualizations(
                run_dir,
                run_id,
                components=components,
                max_samples=max_samples,
                random_state=random_state,
                max_iter=max_iter,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply ICA to stored CEBRA embeddings and generate heatmaps."
    )
    parser.add_argument(
        "run_ids",
        nargs="+",
        help="One or more W&B run IDs (e.g. wwdjevwp).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing Hydra output folders.",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=None,
        help="Number of ICA components to compute (defaults to feasible max).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=512,
        help="Limit the number of samples shown in the heatmap (None keeps all).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Seed for ICA; defaults to dataset or evaluation seed when available.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum number of iterations for FastICA (default: 500).",
    )
    args = parser.parse_args()

    sns.set_theme(style="white")
    plt.switch_backend("Agg")

    _drive(
        args.run_ids,
        args.results_root.resolve(),
        components=args.components,
        max_samples=None if args.max_samples <= 0 else args.max_samples,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
