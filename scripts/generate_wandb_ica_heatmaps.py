import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

try:
    # Optional: used for kurtosis (non-Gaussianity) summary
    from scipy.stats import kurtosis as scipy_kurtosis  # type: ignore
except Exception:
    scipy_kurtosis = None  # scipy not available; we'll skip kurtosis table

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import generate_wandb_visualizations as viz_scripts  # noqa: E402
from src.plotting import prepare_plot_labels  # noqa: E402
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


def _compute_excess_kurtosis(transformed: np.ndarray) -> np.ndarray:
    """Return Fisher excess kurtosis per component (higher magnitude => more non-Gaussian)."""
    if transformed.size == 0:
        return np.array([], dtype=float)
    if scipy_kurtosis is not None:
        try:
            vals = scipy_kurtosis(transformed, axis=0, fisher=True, bias=False)
            return np.asarray(vals, dtype=float)
        except Exception:
            pass

    centered = transformed - transformed.mean(axis=0, keepdims=True)
    variance = np.mean(centered**2, axis=0)
    safe_variance = np.where(variance <= 1e-12, np.nan, variance)
    fourth_moment = np.mean(centered**4, axis=0)
    kurtosis_vals = fourth_moment / (safe_variance**2) - 3.0
    kurtosis_vals = np.where(np.isnan(kurtosis_vals), 0.0, kurtosis_vals)
    return kurtosis_vals


def _compute_label_alignment(
    transformed: np.ndarray, labels: Sequence
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Return alignment score plus between/within variances and label means."""
    if not labels:
        return None, None, None, {}

    first = labels[0]
    if not isinstance(first, (str, int, np.integer)):
        return None, None, None, {}

    label_array = np.asarray([str(label) for label in labels])
    unique_labels = sorted(set(label_array.tolist()))
    if not unique_labels:
        return None, None, None, {}

    n_samples = transformed.shape[0]
    if n_samples == 0:
        return None, None, None, {}

    overall_mean = transformed.mean(axis=0)
    between = np.zeros(transformed.shape[1], dtype=float)
    within = np.zeros(transformed.shape[1], dtype=float)
    label_means: Dict[str, np.ndarray] = {}

    for label in unique_labels:
        mask = label_array == label
        count = int(mask.sum())
        if count == 0:
            continue
        comp_vals = transformed[mask]
        mean = comp_vals.mean(axis=0)
        label_means[label] = mean
        diff = mean - overall_mean
        between += count * diff**2
        centered = comp_vals - mean
        within += count * np.mean(centered**2, axis=0)

    if not label_means:
        return None, None, None, {}

    between /= max(n_samples, 1)
    within /= max(n_samples, 1)
    alignment = between / (within + 1e-12)
    return alignment, between, within, label_means


def _rank_components(
    kurtosis_vals: np.ndarray,
    label_alignment: Optional[np.ndarray],
    *,
    metric: str,
) -> Tuple[str, np.ndarray]:
    """Return metric actually used and sorted indices descending."""
    metric = metric.lower()
    if metric == "label_alignment" and label_alignment is not None:
        order = np.argsort(label_alignment)[::-1]
        return "label_alignment", order
    if metric == "kurtosis":
        pass  # fall through
    # Fallback: use absolute kurtosis
    if kurtosis_vals.size == 0:
        return "kurtosis", np.array([], dtype=int)
    order = np.argsort(np.abs(kurtosis_vals))[::-1]
    return "kurtosis", order


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
    *,
    values: Optional[np.ndarray] = None,
) -> None:
    # Save a simple JSON mapping: component -> excess kurtosis (Fisher)
    if values is None:
        if scipy_kurtosis is None:
            print("[INFO] scipy not available; skipping kurtosis table.")
            return
        try:
            values = scipy_kurtosis(transformed, axis=0, fisher=True, bias=False)
        except Exception as e:
            print(f"[WARN] Failed to compute kurtosis: {e}")
            return
    data = {name: float(v) for name, v in zip(component_names, values)}
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
    label_means: Dict[str, np.ndarray],
    component_names: Sequence[str],
    output_path: Path,
    *,
    order: Sequence,
    component_indices: Optional[Sequence[int]] = None,
    title: str = "Mean ICA components per label",
) -> None:
    if not label_means:
        return

    if order:
        label_order = [str(label) for label in order]
    else:
        label_order = sorted(label_means.keys())

    aggregated: List[np.ndarray] = []
    valid_labels: List[str] = []
    for label in label_order:
        means = label_means.get(label)
        if means is None:
            continue
        aggregated.append(means)
        valid_labels.append(label)

    if not aggregated:
        return

    heatmap_data = np.vstack(aggregated)
    if component_indices is not None:
        heatmap_data = heatmap_data[:, component_indices]
        selected_names = [component_names[idx] for idx in component_indices]
    else:
        selected_names = list(component_names)

    width, height = _figure_size(len(valid_labels), len(selected_names))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="viridis",
        cbar=True,
        xticklabels=selected_names,
        yticklabels=valid_labels,
    )
    ax.set_xlabel("Independent components")
    ax.set_ylabel("Label")
    ax.set_title(title)
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
    ranking_metric: str,
    top_k: int,
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

    labels, _palette, order = prepare_plot_labels(cfg, conditional_data)
    resolved_state = _determine_random_state(cfg, random_state)
    ica_embeddings, ica_model = _apply_ica(
        cebra_embeddings, n_components=components, random_state=resolved_state, max_iter=max_iter
    )
    component_names = [f"IC{i+1}" for i in range(ica_embeddings.shape[1])]

    viz_dir = run_dir / "Linear_ICA"
    viz_dir.mkdir(parents=True, exist_ok=True)

    kurtosis_vals = _compute_excess_kurtosis(ica_embeddings)
    (
        label_alignment_vals,
        between_var,
        within_var,
        label_means,
    ) = _compute_label_alignment(ica_embeddings, labels)
    metric_used, ranking_order = _rank_components(
        kurtosis_vals, label_alignment_vals, metric=ranking_metric
    )
    if ranking_metric.lower() == "label_alignment" and metric_used != "label_alignment":
        print(
            "[WARN] Label alignment ranking requested but labels are not discrete; "
            "falling back to kurtosis ranking."
        )
    if ranking_order.size and top_k > 0:
        limited_k = min(top_k, ranking_order.size)
        selected_indices = ranking_order[:limited_k]
    else:
        selected_indices = np.array([], dtype=int)

    sample_heatmap_path = viz_dir / "cebra_ica_samples.png"
    _save_sample_heatmap(
        ica_embeddings,
        component_names,
        sample_heatmap_path,
        max_samples=max_samples,
    )

    label_heatmap_path = viz_dir / "cebra_ica_labels.png"
    _save_label_heatmap(
        label_means,
        component_names,
        label_heatmap_path,
        order=order,
    )

    if selected_indices.size and label_means:
        topk_heatmap_path = viz_dir / f"cebra_ica_labels_top{selected_indices.size}_{metric_used}.png"
        _save_label_heatmap(
            label_means,
            component_names,
            topk_heatmap_path,
            order=order,
            component_indices=selected_indices.tolist(),
            title=f"Mean ICA components per label (top-{selected_indices.size} by {metric_used})",
        )

    # Additional diagnostics/visualizations
    corr_path = viz_dir / "cebra_ica_component_corr.png"
    _save_component_correlation_heatmap(ica_embeddings, component_names, corr_path)

    hist_path = viz_dir / "cebra_ica_component_hist.png"
    _save_component_histograms(ica_embeddings, component_names, hist_path)

    kurtosis_path = viz_dir / "cebra_ica_kurtosis.json"
    _save_kurtosis_table(
        ica_embeddings,
        component_names,
        kurtosis_path,
        values=kurtosis_vals,
    )

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

    report_path = viz_dir / "cebra_ica_report.json"
    component_report: List[dict] = []
    for idx, name in enumerate(component_names):
        entry: Dict[str, object] = {
            "index": idx,
            "name": name,
            "kurtosis": float(kurtosis_vals[idx]) if kurtosis_vals.size > idx else None,
        }
        if entry["kurtosis"] is not None:
            entry["abs_kurtosis"] = abs(entry["kurtosis"])
        if label_alignment_vals is not None:
            entry["label_alignment"] = float(label_alignment_vals[idx])
            entry["between_variance"] = float(between_var[idx])
            entry["within_variance"] = float(within_var[idx])
            entry["label_means"] = {
                label: float(means[idx]) for label, means in label_means.items()
            }
        component_report.append(entry)

    ranking_order_list = ranking_order.tolist() if ranking_order.size else []
    selected_list = selected_indices.tolist() if selected_indices.size else []
    report_payload = {
        "components": component_report,
        "metrics": {
            "kurtosis": [float(x) for x in kurtosis_vals.tolist()],
            "label_alignment": (
                [float(x) for x in label_alignment_vals.tolist()]
                if label_alignment_vals is not None
                else None
            ),
            "between_variance": (
                [float(x) for x in between_var.tolist()] if between_var is not None else None
            ),
            "within_variance": (
                [float(x) for x in within_var.tolist()] if within_var is not None else None
            ),
        },
        "ranking": {
            "requested_metric": ranking_metric,
            "used_metric": metric_used,
            "order_indices": ranking_order_list,
            "order_names": [component_names[i] for i in ranking_order_list],
            "top_k": top_k,
            "selected_indices": selected_list,
            "selected_names": [component_names[i] for i in selected_list],
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote ICA metric report to {report_path}")


def _drive(
    run_ids: Iterable[str],
    results_root: Path,
    *,
    components: Optional[int],
    max_samples: Optional[int],
    random_state: Optional[int],
    max_iter: int,
    ranking_metric: str,
    top_k: int,
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
                ranking_metric=ranking_metric,
                top_k=top_k,
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
    parser.add_argument(
        "--ranking-metric",
        choices=["kurtosis", "label_alignment"],
        default="kurtosis",
        help="Metric used to rank components for the report and top-k heatmap.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Number of top-ranked components to highlight (0 disables filtering).",
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
        ranking_metric=args.ranking_metric,
        top_k=max(0, args.top_k),
    )


if __name__ == "__main__":
    main()
