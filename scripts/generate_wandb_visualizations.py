import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_schema import AppConfig
from src.data import load_and_prepare_dataset
from src.results import save_interactive_plot, save_static_2d_plots
from src.utils import (
    apply_reproducibility,
    get_embedding_cache_path,
    load_text_embedding,
    save_text_embedding,
)
from src.embeddings import (
    get_embeddings,
    get_last_hidden_state_cache,
    clear_last_hidden_state_cache,
)
from src.cebra_trainer import (
    load_cebra_model,
    transform_cebra,
    normalize_model_architecture,
)


def resolve_layer_index(layer_count: int, requested: Optional[int]) -> int:
    """Replicate main.resolve_layer_index without importing Hydra entrypoint."""
    if layer_count <= 0:
        raise ValueError("Layer cache is empty; cannot select a hidden state layer.")
    index = layer_count - 1 if requested is None else requested
    if index < 0:
        index += layer_count
    if index < 0 or index >= layer_count:
        raise ValueError(
            f"Layer index {requested} is out of bounds for cached tensor with "
            f"{layer_count} layers."
        )
    return index


def _load_cfg(run_dir: Path) -> AppConfig:
    """Hydra writes the resolved config under .hydra/config.yaml."""
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    if not hydra_cfg_path.exists():
        raise FileNotFoundError(f"Cannot locate Hydra config at {hydra_cfg_path}")

    hydra_cfg = OmegaConf.load(hydra_cfg_path)

    default_cfg = OmegaConf.structured(AppConfig)
    cfg = OmegaConf.merge(default_cfg, hydra_cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.cebra.conditional = cfg.cebra.conditional.lower()
    cfg.cebra.model_architecture = normalize_model_architecture(
        getattr(cfg.cebra, "model_architecture", "offset0-model")
    )
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def _find_run_dirs(results_root: Path, run_id: str) -> List[Path]:
    matches: List[Path] = []
    for marker in results_root.rglob("wandb_run_id.txt"):
        if marker.read_text().strip() == run_id:
            matches.append(marker.parent)
    return sorted(matches)


def _prepare_base_embeddings(cfg: AppConfig, ids: Iterable, texts: List[str]) -> np.ndarray:
    """Reuse cached embeddings when possible, otherwise recompute."""
    cache_path = get_embedding_cache_path(cfg)
    cache = load_text_embedding(cache_path)
    if cache is not None:
        cached_ids, cached_embeddings, cached_seed, cached_layer_embeddings = cache
        id_to_index = {str(i): idx for idx, i in enumerate(cached_ids)}
        try:
            selection_indices = np.asarray(
                [id_to_index[str(i)] for i in ids], dtype=int
            )
        except KeyError:
            selection_indices = None

        if selection_indices is not None:
            if cfg.embedding.type == "hf_transformer" and cached_layer_embeddings is not None:
                layer_idx = resolve_layer_index(
                    cached_layer_embeddings.shape[1],
                    getattr(cfg.embedding, "hidden_state_layer", None),
                )
                return cached_layer_embeddings[selection_indices, layer_idx, :]
            return np.asarray(cached_embeddings[selection_indices])

    embeddings = get_embeddings(texts, cfg)
    # get_embeddings caches the full hidden-state tensor internally; stash for reuse
    layer_cache = get_last_hidden_state_cache()
    if layer_cache is not None and layer_cache.shape[0] != embeddings.shape[0]:
        layer_cache = None  # fallback if cache mismatched
    seed = (
        getattr(cfg.dataset, "shuffle_seed", None)
        if getattr(cfg.dataset, "shuffle_seed", None) is not None
        else getattr(getattr(cfg, "evaluation", None), "random_state", None)
    )
    save_text_embedding(
        ids,
        embeddings,
        seed,
        cache_path,
        layer_embeddings=layer_cache,
    )
    clear_last_hidden_state_cache()
    return embeddings


def _load_cebra_embeddings(
    run_dir: Path,
    cfg: AppConfig,
    base_embeddings: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    embeddings_path = run_dir / "cebra_embeddings.npy"
    if embeddings_path.exists():
        return np.load(embeddings_path)

    model_path = run_dir / "cebra_model.pt"
    if not model_path.exists():
        print(
            f"[WARN] Missing cebra_embeddings.npy and cebra_model.pt in {run_dir}. "
            "Skipping visualization."
        )
        return None
    if base_embeddings is None:
        print(
            f"[WARN] Unable to rebuild embeddings for {run_dir} because base embeddings "
            "were not computed."
        )
        return None

    model = load_cebra_model(model_path, cfg, input_dimension=base_embeddings.shape[1])
    return transform_cebra(model, base_embeddings, cfg.device)


def _prepare_labels(
    cfg: AppConfig, conditional_data: np.ndarray
) -> Tuple[List, Optional[dict], List]:
    """Return text labels plus palette/order for plotting."""
    if cfg.cebra.conditional == "discrete":
        data = np.asarray(conditional_data).reshape(-1)
        unique = set(np.unique(data).tolist())
        if unique == {-1, 1}:
            data = np.where(data == -1, 0, 1)
        label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}
        labels = [label_map[int(v)] for v in data]

        vis_cfg = getattr(cfg.dataset, "visualization", None)
        palette = (
            OmegaConf.to_container(vis_cfg.emotion_colors, resolve=True)
            if vis_cfg is not None and getattr(vis_cfg, "emotion_colors", None)
            else None
        )
        order = (
            list(
                OmegaConf.to_container(vis_cfg.emotion_order, resolve=True)
            )
            if vis_cfg is not None and getattr(vis_cfg, "emotion_order", None)
            else sorted({label_map[int(v)] for v in data})
        )
        return labels, palette, order

    values = np.asarray(conditional_data)
    if values.ndim == 2 and values.shape[1] >= 1:
        labels = values[:, 0].tolist()
    else:
        labels = values.reshape(-1).tolist()
    return labels, None, []


def _generate_visualizations(run_dir: Path, run_id: str) -> None:
    print(f"\nProcessing run {run_id} at {run_dir}")
    cfg = _load_cfg(run_dir)
    apply_reproducibility(cfg)

    viz_dir = run_dir / "visualizations"
    conditional_desc = (
        "discrete labels" if cfg.cebra.conditional == "discrete" else "continuous labels"
    )
    interactive_path = viz_dir / f"cebra_interactive_{conditional_desc.replace(' ', '_')}.html"

    # Skip expensive recomputation if we already produced the expected artifacts.
    if interactive_path.exists():
        has_required_outputs = True
        if cfg.cebra.conditional == "discrete":
            expected_static = [
                viz_dir / "static_PCA_plot.png",
                viz_dir / "static_UMAP_plot.png",
            ]
            has_required_outputs = all(path.exists() for path in expected_static)
        if has_required_outputs:
            print(f"[INFO] Visualizations already exist at {viz_dir}; skipping.")
            return

    texts, conditional_data, _time_indices, ids = load_and_prepare_dataset(cfg)

    # Use stored embeddings when present; otherwise rebuild from scratch.
    base_embeddings: Optional[np.ndarray] = None
    embeddings_path = run_dir / "cebra_embeddings.npy"
    if not embeddings_path.exists():
        base_embeddings = _prepare_base_embeddings(cfg, ids, texts)

    cebra_embeddings = _load_cebra_embeddings(run_dir, cfg, base_embeddings)
    if cebra_embeddings is None:
        return
    labels, palette, order = _prepare_labels(cfg, conditional_data)

    viz_dir.mkdir(parents=True, exist_ok=True)

    save_interactive_plot(
        embeddings=cebra_embeddings,
        text_labels=labels,
        output_dim=cfg.cebra.output_dim,
        palette=palette,
        title=f"{cfg.dataset.name} · {run_id} ({conditional_desc})",
        output_path=interactive_path,
    )

    if cfg.cebra.conditional == "discrete":
        hue_order = order if order else sorted(set(labels))
        save_static_2d_plots(
            embeddings=cebra_embeddings,
            text_labels=labels,
            palette=palette,
            title_prefix=f"{cfg.dataset.name} ({run_id})",
            output_dir=viz_dir,
            hue_order=hue_order,
            cfg=cfg,
            log_to_wandb=False,
        )


def _drive(run_ids: Iterable[str], results_root: Path) -> None:
    for run_id in run_ids:
        matches = _find_run_dirs(results_root, run_id)
        if not matches:
            print(f"[WARN] No results directory found for run ID {run_id}")
            continue
        for run_dir in matches:
            _generate_visualizations(run_dir, run_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate local visualizations from stored W&B runs."
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
    args = parser.parse_args()

    _drive(args.run_ids, args.results_root.resolve())


if __name__ == "__main__":
    main()
