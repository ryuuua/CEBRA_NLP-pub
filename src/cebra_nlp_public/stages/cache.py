from pathlib import Path

from ..config_schema import AppConfig
from ..data import load_and_prepare_dataset
from ..embeddings import (
    get_last_cache_usage,
    load_or_generate_embedding_collection,
    load_or_generate_embeddings,
)
from ..label_overlay import (
    load_or_generate_label_overlay_embeddings,
    write_label_overlay_manifest,
)
from ..utils import apply_reproducibility


def run(cfg: AppConfig, output_dir: Path) -> None:
    """Generate and cache embeddings for the configured dataset."""
    apply_reproducibility(cfg)
    if bool(getattr(getattr(cfg, "trajectory_analysis", None), "enabled", False)):
        if str(getattr(cfg.cebra, "conditional", "none")).lower() != "discrete":
            raise ValueError(
                "trajectory_analysis is only supported for discrete CEBRA runs."
            )
        if not (getattr(cfg.dataset, "label_map", None) or {}):
            raise ValueError(
                "trajectory_analysis requires dataset.label_map for discrete labels."
            )

    print("\n--- Cache Stage: Loading dataset ---")
    texts, conditional_data, _, ids = load_and_prepare_dataset(cfg)

    print("\n--- Cache Stage: Generating embeddings ---")
    if getattr(cfg, "embedding_collection", None) and cfg.embedding_collection.embeddings:
        _ = load_or_generate_embedding_collection(
            cfg,
            texts,
            ids,
            labels=conditional_data,
            require_cache=cfg.stage.require_cache,
        )
    else:
        _ = load_or_generate_embeddings(
            cfg,
            texts,
            ids,
            labels=conditional_data,
            require_cache=cfg.stage.require_cache,
        )

    should_cache_overlay = bool(getattr(cfg.label_overlay, "cache_in_cache_stage", True)) and (
        bool(getattr(cfg.label_overlay, "enabled", False))
        or bool(getattr(getattr(cfg, "trajectory_analysis", None), "enabled", False))
    )
    if should_cache_overlay:
        overlay_spec, _overlay_vectors = load_or_generate_label_overlay_embeddings(
            cfg,
            require_cache=cfg.stage.require_cache,
            force_enabled=bool(getattr(cfg.trajectory_analysis, "enabled", False)),
        )
        if overlay_spec is not None:
            cache_usage = get_last_cache_usage() or {}
            manifest_path = output_dir / "label_overlay_manifest.csv"
            write_label_overlay_manifest(
                overlay_spec,
                manifest_path,
                cache_path=str(cache_usage.get("cache_path", "") or ""),
            )
            print(f"Label overlay manifest written to {manifest_path}.")

    summary_path = output_dir / "cache_complete.txt"
    summary_path.write_text("cache_ok\n")
    print(f"Cache stage complete. Summary written to {summary_path}.")
