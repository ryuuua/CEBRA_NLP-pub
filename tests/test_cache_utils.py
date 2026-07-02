from pathlib import Path

from cebra_nlp_public.cache_utils import (
    get_embedding_cache_path,
    resolve_embedding_cache_dir,
)


def test_get_embedding_cache_path_honors_configured_cache_dir(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embedding_cache"
    cfg = {
        "dataset": {"name": "tiny_dataset", "shuffle": False},
        "embedding": {
            "name": "all-MiniLM-L6-v2",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "type": "sentence_transformer",
        },
        "paths": {"embedding_cache_dir": str(cache_dir)},
    }

    resolved_dir = resolve_embedding_cache_dir(cfg)
    cache_path = get_embedding_cache_path(
        cfg,
        dataset_key="tiny_dataset",
        variant_tag="unit",
    )

    assert resolved_dir == cache_dir
    assert cache_path.is_relative_to(cache_dir)
    assert cache_path.name.endswith(".npz")
