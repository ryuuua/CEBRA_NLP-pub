from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from src.utils import get_embedding_cache_path


def _make_cfg(shuffle: bool, seed: Optional[int] = None):
    return OmegaConf.create(
        {
            "paths": {"embedding_cache_dir": "."},
            "dataset": {"name": "dataset", "shuffle": shuffle, "shuffle_seed": seed},
            "embedding": {"name": "embedding"},
        }
    )


def test_get_embedding_cache_path_no_shuffle():
    cfg = _make_cfg(False, 123)
    path = get_embedding_cache_path(cfg)
    assert path == Path("./dataset__embedding.npz")


def test_get_embedding_cache_path_with_seed():
    cfg1 = _make_cfg(True, 1)
    cfg2 = _make_cfg(True, 2)
    path1 = get_embedding_cache_path(cfg1)
    path2 = get_embedding_cache_path(cfg2)
    assert path1 == Path("./dataset__embedding__seed1.npz")
    assert path2 == Path("./dataset__embedding__seed2.npz")
    assert path1 != path2
