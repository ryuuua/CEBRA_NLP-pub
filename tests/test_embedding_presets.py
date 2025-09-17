from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    "config_name, expected_model, expected_dim",
    [
        ("embeddinggemma", "google/embeddinggemma-300M", 1024),
        ("qwen3_embedding", "Qwen/Qwen3-1.5B-Text-Embedding", 1536),
        ("granite_embedding", "ibm-granite/granite-embedding-english-r2", 768),
        ("jina_embedding", "jinaai/jina-embeddings-v2-base-en", 1024),
    ],
)
def test_embedding_preset_configs(config_name, expected_model, expected_dim):
    cfg = OmegaConf.load(Path("conf/embedding") / f"{config_name}.yaml")
    assert cfg["name"] == expected_model
    assert cfg["model_name"] == expected_model
    assert cfg["output_dim"] == expected_dim
