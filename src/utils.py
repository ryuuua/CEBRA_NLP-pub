from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

def get_embedding_cache_path(cfg):
    """
    Generates a unique path for a cached text embedding file.
    Reads the base directory from cfg.paths.embedding_cache_dir,
    with a fallback to 'embedding_cache' for backward compatibility.
    """
    # Get base directory from config, with a default for backward compatibility
    base_dir = OmegaConf.select(cfg, 'paths.embedding_cache_dir', default='embedding_cache')
    
    dataset_name = cfg.dataset.name
    embedding_name = cfg.embedding.name
    
    # Create a filename-safe version of the embedding name
    safe_embedding_name = embedding_name.replace('/', '__')
    
    path = Path(base_dir) / f"{dataset_name}__{safe_embedding_name}.npy"
    return path

def save_text_embedding(embeddings, path: Path):
    """Saves numpy embeddings to the specified path."""
    print(f"Caching text embeddings to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    print("...done.")

def load_text_embedding(path: Path):
    """Loads numpy embeddings from the specified path if it exists."""
    if path.exists():
        print(f"Found cached text embeddings at {path}. Loading...")
        embeddings = np.load(path)
        print("...done.")
        return embeddings
    else:
        print(f"No cached text embeddings found at {path}.")
        return None
