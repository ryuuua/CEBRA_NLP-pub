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
    
    # Store embeddings as .npz to include both ids and vectors
    path = Path(base_dir) / f"{dataset_name}__{safe_embedding_name}.npz"
    return path

def save_text_embedding(ids, embeddings, path: Path):
    """Saves numpy ids and embeddings to the specified path."""
    print(f"Caching text embeddings to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, embeddings=embeddings)
    print("...done.")

def load_text_embedding(path: Path):
    """Loads cached ids and embeddings from the specified path if it exists."""
    if path.exists():
        print(f"Found cached text embeddings at {path}. Loading...")
        data = np.load(path)
        ids = data["ids"]
        embeddings = data["embeddings"]
        print("...done.")
        return ids, embeddings
    else:
        print(f"No cached text embeddings found at {path}.")
        return None
