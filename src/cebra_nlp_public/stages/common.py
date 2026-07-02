from typing import Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from ..config_schema import AppConfig


def align_by_ids(
    source_ids: Sequence,
    source_values: np.ndarray,
    target_ids: Sequence,
) -> np.ndarray:
    """Reorder source_values to match target_ids based on id mapping."""
    id_to_index = {str(item): idx for idx, item in enumerate(source_ids)}
    indices = np.asarray([id_to_index[str(item)] for item in target_ids], dtype=int)
    return np.asarray(source_values)[indices]


def split_with_ids(
    X: np.ndarray,
    labels: np.ndarray,
    time_indices: np.ndarray,
    ids: np.ndarray,
    cfg: AppConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays and return train/valid data plus id splits."""
    indices = np.arange(len(X))
    stratify = labels if cfg.cebra.conditional == "discrete" else None
    (
        X_train,
        X_valid,
        labels_train,
        labels_valid,
        time_train,
        time_valid,
        idx_train,
        idx_valid,
    ) = train_test_split(
        X,
        labels,
        time_indices,
        indices,
        test_size=cfg.evaluation.test_size,
        random_state=cfg.evaluation.random_state,
        stratify=stratify,
    )
    ids_train = ids[idx_train]
    ids_valid = ids[idx_valid]
    return (
        X_train,
        X_valid,
        labels_train,
        labels_valid,
        time_train,
        time_valid,
        ids_train,
        ids_valid,
    )
