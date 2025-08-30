import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config_schema import AppConfig

def load_and_prepare_dataset(cfg: "AppConfig"):
    """
    Loads the dataset specified in the config and prepares texts,
    conditional data (labels or VAD values), and time indices.
    """
    dataset_cfg = cfg.dataset
    print(f"Loading dataset: {dataset_cfg.name}")

    if dataset_cfg.source == "hf":
        dataset = load_dataset(dataset_cfg.hf_path)
    elif dataset_cfg.source == "csv":
        dataset = load_dataset("csv", data_files=dataset_cfg.data_files)
    elif dataset_cfg.source == "kaggle":
        data_path = os.path.join(cfg.paths.kaggle_data_dir, dataset_cfg.data_files)
        dataset = load_dataset("csv", data_files=data_path)
    else:
        raise ValueError(
            f"Unsupported dataset source: {dataset_cfg.source}. Supported sources are 'hf', 'csv', and 'kaggle'."
        )

    # Combine all splits for a comprehensive analysis
    all_splits = [pd.DataFrame(dataset[split]) for split in dataset.keys()]
    df = pd.concat(all_splits, ignore_index=True)
    
    # Special handling for go_emotions: use only the first label
    if dataset_cfg.name == "go_emotions" and dataset_cfg.label_column is not None:
        print("Applying special handling for go_emotions: using only the first label.")
        df[dataset_cfg.label_column] = df[dataset_cfg.label_column].apply(lambda x: x[0])

    if cfg.cebra.conditional == "None":
        # Expect V, A, D columns and drop rows with missing values
        df = df.dropna(subset=[dataset_cfg.text_column, "V", "A", "D"]).reset_index(drop=True)
        conditional_data = df[["V", "A", "D"]].to_numpy(dtype=np.float32)
        texts = df[dataset_cfg.text_column].astype(str)
    else:
        if dataset_cfg.label_column is None:
            raise ValueError(
                "dataset.label_column must be set when cfg.cebra.conditional is not 'None'"
            )
        labels = df[dataset_cfg.label_column]
        df = df.dropna(subset=[dataset_cfg.text_column, dataset_cfg.label_column]).reset_index(drop=True)
        conditional_data = labels.to_numpy()
        texts = df[dataset_cfg.text_column].astype(str)

    texts_list = texts.tolist()
    time_indices = np.arange(len(df))

    return texts_list, conditional_data, time_indices
