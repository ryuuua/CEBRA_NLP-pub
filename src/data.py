import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from src.config_schema import AppConfig # ← この行を追加

def load_and_prepare_dataset(cfg: AppConfig):
    """
    Loads the dataset specified in the config, prepares texts and labels.
    """
    dataset_cfg = cfg.dataset
    print(f"Loading dataset: {dataset_cfg.name}")

    if dataset_cfg.source is None:
        raise ValueError(
            "Dataset source is not specified. Supported sources are 'hf', 'csv', and 'kaggle'."
        )

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
    if dataset_cfg.name == "go_emotions":
        print("Applying special handling for go_emotions: using only the first label.")
        df[dataset_cfg.label_column] = df[dataset_cfg.label_column].apply(lambda x: x[0])

    labels = df[dataset_cfg.label_column].to_numpy()
    texts = df[dataset_cfg.text_column].tolist()

    # Create sequential time indices for each sample
    time_indices = np.arange(len(df))

    return texts, labels, time_indices
