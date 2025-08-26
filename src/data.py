import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import kagglehub
from src.config_schema import AppConfig # ← この行を追加

def load_and_prepare_dataset(cfg: AppConfig):
    """
    Loads the dataset specified in the config, prepares texts and labels.
    """
    dataset_cfg = cfg.dataset
    print(f"Loading dataset: {dataset_cfg.name}")

    if dataset_cfg.source == "kaggle":
        path = kagglehub.dataset_download("kashnitsky/hierarchical-text-classification")
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in Kaggle dataset directory")
        csv_path = os.path.join(path, csv_files[0])
        df = pd.read_csv(csv_path)
    else:
        dataset = load_dataset(dataset_cfg.hf_path)

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
