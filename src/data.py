import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import kagglehub
from src.config_schema import AppConfig

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.config_schema import AppConfig


def load_and_prepare_dataset(cfg: "AppConfig"):
    """Load dataset and prepare texts, conditional data, time indices and IDs."""
    dataset_cfg = cfg.dataset
    conditional_mode = getattr(cfg.cebra, "conditional", "none").lower()
    print(f"Loading dataset: {dataset_cfg.name}")

    if dataset_cfg.source == "hf":
        if dataset_cfg.splits:
            datasets = [load_dataset(dataset_cfg.hf_path, split=split) for split in dataset_cfg.splits]
        else:
            dataset = load_dataset(dataset_cfg.hf_path)
            datasets = [dataset[split] for split in dataset.keys()]
        all_splits = [pd.DataFrame(d) for d in datasets]
        df = pd.concat(all_splits, ignore_index=True)
    elif dataset_cfg.source == "csv":
        dataset = load_dataset("csv", data_files=dataset_cfg.data_files)
        all_splits = [pd.DataFrame(dataset[split]) for split in dataset.keys()]
        df = pd.concat(all_splits, ignore_index=True)
    elif dataset_cfg.source == "kaggle":

        if not dataset_cfg.kaggle_handle:
            raise ValueError(
                "dataset.kaggle_handle must be set when dataset.source is 'kaggle'"
            )
        path = kagglehub.dataset_download(dataset_cfg.kaggle_handle)
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in Kaggle dataset directory")
        csv_path = os.path.join(path, csv_files[0])

        if dataset_cfg.kaggle_handle:
            path = kagglehub.dataset_download(dataset_cfg.kaggle_handle)
        else:
            path = cfg.paths.kaggle_data_dir

        if dataset_cfg.data_files:
            csv_path = os.path.join(path, dataset_cfg.data_files)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Specified data file not found in Kaggle dataset directory: {csv_path}"
                )
        else:
            csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError("No CSV files found in Kaggle dataset directory")
            csv_path = os.path.join(path, csv_files[0])

        df = pd.read_csv(csv_path)
    else:
        raise ValueError(
            f"Unsupported dataset source: {dataset_cfg.source}. Supported sources are 'hf', 'csv', and 'kaggle'."
        )

    # Special handling for go_emotions: use only the first label
    if dataset_cfg.name == "go_emotions" and dataset_cfg.label_column is not None:
        print("Applying special handling for go_emotions: using only the first label.")
        df[dataset_cfg.label_column] = df[dataset_cfg.label_column].apply(lambda x: x[0])

    if dataset_cfg.label_column is not None and not dataset_cfg.multi_label:
        valid_labels = set(dataset_cfg.label_map.keys())
        df = df[df[dataset_cfg.label_column].isin(valid_labels)].reset_index(drop=True)
    if conditional_mode == "none":
        # Expect V, A, D columns and drop rows with missing values
        df = df.dropna(subset=[dataset_cfg.text_column, "V", "A", "D"]).reset_index(drop=True)
    else:
        if dataset_cfg.multi_label:
            if dataset_cfg.label_columns:
                subset_cols = [dataset_cfg.text_column] + dataset_cfg.label_columns
            elif dataset_cfg.label_column is not None:
                subset_cols = [dataset_cfg.text_column, dataset_cfg.label_column]
            else:
                raise ValueError(
                    "multi_label=True requires either label_columns or label_column"
                )
            df = df.dropna(subset=subset_cols).reset_index(drop=True)
        else:
            if dataset_cfg.label_column is None:
                raise ValueError(
                    "dataset.label_column must be set when cfg.cebra.conditional is not 'none'"
                )
            df = df.dropna(subset=[dataset_cfg.text_column, dataset_cfg.label_column]).reset_index(drop=True)

    df = df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    if cfg.dataset.shuffle:
        seed = (
            cfg.dataset.shuffle_seed
            if getattr(cfg.dataset, "shuffle_seed", None) is not None
            else (cfg.evaluation.random_state if hasattr(cfg, "evaluation") else None)
        )
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    ids = df["id"].astype(str).to_numpy()

    if conditional_mode == "none":
        conditional_data = df[["V", "A", "D"]].to_numpy(dtype=np.float32)
    else:
        if dataset_cfg.multi_label:
            label_order = [dataset_cfg.label_map[i] for i in sorted(dataset_cfg.label_map.keys())]
            if dataset_cfg.label_columns:
                ordered_cols = [lbl for lbl in label_order if lbl in dataset_cfg.label_columns]
                if len(ordered_cols) != len(label_order):
                    raise ValueError(
                        "label_columns must contain all labels from label_map"
                    )
                conditional_data = df[ordered_cols].astype(int).to_numpy()
            elif dataset_cfg.label_column is not None and dataset_cfg.label_delimiter:
                delimiter = dataset_cfg.label_delimiter
                mapping = {label: idx for idx, label in enumerate(label_order)}
                label_matrix = np.zeros((len(df), len(label_order)), dtype=int)
                for i, entry in enumerate(df[dataset_cfg.label_column].astype(str)):
                    if pd.isna(entry):
                        continue
                    labels = [lab.strip() for lab in entry.split(delimiter) if lab.strip()]
                    for lab in labels:
                        if lab in mapping:
                            label_matrix[i, mapping[lab]] = 1
                conditional_data = label_matrix
            else:
                raise ValueError(
                    "multi_label=True requires label_columns or label_delimiter with label_column"
                )
        else:
            labels = df[dataset_cfg.label_column]
            conditional_data = labels.to_numpy()

    texts = df[dataset_cfg.text_column].astype(str)
    texts_list = texts.tolist()
    time_indices = np.arange(len(df))

    return texts_list, conditional_data, time_indices, ids
