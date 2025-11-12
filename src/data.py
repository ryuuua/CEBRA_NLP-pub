import os
import urllib.request
import pandas as pd
import numpy as np
from datasets import load_dataset
import kagglehub
from sklearn.datasets import fetch_20newsgroups
from src.config_schema import AppConfig

from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from src.config_schema import AppConfig


_TREC_URLS = {
    "train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}


def _should_use_trec_fallback(error: Exception) -> bool:
    message = str(error)
    return "Dataset scripts are no longer supported" in message or "trust_remote_code" in message


def _download_trec_split(split: str, dataset_cfg) -> pd.DataFrame:
    if split not in _TREC_URLS:
        raise ValueError(f"Unsupported split '{split}' for TREC dataset.")
    url = _TREC_URLS[split]
    with urllib.request.urlopen(url) as response:
        rows = response.read().splitlines()

    label_column = dataset_cfg.label_column or "label"
    text_column = dataset_cfg.text_column
    label_to_id = {label: idx for idx, label in dataset_cfg.label_map.items()}

    records = []
    for row in rows:
        fine_label, _, text = row.replace(b"\xf0", b" ").strip().decode("utf-8").partition(" ")
        coarse_label = fine_label.split(":")[0]
        if coarse_label not in label_to_id:
            raise ValueError(f"Unknown coarse label '{coarse_label}' in TREC data.")
        records.append({text_column: text, label_column: label_to_id[coarse_label]})

    return pd.DataFrame.from_records(records)


def _load_trec_from_source(dataset_cfg, splits: List[str]) -> List[pd.DataFrame]:
    requested_splits = splits or list(_TREC_URLS)
    frames: List[pd.DataFrame] = []
    for split in requested_splits:
        frames.append(_download_trec_split(split, dataset_cfg))
    return frames


def _load_20newsgroups_from_source(dataset_cfg, splits: List[str]) -> List[pd.DataFrame]:
    supported_splits = {"train", "test", "all"}
    requested_splits = splits or ["train", "test"]
    frames: List[pd.DataFrame] = []
    text_column = dataset_cfg.text_column
    label_column = dataset_cfg.label_column or "label"
    expected_label_order = [
        dataset_cfg.label_map[i] for i in sorted(dataset_cfg.label_map.keys())
    ] if dataset_cfg.label_map else None

    for split in requested_splits:
        if split not in supported_splits:
            raise ValueError(
                f"Unsupported split '{split}' for 20 Newsgroups dataset."
            )

        data = fetch_20newsgroups(subset=split, shuffle=False)

        if expected_label_order is not None:
            target_names = list(data.target_names)
            if target_names != expected_label_order:
                raise ValueError(
                    "Configured label_map does not match 20 Newsgroups target names."
                )

        frame = pd.DataFrame({
            text_column: pd.Series(data.data, dtype=str),
            label_column: data.target,
        })
        frames.append(frame)

    return frames


def load_and_prepare_dataset(cfg: "AppConfig"):
    """Load dataset and prepare texts, conditional data, time indices and IDs."""
    dataset_cfg = cfg.dataset
    conditional_mode = getattr(cfg.cebra, "conditional", "none").lower()
    print(f"Loading dataset: {dataset_cfg.name}")

    if dataset_cfg.source == "hf":
        load_kwargs = {}
        if getattr(dataset_cfg, "trust_remote_code", False):
            load_kwargs["trust_remote_code"] = True

        try:
            if dataset_cfg.splits:
                datasets = [
                    load_dataset(
                        dataset_cfg.hf_path,
                        split=split,
                        **load_kwargs,
                    )
                    for split in dataset_cfg.splits
                ]
            else:
                dataset = load_dataset(
                    dataset_cfg.hf_path,
                    **load_kwargs,
                )
                dataset_keys = dataset.keys()
                datasets = [dataset[split] for split in dataset_keys]
        except (RuntimeError, ValueError) as err:
            if dataset_cfg.hf_path == "trec" and _should_use_trec_fallback(err):
                print("Falling back to manual download for the TREC dataset.")
                datasets = _load_trec_from_source(dataset_cfg, dataset_cfg.splits)
            else:
                raise
        all_splits = [pd.DataFrame(d) for d in datasets]
        df = pd.concat(all_splits, ignore_index=True)
    elif dataset_cfg.source == "csv":
        dataset = load_dataset("csv", data_files=dataset_cfg.data_files)
        dataset_keys = dataset.keys()
        all_splits = [pd.DataFrame(dataset[split]) for split in dataset_keys]
        df = pd.concat(all_splits, ignore_index=True)
    elif dataset_cfg.source == "kaggle":
        if not dataset_cfg.kaggle_handle:
            raise ValueError(
                "dataset.kaggle_handle must be set when dataset.source is 'kaggle'"
            )
        
        path = kagglehub.dataset_download(dataset_cfg.kaggle_handle)
        
        # Determine CSV file path: use specified file or find first CSV file
        if dataset_cfg.data_files:
            csv_path = os.path.join(path, dataset_cfg.data_files)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Specified data file not found in Kaggle dataset directory: {csv_path}"
                )
        else:
            all_files = os.listdir(path)
            csv_files = [f for f in all_files if f.endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError("No CSV files found in Kaggle dataset directory")
            csv_path = os.path.join(path, csv_files[0])

        df = pd.read_csv(csv_path)
    elif dataset_cfg.source == "sklearn":
        if dataset_cfg.sklearn_dataset == "20newsgroups":
            datasets = _load_20newsgroups_from_source(dataset_cfg, dataset_cfg.splits)
            df = pd.concat(datasets, ignore_index=True)
        else:
            raise ValueError(
                f"Unsupported sklearn_dataset: {dataset_cfg.sklearn_dataset}."
            )
    else:
        raise ValueError(
            "Unsupported dataset source: "
            f"{dataset_cfg.source}. Supported sources are 'hf', 'csv', 'kaggle', and 'sklearn'."
        )

    # Special handling for go_emotions variants: optionally drop multi-label rows and collapse lists.
    if dataset_cfg.hf_path == "go_emotions" and dataset_cfg.label_column is not None:
        label_col = dataset_cfg.label_column
        
        def _collapse_go_emotions_label(value):
            if isinstance(value, (list, tuple)):
                return value[0] if len(value) > 0 else np.nan
            return value
        
        if dataset_cfg.drop_multi_label_samples:
            print("Applying go_emotions filter: removing multi-label samples.")
            before_count = len(df)
            df = df[df[label_col].apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 1)].reset_index(drop=True)
            after_count = len(df)
            print(f"go_emotions samples after filtering: {after_count} (removed {before_count - after_count}).")
        else:
            print("Applying special handling for go_emotions: using only the first label.")
        
        df[label_col] = df[label_col].apply(_collapse_go_emotions_label)

    if dataset_cfg.label_column is not None and dataset_cfg.label_remap:
        label_col = dataset_cfg.label_column
        remap = dataset_cfg.label_remap
        df = df[df[label_col].isin(remap.keys())].reset_index(drop=True)
        df[label_col] = df[label_col].map(remap).astype(np.int64)

    # Filter to valid labels for single-label datasets
    if dataset_cfg.label_column is not None and not dataset_cfg.multi_label:
        valid_labels = set(dataset_cfg.label_map.keys())
        df = df[df[dataset_cfg.label_column].isin(valid_labels)].reset_index(drop=True)
    
    # Drop rows with missing values based on conditional mode
    if conditional_mode == "none":
        # Expect V, A, D columns and drop rows with missing values
        df = df.dropna(subset=[dataset_cfg.text_column, "V", "A", "D"]).reset_index(drop=True)
    else:
        # Non-none conditional mode: determine columns to check for missing values
        if dataset_cfg.multi_label:
            if dataset_cfg.label_columns:
                subset_cols = [dataset_cfg.text_column] + dataset_cfg.label_columns
            elif dataset_cfg.label_column is not None:
                subset_cols = [dataset_cfg.text_column, dataset_cfg.label_column]
            else:
                raise ValueError(
                    "multi_label=True requires either label_columns or label_column"
                )
        else:
            if dataset_cfg.label_column is None:
                raise ValueError(
                    "dataset.label_column must be set when cfg.cebra.conditional is not 'none'"
                )
            subset_cols = [dataset_cfg.text_column, dataset_cfg.label_column]
        
        df = df.dropna(subset=subset_cols).reset_index(drop=True)

    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    if cfg.dataset.shuffle:
        seed = (
            cfg.dataset.shuffle_seed
            if cfg.dataset.shuffle_seed is not None
            else (cfg.evaluation.random_state if hasattr(cfg, "evaluation") else None)
        )
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    ids = df["id"].astype(str).to_numpy()

    # Extract conditional data based on mode
    if conditional_mode == "none":
        conditional_data = df[["V", "A", "D"]].to_numpy(dtype=np.float32)
    elif dataset_cfg.multi_label:
        # Multi-label mode: extract from columns or delimited strings
        label_order = [dataset_cfg.label_map[i] for i in sorted(dataset_cfg.label_map.keys())]
        
        if dataset_cfg.label_columns:
            # Extract from multiple label columns
            label_columns_set = set(dataset_cfg.label_columns)
            ordered_cols = [lbl for lbl in label_order if lbl in label_columns_set]
            if len(ordered_cols) != len(label_order):
                raise ValueError(
                    "label_columns must contain all labels from label_map"
                )
            conditional_data = df[ordered_cols].astype(int).to_numpy()
        elif dataset_cfg.label_column is not None and dataset_cfg.label_delimiter:
            # Extract from delimited label strings
            delimiter = dataset_cfg.label_delimiter
            mapping = {label: idx for idx, label in enumerate(label_order)}
            label_matrix = np.zeros((len(df), len(label_order)), dtype=int)
            label_column_str = df[dataset_cfg.label_column].astype(str)
            for i, entry in enumerate(label_column_str):
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
        # Single-label mode: extract from single label column
        labels = df[dataset_cfg.label_column]
        conditional_data = labels.to_numpy()

    texts = df[dataset_cfg.text_column].astype(str)
    texts_list = texts.tolist()
    time_indices = np.arange(len(df))

    return texts_list, conditional_data, time_indices, ids
