# pyright: reportAttributeAccessIssue=false, reportCallIssue=false

import os
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
from .config_schema import AppConfig, DatasetConfig

from typing import List


_TREC_URLS = {
    "train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}


def _should_use_trec_fallback(error: Exception) -> bool:
    message = str(error)
    return "Dataset scripts are no longer supported" in message or "trust_remote_code" in message


def _download_trec_split(split: str, dataset_cfg: DatasetConfig) -> pd.DataFrame:
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


def _load_trec_from_source(
    dataset_cfg: DatasetConfig, splits: List[str]
) -> List[pd.DataFrame]:
    requested_splits = splits or list(_TREC_URLS.keys())
    frames: List[pd.DataFrame] = []
    for split in requested_splits:
        frames.append(_download_trec_split(split, dataset_cfg))
    return frames


def _load_20newsgroups_from_source(
    dataset_cfg: DatasetConfig, splits: List[str]
) -> List[pd.DataFrame]:
    from sklearn.datasets import fetch_20newsgroups

    supported_splits = {"train", "test", "all"}
    requested_splits = splits or ["train", "test"]
    frames: List[pd.DataFrame] = []
    text_column = dataset_cfg.text_column
    label_column = dataset_cfg.label_column or "label"
    expected_label_order = None
    if dataset_cfg.label_map:
        expected_label_order = [
            dataset_cfg.label_map[i] for i in sorted(dataset_cfg.label_map.keys())
        ]

    data_home = _resolve_sklearn_data_home()

    for split in requested_splits:
        if split not in supported_splits:
            raise ValueError(
                f"Unsupported split '{split}' for 20 Newsgroups dataset."
            )

        data = fetch_20newsgroups(subset=split, shuffle=False, data_home=data_home)

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


def _resolve_sklearn_data_home() -> str:
    """Return a writable cache directory for scikit-learn datasets."""
    candidates: List[str] = [
        str(Path("artifacts/cache/sklearn_data").resolve()),
        "/tmp/scikit_learn_data",
    ]

    for path in candidates:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(path, os.W_OK):
            return path

    raise PermissionError(
        "Unable to resolve writable scikit-learn data directory. "
        "Run from a writable directory or use a CSV dataset."
    )


def _resolve_kaggle_csv_path(dataset_dir: str, data_files: str | None) -> str:
    if data_files:
        csv_path = os.path.join(dataset_dir, data_files)
        if os.path.exists(csv_path):
            return csv_path
        raise FileNotFoundError(
            f"Specified data file not found in Kaggle dataset directory: {csv_path}"
        )

    csv_files = [name for name in os.listdir(dataset_dir) if name.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in Kaggle dataset directory")
    return os.path.join(dataset_dir, csv_files[0])


def _load_kaggle_dataframe(dataset_cfg: DatasetConfig) -> pd.DataFrame:
    import kagglehub

    if not dataset_cfg.kaggle_handle:
        raise ValueError(
            "dataset.kaggle_handle must be set when dataset.source is 'kaggle'"
        )
    dataset_dir = kagglehub.dataset_download(dataset_cfg.kaggle_handle)
    csv_path = _resolve_kaggle_csv_path(dataset_dir, dataset_cfg.data_files)
    return pd.read_csv(csv_path)


def _load_sklearn_dataframe(dataset_cfg: DatasetConfig) -> pd.DataFrame:
    if dataset_cfg.sklearn_dataset != "20newsgroups":
        raise ValueError(
            f"Unsupported sklearn_dataset: {dataset_cfg.sklearn_dataset}."
        )
    datasets = _load_20newsgroups_from_source(dataset_cfg, dataset_cfg.splits)
    return pd.concat(datasets, ignore_index=True)


def _resolve_conditional_subset_columns(
    *,
    text_col: str,
    label_col: str | None,
    label_columns: list[str],
    multi_label: bool,
) -> list[str]:
    if not multi_label:
        if label_col is None:
            raise ValueError(
                "dataset.label_column must be set when cfg.cebra.conditional is not 'none'"
            )
        return [text_col, label_col]

    if label_columns:
        return [text_col] + label_columns
    if label_col is not None:
        return [text_col, label_col]
    raise ValueError("multi_label=True requires either label_columns or label_column")


def _build_multi_label_conditional_data(
    *,
    df: pd.DataFrame,
    dataset_cfg: DatasetConfig,
    label_col: str | None,
    label_columns: list[str],
) -> np.ndarray:
    label_order = [dataset_cfg.label_map[i] for i in sorted(dataset_cfg.label_map.keys())]

    if label_columns:
        label_columns_set = set(label_columns)
        ordered_cols = [label for label in label_order if label in label_columns_set]
        if len(ordered_cols) != len(label_order):
            raise ValueError("label_columns must contain all labels from label_map")
        return df[ordered_cols].astype(int).to_numpy()

    if label_col is None or not dataset_cfg.label_delimiter:
        raise ValueError(
            "multi_label=True requires label_columns or label_delimiter with label_column"
        )

    delimiter = dataset_cfg.label_delimiter
    mapping = {label: idx for idx, label in enumerate(label_order)}
    label_matrix: np.ndarray = np.zeros((len(df), len(label_order)), dtype=int)
    for row_index, entry in enumerate(df[label_col].astype(str)):
        labels = [label.strip() for label in entry.split(delimiter) if label.strip()]
        for label in labels:
            mapped_index = mapping.get(label)
            if mapped_index is None:
                continue
            label_matrix[row_index, mapped_index] = 1
    return label_matrix


def _load_dataframe_from_source(dataset_cfg: DatasetConfig) -> pd.DataFrame:
    if dataset_cfg.source == "hf":
        from datasets import load_dataset

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
                datasets = [dataset[split] for split in dataset.keys()]
        except (RuntimeError, ValueError) as err:
            if dataset_cfg.hf_path == "trec" and _should_use_trec_fallback(err):
                print("Falling back to manual download for the TREC dataset.")
                datasets = _load_trec_from_source(dataset_cfg, dataset_cfg.splits)
            else:
                raise
        return pd.concat([pd.DataFrame(d) for d in datasets], ignore_index=True)

    if dataset_cfg.source == "csv":
        if not dataset_cfg.data_files:
            raise ValueError("dataset.data_files must be set when dataset.source is 'csv'.")
        data_files = [item.strip() for item in str(dataset_cfg.data_files).split(",") if item.strip()]
        if not data_files:
            raise ValueError("dataset.data_files did not contain any CSV paths.")
        return pd.concat(
            [pd.read_csv(path) for path in data_files],
            ignore_index=True,
        )

    if dataset_cfg.source == "kaggle":
        return _load_kaggle_dataframe(dataset_cfg)

    if dataset_cfg.source == "sklearn":
        return _load_sklearn_dataframe(dataset_cfg)

    raise ValueError(
        "Unsupported dataset source: "
        f"{dataset_cfg.source}. Supported sources are 'hf', 'csv', 'kaggle', and 'sklearn'."
    )


def load_and_prepare_dataset(cfg: "AppConfig"):
    """Load dataset and prepare texts, conditional data, time indices and IDs."""
    dataset_cfg = cfg.dataset
    conditional_mode = getattr(cfg.cebra, "conditional", "none").lower()
    text_col = dataset_cfg.text_column
    label_col = dataset_cfg.label_column
    label_columns = dataset_cfg.label_columns or []
    print(f"Loading dataset: {dataset_cfg.name}")
    df = _load_dataframe_from_source(dataset_cfg)

    # Special handling for go_emotions variants: optionally drop multi-label rows and collapse lists.
    if dataset_cfg.hf_path == "go_emotions" and label_col is not None:
        if dataset_cfg.drop_multi_label_samples:
            print("Applying go_emotions filter: removing multi-label samples.")
            before_count = len(df)
            df = df[df[label_col].apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 1)]
            after_count = len(df)
            print(f"go_emotions samples after filtering: {after_count} (removed {before_count - after_count}).")
        else:
            print("Applying special handling for go_emotions: using only the first label.")

        def _collapse_go_emotions_label(value):
            if isinstance(value, (list, tuple)):
                return value[0] if len(value) > 0 else np.nan
            return value

        df[label_col] = df[label_col].apply(_collapse_go_emotions_label)

    if label_col is not None and dataset_cfg.label_remap:
        remap = dataset_cfg.label_remap
        df = df[df[label_col].isin(remap.keys())]
        df[label_col] = df[label_col].map(remap).astype(np.int64)

    if label_col is not None and not dataset_cfg.multi_label:
        valid_labels = set(dataset_cfg.label_map.keys())
        df = df[df[label_col].isin(valid_labels)]

    if conditional_mode == "none":
        # Expect V, A, D columns and drop rows with missing values
        df = df.dropna(subset=[text_col, "V", "A", "D"])
    else:
        subset_cols = _resolve_conditional_subset_columns(
            text_col=text_col,
            label_col=label_col,
            label_columns=label_columns,
            multi_label=bool(dataset_cfg.multi_label),
        )
        df = df.dropna(subset=subset_cols)

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
            conditional_data = _build_multi_label_conditional_data(
                df=df,
                dataset_cfg=dataset_cfg,
                label_col=label_col,
                label_columns=label_columns,
            )
        else:
            labels = df[label_col]
            conditional_data = labels.to_numpy()

    texts = df[text_col].astype(str)
    texts_list = texts.tolist()
    time_indices = np.arange(len(df))

    conditional_data = _apply_label_randomization(conditional_data, cfg)
    return texts_list, conditional_data, time_indices, ids


def _apply_label_randomization(
    conditional_data: np.ndarray, cfg: AppConfig
) -> np.ndarray:
    """
    Optionally randomize labels used for CEBRA training.
    Supports single-label (1D) data. Multi-label matrices are left untouched.
    """
    rand_cfg = getattr(cfg, "label_randomization", None)
    if rand_cfg is None or getattr(rand_cfg, "mode", "none") == "none":
        return conditional_data

    mode = getattr(rand_cfg, "mode", "none").lower()
    rng_seed = None
    repro_cfg = getattr(cfg, "reproducibility", None)
    if repro_cfg is not None:
        rng_seed = getattr(repro_cfg, "seed", None)
    rng = np.random.default_rng(rng_seed)

    data = np.asarray(conditional_data)
    if data.ndim != 1:
        print(f"[WARN] label_randomization={mode} requested but labels are not 1D; skipping.")
        return conditional_data

    if mode == "permutation":
        shuffled = data.copy()
        rng.shuffle(shuffled)
        return shuffled

    if mode == "random_int":
        num_classes = getattr(rand_cfg, "num_classes", None)
        if num_classes is None:
            uniques = np.unique(data)
            num_classes = len(uniques)
        if num_classes <= 0:
            raise ValueError("label_randomization.num_classes must be positive.")
        return rng.integers(low=0, high=num_classes, size=data.shape[0])

    print(f"[WARN] Unknown label_randomization mode '{mode}'; leaving labels unchanged.")
    return conditional_data
