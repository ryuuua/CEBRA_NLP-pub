#!/usr/bin/env python

"""Utility to summarize text length statistics for configured datasets."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from omegaconf import OmegaConf

try:
    import kagglehub  # type: ignore
except ImportError:  # pragma: no cover
    kagglehub = None

try:
    from sklearn.datasets import fetch_20newsgroups  # type: ignore
except ImportError:  # pragma: no cover
    fetch_20newsgroups = None


def _load_yaml_config(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, MutableMapping):
        raise ValueError(f"Unexpected config structure in {path}")
    # Hydra configs sometimes keep structured defaults; convert to plain dict.
    return OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)  # type: ignore[return-value]


def _ensure_source(cfg: MutableMapping[str, object]) -> None:
    if "source" not in cfg or cfg["source"] is None:
        cfg["source"] = "hf"


def _load_hf_dataset(cfg: Mapping[str, object]) -> Dict[str, pd.DataFrame]:
    hf_path = cfg.get("hf_path")
    if not hf_path:
        raise ValueError("hf_path must be specified for source='hf'")

    load_kwargs = {}
    if cfg.get("trust_remote_code", False):
        load_kwargs["trust_remote_code"] = True

    splits = cfg.get("splits")
    if str(hf_path) == "trec":
        return _load_trec_fallback(cfg, splits)
    try:
        if splits:
            frames = {}
            for split in splits:
                dataset = load_dataset(hf_path, split=split, **load_kwargs)
                frames[str(split)] = dataset.to_pandas()  # type: ignore[assignment]
            return frames

        dataset_dict = load_dataset(hf_path, **load_kwargs)
        frames: Dict[str, pd.DataFrame] = {}
        for split_name, dataset in dataset_dict.items():
            frames[str(split_name)] = dataset.to_pandas()  # type: ignore[assignment]
        return frames
    except (RuntimeError, ValueError) as error:
        if str(hf_path) == "trec" and _should_use_trec_fallback(error):
            return _load_trec_fallback(cfg, splits)
        raise


def _load_csv_dataset(cfg: Mapping[str, object], config_path: Path) -> Dict[str, pd.DataFrame]:
    data_file = cfg.get("data_files")
    if not data_file:
        raise ValueError("data_files must be provided when source='csv'")

    csv_path = Path(str(data_file))
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    return {csv_path.stem: frame}


def _load_kaggle_dataset(cfg: Mapping[str, object]) -> Dict[str, pd.DataFrame]:
    if kagglehub is None:
        raise ImportError("kagglehub is required to download Kaggle datasets")

    handle = cfg.get("kaggle_handle")
    if not handle:
        raise ValueError("kaggle_handle must be provided for source='kaggle'")

    base_path = Path(kagglehub.dataset_download(str(handle)))  # type: ignore[call-arg]
    data_files = cfg.get("data_files")
    frames: Dict[str, pd.DataFrame] = {}
    if data_files:
        relative = Path(str(data_files))
        if relative.is_absolute():
            candidate = relative
        else:
            candidate = (base_path / relative).resolve()
        if candidate.exists():
            frames[candidate.stem] = pd.read_csv(candidate)
            return frames

        search_root = candidate.parent if relative.parent != Path(".") else base_path
        pattern = relative.name
        candidates = list(search_root.glob(pattern))
        if not candidates and "*" not in pattern:
            suffix = relative.suffix
            stem = relative.stem
            wildcard = f"{stem}*{suffix}" if suffix else f"{stem}*"
            candidates = list(search_root.glob(wildcard))
        if not candidates:
            raise FileNotFoundError(f"Could not locate {relative} in Kaggle dataset {handle}")
        for csv_path in candidates:
            if csv_path.is_file() and csv_path.suffix.lower() == ".csv":
                frames[csv_path.stem] = pd.read_csv(csv_path)
        if frames:
            return frames
        raise FileNotFoundError(f"No CSV files matched pattern {relative} in Kaggle dataset {handle}")

    for candidate in base_path.glob("*.csv"):
        frames[candidate.stem] = pd.read_csv(candidate)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in Kaggle dataset directory: {base_path}")
    return frames


def _load_sklearn_dataset(cfg: Mapping[str, object]) -> Dict[str, pd.DataFrame]:
    if fetch_20newsgroups is None:
        raise ImportError("scikit-learn is required for source='sklearn'")

    dataset_name = cfg.get("sklearn_dataset")
    if dataset_name != "20newsgroups":
        raise ValueError(f"Unsupported sklearn_dataset: {dataset_name}")

    requested_splits = cfg.get("splits") or ["train", "test"]
    frames: Dict[str, pd.DataFrame] = {}

    for split in requested_splits:
        subset = str(split)
        data = fetch_20newsgroups(subset=subset, shuffle=False)
        frames[subset] = pd.DataFrame(
            {
                cfg.get("text_column", "text"): pd.Series(data.data, dtype=str),
                cfg.get("label_column", "label"): data.target,
            }
        )
    return frames


def load_dataset_frames(cfg: MutableMapping[str, object], config_path: Path) -> Dict[str, pd.DataFrame]:
    _ensure_source(cfg)
    source = cfg.get("source")
    if source == "hf":
        frames = _load_hf_dataset(cfg)
    elif source == "csv":
        frames = _load_csv_dataset(cfg, config_path)
    elif source == "kaggle":
        frames = _load_kaggle_dataset(cfg)
    elif source == "sklearn":
        frames = _load_sklearn_dataset(cfg)
    else:
        raise ValueError(f"Unsupported dataset source: {source}")

    if cfg.get("hf_path") == "go_emotions" and cfg.get("label_column"):
        frames = {name: _apply_go_emotions_postprocessing(frame.copy(), cfg) for name, frame in frames.items()}

    text_column = cfg.get("text_column")
    if not text_column:
        raise ValueError("text_column must be specified in dataset config")
    for name, frame in frames.items():
        if text_column not in frame.columns:
            fallback = _find_case_insensitive_column(frame.columns, text_column)
            if fallback is None:
                raise KeyError(f"Column '{text_column}' not found in split '{name}'")
            frame.rename(columns={fallback: text_column}, inplace=True)
    return frames


def _find_case_insensitive_column(columns: Iterable[str], target: str) -> Optional[str]:
    lower_target = target.lower()
    for column in columns:
        if str(column).lower() == lower_target:
            return column
    return None


_TREC_URLS = {
    "train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}


def _should_use_trec_fallback(error: Exception) -> bool:
    message = str(error)
    return "Dataset scripts are no longer supported" in message or "trust_remote_code" in message


def _load_trec_fallback(cfg: Mapping[str, object], splits: Optional[Iterable[str]]) -> Dict[str, pd.DataFrame]:
    requested_splits = list(splits) if splits else list(_TREC_URLS.keys())
    frames: Dict[str, pd.DataFrame] = {}
    text_column = cfg.get("text_column", "text")
    label_column = cfg.get("label_column")

    for split in requested_splits:
        if split not in _TREC_URLS:
            raise ValueError(f"Unsupported TREC split '{split}'")
        records: List[Dict[str, object]] = []
        with urllib.request.urlopen(_TREC_URLS[split]) as response:
            rows = response.read().splitlines()
        for row in rows:
            fine_label, _, text = row.replace(b"\xf0", b" ").strip().decode("utf-8").partition(" ")
            coarse_label = fine_label.split(":")[0]
            record: Dict[str, object] = {str(text_column): text}
            if label_column:
                record[str(label_column)] = coarse_label
            records.append(record)
        frames[str(split)] = pd.DataFrame.from_records(records)
    return frames


def _apply_go_emotions_postprocessing(frame: pd.DataFrame, cfg: Mapping[str, object]) -> pd.DataFrame:
    label_column = cfg.get("label_column")
    if label_column not in frame.columns:
        return frame

    if cfg.get("drop_multi_label_samples"):
        mask = frame[label_column].apply(
            lambda value: isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1  # type: ignore[arg-type]
        )
        frame = frame[mask].reset_index(drop=True)

    def _collapse(value: object) -> object:
        if isinstance(value, (list, tuple, np.ndarray)):
            return value[0] if len(value) > 0 else np.nan  # type: ignore[index]
        return value

    frame[label_column] = frame[label_column].apply(_collapse)

    label_remap = cfg.get("label_remap")
    if isinstance(label_remap, Mapping) and label_remap:
        keys = set(label_remap.keys())
        frame = frame[frame[label_column].isin(keys)].reset_index(drop=True)
        frame[label_column] = frame[label_column].map(label_remap).astype(int)
    return frame


def _series_from_frame(frame: pd.DataFrame, text_column: str) -> pd.Series:
    series = frame[text_column].dropna()
    # Ensure we only operate on string content; filter out empty placeholders.
    series = series.astype(str)
    series = series[series.str.strip().astype(bool)]
    return series


def _length_summary(series: pd.Series) -> Mapping[str, object]:
    if series.empty:
        return {
            "char_lengths": {},
            "word_lengths": {},
            "empty_fraction": 0.0,
        }

    lengths_chars = series.str.len()
    lengths_words = series.str.split().map(len)

    def _stats(values: pd.Series) -> Mapping[str, object]:
        return {
            "count": int(values.count()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=1)) if values.count() > 1 else 0.0,
            "min": int(values.min()),
            "max": int(values.max()),
            "p10": float(values.quantile(0.10)),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
        }

    empty_ratio = float((lengths_chars == 0).mean()) if len(lengths_chars) else 0.0
    return {
        "char_lengths": _stats(lengths_chars),
        "word_lengths": _stats(lengths_words),
        "empty_fraction": empty_ratio,
    }


def summarize_dataset(cfg_path: Path) -> Mapping[str, object]:
    cfg = _load_yaml_config(cfg_path)
    dataset_name = cfg.get("name", cfg_path.stem)
    text_column = cfg.get("text_column", "text")
    frames = load_dataset_frames(cfg, cfg_path)

    split_summaries: Dict[str, Mapping[str, object]] = {}
    series_collection: List[pd.Series] = []
    for split_name, frame in frames.items():
        series = _series_from_frame(frame, text_column)
        series_collection.append(series)
        split_summaries[split_name] = _length_summary(series)

    combined_series = pd.concat(series_collection, ignore_index=True) if series_collection else pd.Series(dtype=str)
    overall_summary = _length_summary(combined_series) if not combined_series.empty else {}

    return {
        "dataset": dataset_name,
        "text_column": text_column,
        "num_splits": len(frames),
        "samples_per_split": {split: int(len(frame)) for split, frame in frames.items()},
        "split_summaries": split_summaries,
        "overall_summary": overall_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="+",
        help="Dataset config names (without extension) or explicit paths.",
    )
    parser.add_argument(
        "--conf-root",
        default="conf/dataset",
        help="Directory containing dataset configs when names are provided.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of a human-readable table.",
    )
    return parser.parse_args()


def _resolve_paths(config_args: Iterable[str], conf_root: Path) -> List[Path]:
    resolved: List[Path] = []
    for arg in config_args:
        candidate = Path(arg)
        if candidate.exists():
            resolved.append(candidate.resolve())
            continue
        candidate = (conf_root / f"{arg}.yaml").resolve()
        if candidate.exists():
            resolved.append(candidate)
            continue
        raise FileNotFoundError(f"Could not resolve dataset config '{arg}'")
    return resolved


def _format_summary(summary: Mapping[str, object]) -> str:
    lines: List[str] = []
    dataset_name = summary.get("dataset", "unknown")
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Text column: {summary.get('text_column')}")

    samples = summary.get("samples_per_split", {})
    if isinstance(samples, Mapping):
        split_info = ", ".join(f"{name}={count}" for name, count in samples.items())
        lines.append(f"Samples per split: {split_info}")

    split_summaries = summary.get("split_summaries", {})
    if isinstance(split_summaries, Mapping):
        for split_name, split_summary in split_summaries.items():
            lines.append(f"  [{split_name}]")
            lines.extend(_format_length_block(split_summary, indent="    "))

    overall = summary.get("overall_summary", {})
    if overall:
        lines.append("  [overall]")
        lines.extend(_format_length_block(overall, indent="    "))
    return "\n".join(lines)


def _format_length_block(length_summary: Mapping[str, object], indent: str) -> List[str]:
    lines: List[str] = []
    char_stats = length_summary.get("char_lengths", {})
    word_stats = length_summary.get("word_lengths", {})
    empty_fraction = length_summary.get("empty_fraction", 0.0)

    def _format_stats(label: str, stats: Mapping[str, object]) -> None:
        if not stats:
            return
        ordered_keys = ["mean", "median", "std", "min", "p10", "p25", "p75", "p90", "p95", "max"]
        stats_line = ", ".join(f"{k}={stats.get(k):.2f}" for k in ordered_keys if k in stats)
        lines.append(f"{indent}{label}: {stats_line}")

    _format_stats("chars", char_stats)
    _format_stats("words", word_stats)
    lines.append(f"{indent}empty_fraction: {empty_fraction:.4f}")
    return lines


def main() -> None:
    args = parse_args()
    conf_root = Path(args.conf_root).resolve()
    config_paths = _resolve_paths(args.configs, conf_root)
    summaries = [summarize_dataset(path) for path in config_paths]

    if args.json:
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
    else:
        for idx, summary in enumerate(summaries):
            if idx:
                print("\n" + "-" * 80 + "\n")
            print(_format_summary(summary))


if __name__ == "__main__":
    main()
