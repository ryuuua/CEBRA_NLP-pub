#!/usr/bin/env python
"""
Compute CEBRA consistency scores for a set of runs identified by their W&B run IDs.

The script mirrors the workflow used by the visualization/ICA utilities:
it locates result directories via ``wandb_run_id.txt``, loads the saved
``cebra_embeddings.npy`` files, and evaluates ``cebra.integrations.sklearn.metrics.consistency_score``.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from cebra.integrations.sklearn.metrics import consistency_score
from src.utils import find_run_dirs


def _pick_run_dir(matches: List[Path]) -> Path:
    if not matches:
        raise FileNotFoundError("No run directories containing wandb_run_id.txt were found.")
    if len(matches) == 1:
        return matches[0]
    # Select the most recently modified directory to mimic latest run behaviour.
    return max(matches, key=lambda path: path.stat().st_mtime)


def _load_embeddings(run_dir: Path) -> np.ndarray:
    emb_path = run_dir / "cebra_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} is missing. Run must save embeddings (cebra.save_embeddings=true).")
    return np.load(emb_path)


def _orthogonal_procrustes_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Return (RSS, normalized Procrustes distance) after optimal rotation alignment."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} != {b.shape}; cannot run Procrustes.")
    if a.ndim != 2:
        raise ValueError("Expected 2D embeddings (n_samples, dim).")

    a_center = a - a.mean(axis=0, keepdims=True)
    b_center = b - b.mean(axis=0, keepdims=True)

    cov = b_center.T @ a_center
    U, _, Vt = np.linalg.svd(cov, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = U @ Vt

    aligned = b_center @ R
    residual = a_center - aligned
    rss = float(np.sum(residual ** 2))
    norm = float(np.linalg.norm(a_center))
    distance = float(np.sqrt(rss) / norm) if norm > 0 else float("nan")
    return rss, distance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute pairwise CEBRA consistency scores for saved embeddings.",
    )
    parser.add_argument(
        "run_ids",
        nargs="+",
        help="One or more W&B run IDs to evaluate.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing experiment outputs (default: results/).",
    )
    parser.add_argument(
        "--mode",
        choices=("runs", "datasets"),
        default="runs",
        help="Consistency mode passed to cebra.integrations.sklearn.metrics.consistency_score.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to save scores/pairs/ids_runs as JSON.",
    )
    return parser


def _resolve_embeddings(
    run_ids: Iterable[str],
    results_root: Path,
) -> Tuple[List[np.ndarray], List[str]]:
    embeddings: List[np.ndarray] = []
    resolved_labels: List[str] = []

    for run_id in run_ids:
        matches = find_run_dirs(results_root, run_id)
        if not matches:
            raise FileNotFoundError(
                f"No directories under {results_root} contain wandb_run_id.txt == {run_id}"
            )
        run_dir = _pick_run_dir(matches)
        emb = _load_embeddings(run_dir)
        embeddings.append(emb)
        resolved_labels.append(run_dir.name)
        print(f"[INFO] Loaded embeddings from {run_dir} (shape={emb.shape}).")

    return embeddings, resolved_labels


def _compute_procrustes_metrics(
    embeddings: List[np.ndarray],
    labels: List[str],
) -> List[Tuple[str, str, float, float]]:
    """Compute Procrustes metrics for all pairs of embeddings."""
    procrustes_rows: List[Tuple[str, str, float, float]] = []
    for idx_a, idx_b in combinations(range(len(embeddings)), 2):
        label_a, label_b = labels[idx_a], labels[idx_b]
        try:
            rss, distance = _orthogonal_procrustes_metrics(embeddings[idx_a], embeddings[idx_b])
        except ValueError as exc:
            print(f"[WARN] Skipping Procrustes metrics for {label_a} ↔ {label_b}: {exc}")
            continue
        procrustes_rows.append((label_a, label_b, rss, distance))
    return procrustes_rows


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if len(args.run_ids) < 2:
        raise SystemExit("At least two run IDs are required to compute consistency.")

    embeddings, labels = _resolve_embeddings(args.run_ids, args.results_root.resolve())

    # Build consistency_score arguments
    kwargs = {"embeddings": embeddings, "between": args.mode}
    if args.mode == "datasets":
        kwargs["dataset_ids"] = args.run_ids

    scores, pairs, ids_runs = consistency_score(**kwargs)  # type: ignore[arg-type]

    procrustes_rows = _compute_procrustes_metrics(embeddings, labels)

    print("\nPairwise consistency scores:")
    for pair, score in zip(pairs, scores):
        print(f"  {pair[0]} ↔ {pair[1]} : {float(score):.6f}")

    mean_score = float(np.mean(scores)) if len(scores) > 0 else float("nan")
    print(f"\nMean consistency score: {mean_score:.6f}")

    if procrustes_rows:
        print("\nOrthogonal Procrustes diagnostics (centered embeddings):")
        for name_a, name_b, rss, distance in procrustes_rows:
            print(f"  {name_a} ↔ {name_b} : RSS={rss:.6f}, ProcrustesDist={distance:.6f}")

    if args.json is not None:
        payload = {
            "mode": args.mode,
            "scores": scores.tolist(),
            "pairs": [list(pair) for pair in pairs],
            "ids_runs": ids_runs.tolist(),
        }
        if procrustes_rows:
            payload["procrustes_metrics"] = [
                {
                    "pair": [name_a, name_b],
                    "rss": rss,
                    "procrustes_distance": distance,
                }
                for name_a, name_b, rss, distance in procrustes_rows
            ]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"[INFO] Saved results to {args.json}")


if __name__ == "__main__":
    main()
