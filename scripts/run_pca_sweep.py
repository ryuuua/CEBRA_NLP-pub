#!/usr/bin/env python3
"""Launch PCA runs for the specified dataset × embedding grid."""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from typing import Dict, List, Sequence

# Default datasets from user request
DEFAULT_DATASETS: List[str] = [
    "dair-ai",
    "go_emotions",
    "go_emotions_single_label",
    "go_emotions_6labels",
    "ag_news",
    "twenty_newsgroups",
]

# Embedding names (keep duplicate bert-base-uncased entry as requested)
DEFAULT_EMBEDDINGS: List[str] = [
    "bert-base-uncased",
    "bert-base-uncased",
    "ibm-granite/granite-embedding-english-r2",
    "inaai/jina-embeddings-v2-base-en",
    "Qwen/Qwen3-Embedding-8B",
    "roberta-base",
    "sentence-transformers/all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
]

# Map human-readable embedding names to Hydra config files
EMBEDDING_NAME_TO_CONFIG: Dict[str, str] = {
    "bert-base-uncased": "bert",
    "ibm-granite/granite-embedding-english-r2": "granite_embedding",
    "inaai/jina-embeddings-v2-base-en": "jina_embedding",
    "Qwen/Qwen3-Embedding-8B": "qwen3_embedding",
    "roberta-base": "roberta",
    "sentence-transformers/all-mpnet-base-v2": "mpnet_embedding",
    "all-MiniLM-L6-v2": "sentence_bert",
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run tools/pca_embeddings.py across dataset × embedding combinations."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Hydra dataset configs (default: %(default)s).",
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=DEFAULT_EMBEDDINGS,
        help=(
            "Embedding names matching YAML `name` fields. "
            "Duplicates are allowed and preserved (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter).",
    )
    parser.add_argument(
        "--pca-script",
        default="tools/pca_embeddings.py",
        help="Hydra entry point for PCA (default: %(default)s).",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Additional Hydra overrides appended after dataset/embedding "
            "(e.g. --extra-args pca_analysis.export_dir=results_PCA/custom)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running even if a command fails (default: stop on first failure).",
    )
    return parser.parse_args(argv)


def _resolve_embedding_config(name: str) -> str:
    if name not in EMBEDDING_NAME_TO_CONFIG:
        known = ", ".join(sorted(EMBEDDING_NAME_TO_CONFIG))
        raise ValueError(
            f"Embedding '{name}' is not mapped to a Hydra config. "
            f"Known names: {known}. Update EMBEDDING_NAME_TO_CONFIG as needed."
        )
    return EMBEDDING_NAME_TO_CONFIG[name]


def _setup_wandb_env() -> Dict[str, str]:
    """Set up WANDB environment variables if not already configured."""
    env = os.environ.copy()
    # Set WANDB_MODE to offline if no WANDB configuration is present
    if not any(key in env for key in ("WANDB_API_KEY", "WANDB_MODE", "WANDB_DISABLED")):
        env["WANDB_MODE"] = "offline"
    return env


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if not args.datasets:
        print("[WARN] No datasets specified; nothing to run.")
        return 0
    if not args.embeddings:
        print("[WARN] No embeddings specified; nothing to run.")
        return 0

    env = _setup_wandb_env()

    embeddings_with_index = list(enumerate(args.embeddings))
    combos = list(itertools.product(args.datasets, embeddings_with_index))
    total = len(combos)

    for idx, (dataset, (emb_idx, embedding_name)) in enumerate(combos, start=1):
        embedding_cfg = _resolve_embedding_config(embedding_name)
        cmd = [
            args.python,
            args.pca_script,
            f"dataset={dataset}",
            f"embedding={embedding_cfg}",
        ] + args.extra_args

        printable = " ".join(cmd)
        prefix = (
            f"[{idx}/{total}] dataset={dataset} "
            f"embedding={embedding_name} (entry #{emb_idx + 1})"
        )
        print(f"{prefix}\n  -> {printable}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(
                f"[ERROR] Command failed with exit code {result.returncode}.",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                return result.returncode

    print(f"Completed {total} PCA run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
