#!/usr/bin/env python3
"""Run main.py for the requested dataset × embedding grid."""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from typing import Dict, List, Sequence


DEFAULT_DATASETS: List[str] = [
    "dair-ai",
    "go_emotions",
    "go_emotions_single_label",
    "go_emotions_6labels",
    "ag_news",
    "twenty_newsgroups",
]

EMBEDDING_NAME_TO_CONFIG: Dict[str, str] = {
    "bert-base-uncased": "bert",
    "ibm-granite/granite-embedding-english-r2": "granite_embedding",
    "inaai/jina-embeddings-v2-base-en": "jina_embedding",
    "Qwen/Qwen3-Embedding-8B": "qwen3_embedding",
    "roberta-base": "roberta",
    "sentence-transformers/all-mpnet-base-v2": "mpnet_embedding",
    "all-MiniLM-L6-v2": "sentence_bert",
}

DEFAULT_EMBEDDINGS: List[str] = [
    "bert-base-uncased",
    "ibm-granite/granite-embedding-english-r2",
    "inaai/jina-embeddings-v2-base-en",
    "Qwen/Qwen3-Embedding-8B",
    "roberta-base",
    "sentence-transformers/all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially launch main.py runs over the dataset × embedding grid. "
            "Datasets and embeddings can be restricted via CLI flags."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Hydra dataset configs to use (default: %(default)s).",
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=DEFAULT_EMBEDDINGS,
        help=(
            "Human-readable embedding names (matching the YAML `name` fields). "
            "These are mapped to Hydra config files via EMBEDDING_NAME_TO_CONFIG."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for launching main.py (default: current interpreter).",
    )
    parser.add_argument(
        "--main-script",
        default="main.py",
        help="Entry point to execute for each run (default: %(default)s).",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Any additional Hydra overrides to append after dataset/embedding "
            "(e.g. --extra-args cebra.output_dim=8 evaluation.enable_plots=false)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep iterating even if a run fails (default: stop at first failure).",
    )
    return parser.parse_args(argv)


def _resolve_embedding_config(name: str) -> str:
    if name not in EMBEDDING_NAME_TO_CONFIG:
        known = ", ".join(sorted(EMBEDDING_NAME_TO_CONFIG))
        raise ValueError(
            f"Unknown embedding '{name}'. Update EMBEDDING_NAME_TO_CONFIG or pass "
            f"one of: {known}."
        )
    return EMBEDDING_NAME_TO_CONFIG[name]


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if not args.datasets:
        print("[WARN] No datasets specified; nothing to run.")
        return 0
    if not args.embeddings:
        print("[WARN] No embeddings specified; nothing to run.")
        return 0

    env = os.environ.copy()
    if (
        "WANDB_API_KEY" not in env
        and "WANDB_MODE" not in env
        and "WANDB_DISABLED" not in env
    ):
        env["WANDB_MODE"] = "offline"

    combos = list(itertools.product(args.datasets, args.embeddings))
    total = len(combos)

    for idx, (dataset, embedding_name) in enumerate(combos, start=1):
        embedding_cfg = _resolve_embedding_config(embedding_name)
        cmd = [
            args.python,
            args.main_script,
            f"dataset={dataset}",
            f"embedding={embedding_cfg}",
        ] + args.extra_args

        printable = " ".join(cmd)
        prefix = f"[{idx}/{total}] dataset={dataset} embedding={embedding_name}"
        print(f"{prefix}\n  -> {printable}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            msg = (
                f"[ERROR] Command failed with exit code {result.returncode}. "
                "Use --continue-on-error to keep going."
            )
            print(msg, file=sys.stderr)
            if not args.continue_on_error:
                return result.returncode

    print(f"Completed {total} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
