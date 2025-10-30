#!/usr/bin/env python
"""
Pre-compute and cache text embeddings for a given dataset/embedding configuration.

Launch with Hydra overrides, e.g.:
  torchrun --standalone --nproc_per_node=2 scripts/cache_embeddings.py dataset=go_emotions_6labels embedding=bert

The script splits the dataset across distributed ranks, computes embeddings on each
GPU, gathers the results on rank 0, and stores them via src.utils.save_text_embedding.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import hydra

from src.config_schema import AppConfig
from src.data import load_and_prepare_dataset
from src.embeddings import (
    clear_last_hidden_state_cache,
    get_embeddings,
    get_last_hidden_state_cache,
)
from src.utils import (
    apply_reproducibility,
    get_embedding_cache_path,
    load_text_embedding,
    save_text_embedding,
)


def _compute_shuffle_seed(cfg: AppConfig) -> Optional[int]:
    if getattr(cfg.dataset, "shuffle", False):
        return getattr(cfg.dataset, "shuffle_seed", None)
    return getattr(cfg.evaluation, "random_state", None)


def _should_skip_cache(ids: np.ndarray, seed: Optional[int], cache_path) -> bool:
    cache = load_text_embedding(cache_path)
    if cache is None:
        return False
    cached_ids, _, cached_seed, _ = cache
    if cached_seed != seed:
        print("Cached embeddings have a different shuffle seed; regenerating.")
        return False

    ids_str = np.asarray(ids, dtype=str)
    cached_ids_str = np.asarray(cached_ids, dtype=str)
    if ids_str.shape != cached_ids_str.shape or not np.array_equal(
        ids_str, cached_ids_str
    ):
        print("Cached embeddings do not match dataset ids; regenerating.")
        return False

    print("Embedding cache already up to date. Skipping recomputation.")
    return True


def _split_range(count: int, world_size: int, rank: int) -> tuple[int, int]:
    if world_size <= 1:
        return 0, count
    block = (count + world_size - 1) // world_size
    start = rank * block
    end = min(start + block, count)
    return start, end


def _gather_chunks(payload: Dict[str, Any], world_size: int, rank: int) -> List[Dict[str, Any]]:
    if world_size <= 1:
        return [payload]
    gather_list: Optional[List[Dict[str, Any]]] = None
    if rank == 0:
        gather_list = [dict() for _ in range(world_size)]
    dist.gather_object(payload, gather_list if rank == 0 else None, dst=0)
    if rank == 0 and gather_list is not None:
        return gather_list
    return []


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def cache_embeddings(cfg: AppConfig) -> None:
    default_cfg = OmegaConf.structured(AppConfig)
    cfg = OmegaConf.merge(default_cfg, cfg)
    OmegaConf.set_struct(cfg, False)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.ddp.world_size = world_size
    cfg.ddp.rank = rank
    cfg.ddp.local_rank = local_rank

    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        if torch.cuda.is_available():
            cfg.device = f"cuda:{local_rank}"
    else:
        if torch.cuda.is_available():
            cfg.device = "cuda"

    apply_reproducibility(cfg)
    is_main = rank == 0

    print(f"[Rank {rank}] Loading dataset...")
    texts, _, _, ids = load_and_prepare_dataset(cfg)
    total_count = len(texts)
    start, end = _split_range(total_count, world_size, rank)
    local_texts = texts[start:end]
    local_ids = ids[start:end]

    cache_path = get_embedding_cache_path(cfg)
    seed = _compute_shuffle_seed(cfg)

    skip_recompute = False
    if is_main:
        skip_recompute = _should_skip_cache(ids, seed, cache_path)
    if world_size > 1:
        flag_tensor = torch.tensor(int(skip_recompute), device="cpu")
        dist.broadcast(flag_tensor, src=0)
        skip_recompute = bool(flag_tensor.item())

    if skip_recompute:
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    print(f"[Rank {rank}] Computing embeddings for samples {start}:{end} ...")
    if len(local_texts) == 0:
        output_dim = getattr(cfg.embedding, "output_dim", 0) or 0
        local_embeddings = np.zeros((0, output_dim), dtype=np.float32)
        layer_cache = None
    else:
        local_embeddings = get_embeddings(local_texts, cfg)
        layer_cache = get_last_hidden_state_cache()
    clear_last_hidden_state_cache()

    payload = {
        "ids": np.asarray(local_ids),
        "embeddings": np.asarray(local_embeddings),
        "layer_embeddings": None if layer_cache is None else np.asarray(layer_cache),
    }

    chunks = _gather_chunks(payload, world_size, rank)
    if is_main:
        ordered_ids: List[np.ndarray] = []
        ordered_embeddings: List[np.ndarray] = []
        ordered_layers: List[np.ndarray] = []
        has_layer_data = True

        for chunk in chunks:
            ids_chunk = np.asarray(chunk["ids"])
            emb_chunk = np.asarray(chunk["embeddings"])
            layer_chunk = chunk.get("layer_embeddings")
            if ids_chunk.size == 0:
                continue
            ordered_ids.append(ids_chunk)
            ordered_embeddings.append(emb_chunk)
            if layer_chunk is None:
                has_layer_data = False
            ordered_layers.append(np.asarray(layer_chunk) if layer_chunk is not None else None)

        if ordered_ids:
            full_ids = np.concatenate(ordered_ids, axis=0)
            full_embeddings = np.concatenate(ordered_embeddings, axis=0)
            if has_layer_data and all(layer is not None for layer in ordered_layers):
                full_layers = np.concatenate([layer for layer in ordered_layers if layer is not None], axis=0)
            else:
                full_layers = None

            save_text_embedding(
                full_ids,
                full_embeddings,
                seed,
                cache_path,
                layer_embeddings=full_layers,
            )
        else:
            print("No embeddings were produced. Nothing to cache.")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    cache_embeddings()
