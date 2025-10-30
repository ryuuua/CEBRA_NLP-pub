#!/usr/bin/env bash
set -u
set -o pipefail

DATASET="go_emotions_6labels"
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "[ERROR] GPU_LIST must specify at least one GPU id (e.g. 0,1)." >&2
  exit 1
fi
NPROC=${#GPU_IDS[@]}

EMBEDDINGS=(
  bert
  roberta
  mpnet_embedding
  sentence_bert
  qwen3_embedding
  embeddinggemma
  granite_embedding
  jina_embedding
)

echo "[INFO] Using GPUs: $GPU_LIST (nproc_per_node=$NPROC)"
echo "[INFO] Target dataset: $DATASET"

EXTRA_ARGS=("$@")

for emb in "${EMBEDDINGS[@]}"; do
  echo
  echo "[INFO] ===== Embedding: $emb ====="
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  torchrun --standalone --nproc_per_node="$NPROC" \
    scripts/cache_embeddings.py \
      dataset="$DATASET" \
      embedding="$emb" \
      "${EXTRA_ARGS[@]}"
done

echo
echo "[INFO] Embedding cache generation complete."
