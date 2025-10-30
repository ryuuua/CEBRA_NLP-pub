#!/usr/bin/env bash
# Cache go_emotions_6labels embeddings for all configured models using distributed inference.
set -u
set -o pipefail

DATASET="go_emotions_6labels"
GPU_LIST="${GPU_LIST:-0,1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
EXTRA_ARGS=("$@")

IFS=',' read -r -a GPU_IDS <<<"$GPU_LIST"
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "[ERROR] GPU_LIST must contain at least one GPU id (e.g. 0,1)." >&2
  exit 1
fi
NPROC=${NPROC_PER_NODE:-${#GPU_IDS[@]}}
if [[ $NPROC -ne ${#GPU_IDS[@]} ]]; then
  echo "[WARN] NPROC_PER_NODE ($NPROC) differs from GPU count (${#GPU_IDS[@]}). CUDA_VISIBLE_DEVICES will still limit usable devices." >&2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

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
echo "[INFO] Distributed settings -> nnodes=$NNODES, node_rank=$NODE_RANK, master_addr=$MASTER_ADDR, master_port=$MASTER_PORT"
echo "[INFO] Target dataset: $DATASET"

run_embedding() {
  local emb="$1"
  echo
  echo "[INFO] ===== Embedding: $emb ====="
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  torchrun \
    --standalone \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    scripts/cache_embeddings.py \
      dataset="$DATASET" \
      embedding="$emb" \
      "${EXTRA_ARGS[@]}"
}

for embedding in "${EMBEDDINGS[@]}"; do
  run_embedding "$embedding"
done

echo
echo "[INFO] Embedding cache generation complete."
