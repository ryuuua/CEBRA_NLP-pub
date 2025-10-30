#!/usr/bin/env bash
set -u
set -o pipefail

# ====== 固定設定 ======
# 対象データセットと使用する埋め込み
DATASETS=(go_emotions_6labels)
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
MODELS=(offset1-model-mse-lr offset1-model-v4-lr)

# --- dataset|model ごとの出力次元 ---
declare -A DIMS_MAP
DIMS_MAP["go_emotions_single_label|offset1-model-mse-lr"]="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 27 28"
DIMS_MAP["go_emotions_single_label|offset1-model-v4-lr"]="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 27 28"
DIMS_MAP["go_emotions_6labels|offset1-model-mse-lr"]="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 27 28"
DIMS_MAP["go_emotions_6labels|offset1-model-v4-lr"]="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 27 28"

# --- iteration 設定 ---
DEFAULT_ITERS=15000
declare -A ITERS
ITERS["go_emotions_single_label|offset1-model-mse-lr"]=15000
ITERS["go_emotions_single_label|offset1-model-v4-lr"]=20000
ITERS["go_emotions_6labels|offset1-model-mse-lr"]=15000
ITERS["go_emotions_6labels|offset1-model-v4-lr"]=20000

# --- W&B プロジェクト名 ---
declare -A WANDB_PROJECTS
WANDB_PROJECTS["go_emotions_single_label|offset1-model-mse-lr"]="CEBRA_NLP_Experiment_goemo_single_embedeuc"
WANDB_PROJECTS["go_emotions_single_label|offset1-model-v4-lr"]="CEBRA_NLP_Experiment_goemo_single_cos"
WANDB_PROJECTS["go_emotions_6labels|offset1-model-mse-lr"]="CEBRA_NLP_Experiment_goemo_6labels_euc"
WANDB_PROJECTS["go_emotions_6labels|offset1-model-v4-lr"]="CEBRA_NLP_Experiment_goemo_6labels_cos"
DEFAULT_WANDB_PROJECT="CEBRA_NLP_Experiment_Default"

# ====== 実行環境 ======
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "[ERROR] GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi
DEFAULT_NPROC=${#GPU_IDS[@]}
NPROC_PER_NODE="${NPROC_PER_NODE:-$DEFAULT_NPROC}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo,eth0}"
export PYTHONFAULTHANDLER=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

# W&B キー未設定時はオフラインにフォールバック
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" && -z "${WANDB_DISABLED:-}" ]]; then
  export WANDB_MODE=offline
  echo "[INFO] WANDB_API_KEY not found. Falling back to WANDB_MODE=offline."
fi

# ====== ログ環境 ======
STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR="runs_${STAMP}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.csv"
echo "run_id,dataset,model,embedding,dim,iterations,wandb_project,exit_code,start_ts,end_ts,retries,master_port" > "$SUMMARY"

trap 'echo "SIGINT: killing children..."; pkill -P $$; exit 130' INT

# ====== ヘルパ ======

get_iters() {
  local dataset="$1" model="$2"
  local key="${dataset}|${model}"
  if [[ -n "${ITERS[$key]+x}" ]]; then echo "${ITERS[$key]}"; else echo "$DEFAULT_ITERS"; fi
}

get_dims_for_pair() {
  local dataset="$1" model="$2"
  local key="${dataset}|${model}"
  if [[ -n "${DIMS_MAP[$key]+x}" ]]; then
    echo "${DIMS_MAP[$key]}"
  else
    echo "2 3 4 5 6 7 8 9 10 12 15 20"
  fi
}

get_wandb_project() {
  local dataset="$1" model="$2"
  local key="${dataset}|${model}"
  if [[ -n "${WANDB_PROJECTS[$key]+x}" ]]; then echo "${WANDB_PROJECTS[$key]}"; else echo "$DEFAULT_WANDB_PROJECT"; fi
}

pick_free_port() {
  for p in $(seq 29513 29999); do
    if ! ss -ltn "( sport = :$p )" 2>/dev/null | grep -q ":$p "; then
      echo "$p"; return 0
    fi
  done
  echo 29513
}

run_one () {
  local dataset="$1" model="$2" emb="$3" dim="$4" iters="$5" wandb_project="$6"
  local runid="${dataset}_${model}_${emb}_d${dim}"
  local start_ts end_ts rc=0 retries=0
  local errfile="/tmp/torchelastic_errors_${dataset}_${model}_${emb}_d${dim}.txt"

  echo "[RUN] $runid ($wandb_project)"
  start_ts=$(date +%Y-%m-%dT%H:%M:%S)

  for attempt in 1 2 3; do
    retries=$((attempt-1))
    local mport; mport="$(pick_free_port)"

    TORCHELASTIC_ERROR_FILE="$errfile" \
    CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    timeout 7h \
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" --master_port="$mport" \
      main.py \
        wandb.project="$wandb_project" \
        dataset="$dataset" \
        embedding="$emb" \
        cebra="$model" \
        cebra.output_dim="$dim" \
        cebra.max_iterations="$iters" \
        consistency_check.enabled=true \
        evaluation.enable_plots=false \
        reproducibility.deterministic=true \
        reproducibility.cudnn_benchmark=false \
      >"$LOGDIR/$runid.out" 2>"$LOGDIR/$runid.err"

    rc=$?
    if [[ $rc -eq 0 ]]; then
      break
    else
      echo "[WARN] $runid failed with rc=$rc (attempt $attempt/3). Checking for hints..."
      tail -n 50 "$LOGDIR/$runid.err" || true
      [[ -f "$errfile" ]] && { echo "--- TORCHELASTIC_ERROR_FILE tail ---"; tail -n 50 "$errfile" || true; }
      sleep 5
    fi
  done

  end_ts=$(date +%Y-%m-%dT%H:%M:%S)
  echo "$runid,$dataset,$model,$emb,$dim,$iters,$wandb_project,$rc,$start_ts,$end_ts,$retries,${mport:-}" >> "$SUMMARY"
}

# ====== 実行ループ ======
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    iters="$(get_iters "$dataset" "$model")"
    wandb_project="$(get_wandb_project "$dataset" "$model")"
    read -r -a dims <<<"$(get_dims_for_pair "$dataset" "$model")"
    for emb in "${EMBEDDINGS[@]}"; do
      for dim in "${dims[@]}"; do
        run_one "$dataset" "$model" "$emb" "$dim" "$iters" "$wandb_project"
      done
    done
  done
done

echo "All done. Summary -> $SUMMARY"
