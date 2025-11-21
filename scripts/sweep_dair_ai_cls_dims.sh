#!/usr/bin/env bash
set -euo pipefail

# Sweep cebra.output_dim=2..20 for the dair-ai dataset on two GPUs in parallel.
# GPU0 and GPU1 are configurable via env vars GPU0/GPU1 (defaults: 0 and 1).

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LOG_DIR="${LOG_DIR:-runs_dair_ai_cls_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"

BASE_CMD=(
  python main.py -m
  dataset=dair-ai
  embedding=bert
  embedding.pooling=cls
  visualization
  cebra=offset1-model-mse-lr
  evaluation.enable_plots=true
  wandb.project="CEBRA_NLP_CLS_Experiment-dair-ai"
)

run_subset() {
  local gpu_id="$1"; shift
  local dims=("$@")
  for dim in "${dims[@]}"; do
    echo "[GPU ${gpu_id}] cebra.output_dim=${dim}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      "${BASE_CMD[@]}" \
      cebra.output_dim="${dim}" \
      wandb.name="dair-ai-cls-d${dim}" \
      > "${LOG_DIR}/dim${dim}.out" 2> "${LOG_DIR}/dim${dim}.err"
  done
}

GPU0_DIMS=($(seq 2 2 20))
GPU1_DIMS=($(seq 3 2 19))

run_subset "$GPU0" "${GPU0_DIMS[@]}" &
PID0=$!
run_subset "$GPU1" "${GPU1_DIMS[@]}" &
PID1=$!

wait "$PID0" "$PID1"
echo "All sweeps finished. Logs -> ${LOG_DIR}"
