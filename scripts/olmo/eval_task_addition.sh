#!/bin/bash
# Merge + evaluate OLMo models via olmes, then collect results.
#
# Prerequisites: run scripts/olmo/download_models.sh first.
#
# Usage:
#   bash scripts/olmo/eval_task_addition.sh
set -euo pipefail

# 0. Setup environment
source "$SCRATCH/eigcov/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL="Olmo-3-7b"
METHODS=(eigcov tsv mean isoc)
TASK_DIRS=(
  "checkpoints/${MODEL}/Math"
  "checkpoints/${MODEL}/Code"
  "checkpoints/${MODEL}/IF"
)

# ── OLMES ─────────────────────────────────────────────────────────────────────
OLMES_TASKS=(
  "codex_humaneval::tulu"
  "codex_humanevalplus::tulu"
  "ifeval::tulu"
  "aime:zs_cot_r1::pass_at_32_2024_deepseek"
  "aime:zs_cot_r1::pass_at_32_2025_deepseek"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 16384}'
GPUS=4
BATCH_SIZE=64
NUM_WORKERS=1

# ── Merge + Evaluate ────────────────────────────────────────────────────────
for method in "${METHODS[@]}"; do
  MERGED_DIR="checkpoints/${MODEL}/${method}"
  RESULTS_DIR="results/${MODEL}-${method}"

  echo "============================================================"
  echo "Method: ${method}"
  echo "Merged: ${MERGED_DIR}"
  echo "Results: ${RESULTS_DIR}"
  echo "============================================================"

  # 1. Merge (skip if already done)
  if [[ -d "$MERGED_DIR" ]]; then
    echo ">>> Skipping merge: ${MERGED_DIR} already exists"
  else
    python scripts/olmo/merge.py \
      --task-dirs "${TASK_DIRS[@]}" \
      --merge-func "$method" \
      --output-dir "$MERGED_DIR"
  fi

  # 2. Evaluate (skip if already done)
  if ls "$RESULTS_DIR"/*-metrics-all.json &>/dev/null; then
    echo ">>> Skipping eval: ${RESULTS_DIR} already has results"
  else
    echo ">>> Evaluating: Batch size = $BATCH_SIZE, Number of workers = $NUM_WORKERS, GPUs = $GPUS"
    echo ">>> Model: $MERGED_DIR, tasks: ${OLMES_TASKS[@]}"
    olmes \
      --model "$MERGED_DIR" \
      --task "${OLMES_TASKS[@]}" \
      --output-dir "$RESULTS_DIR" \
      --gpus "$GPUS" \
      --model-type vllm \
      --model-args "$OLMES_MODEL_ARGS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS"
  fi
done
