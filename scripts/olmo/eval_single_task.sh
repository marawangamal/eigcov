#!/bin/bash
# Evaluate each OLMo RL-Zero expert on its corresponding task.
#
# Usage:
#   bash scripts/olmo/eval_single_task.sh

set -euo pipefail

# 0. Setup environment
source "$SCRATCH/eigcov/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# ── Defaults ──────────────────────────────────────────────────────────────────
declare -A MODEL_TASKS
MODEL_TASKS[allenai/Olmo-3-7B-RL-Zero-Math]="aime:zs_cot_r1::pass_at_32_2024_deepseek aime:zs_cot_r1::pass_at_32_2025_deepseek"
MODEL_TASKS[allenai/Olmo-3-7B-RL-Zero-Code]="codex_humaneval::tulu codex_humanevalplus::tulu"
MODEL_TASKS[allenai/Olmo-3-7B-RL-Zero-IF]="ifeval::tulu"

for MODEL_ID in "${!MODEL_TASKS[@]}"; do
  TASKS=${MODEL_TASKS[$MODEL_ID]}
  MODEL_FNAME="$(echo $MODEL_ID | tr '/' '-')"
  OUTPUT_DIR="results-rl/${MODEL_FNAME}"

  echo "============================================================"
  echo "Model      : ${MODEL_ID}"
  echo "Tasks      : ${TASKS}"
  echo "Output     : ${OUTPUT_DIR}"
  echo "============================================================"

  olmes \
    --model $MODEL_ID \
    --task $TASKS \
    --output-dir $OUTPUT_DIR \
    --gpus 4 \
    --model-type vllm \
    --model-args '{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 16384}' \
    --batch-size 128
done

