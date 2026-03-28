#!/bin/bash
# Evaluate lora finetuned models in one shot.
#
# Usage:
#   bash scripts/lora/eval_single_task.sh

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
ADAPTER_IDS=(
  juzhengz/LoRI-S_code_llama3_rank_32
  juzhengz/LoRI-S_math_llama3_rank_32
)
BASE_MODEL_ID=meta-llama/Meta-Llama-3-8B

for MODEL_ID in "${MODEL_IDS[@]}"; do
  MODEL_FNAME="$(echo $MODEL_ID | tr '/' '-')"
  ADAPTER_FNAME="$(echo $ADAPTER_ID | tr '/' '-')"
  OUTPUT_DIR="results-lora/${ADAPTER_FNAME}"

  echo "============================================================"
  echo "Model      : ${MODEL_ID}"
  echo "Output     : ${OUTPUT_DIR}"
  echo "============================================================"
  
  olmes \
    --model $MODEL_ID \
    --task \
      codex_humaneval::tulu \
      gsm8k::tulu \
    --output-dir $OUTPUT_DIR \
    --gpus 4 \
    --model-type vllm \
    --model-args '{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}' \
    --batch-size 128
done

