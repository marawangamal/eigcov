#!/bin/bash
# Evaluate a model on Twin-Merging benchmarks using lm-evaluation-harness.
#
# Usage:
#   bash scripts/twin/eval.sh <model_path_or_hf_id> <output_dir>
#
# Example:
#   bash scripts/twin/eval.sh checkpoints/twin/Qwen-14B-eigcov results/twin/eigcov
#
# Requires: pip install lm-eval

set -euo pipefail

MODEL=${1:?Usage: bash scripts/twin/eval.sh <model_path> <output_dir>}
OUTPUT_DIR=${2:?Usage: bash scripts/twin/eval.sh <model_path> <output_dir>}

# Task names in lm-evaluation-harness.
# Verify with: lm_eval --tasks list 2>/dev/null | grep -i "bbq\|cnn\|mmlu\|truthful"
TASKS="bbq,cnn_dailymail,mmlu,truthfulqa_mc2"

echo "============================================================"
echo "Model      : ${MODEL}"
echo "Output     : ${OUTPUT_DIR}"
echo "Tasks      : ${TASKS}"
echo "============================================================"

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=float16" \
    --tasks "${TASKS}" \
    --batch_size 4 \
    --output_path "${OUTPUT_DIR}"

echo "Results saved to ${OUTPUT_DIR}"
