#!/bin/bash
# Merge + evaluate rl models in one shot.
#
# For each merge method: runs merge.py, optionally uploads to HF Hub,
# evaluates via olmes, then collects all results into a summary table.
#
# Usage:
#   bash scripts/rl/eval_task_addition.sh \
#     --pretrained-dir checkpoints/rl/meta-llama-Meta-Llama-3.1-8B \
#     --finetuned-dirs \
#       checkpoints/rl/pmahdavi-Llama-3.1-8B-math-reasoning \
#       checkpoints/rl/pmahdavi-Llama-3.1-8B-coding \
#       checkpoints/rl/pmahdavi-Llama-3.1-8B-precise-if \
#       checkpoints/rl/pmahdavi-Llama-3.1-8B-general \
#       checkpoints/rl/pmahdavi-Llama-3.1-8B-knowledge-recall \
#     --merge-funcs "eigcov tsv isoc mean" \
#     --gpus 4

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_IDS=(
  allenai/Olmo-3-7B-RL-Zero-Math
  allenai/Olmo-3-7B-RL-Zero-Code
  allenai/Olmo-3-7B-RL-Zero-IF
)

for MODEL_ID in "${MODEL_IDS[@]}"; do
  MODEL_FNAME="$(echo $MODEL_ID | tr '/' '-')"
  OUTPUT_DIR="results-rl/${MODEL_FNAME}"

  echo "============================================================"
  echo "Model      : ${MODEL_ID}"
  echo "Output     : ${OUTPUT_DIR}"
  echo "============================================================"
  
  olmes \
    --model $MODEL_ID \
    --task \
      codex_humaneval::tulu \
      codex_humanevalplus::tulu \
      ifeval::tulu \
      aime:zs_cot_r1::pass_at_32_2024_deepseek \
      aime:zs_cot_r1::pass_at_32_2025_deepseek \
    --output-dir $OUTPUT_DIR \
    --gpus 4 \
    --model-type vllm \
    --model-args '{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 16384}' \
    --batch-size 128
done

