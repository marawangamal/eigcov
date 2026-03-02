#!/bin/bash
#SBATCH --job-name=finetune_vision_models
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# set -euo pipefail
# mkdir -p logs

# 0. Setup environment
source "$SCRATCH/eigcov/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

MODELS=(t5-base)
FT_MODES=(lora)

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1. Finetune model
    echo "[BASH] Running finetune.py | model: $MODEL | ft mode: $FT_MODE"
    python scripts/language/finetune.py \
    --finetuning-mode="$FT_MODE" \
    --model="$MODEL" \
    --world-size=1 \
    --hf-cache-dir=$SCRATCH/hf_cache 

  done
done

