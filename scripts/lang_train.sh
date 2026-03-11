#!/bin/bash
#SBATCH --job-name=finetune_lang_models
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# 0. Setup environment
cd $SCRATCH/eigcov
mkdir -p logs
source .venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

HF_CACHE_DIR="$SCRATCH/hf_cache"

MODELS=(t5-large)
FT_MODES=(lora)

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1. Finetune model
    echo "[BASH] Running finetune.py | model: $MODEL | ft mode: $FT_MODE"
    python scripts/language/finetune.py \
      --finetuning-mode="$FT_MODE" \
      --model="$MODEL" \
      --world-size=1 \
      --hf-cache-dir="$HF_CACHE_DIR"

  done
done
