#!/bin/bash
#SBATCH --job-name=finetune_lang_models
#SBATCH --array=0-3
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# 0. Setup environment
cd $SCRATCH/eigcov
mkdir -p logs
source .venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

HF_CACHE_DIR="$SCRATCH/hf_cache"

# ── Map array task ID to (model, ft_mode) ──
# 0: t5-base/standard   1: t5-base/lora
# 2: t5-large/standard  3: t5-large/lora
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
FT_MODES=(lora standard)

NUM_FT_MODES=${#FT_MODES[@]}
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / NUM_FT_MODES))]}
FT_MODE=${FT_MODES[$((SLURM_ARRAY_TASK_ID % NUM_FT_MODES))]}

echo "[BASH] Running finetune.py | model: $MODEL | ft mode: $FT_MODE"
python scripts/language/finetune.py \
  --finetuning-mode="$FT_MODE" \
  --model="$MODEL" \
  --world-size=1 \
  --hf-cache-dir="$HF_CACHE_DIR"
