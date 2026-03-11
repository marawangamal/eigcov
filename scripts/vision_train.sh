#!/bin/bash
#SBATCH --job-name=finetune_vision_models
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# 0. Setup environment
source "$SCRATCH/eigcov/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

DATA_DIR="$SLURM_TMPDIR/datasets"
OPENCLIP_DIR="$SCRATCH/openclip"

if [ ! -d "$DATA_DIR" ]; then
  cp vit_datasets_08.zip "$SLURM_TMPDIR/"
  unzip -q "$SLURM_TMPDIR/vit_datasets_08.zip" -d "$SLURM_TMPDIR/"
fi

# ── Option A: train all models/modes, no intermediate checkpoints (default) ──
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
FT_MODES=(standard lora)
CHECKPOINT_EVERY=-1   # -1 = disabled

# # ── Option B: single model, save a checkpoint every N steps ──────────────────
# MODELS=(ViT-B-16)
# FT_MODES=(standard)
# CHECKPOINT_EVERY=200

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1. Finetune model
    echo "[BASH] Running finetune.py | model: $MODEL | ft mode: $FT_MODE | checkpoint-every: $CHECKPOINT_EVERY"
    python scripts/vision/finetune.py \
      --finetuning-mode="$FT_MODE" \
      --model="$MODEL" \
      --world-size=1 \
      --num-workers=1 \
      --checkpoint-every="$CHECKPOINT_EVERY" \
      --openclip-cachedir="$OPENCLIP_DIR" \
      --data-location="$DATA_DIR" \
      --save="$SCRATCH/eigcov/checkpoints"

  done
done
