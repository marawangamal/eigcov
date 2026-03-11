#!/bin/bash
#SBATCH --job-name=finetune_vision_models
#SBATCH --array=0-5
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

# 0. Setup environment
source "$SCRATCH/eigcov/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

if [ ! -d "$SLURM_TMPDIR/datasets" ]; then
  cp vit_datasets_08.zip "$SLURM_TMPDIR/"
  unzip -q "$SLURM_TMPDIR/vit_datasets_08.zip" -d "$SLURM_TMPDIR/"
fi

# ── Map array task ID to (model, ft_mode) ──
# 0: ViT-B-16/standard  1: ViT-B-16/lora
# 2: ViT-B-32/standard  3: ViT-B-32/lora
# 4: ViT-L-14/standard  5: ViT-L-14/lora
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
FT_MODES=(standard lora)
CHECKPOINT_EVERY=-1   # -1 = disabled

NUM_FT_MODES=${#FT_MODES[@]}
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / NUM_FT_MODES))]}
FT_MODE=${FT_MODES[$((SLURM_ARRAY_TASK_ID % NUM_FT_MODES))]}

echo "[BASH] Running finetune.py | model: $MODEL | ft mode: $FT_MODE | checkpoint-every: $CHECKPOINT_EVERY"
python scripts/vision/finetune.py \
  --finetuning-mode="$FT_MODE" \
  --model="$MODEL" \
  --world-size=1 \
  --num-workers=1 \
  --checkpoint-every="$CHECKPOINT_EVERY" \
  --openclip-cachedir="$SCRATCH/openclip" \
  --data-location="$SLURM_TMPDIR/datasets" \
  --save="$SCRATCH/eigcov/checkpoints-v2"



# EXPERIMENT LOG:
# 2026-03-08 07:00 - 8904571_[0-3] 
# squeue -u marawan.gamal -o "%.18i %.12u %.10P %.40j %.2t %.19S %.10M %.6D %.8C %.20R"