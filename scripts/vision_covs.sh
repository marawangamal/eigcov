#!/bin/bash
#SBATCH --job-name=vision_covs
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

if [ ! -d "$SLURM_TMPDIR/datasets" ]; then
  cp vit_datasets_08.zip "$SLURM_TMPDIR/"
  unzip -q "$SLURM_TMPDIR/vit_datasets_08.zip" -d "$SLURM_TMPDIR/"
fi

MODELS=(ViT-B-16)
FT_MODES=(standard)
BATCH_SIZE=1
NUM_BATCHES="1,10,100,500,1000"

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do
    echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE"
    python scripts/vision/covariance.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --cov-split=train \
      --cov-num-batches="$NUM_BATCHES" \
      --cov-batch-size=$BATCH_SIZE \
      --mha=split \
      --cov-type=sm \
      --cov-estimator=full \
      --data-location="$SLURM_TMPDIR/datasets" \
      --openclip-cachedir="$SCRATCH/openclip"
  done
done
