#!/bin/bash
#SBATCH --job-name=vision_covs_nsamples
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL="ViT-B-16"
NSAMPLES="1,10,100,500,1000"
BATCH_SIZE=1
SPLIT="train"
MHA="split"
COV_TYPE="sm"
COV_ESTIMATOR="full"
# ──────────────────────────────────────────────────────────────────────────────

export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

if [ ! -d "$SLURM_TMPDIR/datasets" ]; then
  cp vit_datasets_08.zip "$SLURM_TMPDIR/"
  unzip -q "$SLURM_TMPDIR/vit_datasets_08.zip" -d "$SLURM_TMPDIR/"
fi

python scripts/vision/covariance.py \
    --model="$MODEL" \
    --cov-split="$SPLIT" \
    --cov-num-batches="$NSAMPLES" \
    --cov-batch-size="$BATCH_SIZE" \
    --mha="$MHA" \
    --cov-type="$COV_TYPE" \
    --cov-estimator="$COV_ESTIMATOR" \
    --openclip-cachedir="$SCRATCH/openclip" \
    --data-location="$SLURM_TMPDIR/datasets"
