#!/bin/bash
#SBATCH --job-name=vision_covs
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL="ViT-B-16"
ROOTDIR="$SCRATCH/eigcov"   # repo root on the cluster

DATASETS=(
    "Cars"
    "DTD"
    "EuroSAT"
    "GTSRB"
    "MNIST"
    "RESISC45"
    "SUN397"
    "SVHN"
)

NUM_BATCHES=10
BATCH_SIZE=32
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

for DATASET in "${DATASETS[@]}"; do
    CKPT_DIR="$ROOTDIR/checkpoints/${MODEL}/${DATASET}Val"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "WARNING: checkpoint directory not found: $CKPT_DIR, skipping $DATASET"
        continue
    fi

    for ckpt in "$CKPT_DIR"/checkpoint_*.pt; do
        [ -f "$ckpt" ] || continue
        ckpt_name=$(basename "$ckpt" .pt)   # e.g. checkpoint_0, finetuned, zeroshot

        echo "[BASH] covariance.py | model: $MODEL | dataset: $DATASET | ckpt: $ckpt_name"
        python scripts/vision/covariance.py \
            --model="$MODEL" \
            --load="$ckpt" \
            --eval-datasets="$DATASET" \
            --cov-split="$SPLIT" \
            --cov-num-batches="$NUM_BATCHES" \
            --cov-batch-size="$BATCH_SIZE" \
            --mha="$MHA" \
            --cov-type="$COV_TYPE" \
            --cov-estimator="$COV_ESTIMATOR" \
            --openclip-cachedir="$SCRATCH/openclip" \
            --data-location="$SLURM_TMPDIR/datasets"
    done
done
