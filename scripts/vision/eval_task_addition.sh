#!/bin/bash
#SBATCH --job-name=eval_vision_models
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

# Common parameters
NUM_BATCHES=10
BATCH_SIZE=32

# ===== Hyperparameter-optimized experiments =====
# Only evaluate TA (sum) since other methods do not require HP tuning.
# Results are written to a separate database to avoid mixing with the
# default runs.
# MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
# METHODS=(sum)
# FT_MODES=(lora)
# HPO='{"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'

# # ===== EigCov General sweep =====
# # Sweep lam with alpha_weighted + cov_weighted enabled.
# MODELS=(ViT-B-16)
# METHODS=(eigcov_general)
# FT_MODES=(standard)
# HPO='{"lam": [0.0001, 0.001, 0.01], "alpha_weighted": [true], "cov_weighted": [true]}'

# ===== Default experiments (no hyperparameter tuning) =====
# Evaluate all merging methods using their default settings.
# Results are stored in the main results database.
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
METHODS=(tsv eigcov regmean isoc mean sum04)
FT_MODE=standard
HPO=""


for MODEL in "${MODELS[@]}"; do
  # Evaluate task addition w/ diff merge methods
  for method in "${METHODS[@]}"; do

    # 2a. Run covariance/fisher script if needed
    if [ "$method" = "regmean" ]; then
      echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/vision/covariance.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --mha=split
    elif [ "$method" = "fisher" ]; then
      echo "[BASH] Running fisher.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/vision/fisher.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE"
    fi

    # 2b. Evaluate task addition
    echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
    python scripts/vision/eval_task_addition.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --data-location="$DATA_DIR" \
      --merge-func="$method" \
      --mha=split \
      ${HPO:+--hpo="$HPO"}

  done
done

