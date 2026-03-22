#!/bin/bash
#SBATCH --job-name=eval_vision_covs
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

# ── Configuration ────────────────────────────────────────────────────────
MODELS=(ViT-B-16)
FT_MODES=(standard)
BATCH_SIZE=1
NUM_BATCHES_LIST=(1 10 100 500 1000)
RESULTS_DB="results/results.jsonl"
ESTIMATOR=avg
# ─────────────────────────────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1. EigCov (data-free, run once)
    echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: eigcov"
    python scripts/vision/eval_task_addition.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --data-location="$DATA_DIR" \
      --merge-func=eigcov \
      --mha=split \
      --cov-dir=None \
      --results-db="$RESULTS_DB"

    # 2. Collect covariances for all sample counts in one pass
    NUM_BATCHES_CSV=$(IFS=,; echo "${NUM_BATCHES_LIST[*]}")
    echo "[BASH] Running covariance.py | model: $MODEL | n: $NUM_BATCHES_CSV | b: $BATCH_SIZE"
    python scripts/vision/covariance.py \
      --model="$MODEL" \
      --cov-split=train \
      --cov-num-batches="$NUM_BATCHES_CSV" \
      --cov-batch-size="$BATCH_SIZE" \
      --mha=split \
      --cov-type=sm \
      --cov-estimator="$ESTIMATOR" \
      --finetuning-mode="$FT_MODE" \
      --openclip-cachedir="$OPENCLIP_DIR" \
      --data-location="$DATA_DIR"

    # 3. Evaluate RegMean at each sample count
    for NUM_BATCHES in "${NUM_BATCHES_LIST[@]}"; do
      COV_DIR="results/$MODEL/covariances_strain_n${NUM_BATCHES}_b${BATCH_SIZE}_tsm_attnsplit_e${ESTIMATOR}_ft${FT_MODE}"

      echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: regmean | n: $NUM_BATCHES"
      python scripts/vision/eval_task_addition.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --data-location="$DATA_DIR" \
        --merge-func=regmean \
        --mha=split \
        --cov-dir="$COV_DIR" \
        --results-db="$RESULTS_DB"
    done
  done
done


# DEBUG:

# Test normalized accuracy: 0.9189491817263742
# Test absolute accuracy: 0.8533028547806022
# python scripts/vision/eval_task_addition.py \
#   --model="ViT-B-16" \
#   --finetuning-mode="standard" \
#   --data-location="$SLURM_TMPDIR/datasets" \
#   --merge-func=regmean \
#   --mha=split \
#   --save="checkpoints-v1/ViT-B-16" \
#   --cov-dir="results-v1/ViT-B-16/covariances_strain_n1000_b1_tsm_attnsplit_efull_ftstandard"


# Test normalized accuracy: 0.9265262701184132
# Test absolute accuracy: 0.8602708905502066
# python scripts/vision/eval_task_addition.py \
#   --model="ViT-B-16" \
#   --finetuning-mode="standard" \
#   --data-location="$SLURM_TMPDIR/datasets" \
#   --merge-func=regmean \
#   --mha=split \
#   --save="checkpoints-v2/ViT-B-16" \
#   --cov-dir="results-v2/ViT-B-16/covariances_strain_n1000_b1_tsm_attnsplit_efull_ftstandard"


# Test normalized accuracy: 0.9185090906665918
# Test absolute accuracy: 0.8528623753209148
# python scripts/vision/eval_task_addition.py   --model="ViT-B-16"   --finetuning-mode="standard"   --data-location="$SLURM_TMPDIR/datasets"   --merge-func=regmean   --mha=split   --save="checkpoints-v1/ViT-B-16"   --cov-dir="results-v1/ViT-B-16/covariances_strain_n500_b1_tsm_attnsplit_efull_ftstandard"
