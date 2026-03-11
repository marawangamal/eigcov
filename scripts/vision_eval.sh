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

# # ===== Hyperparameter-optimized experiments =====
# # Only evaluate TA (sum) since other methods do not require HP tuning.
# # Results are written to a separate database to avoid mixing with the
# # default runs.
# MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
# METHODS=(sum)
# FT_MODES=(lora)
# RESULTS_DB="results/results-hpopt.jsonl"
# COEFF_START=0.0
# COEFF_END=1.0
# N_EVAL_POINTS=11

# ===== Default experiments (no hyperparameter tuning) =====
# Evaluate all merging methods using their default settings.
# Results are stored in the main results database.
MODELS=(ViT-B-16)
# METHODS=(eigcov isoc_mean knots_isoc_mean tsv knots_tsv regmean sum mean)
METHODS=(eigcov_weighted)
FT_MODES=(standard)
RESULTS_DB="results/results.jsonl"
COEFF_START=1.0
COEFF_END=1.0
N_EVAL_POINTS=1


for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1a. Evaluate single task (finetuned)
    case "$FT_MODE" in
      lora)    _ST_FT="checkpoints/$MODEL/lora_ft_accuracies.json" ;;
      linear)  _ST_FT="checkpoints/$MODEL/linear_ft_accuracies.json" ;;
      posthoc) _ST_FT="checkpoints/$MODEL/posthoc_ft_accuracies.json" ;;
      *)       _ST_FT="checkpoints/$MODEL/ft_accuracies.json" ;;
    esac
    if [ -f "$_ST_FT" ]; then
      echo "[BASH] Skipping eval_single_task.py | $_ST_FT already exists"
    else
      echo "[BASH] Running eval_single_task.py | model: $MODEL | ft mode: $FT_MODE"
      python scripts/vision/eval_single_task.py \
        --finetuning-mode="$FT_MODE" \
        --model="$MODEL" \
        --openclip-cachedir="$OPENCLIP_DIR" \
        --data-location="$DATA_DIR" \
        --results-db="$RESULTS_DB"
    fi

    # 1b. Evaluate single task (zeroshot)
    _ST_ZS="checkpoints/$MODEL/zeroshot_accuracies.json"
    if [ -f "$_ST_ZS" ]; then
      echo "[BASH] Skipping eval_single_task.py (zeroshot) | $_ST_ZS already exists"
    else
      echo "[BASH] Running eval_single_task.py | model: $MODEL | ft mode: none"
      python scripts/vision/eval_single_task.py \
        --finetuning-mode="none" \
        --model="$MODEL" \
        --openclip-cachedir="$OPENCLIP_DIR" \
        --data-location="$DATA_DIR" \
        --results-db="$RESULTS_DB"
    fi

    # 2. Evaluate task addition w/ diff merge methods
    for method in "${METHODS[@]}"; do

      # 2a. Run covariance/fisher script if needed
      if [ "$method" = "regmean" ]; then
        echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
        python scripts/vision/covariance.py \
          --model="$MODEL" \
          --cov-split=train \
          --cov-num-batches="$NUM_BATCHES" \
          --cov-batch-size="$BATCH_SIZE" \
          --mha=split \
          --cov-type=sm \
          --cov-estimator=full \
          --finetuning-mode="$FT_MODE" \
          --openclip-cachedir="$OPENCLIP_DIR" \
          --data-location="$DATA_DIR"
      elif [ "$method" = "fisher" ]; then
        echo "[BASH] Running fisher.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
        python scripts/vision/fisher.py \
          --model="$MODEL" \
          --cov-split=train \
          --cov-num-batches="$NUM_BATCHES" \
          --cov-batch-size="$BATCH_SIZE" \
          --finetuning-mode="$FT_MODE" \
          --openclip-cachedir="$OPENCLIP_DIR" \
          --data-location="$DATA_DIR"
      fi

      # 2b. Set cov-dir (only meaningful for regmean/fisher)
      if [ "$method" = "fisher" ]; then
        COV_DIR="results/$MODEL/fisher_strain_n${NUM_BATCHES}_b${BATCH_SIZE}_ft${FT_MODE}"
      elif [ "$method" = "regmean" ]; then
        COV_DIR="results/$MODEL/covariances_strain_n${NUM_BATCHES}_b${BATCH_SIZE}_tsm_attnsplit_efull_ft${FT_MODE}"
      else
        COV_DIR="None"
      fi

      # 2c. Evaluate task addition
      echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | coeff start: $COEFF_START | coeff end: $COEFF_END | n eval points: $N_EVAL_POINTS"
      python scripts/vision/eval_task_addition.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --data-location="$DATA_DIR" \
        --merge-func="$method" \
        --mha=split \
        --cov-dir="$COV_DIR" \
        --results-db="$RESULTS_DB" \
        --coeff-start="$COEFF_START" \
        --coeff-end="$COEFF_END" \
        --n-eval-points="$N_EVAL_POINTS"

    done
  done
done