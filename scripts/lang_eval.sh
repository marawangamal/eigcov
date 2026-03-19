#!/bin/bash
#SBATCH --job-name=eval_lang_models
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

HF_CACHE_DIR="$SCRATCH/hf_cache"

# Common parameters
NUM_BATCHES=10
BATCH_SIZE=32

# # ===== Hyperparameter-optimized experiments =====
# # Only evaluate TA (sum) since other methods do not require HP tuning.
# # Results are written to a separate database to avoid mixing with the
# # default runs.
# MODELS=(t5-base)
# METHODS=(tsv eigcov)
# FT_MODES=(standard)
# RESULTS_DB="results/results-hpopt.jsonl"
# HPO='{"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'

# ===== Default experiments (no hyperparameter tuning) =====
# Evaluate all merging methods using their default settings.
# Results are stored in the main results database.
MODELS=(t5-large)
METHODS=(tsv eigcov regmean)
FT_MODES=(standard)
RESULTS_DB="results/results.jsonl"
HPO=""


for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1a. Evaluate single task (finetuned)
    case "$FT_MODE" in
      standard) _ST_FT="checkpoints/$MODEL/ft_accuracies.json" ;;
      linear)   _ST_FT="checkpoints/$MODEL/linear_ft_accuracies.json" ;;
      lora)     _ST_FT="checkpoints/$MODEL/lora_ft_accuracies.json" ;;
      *)        echo "Unknown finetuning mode: $FT_MODE"; exit 1 ;;
    esac
    if [ -f "$_ST_FT" ]; then
      echo "[BASH] Skipping eval_single_task.py | $_ST_FT already exists"
    else
      echo "[BASH] Running eval_single_task.py | model: $MODEL | ft mode: $FT_MODE"
      python scripts/language/eval_single_task.py \
        --finetuning-mode="$FT_MODE" \
        --hf-cache-dir="$HF_CACHE_DIR" \
        --model="$MODEL" \
        --results-db="$RESULTS_DB"
    fi

    # 1b. Evaluate single task (zeroshot)
    _ST_ZS="checkpoints/$MODEL/zeroshot_accuracies.json"
    if [ -f "$_ST_ZS" ]; then
      echo "[BASH] Skipping eval_single_task.py (zeroshot) | $_ST_ZS already exists"
    else
      echo "[BASH] Running eval_single_task.py | model: $MODEL | ft mode: none"
      python scripts/language/eval_single_task.py \
        --finetuning-mode="none" \
        --hf-cache-dir="$HF_CACHE_DIR" \
        --model="$MODEL" \
        --results-db="$RESULTS_DB"
    fi

    # 2. Evaluate task addition w/ diff merge methods
    for method in "${METHODS[@]}"; do

      # 2a. Run covariance.py if regmean method is used
      if [ "$method" = "regmean" ]; then
        echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
        python scripts/language/covariance.py \
          --model="$MODEL" \
          --finetuning-mode="$FT_MODE" \
          --cov-split=train \
          --cov-num-batches="$NUM_BATCHES" \
          --cov-batch-size="$BATCH_SIZE" \
          --cov-type=sm \
          --cov-estimator=full
      fi

      # 2b. Set cov-dir (only meaningful for regmean/fisher)
      if [ "$method" = "fisher" ]; then
        COV_DIR="checkpoints/$MODEL/_fishers/fisher_strain_n${NUM_BATCHES}_b${BATCH_SIZE}_ft${FT_MODE}"
      elif [ "$method" = "regmean" ]; then
        COV_DIR="checkpoints/$MODEL/_covariances/covariances_strain_n${NUM_BATCHES}_b${BATCH_SIZE}_tsm_efull_ft${FT_MODE}"
      else
        COV_DIR="None"
      fi

      # 2c. Evaluate task addition
      echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/language/eval_task_addition.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --merge-func="$method" \
        --cov-dir="$COV_DIR" \
        --results-db="$RESULTS_DB" \
        --hf-cache-dir="$HF_CACHE_DIR" \
        ${HPO:+--hpo="$HPO"}

    done
  done
done
