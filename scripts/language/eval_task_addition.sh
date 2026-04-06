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

# ===== Default experiments (no hyperparameter tuning) =====
MODELS=(t5-large)
METHODS=(eigcov tsv)
FT_MODE=standard
HPO=""

for MODEL in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do

    # Run covariance script if needed
    if [ "$method" = "regmean" ]; then
      echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/language/covariance.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --overwrite
    fi

    # Run fisher collection if needed
    if [ "$method" = "fisher" ]; then
      echo "[BASH] Running fisher.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/language/fisher.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --overwrite
    fi

    # Evaluate task addition
    echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
    python scripts/language/eval_task_addition.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --merge-func="$method" \
      ${HPO:+--hpo="$HPO"}

  done
done
