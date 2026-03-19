#!/bin/bash
#SBATCH --job-name=eval_lang_hpo
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# set -euo pipefail
# mkdir -p logs

# 0. Setup environment
source "$SCRATCH/eigcov/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

HF_CACHE_DIR="$SCRATCH/hf_cache"

# ── Configuration ────────────────────────────────────────────────────────
MODELS=(t5-large)
METHODS=(eigcov_general)
FT_MODES=(standard)
RESULTS_DB="results/results.jsonl"
SAVE="checkpoints"

HPOS=(
  # Regularized experiments.
  # EigCov (default)
  '{"alpha_weighted": [false], "cov_weighted": [false], "lam": [0.00001, 0.0001, 0.001, 0.01], "solver": ["solve"]}'
  # EigCov (normalized covariance)
  '{"alpha_weighted": [false], "cov_weighted": [true], "lam": [0.00001, 0.0001, 0.001, 0.01], "solver": ["solve"]}'
  # EigCov (normalized objective)
  '{"alpha_weighted": [true], "cov_weighted": [false], "lam": [0.00001, 0.0001, 0.001, 0.01], "solver": ["solve"]}'

  # Unregularized experiments.
  # EigCov (default)
  '{"alpha_weighted": [false], "cov_weighted": [false], "solver": ["solve"]}'
  # EigCov (normalized covariance)
  '{"alpha_weighted": [false], "cov_weighted": [true], "solver": ["solve"]}'
  # EigCov (normalized objective)
  '{"alpha_weighted": [true], "cov_weighted": [false], "solver": ["solve"]}'
)
# ─────────────────────────────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1a. Evaluate single task (finetuned)
    case "$FT_MODE" in
      standard) _ST_FT="$SAVE/$MODEL/ft_accuracies.json" ;;
      linear)   _ST_FT="$SAVE/$MODEL/linear_ft_accuracies.json" ;;
      lora)     _ST_FT="$SAVE/$MODEL/lora_ft_accuracies.json" ;;
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
        --results-db="$RESULTS_DB" \
        --save="$SAVE"
    fi

    # 1b. Evaluate single task (zeroshot)
    _ST_ZS="$SAVE/$MODEL/zeroshot_accuracies.json"
    if [ -f "$_ST_ZS" ]; then
      echo "[BASH] Skipping eval_single_task.py (zeroshot) | $_ST_ZS already exists"
    else
      echo "[BASH] Running eval_single_task.py | model: $MODEL | ft mode: none"
      python scripts/language/eval_single_task.py \
        --finetuning-mode="none" \
        --hf-cache-dir="$HF_CACHE_DIR" \
        --model="$MODEL" \
        --results-db="$RESULTS_DB" \
        --save="$SAVE"
    fi

    # 2. Loop over methods and HPO configs
    for METHOD in "${METHODS[@]}"; do
      for HPO in "${HPOS[@]}"; do
        echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $METHOD | hpo: $HPO"
        python scripts/language/eval_task_addition.py \
          --model="$MODEL" \
          --finetuning-mode="$FT_MODE" \
          --merge-func="$METHOD" \
          --cov-dir="None" \
          --results-db="$RESULTS_DB" \
          --hf-cache-dir="$HF_CACHE_DIR" \
          --hpo="$HPO" \
          --save="$SAVE"
      done
    done

  done
done


# # Prototype evaluation
# python scripts/language/eval_task_addition.py \
#   --model="t5-base" \
#   --finetuning-mode="standard" \
#   --merge-func="eigcov" \
#   --cov-dir="None" \
#   --results-db="results/results-tests.jsonl" \
#   --hf-cache-dir="$SCRATCH/hf_cache" \
#   --hpo='{"alpha_weighted": [false], "cov_weighted": [false], "lam": [0.00001, 0.1, 1.0]}'