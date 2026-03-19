#!/bin/bash
#SBATCH --job-name=eval_lang_merge_time
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# set -euo pipefail
# mkdir -p logs

# 0. Setup environment
source "$SCRATCH/eigcov/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

HF_CACHE_DIR="$SCRATCH/hf_cache"
RESULTS_DB="results/results-latency.jsonl"
SAVE="checkpoints"

# ── Models / finetuning modes ───────────────────────────────────────────
MODELS=(t5-large)
FT_MODES=(standard)

# ── Default methods (from lang_eval.sh) ─────────────────────────────────
DEFAULT_METHODS=(tsv eigcov isoc mean sum)

# ── HPO methods (from lang_eval_hpo.sh) ─────────────────────────────────
HPO_METHODS=(eigcov_gd)
HPO_CONFIGS=(
  '{"alpha_weighted": [false], "cov_weighted": [false], "lam": [0.00001]}'
  '{"alpha_weighted": [false], "cov_weighted": [true], "lam": [0.00001]}'
  '{"alpha_weighted": [true], "cov_weighted": [false], "lam": [0.00001]}'
)

# ── HPO methods (from lang_eval_hpo_2.sh) ───────────────────────────────
HPO2_METHODS=(eigcov_general)
HPO2_CONFIGS=(
  '{"alpha_weighted": [false], "cov_weighted": [false], "lam": [0.00001], "solver": ["solve"]}'
  '{"alpha_weighted": [false], "cov_weighted": [true], "lam": [0.00001], "solver": ["solve"]}'
  '{"alpha_weighted": [true], "cov_weighted": [false], "lam": [0.00001], "solver": ["solve"]}'
)

# ─────────────────────────────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1. Default methods (no HPO, no cov-dir)
    for METHOD in "${DEFAULT_METHODS[@]}"; do
      echo "[BASH] Timing merge | model: $MODEL | ft mode: $FT_MODE | method: $METHOD"
      python scripts/language/eval_merge_time.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --merge-func="$METHOD" \
        --cov-dir="None" \
        --results-db="$RESULTS_DB" \
        --hf-cache-dir="$HF_CACHE_DIR" \
        --save="$SAVE"
    done

    # 2. HPO methods (eigcov_gd variants)
    for METHOD in "${HPO_METHODS[@]}"; do
      for HPO in "${HPO_CONFIGS[@]}"; do
        echo "[BASH] Timing merge | model: $MODEL | ft mode: $FT_MODE | method: $METHOD | hpo: $HPO"
        python scripts/language/eval_merge_time.py \
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

    # 3. HPO methods (eigcov_general variants)
    for METHOD in "${HPO2_METHODS[@]}"; do
      for HPO in "${HPO2_CONFIGS[@]}"; do
        echo "[BASH] Timing merge | model: $MODEL | ft mode: $FT_MODE | method: $METHOD | hpo: $HPO"
        python scripts/language/eval_merge_time.py \
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
