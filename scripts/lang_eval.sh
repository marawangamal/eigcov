#!/bin/bash
#SBATCH --job-name=eval_vision_models
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

# Common parameters
NUM_BATCHES=10
BATCH_SIZE=32

# ===== Hyperparameter-optimized experiments =====
# Only evaluate TA (sum) since other methods do not require HP tuning.
# Results are written to a separate database to avoid mixing with the
# default runs.
MODELS=(t5-base)
METHODS=(sum)
FT_MODES=(lora)
RESULTS_DB="results/results-hpopt.jsonl"
COEFF_START=0.0
COEFF_END=1.0
N_EVAL_POINTS=11

# # # ===== Default experiments (no hyperparameter tuning) =====
# # Evaluate all merging methods using their default settings.
# # Results are stored in the main results database.
# MODELS=(t5-base)
# METHODS=(regmean)
# # Next do lor and standard for t5-large
# FT_MODES=(standard)
# RESULTS_DB="results/results.jsonl"
# COEFF_START=1.0
# COEFF_END=1.0
# N_EVAL_POINTS=1


for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1a. Evaluate single task
    case "$FT_MODE" in
      standard) _ST_JSON="checkpoints/$MODEL/ft_accuracies.json" ;;
      linear)   _ST_JSON="checkpoints/$MODEL/linear_ft_accuracies.json" ;;
      lora)     _ST_JSON="checkpoints/$MODEL/lora_ft_accuracies.json" ;;
    esac
    if [ -f "$_ST_JSON" ]; then
      echo "[BASH] Skipping eval_single_task.py | $_ST_JSON already exists"
    else
      echo "[BASH] Running eval_single_task.py | model: $MODEL | ft mode: $FT_MODE"
      python scripts/language/eval_single_task.py \
        --finetuning-mode="$FT_MODE" \
        --hf-cache-dir=$SCRATCH/hf_cache \
        --model="$MODEL"
    fi

    # 1b. Evaluate single task (zeroshot)
    _ZS_JSON="checkpoints/$MODEL/zeroshot_accuracies.json"
    if [ -f "$_ZS_JSON" ]; then
      echo "[BASH] Skipping eval_single_task.py (zeroshot) | $_ZS_JSON already exists"
    else
      python scripts/language/eval_single_task.py \
          --finetuning-mode="none" \
          --hf-cache-dir=$SCRATCH/hf_cache \
          --model="$MODEL"
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

      # 2b. Evaluate task addition
      echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | coeff start: $COEFF_START | coeff end: $COEFF_END | n eval points: $N_EVAL_POINTS"
      if [ "$method" = "regmean" ]; then
        python scripts/language/eval_task_addition.py \
          --model="$MODEL" \
          --finetuning-mode="$FT_MODE" \
          --merge-func="$method" \
          --cov-dir="results/$MODEL/covariances_strain_n${NUM_BATCHES}_b${BATCH_SIZE}_tsm_efull_ft${FT_MODE}" \
          --results-db="$RESULTS_DB" \
          --coeff-start="$COEFF_START" \
          --coeff-end="$COEFF_END" \
          --n-eval-points="$N_EVAL_POINTS"
      else
        python scripts/language/eval_task_addition.py \
          --model="$MODEL" \
          --finetuning-mode="$FT_MODE" \
          --merge-func="$method" \
          --results-db="$RESULTS_DB" \
          --coeff-start="$COEFF_START" \
          --coeff-end="$COEFF_END" \
          --n-eval-points="$N_EVAL_POINTS"
      fi

    done
  done
done



# # Prototype evaluation
# python scripts/language/eval_single_task.py \
# --model=t5-base \
# --hf-cache-dir=$SCRATCH/hf_cache \
# --finetuning-mode=standard


# python scripts/vision/eval_single_task.py \
#   --finetuning-mode=lora \
#   --model=ViT-L-14 \
#   --openclip-cachedir=$SCRATCH/openclip \
#   --data-location=$SLURM_TMPDIR/datasets \

# # Prototype evaluation
# python scripts/language/eval_task_addition.py \
# --model=t5-base \
# --finetuning-mode=standard \
# --merge-func=regmean \
# --cov-dir="results/t5-base/covariances_strain_n10_b32_tsm_efull_ftstandard"
