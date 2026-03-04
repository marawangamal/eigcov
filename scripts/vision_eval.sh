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

# # 0. Setup environment
# source "$SCRATCH/eigcov/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

if [ ! -d "$SLURM_TMPDIR/datasets" ]; then
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
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
METHODS=(sum)
FT_MODES=(standard)
RESULTS_DB="results/results-hpopt.jsonl"
COEFF_START=0.0
COEFF_END=1.0
N_EVAL_POINTS=11

# ===== Default experiments (no hyperparameter tuning) =====
# Evaluate all merging methods using their default settings.
# Results are stored in the main results database.
# MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
# METHODS=(eigcov isoc_mean knots_isoc_mean tsv knots_tsv regmean sum mean)
# FT_MODES=(lora standard)
# RESULTS_DB="results/results.jsonl"
# COEFF_START=1.0
# COEFF_END=1.0
# N_EVAL_POINTS=1


for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1. Evaluate single task
    case "$FT_MODE" in
      lora)    _ST_JSON="checkpoints/$MODEL/lora_ft_accuracies.json" ;;
      linear)  _ST_JSON="checkpoints/$MODEL/linear_ft_accuracies.json" ;;
      posthoc) _ST_JSON="checkpoints/$MODEL/posthoc_ft_accuracies.json" ;;
      *)       _ST_JSON="checkpoints/$MODEL/ft_accuracies.json" ;;
    esac
    if [ -f "$_ST_JSON" ]; then
      echo "[BASH] Skipping eval_single_task.py | $_ST_JSON already exists"
    else
      echo "[BASH] Running eval_single_task.py | model: $MODEL | ft mode: $FT_MODE"
      python scripts/vision/eval_single_task.py \
        --finetuning-mode="$FT_MODE" \
        --model="$MODEL" \
        --openclip-cachedir="$SCRATCH/openclip" \
        --data-location="$SLURM_TMPDIR/datasets" \
        --results-db="$RESULTS_DB"
    fi

    # 2. Evaluate task addition w/ diff merge methods
    for method in "${METHODS[@]}"; do

      # 2a. Run covariance.py if regmean method is used
      if [ "$method" = "regmean" ]; then
        echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
        python scripts/vision/covariance.py \
          --model="$MODEL" \
          --cov-split=train \
          --cov-num-batches="$NUM_BATCHES" \
          --cov-batch-size="$BATCH_SIZE" \
          --mha=split \
          --cov-type=sm \
          --cov-estimator=full
      fi

      # 2c. Evaluate task addition (fixed coeff)
      echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | coeff start: $COEFF_START | coeff end: $COEFF_END | n eval points: $N_EVAL_POINTS"
      python scripts/vision/eval_task_addition.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --merge-func="$method" \
        --mha=split \
        --cov-dir="results/$MODEL/covariances_strain_n${NUM_BATCHES}_b${BATCH_SIZE}_tsm_attnsplit_efull_ft${FT_MODE}" \
        --results-db="$RESULTS_DB" \
        --coeff-start="$COEFF_START" \
        --coeff-end="$COEFF_END" \
        --n-eval-points="$N_EVAL_POINTS"

    done
  done
done



# # Prototype evaluation
# python scripts/vision/eval_task_addition.py \
#         --model=ViT-B-16 \
#         --finetuning-mode=lora \
#         --merge-func=mean \
#         --mha=split


# python scripts/vision/eval_task_addition.py \
#   --model=ViT-B-16 \
#   --finetuning-mode=standard \
#   --merge-func=sum \
#   --coeff-start=0.0 \
#   --coeff-end=1.0 \
#   --n-eval-points=11