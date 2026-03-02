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
# export PYTHONPATH="$PYTHONPATH:$PWD"
# export SSL_CERT_DIR=/etc/ssl/certs

MODELS=(t5-base)
METHODS=(regmean)
FT_MODES=(lora)
RESULTS_DB="results/results.jsonl"
NUM_BATCHES=10
BATCH_SIZE=32

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1a. Evaluate single task
    echo "[BASH] Running eval_single_task.py | model: $MODEL | ft mode: $FT_MODE"
    python scripts/language/eval_single_task.py \
      --finetuning-mode="$FT_MODE" \
      --hf-cache-dir=$SCRATCH/hf_cache \
      --model="$MODEL" 

    # 1b. Evaluate single task (zeroshot)
    python scripts/language/eval_single_task.py \
        --finetuning-mode="none" \
        --hf-cache-dir=$SCRATCH/hf_cache \
        --model="$MODEL" 

    # 2. Evaluate task addition w/ diff merge methods
    for method in "${METHODS[@]}"; do

      # 2a. Run covariance.py if regmean method is used
      if [ "$method" = "regmean" ]; then
        echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
        python scripts/language/covariance.py \
          --model="$MODEL" \
          --cov-split=train \
          --cov-num-batches="$NUM_BATCHES" \
          --cov-batch-size="$BATCH_SIZE" \
          --cov-type=sm \
          --cov-estimator=full
      fi

      if [ "$method" = "regmean" ]; then
        python scripts/language/eval_task_addition.py \
          --model="$MODEL" \
          --finetuning-mode="$FT_MODE" \
          --merge-func="$method" \
          --cov-dir="results/$MODEL/covariances_strain_n${NUM_BATCHES}_b${BATCH_SIZE}_tsm_efull_ft${FT_MODE}" \
          --results-db="$RESULTS_DB"
      else
        python scripts/language/eval_task_addition.py \
          --model="$MODEL" \
          --finetuning-mode="$FT_MODE" \
          --merge-func="$method" \
          --results-db="$RESULTS_DB"
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