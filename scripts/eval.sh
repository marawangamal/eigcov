#!/bin/bash
#SBATCH --job-name=vitb32_lora_merge
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# 0. Setup environment
source $SCRATCH/eigcov/.venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
cp vit_datasets_08.zip $SLURM_TMPDIR/
unzip -q $SLURM_TMPDIR/vit_datasets_08.zip -d $SLURM_TMPDIR/

MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)  
METHODS=(eigcov isoc_mean knots_isoc_mean tsv knots_tsv regmean)
FT_MODES=(standard lora)
RESULTS_DB="results/results.jsonl"
NUM_BATCHES=10
BATCH_SIZE=32

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1. Evaluate single task to get results/model/ftmode_accuracies.json
    echo "-------------------------------------------------"
    echo "Evaluating single task for $MODEL with $FT_MODE"
    echo "-------------------------------------------------"
    python scripts/vision/eval_single_task.py \
    --finetuning-mode=$FT_MODE \
    --model=$MODEL \
    --openclip-cachedir=$SCRATCH/openclip \
    --data-location=$SLURM_TMPDIR/datasets \
    --results-db=$RESULTS_DB

    # 2. Evaluate task addition w/ diff merge methods
    for method in "${METHODS[@]}"; do

        # 2a. Run covariance.py if regmean method is used
        if [ "$method" = "regmean" ]; then
            echo "Running covariance.py for $MODEL with $FT_MODE and $method"
            python scripts/vision/covariance.py \
            --model="$MODEL" \
            --cov-split=train \
            --cov-num-batches="$NUM_BATCHES" \
            --cov-batch-size="$BATCH_SIZE" \
            --mha=split \
            --cov-type=sm \
            --cov-estimator=full
        fi

        # 2b. Evaluate task addition w/ diff merge methods
        echo "Evaluating task addition for $MODEL with $FT_MODE and $method"
        python scripts/vision/eval_task_addition.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --merge-func="$method" \
        --mha=split \
        --cov-dir="results/$MODEL/covariances_strain_n$NUM_BATCHES_b$BATCH_SIZE_tsm_attnsplit_efull_ft$FT_MODE" \
        --results-db=$RESULTS_DB
    done
  done
done