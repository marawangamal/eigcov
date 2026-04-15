#!/usr/bin/env bash
# Minimal end-to-end smoke test: finetune a couple datasets briefly, then merge & eval.
# Writes to dedicated test dirs so it doesn't collide with real runs.

set -euo pipefail


# Setup environment
export PYTHONPATH="$PYTHONPATH:$(pwd)" # Add src to python path
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
source "$SCRATCH/eigcov/.venv-vl/bin/activate"

# Set vars
TEST_CKPT_DIR="testing-checkpoints"
TEST_RESULTS_DIR="testing-results"
MODEL="ViT-B-32"
DATASETS="MNIST,SVHN"
DATA_DIR="$SLURM_TMPDIR/datasets"
CACHE_DIR="$SCRATCH/openclip"

# Prepare datasets
if [ ! -d "$DATA_DIR" ]; then
  cp vit_datasets_08.zip "$SLURM_TMPDIR/"
  unzip -q "$SLURM_TMPDIR/vit_datasets_08.zip" -d "$SLURM_TMPDIR/"
fi

echo "=== [1/2] finetune (max-steps=2 on $DATASETS) ==="
python scripts/vision/finetune.py \
    --model="$MODEL" \
    --finetuning-mode=standard \
    --train-dataset="$DATASETS" \
    --save="$TEST_CKPT_DIR" \
    --cache-dir="$CACHE_DIR" \
    --max-steps=2 \
    --batch-size=16 \
    --num-workers=1

echo "=== [2/2] eval_task_addition ==="
python scripts/vision/eval_task_addition.py \
    --model="$MODEL" \
    --finetuning-mode=standard \
    --eval-datasets="$DATASETS" \
    --save="$TEST_CKPT_DIR/$MODEL/max_steps_2" \
    --cache-dir="$CACHE_DIR" \
    --results-dir="$TEST_RESULTS_DIR" \
    --eval-val-max-batches=1 \
    --overwrite

RESULTS_FILE="$TEST_RESULTS_DIR/$MODEL-sum/metrics.json"
if [[ ! -f "$RESULTS_FILE" ]]; then
    echo "FAIL: expected $RESULTS_FILE to exist"
    exit 1
fi
echo "PASS: $RESULTS_FILE"
