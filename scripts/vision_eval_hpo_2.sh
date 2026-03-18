#!/bin/bash
#SBATCH --job-name=eval_vision_hpo
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

# ── Configuration ────────────────────────────────────────────────────────
MODEL=ViT-B-16
METHOD=eigcov_gd
FT_MODE=standard
RESULTS_DB="results/results.jsonl"
SAVE="checkpoints"

HPOS=(
  # EigCov (default)
  '{"alpha_weighted": [false], "cov_weighted": [false], "lam": [0.00001, 0.0001, 0.001, 0.01]}'
  # EigCov (normalized covariance)
  '{"alpha_weighted": [false], "cov_weighted": [true], "lam": [0.01]}'
  # # EigCov (normalized objective)
  '{"alpha_weighted": [true], "cov_weighted": [false], "lam": [0.00001, 0.0001, 0.001, 0.01]}'

  # Unregularized experiments.
  # EigCov (default)
  '{"alpha_weighted": [false], "cov_weighted": [false]}'
  # EigCov (normalized covariance)
  '{"alpha_weighted": [false], "cov_weighted": [true]}'
  # EigCov (normalized objective)
  '{"alpha_weighted": [true], "cov_weighted": [false]}'
)
# ─────────────────────────────────────────────────────────────────────────

# 1a. Evaluate single task (finetuned)
case "$FT_MODE" in
  lora)    _ST_FT="$SAVE/$MODEL/lora_ft_accuracies.json" ;;
  linear)  _ST_FT="$SAVE/$MODEL/linear_ft_accuracies.json" ;;
  posthoc) _ST_FT="$SAVE/$MODEL/posthoc_ft_accuracies.json" ;;
  *)       _ST_FT="$SAVE/$MODEL/ft_accuracies.json" ;;
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
    --save="$SAVE"
fi

# 1b. Evaluate single task (zeroshot)
_ST_ZS="$SAVE/$MODEL/zeroshot_accuracies.json"
if [ -f "$_ST_ZS" ]; then
  echo "[BASH] Skipping eval_single_task.py (zeroshot) | $_ST_ZS already exists"
else
  echo "[BASH] Running eval_single_task.py | model: $MODEL | ft mode: none"
  python scripts/vision/eval_single_task.py \
    --finetuning-mode="none" \
    --model="$MODEL" \
    --openclip-cachedir="$OPENCLIP_DIR" \
    --data-location="$DATA_DIR" \
    --save="$SAVE"
fi

# 2. Loop over HPO configs
for HPO in "${HPOS[@]}"; do
  echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $METHOD | hpo: $HPO"
  python scripts/vision/eval_task_addition.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE" \
    --data-location="$DATA_DIR" \
    --merge-func="$METHOD" \
    --results-db="$RESULTS_DB" \
    --hpo="$HPO" \
    --save="$SAVE"
done
