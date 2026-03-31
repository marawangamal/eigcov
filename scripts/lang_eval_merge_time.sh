#!/bin/bash
#SBATCH --job-name=eval_lang_merge_time
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# 0. Setup environment
source "$SCRATCH/eigcov/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

HF_CACHE_DIR="$SCRATCH/hf_cache"
RESULTS_DB="results-tracked/results-latency.jsonl"
SAVE="checkpoints"

# ── Models / finetuning modes ───────────────────────────────────────────
MODELS=(t5-large)
NUM_BATCHES=10
BATCH_SIZE=32
FT_MODE=standard

# ── Default methods (from lang_eval.sh) ─────────────────────────────────
DEFAULT_METHODS=(tsv eigcov isoc mean sum regmean)

for MODEL in "${MODELS[@]}"; do
    for METHOD in "${DEFAULT_METHODS[@]}"; do
      COV_DIR="checkpoints/$MODEL/_covariances/covariances_strain_n${NUM_BATCHES}_b${BATCH_SIZE}_tsm_efull_ft${FT_MODE}"
      echo "[BASH] Timing merge | model: $MODEL | ft mode: $FT_MODE | method: $METHOD"
      python scripts/language/eval_merge_time.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --merge-func="$METHOD" \
        --cov-dir="$COV_DIR" \
        --results-db="$RESULTS_DB" \
        --hf-cache-dir="$HF_CACHE_DIR" \
        --save="$SAVE"
    done
done


# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
# +-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA L40S                    On  |   00000000:61:00.0 Off |                    0 |
# | N/A   26C    P8             33W /  325W |       0MiB /  46068MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+

# +-----------------------------------------------------------------------------------------+
# | Processes:                                                                              |
# |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
# |        ID   ID                                                               Usage      |
# |=========================================================================================|
# |  No running processes found                                                             |
# +-----------------------------------------------------------------------------------------+