#!/bin/bash
# Download OLMo models and organize into checkpoint structure matching vision layout:
#
#   checkpoints/Olmo-3-7b/
#     pretrained/          (param folder — shared)
#     Math/
#       zeroshot/          (symlink → pretrained)
#       finetuned/         (param folder)
#     Code/
#       zeroshot/          (symlink → pretrained)
#       finetuned/         (param folder)
#     IF/
#       zeroshot/          (symlink → pretrained)
#       finetuned/         (param folder)
#
# Usage:
#   bash scripts/olmo/download_models.sh
set -euo pipefail

MODEL=
PRETRAINED_ID="allenai/Olmo-3-1025-7B"
FINETUNED_IDS=(
  "allenai/Olmo-3-7B-RL-Zero-Math"
  "allenai/Olmo-3-7B-RL-Zero-Code"
  "allenai/Olmo-3-7B-RL-Zero-IF"
)

BASE="checkpoints/Olmo-3-7b"
PRETRAINED_DIR="${BASE}/pretrained"

# 1. Download pretrained model
if [[ -d "$PRETRAINED_DIR" ]]; then
  echo ">>> Skipping pretrained: ${PRETRAINED_DIR} already exists"
else
  echo ">>> Downloading pretrained: ${PRETRAINED_ID}"
  python scripts/olmo/save_model_param_folder.py --model "$PRETRAINED_ID" --output-dir "$PRETRAINED_DIR"
fi

# 2. Download finetuned models and symlink zeroshot
for hf_id in "${FINETUNED_IDS[@]}"; do
  task="${hf_id##*-}"  # extract last segment: Math, Code, IF

  # Download finetuned
  ft_dir="${BASE}/${task}/finetuned"
  if [[ -d "$ft_dir" ]]; then
    echo ">>> Skipping ${task}/finetuned: already exists"
  else
    echo ">>> Downloading ${task}/finetuned: ${hf_id}"
    python scripts/olmo/save_model_param_folder.py --model "$hf_id" --output-dir "$ft_dir"
  fi

  # Symlink zeroshot → pretrained (always refresh to handle re-downloads)
  zs_link="${BASE}/${task}/zeroshot"
  rm -f "$zs_link"
  ln -s "$(realpath "$PRETRAINED_DIR")" "$zs_link"
  echo ">>> Symlinked ${task}/zeroshot → $(realpath "$PRETRAINED_DIR")"
done
