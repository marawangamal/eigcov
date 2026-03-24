#!/bin/bash

export HF_HOME=$SCRATCH/huggingface

METHODS="eigcov tsv isoc mean"

for method in $METHODS; do
  echo "=== Merging with method: $method ==="
  # python scripts/nlg/merge.py \
  #   --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
  #   --finetuned-dirs \
  #     checkpoints/nlg/pmahdavi-Llama-3.1-8B-math-reasoning \
  #     checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding \
  #     checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if \
  #     checkpoints/nlg/pmahdavi-Llama-3.1-8B-general \
  #     checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall \
  #   --merge-func "$method" \
  #   --output-dir "checkpoints/nlg/pmahdavi-Llama-3.1-8B-$method"

  echo "=== Uploading $method to HF Hub ==="
  hf upload "mremila/pmahdavi-Llama-3.1-8B-20K-$method" \
    "checkpoints/nlg/pmahdavi-Llama-3.1-8B-$method" \
    --repo-type model
  echo ""
done
