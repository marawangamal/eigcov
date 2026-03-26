#!/bin/bash
# Merge + evaluate NLG models in one shot.
#
# For each merge method: runs merge.py, optionally uploads to HF Hub,
# evaluates via olmes, then collects all results into a summary table.
#
# Usage:
#   bash scripts/nlg/eval_task_addition.sh \
#     --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
#     --finetuned-dirs \
#       checkpoints/nlg/pmahdavi-Llama-3.1-8B-math-reasoning \
#       checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding \
#       checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if \
#       checkpoints/nlg/pmahdavi-Llama-3.1-8B-general \
#       checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall \
#     --merge-funcs "eigcov tsv isoc mean" \
#     --gpus 4

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PRETRAINED_DIR=""
FINETUNED_DIRS=()
MERGE_FUNCS="eigcov tsv isoc mean"
OUTPUT_BASE="checkpoints/nlg"
RESULTS_BASE="results-nlg"
GPUS=4
UPLOAD=""          # HF Hub user/org prefix; empty = skip upload
MERGE_KWARGS=""    # JSON string forwarded to merge.py --merge-kwargs
IGNORE_KEYS=""     # space-separated keys forwarded to merge.py --ignore-keys
NO_CODE=""         # set to 1 to pass --no-code to collect_results.py
BATCH_SIZE=128     # olmes batch size

# ── Olmes config ──────────────────────────────────────────────────────────────
OLMES_TASKS=(
  "codex_humaneval::tulu"
  "codex_humanevalplus::tulu"
  "gsm8k::tulu"
  "drop::llama3"
  "minerva_math::tulu"
  "ifeval::tulu"
  "popqa::tulu"
  "bbh:cot-v1::tulu"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}'

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pretrained-dir)
      PRETRAINED_DIR="$2"; shift 2 ;;
    --finetuned-dirs)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        FINETUNED_DIRS+=("$1"); shift
      done
      ;;
    --merge-funcs)
      MERGE_FUNCS="$2"; shift 2 ;;
    --output-base)
      OUTPUT_BASE="$2"; shift 2 ;;
    --results-base)
      RESULTS_BASE="$2"; shift 2 ;;
    --gpus)
      GPUS="$2"; shift 2 ;;
    --upload)
      UPLOAD="$2"; shift 2 ;;
    --merge-kwargs)
      MERGE_KWARGS="$2"; shift 2 ;;
    --ignore-keys)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        IGNORE_KEYS+="$1 "; shift
      done
      IGNORE_KEYS="${IGNORE_KEYS% }"  # trim trailing space
      ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --no-code)
      NO_CODE=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "$PRETRAINED_DIR" ]]; then
  echo "Error: --pretrained-dir is required" >&2; exit 1
fi
if [[ ${#FINETUNED_DIRS[@]} -eq 0 ]]; then
  echo "Error: --finetuned-dirs requires at least one directory" >&2; exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
MODEL_PREFIX="$(basename "$PRETRAINED_DIR")"
RESULT_DIRS=()

for method in $MERGE_FUNCS; do
  OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_PREFIX}-${method}"
  RESULTS_DIR="${RESULTS_BASE}/${MODEL_PREFIX}-${method}"

  echo "============================================================"
  echo "Method     : ${method}"
  echo "Output     : ${OUTPUT_DIR}"
  echo "Results    : ${RESULTS_DIR}"
  echo "============================================================"

  # 1. Merge (skip if output dir already exists)
  if [[ -d "$OUTPUT_DIR" ]]; then
    echo ">>> Skipping merge: ${OUTPUT_DIR} already exists"
  else
    MERGE_CMD=(
      python scripts/nlg/merge.py
      --pretrained-dir "$PRETRAINED_DIR"
      --finetuned-dirs "${FINETUNED_DIRS[@]}"
      --merge-func "$method"
      --output-dir "$OUTPUT_DIR"
    )
    if [[ -n "$MERGE_KWARGS" ]]; then
      MERGE_CMD+=(--merge-kwargs "$MERGE_KWARGS")
    fi
    if [[ -n "$IGNORE_KEYS" ]]; then
      MERGE_CMD+=(--ignore-keys $IGNORE_KEYS)
    fi

    echo ">>> Merging with ${method} ..."
    "${MERGE_CMD[@]}"
  fi

  # 2. Optional HF Hub upload
  if [[ -n "$UPLOAD" ]]; then
    REPO_NAME="${UPLOAD}/$(basename "$OUTPUT_DIR")"
    echo ">>> Uploading to ${REPO_NAME} ..."
    hf upload "$REPO_NAME" "$OUTPUT_DIR" --repo-type model
  fi

  # 3. Evaluate via olmes (skip if results already exist)
  if ls "$RESULTS_DIR"/*-metrics.json &>/dev/null; then
    echo ">>> Skipping eval: ${RESULTS_DIR} already has results"
  else
    echo ">>> Evaluating with olmes ..."
    olmes \
      --model "$OUTPUT_DIR" \
      --task "${OLMES_TASKS[@]}" \
      --output-dir "$RESULTS_DIR" \
      --gpus "$GPUS" \
      --model-type vllm \
      --model-args "$OLMES_MODEL_ARGS" \
      --batch-size "$BATCH_SIZE"
  fi

  RESULT_DIRS+=("$RESULTS_DIR")
  echo ""
done

# 4. Collect results across all methods
echo "============================================================"
echo "Collecting results ..."
echo "============================================================"
COLLECT_CMD=(python scripts/nlg/collect_results.py --dirs "${RESULT_DIRS[@]}")
if [[ -n "$NO_CODE" ]]; then
  COLLECT_CMD+=(--no-code)
fi
"${COLLECT_CMD[@]}"
