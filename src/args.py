"""Unified CLI argument parser shared across vision, language, and OLMo pipelines.

Sections below group flags by purpose. Pipeline-specific flags (e.g. --capability,
--data-location, --max-seq-len) live alongside shared flags in the same parser; a
script that doesn't read a given flag simply ignores it.
"""

import argparse
import json
import os

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    # ─── Model & paths ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16",
        help="Model name (vision: OpenCLIP arch e.g. ViT-B-16; language: HF id e.g. t5-base).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save checkpoints / classifiers.",
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load classifier(s).",
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.join(os.environ.get("SLURM_TMPDIR", "/tmp"), "datasets"),
        help="Vision: root dir for image datasets.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join(os.environ.get("SCRATCH", os.path.expanduser("~/.cache")), "models"),
        help="Cache dir for downloaded model weights (vision: openclip cache; language/OLMo: HF_HOME).",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default=None,
        help="Vision-only: dir for caching encoded image features.",
    )

    # ─── Datasets ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to fine-tune on.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Comma-separated.",
    )

    # ─── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--num-grad-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument("--warmup-length", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=75000,
        help="Language: total number of training gradient steps.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Language: maximum sequence length for the tokenizer.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap on optimizer steps for training. None means train all epochs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--port", type=int, default=12355, help="Port for distributed training."
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=-1,
        help="Checkpoint / validation evaluation every N steps. -1 disables.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=-1,
        help="Keep only the most recent N checkpoints. -1 keeps all.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop training after this many checkpoints without validation improvement.",
    )

    # ─── Finetuning mode ──────────────────────────────────────────────────────
    parser.add_argument(
        "--finetuning-mode",
        default="standard",
        choices=["standard", "linear", "posthoc", "lora", "none"],
        help="Finetuning strategy.",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=float, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")

    # ─── Merging ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--merge-func",
        type=str,
        default="sum",
        help="Name of tensor-level merge function in src.merging for task arithmetic.",
    )
    parser.add_argument(
        "--hpo",
        type=json.loads,
        default=None,
        help='JSON dict of HP grid. Example: \'{"lam": [0.0, 0.1, 0.2]}\'.',
    )
    parser.add_argument(
        "--ignore-keys",
        nargs="+",
        default=None,
        help="Substrings of parameter keys to exclude from merging (averaged instead).",
    )
    parser.add_argument(
        "--merge-kwargs",
        type=json.loads,
        default={},
        help="JSON dict of extra kwargs forwarded to the merge function.",
    )

    # ─── Statistics (covariance / fisher) ─────────────────────────────────────
    parser.add_argument(
        "--cov-split",
        type=str,
        default="train",
        help="Dataset split for covariance collection.",
    )
    parser.add_argument(
        "--cov-num-batches",
        type=lambda x: [int(v) for v in x.split(",")],
        default=[10],
        help="Max batches for covariance collection. Comma-separated for multiple snapshots.",
    )
    parser.add_argument(
        "--cov-batch-size", type=int, default=32, help="Batch size for covariance collection."
    )
    parser.add_argument(
        "--cov-type",
        choices=["cov", "sm"],
        default="sm",
        help="Centered covariance ('cov') or uncentered second moment ('sm').",
    )
    parser.add_argument(
        "--cov-estimator",
        choices=["sampled", "full", "avg"],
        default="full",
        help="How to estimate the covariance per layer.",
    )
    parser.add_argument(
        "--mha",
        choices=["packed", "split"],
        default=None,
        help="Vision: replace nn.MultiheadAttention with custom split/packed Q/K/V module.",
    )

    # ─── Evaluation ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--eval-val-max-batches",
        type=int,
        default=50,
        help="Max batches in phase 1 (coefficient selection).",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Path to a JSON-lines results database file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite cached results / existing files.",
    )
    parser.add_argument(
        "--control-threshold",
        type=float,
        default=0.95,
        help="Language: threshold for control metric in task negation.",
    )

    # ─── Vision experimental ──────────────────────────────────────────────────
    parser.add_argument(
        "--grad-cross-matrix",
        action="store_true",
        default=False,
        help="Per-layer matrix measurement of gradient cross-term condition.",
    )

    # ─── OLMo / NLG ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="OLMo: output directory (finetune checkpoints / merged model).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=None,
        help="OLMo merge: directory to copy tokenizer from.",
    )
    parser.add_argument(
        "--capability",
        type=str,
        default=None,
        help="OLMo finetune/covariance: capability subset (math, code, if, all).",
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="OLMo: use bfloat16."
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.03, help="OLMo: warmup ratio."
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        default=False,
        help="OLMo finetune: enable FSDP full sharding.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="OLMo finetune: resume training from latest checkpoint in output dir.",
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
