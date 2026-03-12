import argparse
import os

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="t5-base",
        help="The HuggingFace model name (e.g. t5-base).",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Directory for caching HuggingFace models and datasets.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=str,
        help="Which dataset to fine-tune on.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load a checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--num-grad-accumulation",
        type=int,
        default=16,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=500,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=75000,
        help="Total number of training gradient steps.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length for the tokenizer.",
    )
    parser.add_argument(
        "--finetuning-mode",
        default="standard",
        choices=["none", "standard", "linear", "lora"],
        help="Whether to use standard or linearized fine-tuning, or 'none' for zeroshot.",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=float, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="all-linear",
        help="Comma-separated list of target modules for LoRA.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12355,
        help="Port for distributed training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--coeff-start",
        type=float,
        default=1.0,
        help="Start coefficient for evaluation.",
    )
    parser.add_argument(
        "--coeff-end",
        type=float,
        default=1.0,
        help="End coefficient for evaluation.",
    )
    parser.add_argument(
        "--n-eval-points",
        type=int,
        default=1,
        help="Number of evaluation points for coefficient search in task arithmetic.",
    )
    parser.add_argument(
        "--control-threshold",
        type=float,
        default=0.95,
        help="Threshold for control metric in task negation.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=1000,
        help="Evaluate on validation and checkpoint every N gradient steps.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop training after this many checkpoints without validation improvement.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--merge-func",
        type=str,
        default="sum",
        help="Task vector merge function (e.g. sum, mean, ties, regmean).",
    )
    parser.add_argument(
        "--cov-dir",
        type=str,
        default=None,
        help="Directory containing per-dataset covariance .npz files.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Path to a JSON-lines results database file.",
    )
    parser.add_argument(
        "--eval-val-split",
        type=str,
        default="validation",
        help="Split used for coefficient selection (validation or train).",
    )
    parser.add_argument(
        "--eval-test-split",
        type=str,
        default="test",
        help="Split used for final evaluation.",
    )
    parser.add_argument(
        "--eval-val-max-batches",
        type=int,
        default=None,
        help="Cap number of batches during coefficient selection (None = all).",
    )

    # Covariance collection arguments
    parser.add_argument(
        "--cov-split",
        type=str,
        default="train",
        help="Dataset split to use for covariance collection.",
    )
    parser.add_argument(
        "--cov-num-batches",
        type=lambda x: [int(v) for v in x.split(",")],
        default=[10],
        help="Max number of batches for covariance collection. Comma-separated list for multiple snapshots, e.g. '1,10,100,500,1000' (default: 10).",
    )
    parser.add_argument(
        "--cov-batch-size",
        type=int,
        default=32,
        help="Batch size for covariance collection.",
    )
    parser.add_argument(
        "--cov-type",
        type=str,
        default="sm",
        choices=["cov", "sm"],
        help="Covariance type: centered ('cov') or uncentered second moment ('sm').",
    )
    parser.add_argument(
        "--cov-estimator",
        type=str,
        default="sampled",
        choices=["sampled", "full"],
        help="Covariance estimator: 'sampled' (one token) or 'full' (all tokens).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing covariance files.",
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]

    return parsed_args
