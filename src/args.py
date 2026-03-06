import argparse
import os

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.join(os.environ["SLURM_TMPDIR"], "datasets"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-grad-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",  # noqa: E501
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="None",
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.join(os.environ["SCRATCH"], "openclip"),
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=-1,
        help="How often to checkpoint the model.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=-1,
        help="Keep only the most recent N checkpoints. -1 means keep all checkpoints.",
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
        "--finetuning-mode",
        default="standard",
        choices=["standard", "linear", "posthoc", "lora", "none"],
        help="Whether to use linearized models or not.",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=float, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    # TODO: make these proper regexes
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        # default="out_proj,c_fc,c_proj",
        default="all-linear",  # all linear layers
        help="Comma-separated list of target modules for LoRA.",
    )
    parser.add_argument(
        "--lora-target-parameters",
        type=str,
        # default="in_proj_weight",
        default=None,
        help="Comma-separated list of target parameters for LoRA.",
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
        help="Number of evaluation points used to find optimal coefficient in task arithmetic.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, overwrite cached results instead of loading from cache.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--merge-func",
        type=str,
        default="sum",
        help="Name of tensor-level merge function in src.merging for task arithmetic.",
    )
    parser.add_argument(
        "--cov-dir",
        type=str,
        default=None,
        help=(
            "Directory of per-dataset covariance .npz files produced by scripts/covariance.py. "
            "Expected layout: {cov_dir}/covariance_{dataset}.npz. "
            "Used by eval_task_addition.py and merging.py."
        ),
    )
    parser.add_argument(
        "--cov-split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Dataset split for covariance collection [train, test] (default: train).",
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
        help="Batch size for covariance collection (default: 32).",
    )
    parser.add_argument(
        "--cov-type",
        choices=["cov", "sm"],  # covariance vs second moment (uncentered)
        default="sm",
        help="Type of covariance to collect (default: cov).",
    )
    parser.add_argument(
        "--cov-estimator",
        choices=["sampled", "full"],
        default="full",
        help=(
            "How to estimate the covariance per layer. "
            "'sampled': add a Dx1 vector per sample (one random token position); "
            "'full': add the full DxT matrix per sample (default: sampled)."
        ),
    )
    parser.add_argument(
        "--mha",
        choices=["packed", "split"],
        default=None,
        help="Replace nn.MultiheadAttention in the visual encoder with the custom MultiHeadAttention [packed, split] (default: None).",
    )
    parser.add_argument(
        "--eval-val-split",
        choices=["val", "test", "train"],
        default="val",
        help="Split for phase 1 (coefficient selection) in eval_task_addition (default: val).",
    )
    parser.add_argument(
        "--eval-test-split",
        choices=["val", "test", "train"],
        default="test",
        help="Split for phase 2 (reported metrics) in eval_task_addition (default: test).",
    )
    parser.add_argument(
        "--eval-val-max-batches",
        type=int,
        default=50,
        help="Max number of batches to use in phase 1 (coefficient selection). Default: 50.",
    )
    parser.add_argument(
        "--cosine-samples",
        type=int,
        default=0,
        help=(
            "If > 0, track cosine similarity between consecutive optimizer-step gradients "
            "using the first N trainable parameter tensors. Results saved to {ckpdir}/cosine_sim.npz."
        ),
    )
    parser.add_argument(
        "--grad-cross-ip",
        action="store_true",
        default=False,
        help=(
            "Track cross-sample gradient inner product E[g_k^T g_k'] for k!=k' during training. "
            "Requires batch_size=1. Saves results to {ckpdir}/grad_cross_ip.pt."
        ),
    )
    parser.add_argument(
        "--mid-checkpoint-step",
        type=int,
        default=None,
        help="Training step of the intermediate checkpoint used for eigcov covariance (e.g. 500).",
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
