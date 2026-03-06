"""Precompute per-layer mid-checkpoint covariance matrices for use with regmean.

For each dataset and each 2D weight matrix, computes:
    Delta_k = W_T - W_k          (weight delta from mid checkpoint to finetuned)
    C_k     = Delta_k.T @ Delta_k  (covariance used by regmean merge)

Keys in the output .npz match the format expected by merge_regmean, which looks
them up via param_key_to_cov_key():  "image_encoder.<module_path>" (no ".weight"
suffix).

Output directory follows the project convention:
    results/{model}/covariances_eigcov_k{step}_ft{mode}

Usage:
python scripts/vision/eigcov.py \
--model ViT-B-16 \
--mid-checkpoint-step 500

    python scripts/vision/eval_task_addition.py \
        --merge-func regmean \
        --cov-dir results/ViT-B-16/covariances_eigcov_k500_ftstandard
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.args import parse_arguments


ALL_DATASETS = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]


def load_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return obj


def param_key_to_cov_key(key: str) -> str:
    return "image_encoder." + key.replace(".weight", "")


if __name__ == "__main__":
    args = parse_arguments()

    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

    datasets = args.eval_datasets if args.eval_datasets is not None else ALL_DATASETS

    cov_dir = f"results/{args.model}/covariances_eigcov_k{args.mid_checkpoint_step}_ft{args.finetuning_mode}"
    os.makedirs(cov_dir, exist_ok=True)
    print(f"Covariance directory: {cov_dir}")

    for dataset in datasets:
        dataset_dir = f"{args.save}/{dataset}Val"

        if args.finetuning_mode == "lora":
            ft_path = f"{dataset_dir}/lora_finetuned.pt"
        else:
            ft_path = f"{dataset_dir}/finetuned.pt"
        mid_path = f"{dataset_dir}/checkpoint_{args.mid_checkpoint_step}.pt"

        if not os.path.exists(ft_path):
            print(f"[skip] {ft_path} not found")
            continue
        if not os.path.exists(mid_path):
            print(f"[skip] {mid_path} not found")
            continue

        print(f"Processing {dataset} ...")
        ft_sd = load_state_dict(ft_path)
        mid_sd = load_state_dict(mid_path)

        covs = {}
        for key in ft_sd:
            if ft_sd[key].dtype in (torch.int64, torch.uint8):
                continue
            if ft_sd[key].ndim != 2:
                continue
            delta = (ft_sd[key] - mid_sd[key]).float()
            cov_key = param_key_to_cov_key(key)
            covs[cov_key] = (delta.T @ delta).numpy()

        cov_path = f"{cov_dir}/covariance_{dataset}.npz"
        np.savez(cov_path, **covs)
        print(f"  Saved {len(covs)} covariance matrices -> {cov_path}")

        del ft_sd, mid_sd
        torch.cuda.empty_cache()
