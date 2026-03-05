"""Computes per-parameter diagonal Fisher Information Matrix approximation.

For each task, saves an npz with keys matching param_key_to_cov_key:
  - image_encoder.<layer>: (Do*Di,) flattened averaged squared gradient
  - n_batches: number of batches used

The empirical Fisher diagonal is estimated as:
  F_i = (1/N) * sum_n (dL_n/d theta_i)^2

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/vision/fisher.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets
python scripts/vision/fisher.py --model=ViT-B-16 --finetuning-mode=lora --cov-split=train --cov-num-batches=1,10,100 --cov-batch-size=32 ...
"""

import gc
import os
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.vision.heads import get_classification_head
from src.vision.modeling import ImageClassifier
from src.args import parse_arguments
from src.vision.datasets.registry import get_dataset
from src import mhap, mhas


def compute_fisher(encoder, dataset_name, args, on_batch=None, on_end=None):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(encoder, classification_head)
    model.freeze_head()
    model.eval()
    model.to(args.model_device)

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.cov_batch_size,
        num_workers=args.num_workers,
    )
    split = args.cov_split
    max_num_batches = max(args.cov_num_batches)
    loader = dataset.train_loader if split == "train" else dataset.test_loader
    dataset_size = len(loader.dataset)
    print(f"    {dataset_size} samples (split={split})")

    loss_fn = torch.nn.CrossEntropyLoss()

    # Accumulate squared gradients per parameter (running sum, averaged at save time)
    fisher = {
        name: torch.zeros_like(param, device="cpu")
        for name, param in model.image_encoder.named_parameters()
    }

    n_batches = 0
    for images, labels in tqdm(loader, desc="Computing Fisher"):
        if n_batches >= max_num_batches:
            break
        images, labels = images.to(args.model_device), labels.to(args.model_device)
        model.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()

        for name, param in model.image_encoder.named_parameters():
            if param.grad is not None:
                fisher[name] += (param.grad.detach() ** 2).cpu()

        n_batches += 1
        if on_batch is not None:
            on_batch(fisher, n_batches)

    print(f"    Used {n_batches} batches (max={max_num_batches})")

    if on_end is not None:
        on_end(fisher, n_batches)

    model.cpu()
    del model, dataset, loader
    gc.collect()

    return fisher, n_batches


if __name__ == "__main__":
    args = parse_arguments()
    args.save = f"checkpoints/{args.model}"
    args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]

    ss_num_batches = max(args.cov_num_batches)
    ss_batch_size = args.cov_batch_size
    ss_split = args.cov_split
    ss_fm = args.finetuning_mode
    fisher_dir = f"results/{args.model}/fisher_s{ss_split}_n{ss_num_batches}_b{ss_batch_size}_ft{ss_fm}"
    os.makedirs(fisher_dir, exist_ok=True)

    def make_fisher_dir(n):
        return (
            f"results/{args.model}/fisher_s{ss_split}_n{n}_b{ss_batch_size}_ft{ss_fm}"
        )

    print(f"Fisher directory: {fisher_dir}")
    for task in tasks:
        fisher_path = f"{fisher_dir}/fisher_{task}.npz"
        all_cached = all(
            os.path.exists(f"{make_fisher_dir(n)}/fisher_{task}.npz")
            for n in args.cov_num_batches
        )
        if all_cached and not args.overwrite:
            print(f"Skipping {task} (cached)")
            continue

        print(f"\nCollecting Fisher for {task}")
        if args.finetuning_mode == "lora":
            pretrained_checkpoint = f"{args.save}/{task}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{task}Val/lora_finetuned.pt"
            tv = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            encoder = tv.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        elif args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{task}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{task}Val/linear_finetuned.pt"
            pretrained_nonlinear_checkpoint = f"{args.save}/{task}Val/zeroshot.pt"
            nonlinear_encoder = torch.load(
                pretrained_nonlinear_checkpoint, map_location="cpu", weights_only=False
            )
            param_names = [n for n, _ in nonlinear_encoder.named_parameters()]
            del nonlinear_encoder
            tv = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            encoder = tv.apply_to_nonlinear(
                pretrained_nonlinear_checkpoint, param_names, scaling_coef=1.0
            )
        else:
            pretrained_checkpoint = f"{args.save}/{task}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{task}Val/finetuned.pt"
            tv = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            encoder = tv.apply_to(pretrained_checkpoint, scaling_coef=1.0)

        del tv

        # swap mha
        if args.mha is not None:
            swap_fn = {
                "packed": mhap.swap_mha,
                "split": mhas.swap_mha,
            }[args.mha]
            encoder = swap_fn(encoder)

        thresholds = set(n for n in args.cov_num_batches if n < ss_num_batches)

        def on_batch(fisher, n_batches):
            if n_batches in thresholds:
                snap_path = f"{make_fisher_dir(n_batches)}/fisher_{task}.npz"
                os.makedirs(os.path.dirname(snap_path), exist_ok=True)
                snap = {
                    k.replace(".weight", ""): (v / n_batches).cpu().numpy().flatten()
                    for k, v in fisher.items()
                }
                snap["n_batches"] = n_batches
                np.savez(snap_path, **snap)
                print(f"  Saved snapshot (n={n_batches}) to {snap_path}")

        def on_end(fisher, n_batches, _fisher_path=fisher_path):
            # two loops to save memory: divide in-place, then build saveable as views
            for v in fisher.values():
                v.div_(n_batches)
            fisher = {
                k.replace(".weight", ""): v.cpu().numpy() for k, v in fisher.items()
            }
            fisher["n_batches"] = n_batches
            np.savez(_fisher_path, **fisher)
            print(f"  Saved to {_fisher_path}")

        compute_fisher(encoder, f"{task}Val", args, on_batch=on_batch, on_end=on_end)
        del encoder
        gc.collect()
