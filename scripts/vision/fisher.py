"""Computes per-parameter diagonal Fisher Information Matrix approximation.

For each task, saves a .pt dict with keys matching param_key_to_cov_key:
  - image_encoder.<layer>: (Do*Di,) flattened averaged squared gradient
  - n_batches: number of batches used

The empirical Fisher diagonal is estimated as:
  F_i = (1/N) * sum_n (dL_n/d theta_i)^2

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/vision/fisher.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets
"""

import gc
import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.vision.heads import get_classification_head
from src.vision.modeling import ImageClassifier
from src.args import parse_arguments
from src.vision.datasets.registry import get_dataset
from src import mhap, mhas


def compute_fisher(encoder, dataset_name, args, on_end=None):
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
    args.cov_batch_size = 1
    args.cov_num_batches = [100]

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

    for task in tasks:
        checkpoint_dir = f"{args.save}/{task}Val"
        fisher_path = os.path.join(checkpoint_dir, "fisher.pt")

        if os.path.exists(fisher_path) and not args.overwrite:
            print(f"Skipping {task} (cached: {fisher_path})")
            continue

        print(f"\nCollecting Fisher for {task}")
        if args.finetuning_mode == "linear":
            checkpoint_dir_path = checkpoint_dir
            zeroshot_path = os.path.join(checkpoint_dir_path, "zeroshot.pt")
            nonlinear_encoder = torch.load(
                zeroshot_path, map_location="cpu", weights_only=False
            )
            param_names = [n for n, _ in nonlinear_encoder.named_parameters()]
            del nonlinear_encoder
            tv = LinearizedTaskVector(checkpoint_dir=checkpoint_dir_path)
            encoder = tv.apply_to_nonlinear(
                checkpoint_dir_path, param_names, scaling_coef=1.0
            )
        else:
            tv = NonLinearTaskVector(checkpoint_dir=checkpoint_dir)
            encoder = tv.apply_to(checkpoint_dir, scaling_coef=1.0)

        del tv

        # swap mha
        if args.mha is not None:
            swap_fn = {
                "packed": mhap.swap_mha,
                "split": mhas.swap_mha,
            }[args.mha]
            encoder = swap_fn(encoder)

        def on_end(fisher, n_batches, _fisher_path=fisher_path):
            # Divide in-place, then build saveable dict
            for v in fisher.values():
                v.div_(n_batches)
            saveable = {"image_encoder." + k.replace(".weight", ""): v.cpu() for k, v in fisher.items()}
            saveable["n_batches"] = n_batches
            torch.save(saveable, _fisher_path)
            print(f"  Saved to {_fisher_path}")

        compute_fisher(encoder, f"{task}Val", args, on_end=on_end)
        del encoder
        gc.collect()
