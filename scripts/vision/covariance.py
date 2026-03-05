"""Collects per-layer ||dL/dy||² and sampled entries of yy^T per sample.

For each layer, saves:
  - g_sq/<layer>:         (N,) array of ||dL/dy||²
  - aat_samples/<layer>:  (N, K) array of sampled entries of yy^T
  - index_pairs/<layer>:  (K, 2) array of which (i,j) pairs were sampled

Activations are flattened (seq_len * D) so aa^T captures the full structure.
K = 32 (i,j) pairs, randomly sampled with a fixed seed for reproducibility.

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/vision/covariance.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets
python scripts/vision/covariance.py --cov-split train --cov-num-batches 100 --cov-batch-size 32 --mha=split ...

# Generate covariance matrices
python scripts/vision/covariance.py --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets \
--model=ViT-B-16 --cov-split train --cov-num-batches 100 --cov-batch-size 32 --mha=split 

"""

import gc
import os
import sys
from pathlib import Path
from typing import Dict
import torch
import numpy as np
from tqdm import tqdm


# Add to python path
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.vision.heads import get_classification_head
from src.vision.modeling import ImageClassifier
from src.args import parse_arguments
from src.vision.datasets.registry import get_dataset
from src import mhap, mhas
from src.covariance import OnlineCovariance, register_hooks


def compute_covs(encoder, dataset_name, args, on_batch=None, on_end=None):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(encoder, classification_head)
    model.freeze_head()
    # model.train()
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

    cobjs, handles = register_hooks(
        model,
        cov_device=args.cov_device,
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        extra_module_types=(
            mhap.MultiHeadAttentionPacked,
            mhas.MultiHeadAttentionSplit,
        ),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    total_batches = len(loader) if max_num_batches is None else None
    n_batches = 0
    with torch.no_grad():
        for images, labels in tqdm(
            loader,
            desc="Computing covariance",
            total=total_batches,
        ):
            if max_num_batches is not None and n_batches >= max_num_batches:
                break
            images, labels = images.cuda(), labels.cuda()
            model.zero_grad()
            _ = loss_fn(model(images), labels)
            n_batches += 1
            if on_batch is not None:
                on_batch(cobjs, n_batches)
    print(f"    Used {n_batches} batches (max={max_num_batches})")

    del model, dataset, loader, handles
    gc.collect()

    if on_end is not None:
        on_end(cobjs)


if __name__ == "__main__":
    args = parse_arguments()
    args.save = f"checkpoints/{args.model}"
    args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cov_device = torch.device("cpu")

    all_tasks = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
    tasks = args.eval_datasets if args.eval_datasets is not None else all_tasks

    ss_num_batches = max(args.cov_num_batches)
    ss_batch_size = args.cov_batch_size
    ss_type = args.cov_type
    ss_attn = args.mha
    ss_split = args.cov_split
    ss_estimator = args.cov_estimator
    ss_fm = Path(args.load).stem if args.load is not None else args.finetuning_mode
    cov_dir = f"results/{args.model}/covariances_s{ss_split}_n{ss_num_batches}_b{ss_batch_size}_t{ss_type}_attn{ss_attn}_e{ss_estimator}_ft{ss_fm}"
    os.makedirs(cov_dir, exist_ok=True)

    def make_cov_dir(n):
        return f"results/{args.model}/covariances_s{ss_split}_n{n}_b{ss_batch_size}_t{ss_type}_attn{ss_attn}_e{ss_estimator}_ft{ss_fm}"

    print(f"Covariance directory: {cov_dir}")
    for task in tasks:
        cov_path = f"{cov_dir}/covariance_{task}.npz"
        all_cached = all(
            os.path.exists(f"{make_cov_dir(n)}/covariance_{task}.npz")
            for n in args.cov_num_batches
        )
        if all_cached and not args.overwrite:
            print(f"Skipping {task} (cached)")
            continue

        print(f"\nCollecting covariance for {task}")
        if args.load is not None:
            # Direct checkpoint path supplied via --load; skip task-vector construction.
            encoder = torch.load(args.load, map_location="cpu", weights_only=False)
        elif args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{task}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{task}Val/linear_finetuned.pt"
            pretrained_nonlinear_checkpoint = f"{args.save}/{task}Val/zeroshot.pt"

            # Get param names
            nonlinear_encoder = torch.load(
                pretrained_nonlinear_checkpoint, map_location="cpu", weights_only=False
            )
            param_names = [n for n, _ in nonlinear_encoder.named_parameters()]
            del nonlinear_encoder

            tv = LinearizedTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                covariance_path=cov_path,
            )
            encoder = tv.apply_to_nonlinear(
                pretrained_nonlinear_checkpoint, param_names, scaling_coef=1.0
            )
            del tv
        elif args.finetuning_mode == "lora":
            pretrained_checkpoint = f"{args.save}/{task}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{task}Val/lora_finetuned.pt"
            tv = NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
            )
            encoder = tv.apply_to(pretrained_checkpoint, scaling_coef=1.0)
            del tv
        else:
            pretrained_checkpoint = f"{args.save}/{task}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{task}Val/finetuned.pt"
            tv = NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
            )
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

        def on_batch(cobjs, n_batches):
            if n_batches in thresholds:
                snap_path = f"{make_cov_dir(n_batches)}/covariance_{task}.npz"
                os.makedirs(os.path.dirname(snap_path), exist_ok=True)
                snap = {
                    k: v.cov.cpu().numpy() if isinstance(v, OnlineCovariance) else v
                    for k, v in cobjs.items()
                }
                snap.update(
                    {
                        f"{k}_n": v.n
                        for k, v in cobjs.items()
                        if isinstance(v, OnlineCovariance)
                    }
                )
                np.savez(snap_path, **snap)
                print(f"  Saved snapshot (n={n_batches}) to {snap_path}")

        def on_end(cobjs):
            # Convert cobjs to saveable arrays (np.savez can't save tuples of inhomogeneous shapes)
            # two loops to save memory
            for lname in list(cobjs):
                cobjs[f"{lname}_n"] = cobjs[lname].n
            saveable = {
                k: v.cov.cpu().numpy() if isinstance(v, OnlineCovariance) else v
                for k, v in cobjs.items()
            }
            np.savez(cov_path, **saveable)
            print(f"  Saved to {cov_path}")

        compute_covs(encoder, f"{task}Val", args, on_batch=on_batch, on_end=on_end)
        del encoder
        gc.collect()
