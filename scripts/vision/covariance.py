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
from tqdm import tqdm


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
from src.utils import get_prefix


def compute_covs(encoder, dataset_name, args, on_end=None):
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
    print(f"    Used {n_batches} batches (max={max_num_batches})")

    del model, dataset, loader, handles
    gc.collect()

    if on_end is not None:
        on_end(cobjs)


if __name__ == "__main__":
    args = parse_arguments()
    if args.save is None:
        args.save = f"checkpoints/{args.model}"
    prefix = get_prefix(args.finetuning_mode)
    args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cov_device = torch.device("cpu")

    all_tasks = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]
    tasks = args.eval_datasets if args.eval_datasets is not None else all_tasks

    for task in tasks:
        checkpoint_dir = f"{args.save}/{task}Val"
        cov_path = os.path.join(checkpoint_dir, "covariance.pt")

        if os.path.exists(cov_path) and not args.overwrite:
            print(f"Skipping {task} (cached: {cov_path})")
            continue

        print(f"\nCollecting covariance for {task}")
        if args.load is not None:
            # Direct checkpoint path supplied via --load; skip task-vector construction.
            encoder = torch.load(args.load, map_location="cpu", weights_only=False)
        elif args.finetuning_mode == "linear":
            zeroshot_path = os.path.join(checkpoint_dir, "zeroshot.pt")

            # Get param names from nonlinear model
            nonlinear_encoder = torch.load(
                zeroshot_path, map_location="cpu", weights_only=False
            )
            param_names = [n for n, _ in nonlinear_encoder.named_parameters()]
            del nonlinear_encoder

            tv = LinearizedTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
            encoder = tv.apply_to_nonlinear(
                checkpoint_dir, param_names, scaling_coef=1.0
            )
            del tv
        else:
            tv = NonLinearTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
            encoder = tv.apply_to(checkpoint_dir, scaling_coef=1.0)
            del tv

        # swap mha
        if args.mha is not None:
            swap_fn = {
                "packed": mhap.swap_mha,
                "split": mhas.swap_mha,
            }[args.mha]
            encoder = swap_fn(encoder)

        def on_end(cobjs, _cov_path=cov_path):
            # Convert OnlineCovariance objects to tensors
            saveable = {}
            for lname in list(cobjs):
                if isinstance(cobjs[lname], OnlineCovariance):
                    saveable[lname] = cobjs[lname].cov.cpu()
                    saveable[f"{lname}_n"] = cobjs[lname].n
                else:
                    saveable[lname] = cobjs[lname]
            torch.save(saveable, _cov_path)
            print(f"  Saved to {_cov_path}")

        compute_covs(encoder, f"{task}Val", args, on_end=on_end)
        del encoder
        gc.collect()
