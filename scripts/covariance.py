"""Collects per-layer ||dL/dy||² and sampled entries of yy^T per sample.

For each layer, saves:
  - g_sq/<layer>:         (N,) array of ||dL/dy||²
  - aat_samples/<layer>:  (N, K) array of sampled entries of yy^T
  - index_pairs/<layer>:  (K, 2) array of which (i,j) pairs were sampled

Activations are flattened (seq_len * D) so aa^T captures the full structure.
K = 32 (i,j) pairs, randomly sampled with a fixed seed for reproducibility.

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/covariance.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets
python scripts/covariance.py --cov-split train --cov-num-batches 100 ...
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Add to python path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.task_vectors import NonLinearTaskVector
from src.heads import get_classification_head
from src.modeling import ImageClassifier
from src.args import parse_arguments
from src.datasets.registry import get_dataset


class OnlineCovariance:
    def __init__(self, dim1, dim2=1, device="cpu"):
        # Store on CPU to avoid GPU memory issues
        self.device = device
        self.meanx = torch.zeros((dim1, dim2), device=device)
        self.meany = torch.zeros((dim1, dim2), device=device)
        self.C = torch.zeros((dim1, dim1), device=device)
        self.n = 0

    @property
    def cov(self):
        # Population covariance
        return self.C / self.n

    @property
    def cov_sample(self):
        # Sample covariance
        return self.C / (self.n - 1)

    def add(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        self.n += 1
        self.dx = x - self.meanx
        self.meanx += self.dx / self.n
        self.meany += (y - self.meany) / self.n
        self.C += self.dx @ (y - self.meany).T


def register_hooks(model, cov_device="cpu"):
    cobjs = {}  # covariance objects
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) and not isinstance(
            module, torch.nn.MultiheadAttention
        ):
            # Only collect covariance for linear layers and multihead attention layers
            continue

        def make_hook(n):
            def hook(mod, inp, out):
                x = inp[0] if isinstance(inp, (tuple, list)) else inp
                if not isinstance(x, torch.Tensor):
                    return
                T, B, D = x.shape  # adjust to your real shape
                if n not in cobjs:
                    cobjs[n] = OnlineCovariance(D, T, device=cov_device)
                cobj = cobjs[n]
                for b in range(B):
                    cobj.add(x[:, b].T, x[:, b].T)

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))
    return cobjs, handles


def compute_covs(encoder, dataset_name, args, model_device="cuda", cov_device="cpu"):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(encoder, classification_head)
    model.freeze_head()
    model.train()
    model.to(model_device)

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.cov_batch_size,
        num_workers=args.num_workers,
    )
    split = args.cov_split
    max_num_batches = args.cov_num_batches
    loader = dataset.train_loader if split == "train" else dataset.test_loader
    dataset_size = len(loader.dataset)
    print(f"    {dataset_size} samples (split={split})")

    cobjs, handles = register_hooks(model, cov_device=cov_device)
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
            loss = loss_fn(model(images), labels)
            n_batches += 1
    if max_num_batches is not None:
        print(f"    Used {n_batches} batches (cov_num_batches={max_num_batches})")

    for h in handles:
        h.remove()
    model.cpu()

    return cobjs


if __name__ == "__main__":
    args = parse_arguments()
    args.save = f"checkpoints/{args.model}"

    results_dir = f"results/{args.model}"
    os.makedirs(results_dir, exist_ok=True)

    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cov_device = torch.device("cpu")

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
    pretrained_ckpt = f"checkpoints/{args.model}/{tasks[0]}Val/zeroshot.pt"

    n_suffix = args.cov_num_batches if args.cov_num_batches is not None else "all"
    b = args.cov_batch_size
    for task in tasks:
        cache_path = (
            f"{results_dir}/covariance_{task}_{args.cov_split}_b{b}_n{n_suffix}.npz"
        )
        if os.path.exists(cache_path) and not args.overwrite:
            print(f"Skipping {task} (cached)")
            continue

        print(f"\nCollecting covariance for {task}")
        tv = NonLinearTaskVector(
            pretrained_ckpt, f"checkpoints/{args.model}/{task}Val/finetuned.pt"
        )
        encoder = tv.apply_to(pretrained_ckpt, scaling_coef=1.0)
        del tv

        cobjs = compute_covs(
            encoder,
            f"{task}Val",
            args,
            model_device=model_device,
            cov_device=cov_device,
        )
        del encoder

        # Convert cobjs to covs (in place)
        cobjs = {n: cobj.cov.cpu().numpy() for n, cobj in cobjs.items()}
        np.savez(cache_path, **cobjs)
        print(f"  Cached to {cache_path}")
