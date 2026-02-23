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
python scripts/covariance.py --cov-split train --cov-num-batches 100 --cov-batch-size 32 --mha=split ...

# Generate covariance matrices
python scripts/covariance.py --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets \
--model=ViT-B-16 --cov-split train --cov-num-batches 100 --cov-batch-size 32 --mha=split 

"""

import gc
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
import src.mhap as mhap
import src.mhas as mhas


class OnlineCovariance:
    def __init__(self, dim1, dim2=1, device="cpu", mode="cov"):
        # Store on CPU to avoid GPU memory issues
        self.device = device
        self.meanx = torch.zeros((dim1, dim2), device=device)
        self.meany = torch.zeros((dim1, dim2), device=device)
        self.C = torch.zeros((dim1, dim1), device=device)
        self.n = 0
        self.add = {
            "cov": self._add_cov,
            "sm": self._add_second_moment,
        }[mode]

    @property
    def cov(self):
        # Population covariance
        return self.C / self.n

    @property
    def cov_sample(self):
        # Sample covariance
        return self.C / (self.n - 1)

    def _add_cov(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        self.n += 1
        # NOTE: removed as instance attr (self.dx -> dx)
        dx = x - self.meanx
        self.meanx += dx / self.n
        self.meany += (y - self.meany) / self.n
        self.C += dx @ (y - self.meany).T

    def _add_second_moment(self, x, y):
        # x is (D, T), y is (D, T)
        x = x.to(self.device)
        y = y.to(self.device)
        self.n += 1

        # Uncentered second moment: E[X.T X]
        self.C += x @ y.T


def register_hooks(model, args):
    cobjs = {}  # covariance objects
    handles = []
    for name, module in model.named_modules():
        if not isinstance(
            module,
            (
                torch.nn.Linear,
                torch.nn.MultiheadAttention,
                mhap.MultiHeadAttentionPacked,
                mhas.MultiHeadAttentionSplit,
            ),
        ):
            # Only collect covariance for linear layers and multihead attention layers
            continue

        def make_hook(n):
            def hook(mod, inp, out):
                x = inp[0] if isinstance(inp, (tuple, list)) else inp
                if not isinstance(x, torch.Tensor):
                    return
                T, B, D = x.shape  # adjust to your real shape

                if args.cov_estimator == "sampled":
                    # Add Dx1 vectors (one random token position per sample)
                    if n not in cobjs:
                        cobjs[n] = OnlineCovariance(
                            D, device=args.cov_device, mode=args.cov_type
                        )
                    cobj = cobjs[n]
                    for b in range(B):
                        j = torch.randint(0, T, (1,)).item()
                        cobj.add(x[j : j + 1, b].T, x[j : j + 1, b].T)
                else:
                    # Add DxT vectors (full sequence per sample)
                    if n not in cobjs:
                        cobjs[n] = OnlineCovariance(
                            D, T, device=args.cov_device, mode=args.cov_type
                        )
                    cobj = cobjs[n]
                    for b in range(B):
                        cobj.add(x[:, b].T, x[:, b].T)

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))
    return cobjs, handles


def compute_covs(encoder, dataset_name, args):
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
    max_num_batches = args.cov_num_batches
    loader = dataset.train_loader if split == "train" else dataset.test_loader
    dataset_size = len(loader.dataset)
    print(f"    {dataset_size} samples (split={split})")

    cobjs, handles = register_hooks(model, args)
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
    if max_num_batches is not None:
        print(f"    Used {n_batches} batches (cov_num_batches={max_num_batches})")

    for h in handles:
        h.remove()
    model.cpu()

    return cobjs


if __name__ == "__main__":
    args = parse_arguments()
    args.save = f"checkpoints/{args.model}"
    args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cov_device = torch.device("cpu")

    tasks = [
        # "Cars",
        # "DTD",
        # "EuroSAT",
        # "GTSRB",
        # "MNIST",
        # "RESISC45",
        "SUN397",
        "SVHN",
    ]
    pretrained_ckpt = f"checkpoints/{args.model}/{tasks[0]}Val/zeroshot.pt"

    ss_num_batches = args.cov_num_batches if args.cov_num_batches is not None else "all"
    ss_batch_size = args.cov_batch_size
    ss_type = args.cov_type
    ss_attn = args.mha
    ss_split = args.cov_split
    ss_estimator = args.cov_estimator
    cov_dir = f"results/{args.model}/covariances_s{ss_split}_n{ss_num_batches}_b{ss_batch_size}_t{ss_type}_attn{ss_attn}_e{ss_estimator}"
    os.makedirs(cov_dir, exist_ok=True)

    print(f"Covariance directory: {cov_dir}")
    for task in tasks:
        cov_path = f"{cov_dir}/covariance_{task}.npz"
        if os.path.exists(cov_path) and not args.overwrite:
            print(f"Skipping {task} (cached)")
            continue

        print(f"\nCollecting covariance for {task}")
        tv = NonLinearTaskVector(
            pretrained_ckpt, f"checkpoints/{args.model}/{task}Val/finetuned.pt"
        )
        encoder = tv.apply_to(pretrained_ckpt, scaling_coef=1.0)
        del tv

        # swap mha
        if args.mha is not None:
            swap_fn = {
                "packed": mhap.swap_mha,
                "split": mhas.swap_mha,
            }[args.mha]
            encoder = swap_fn(encoder)

        cobjs = compute_covs(encoder, f"{task}Val", args)
        del encoder

        # Convert cobjs to saveable arrays (np.savez can't save tuples of inhomogeneous shapes)
        save_dict = {}
        for lname, cobj in cobjs.items():
            save_dict[lname] = cobj.cov.cpu().numpy()
            save_dict[f"{lname}_n"] = cobj.n
        np.savez(cov_path, **save_dict)
        print(f"  Saved to {cov_path}")
        del cobjs, save_dict
        gc.collect()
