"""Empirical check of cross-sample gradient orthogonality (Theorem, condition i).

Runs one epoch of SGD with batch_size=1. At each step k the per-sample gradient is
    g_k = nabla_W L(x_k; theta_k)

We estimate  E[g_k^T g_{k'}]  for k != k' (different samples, different model states)
using the identity:
    sum_{k != k'} g_k^T g_{k'} = ||sum_k g_k||^2 - sum_k ||g_k||^2

so:
    mean_cross_ip = (||sum_k g_k||^2 - sum_k ||g_k||^2) / (K * (K-1))

This requires only O(P) extra memory (one running sum vector + one scalar).
If condition (i) holds, mean_cross_ip should be ~0 relative to mean ||g_k||^2.

Usage:
    python -m scripts.vision.grad_cross_ip \
        --model ViT-B-16 \
        --train-dataset MNIST \
        --data-location datasets \
        --save checkpoints \
        --finetuning-mode standard \
        --lr 1e-5 \
        --max-steps 2000
"""

import os

import torch

from src.args import parse_arguments
from src.vision.datasets.common import get_dataloader, maybe_dictionarize
from src.vision.datasets.registry import get_dataset
from src.vision.heads import get_classification_head
from src.vision.linearize import LinearizedImageEncoder
from src.vision.modeling import ImageClassifier, ImageEncoder
from src.utils import LabelSmoothing


def _flat_grad(model):
    """Return all requires_grad gradients concatenated into a 1-D CPU float32 tensor."""
    parts = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            parts.append(p.grad.detach().cpu().view(-1).float())
    return torch.cat(parts)


def compute_cross_grad_ip(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)
    linearized_finetuning = args.finetuning_mode == "linear"

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    if args.load is not None and args.load.endswith(".pt"):
        image_encoder = (
            LinearizedImageEncoder.load(args.load)
            if linearized_finetuning
            else ImageEncoder.load(args.load)
        )
    else:
        print("Building image encoder.")
        image_encoder = (
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    classification_head = get_classification_head(args, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    model = model.cuda()

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {num_trainable:,}")

    # ------------------------------------------------------------------
    # Data  (force batch_size=1)
    # ------------------------------------------------------------------
    args.batch_size = 1
    dataset = get_dataset(
        train_dataset,
        model.train_preprocess,
        location=args.data_location,
        batch_size=1,
        num_workers=args.num_workers,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)

    loss_fn = LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    # ------------------------------------------------------------------
    # Running accumulators
    # ------------------------------------------------------------------
    grad_sum = torch.zeros(num_trainable)  # sum_k g_k  (CPU, float32)
    grad_sq_norm_sum = 0.0                 # sum_k ||g_k||^2
    K = 0

    max_steps = getattr(args, "max_steps", None)
    total = len(data_loader) if max_steps is None else min(max_steps, len(data_loader))
    print(f"Running {total} SGD steps (one sample each)...")

    model.train()
    for i, batch in enumerate(data_loader):
        if max_steps is not None and i >= max_steps:
            break

        batch = maybe_dictionarize(batch)
        inputs = batch["images"].cuda()
        labels = batch["labels"].cuda()

        optimizer.zero_grad()
        loss = loss_fn(model(inputs), labels)
        loss.backward()

        g = _flat_grad(model)        # shape (P,), CPU
        grad_sum += g
        grad_sq_norm_sum += g.dot(g).item()
        K += 1

        optimizer.step()

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}]  loss={loss.item():.4f}", flush=True)

    # ------------------------------------------------------------------
    # Estimate cross-sample mean inner product
    #   mean_cross_ip = (||sum g_k||^2 - sum ||g_k||^2) / (K*(K-1))
    # ------------------------------------------------------------------
    grad_sum_sq_norm = grad_sum.dot(grad_sum).item()
    mean_self_ip = grad_sq_norm_sum / K
    cross_ip_total = grad_sum_sq_norm - grad_sq_norm_sum
    mean_cross_ip = cross_ip_total / (K * (K - 1))
    normalized = mean_cross_ip / mean_self_ip  # should be ~0

    print("\n=== Cross-sample gradient orthogonality (condition i) ===")
    print(f"  Steps K                   : {K}")
    print(f"  mean ||g_k||^2            : {mean_self_ip:.6e}   <- self inner product")
    print(f"  mean g_k^T g_k' (k!=k')  : {mean_cross_ip:.6e}   <- cross inner product")
    print(f"  normalized cross-IP       : {normalized:.6e}   <- should be ~0")
    print("=========================================================")

    os.makedirs(ckpdir, exist_ok=True)
    save_path = os.path.join(ckpdir, "grad_cross_ip.pt")
    torch.save(
        {
            "K": K,
            "mean_self_ip": mean_self_ip,
            "mean_cross_ip": mean_cross_ip,
            "normalized_cross_ip": normalized,
            "grad_sum_sq_norm": grad_sum_sq_norm,
            "grad_sq_norm_sum": grad_sq_norm_sum,
        },
        save_path,
    )
    print(f"Saved results to {save_path}")
    return mean_cross_ip, mean_self_ip, normalized


if __name__ == "__main__":
    import argparse as _ap

    # Extend parse_arguments with --max-steps
    import sys

    # Inject --max-steps before parse_arguments consumes argv
    parser_extra = _ap.ArgumentParser(add_help=False)
    parser_extra.add_argument("--max-steps", type=int, default=None)
    extra, remaining = parser_extra.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    args = parse_arguments()
    args.max_steps = extra.max_steps

    if args.save is None:
        args.save = f"checkpoints/{args.model}"
    else:
        args.save = os.path.join(args.save, args.model)
    if args.seed is not None:
        args.save = os.path.join(args.save, f"seed_{args.seed}")

    train_datasets = args.train_dataset or ["MNIST"]
    for dataset in train_datasets:
        args.train_dataset = dataset + "Val"
        print("=" * 80)
        print(f"Cross-grad IP | {args.model} on {dataset} ({args.finetuning_mode})")
        print("=" * 80)
        compute_cross_grad_ip(args)
