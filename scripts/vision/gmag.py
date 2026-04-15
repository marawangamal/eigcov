"""Computes per-layer gradient magnitude distributions w.r.t. activations.

For each layer, collects ||dL/dy||² per sample across the dataset,
saving the full distribution (not just the mean) for downstream analysis.

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/gmag.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets
"""

import os
import torch
import numpy as np

from src.vision.task_vectors import NonLinearTaskVector
from src.vision.heads import get_classification_head
from src.vision.modeling import ImageClassifier
from src.args import parse_arguments
from src.utils import get_prefix
from src.vision.datasets.registry import get_dataset


def register_activation_hooks(model):
    """Register forward hooks that retain_grad on activations.

    Returns:
        activations: dict populated with {layer_name: output} during forward.
        handles: list of hook handles (call .remove() to clean up).
    """
    activations = {}
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue

        def make_hook(n):
            def hook(mod, inp, out):
                # Handle tuple outputs (e.g., attention returning (output, weights))
                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(out, torch.Tensor):
                    out.retain_grad()
                    activations[n] = out

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))
    return activations, handles


def compute_gradient_magnitudes(encoder, dataset_name, args):
    """Compute per-layer ||dL/dy||² for each sample in the dataset.

    Returns:
        layer_gmag: dict mapping layer_name -> np.array of shape (N,)
            where N is the number of samples.
    """
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(encoder, classification_head)
    model.freeze_head()
    model.train()
    model.cuda()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    print(f"    {len(dataset.test_loader.dataset)} samples")

    activations, handles = register_activation_hooks(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    layer_samples = {}  # layer_name -> list of per-sample ||dL/dy||²

    for images, labels in dataset.test_loader:
        images, labels = images.cuda(), labels.cuda()
        model.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()

        for name, act in activations.items():
            if act.grad is not None:
                # ||dL/dy||² for this sample
                # Sum over ALL dimensions since batch_size=1
                grad_mag = act.grad.pow(2).sum().item()

                if name not in layer_samples:
                    layer_samples[name] = []
                layer_samples[name].append(grad_mag)

    for h in handles:
        h.remove()
    model.cpu()

    return {name: np.array(vals) for name, vals in layer_samples.items()}


if __name__ == "__main__":
    args = parse_arguments()
    # HACK: Some command line arguments are overwritten by defaults here.
    # NOTE: Batch size must be one for correctness. We want to visualize the distribution
    # of ||dL/dy||² over samples
    args.batch_size = 1
    args.save = f"checkpoints/{args.model}"
    prefix = get_prefix(args.finetuning_mode)

    model = args.model
    results_dir = f"results/{model}"
    os.makedirs(results_dir, exist_ok=True)

    datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]
    print(f"Results will be saved to: {results_dir}/")

    for ds in datasets:
        cache_path = f"{results_dir}/gmag_{ds}.npz"
        if os.path.exists(cache_path) and not args.overwrite:
            print(f"Skipping {ds} (cached at {cache_path})")
            continue

        print(f"\nComputing gradient magnitudes for {ds}")
        checkpoint_dir = f"checkpoints/{model}/{ds}Val"
        tv = NonLinearTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
        encoder = tv.apply_to(checkpoint_dir, scaling_coef=1.0)
        del tv

        layer_gmag = compute_gradient_magnitudes(encoder, f"{ds}Val", args)
        del encoder

        np.savez(cache_path, **layer_gmag)
        print(f"  Cached to {cache_path}")

        print(f"\n  {'Layer':<60} {'mean':>12} {'std':>12} {'median':>12} {'cv':>12}")
        print(f"  {'-' * 110}")
        for name, vals in layer_gmag.items():
            cv = vals.std() / vals.mean() if vals.mean() > 0 else 0
            print(
                f"  {name:<60} {vals.mean():>12.6f} {vals.std():>12.6f} "
                f"{np.median(vals):>12.6f} {cv:>12.4f}"
            )

        # Summary statistics
        all_cvs = [
            vals.std() / vals.mean() for vals in layer_gmag.values() if vals.mean() > 0
        ]
        print(f"\n  Mean CV across layers: {np.mean(all_cvs):.4f}")
        print(f"  Median CV across layers: {np.median(all_cvs):.4f}")
        print(
            f"  % layers with CV < 0.5: {100 * np.mean(np.array(all_cvs) < 0.5):.1f}%"
        )
        print(
            f"  % layers with CV < 1.0: {100 * np.mean(np.array(all_cvs) < 1.0):.1f}%"
        )
