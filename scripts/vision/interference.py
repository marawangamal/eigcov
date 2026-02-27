"""Computes per-layer interference when merging multiple task vectors.

Interference measures how much the merged model's activations deviate from
each single-task model's activations:

    L(W, {W_t}) = Σ_t E_{x~D_t} [||Wx - W_t x||²]

where W is the merged model (θ_0 + α*Στ_t) and W_t is a single-task model
(θ_0 + α*τ_t). This is computed per layer using forward hooks.
"""

import os
import torch
import numpy as np
from collections import OrderedDict

from src.vision.task_vectors import NonLinearTaskVector
from src.args import parse_arguments
from src.vision.datasets.registry import get_dataset


def register_hooks(model):
    """Register forward hooks on all named modules to capture activations.

    Returns:
        activations: OrderedDict that will be populated with {layer_name: output}
            during a forward pass.
        handles: list of hook handles (call .remove() to clean up).
    """
    activations = OrderedDict()
    handles = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # Only capture tensor outputs (skip tuples, etc.)
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()

        return hook_fn

    for name, module in model.named_modules():
        if name == "":
            continue
        h = module.register_forward_hook(make_hook(name))
        handles.append(h)

    return activations, handles


def compute_layerwise_interference(
    merged_encoder,
    single_encoder,
    dataset_name,
    args,
):
    """Compute per-layer interference: E_x[||W_merged x - W_single x||²].

    Args:
        merged_encoder: encoder with merged task vectors applied.
        single_encoder: encoder with only this task's vector applied.
        dataset_name: name of the dataset to draw samples from.
        args: parsed arguments.

    Returns:
        layer_interference: dict mapping layer_name -> mean squared activation diff.
    """
    dataset = get_dataset(
        dataset_name,
        merged_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    test_loader = dataset.test_loader
    print(f"    {len(dataset.test_loader.dataset)} samples")

    # Register hooks on both models
    merged_acts, merged_handles = register_hooks(merged_encoder)
    single_acts, single_handles = register_hooks(single_encoder)

    merged_encoder.eval()
    single_encoder.eval()
    merged_encoder.cuda()
    single_encoder.cuda()

    # Accumulate per-layer squared differences
    layer_sum = {}
    total_samples = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.cuda()
            batch_size = images.size(0)

            # Forward pass through both — hooks capture activations
            merged_encoder(images)
            single_encoder(images)

            # Compute ||merged_act - single_act||² for each shared layer
            for name in merged_acts:
                if name not in single_acts:
                    continue
                diff = merged_acts[name] - single_acts[name]
                # Sum of squared L2 norms over the batch
                sq_diff = diff.reshape(batch_size, -1).pow(2).sum(dim=1).sum().item()
                layer_sum[name] = layer_sum.get(name, 0.0) + sq_diff

            total_samples += batch_size

    # Clean up hooks
    for h in merged_handles + single_handles:
        h.remove()

    merged_encoder.cpu()
    single_encoder.cpu()

    # Average over samples
    layer_interference = {name: val / total_samples for name, val in layer_sum.items()}
    return layer_interference


def compute_interference(
    pretrained_checkpoint,
    finetuned_checkpoints,
    dataset_names,
    args,
    alpha=1.0,
    cache_path=None,
):
    """Compute total per-layer interference for a given alpha.

    Loads one task vector at a time to keep memory usage low. First builds
    the merged task vector incrementally (Σ τ_t), then loops again to
    compare merged vs each single-task model.

    Args:
        pretrained_checkpoint: path to the pretrained (zeroshot) checkpoint.
        finetuned_checkpoints: list of paths to finetuned checkpoints.
        dataset_names: list of dataset names corresponding to each task.
        args: parsed arguments.
        alpha: scaling coefficient applied to all task vectors.
        cache_path: if provided, save/load results to/from this path.

    Returns:
        total_interference: dict mapping layer_name -> total interference.
    """
    if cache_path is not None and os.path.exists(cache_path) and not args.overwrite:
        print(f"Loading cached interference from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return dict(data["interference"].item())

    T = len(finetuned_checkpoints)

    # Pass 1: Build merged task vector incrementally (one at a time)
    print("  Building merged task vector...")
    merged_vector = None
    for t in range(T):
        tv = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoints[t])
        merged_vector = tv if merged_vector is None else merged_vector + tv
        print(f"    Added task vector {t + 1}/{T}: {dataset_names[t]}")

    merged_encoder = merged_vector.apply_to(pretrained_checkpoint, scaling_coef=alpha)
    del merged_vector  # Free memory

    # Pass 2: Compare merged vs each single-task model (one at a time)
    total_interference = {}
    for t in range(T):
        print(f"  Computing interference for task {t + 1}/{T}: {dataset_names[t]}")

        tv_t = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoints[t])
        single_encoder = tv_t.apply_to(pretrained_checkpoint, scaling_coef=alpha)
        del tv_t

        layer_interference = compute_layerwise_interference(
            merged_encoder, single_encoder, dataset_names[t], args
        )
        del single_encoder

        for name, val in layer_interference.items():
            total_interference[name] = total_interference.get(name, 0.0) + val

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, interference=total_interference, alpha=alpha)
        print(f"Cached interference to {cache_path}")

    return total_interference


# ---- Main ----
if __name__ == "__main__":
    args = parse_arguments()
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.save = f"checkpoints/{args.model}"

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

    pretrained_ckpt = f"checkpoints/{model}/{datasets[0]}Val/zeroshot.pt"
    ft_ckpts = [f"checkpoints/{model}/{ds}Val/finetuned.pt" for ds in datasets]
    dataset_names = [f"{ds}Val" for ds in datasets]

    alpha_range = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, ..., 1.0]

    # Compute interference for each alpha
    all_results = {}
    for alpha in alpha_range:
        print(f"\n{'=' * 77}")
        print(f"Computing interference with α={alpha:.2f}")
        print(f"{'=' * 77}")
        interference = compute_interference(
            pretrained_ckpt,
            ft_ckpts,
            dataset_names,
            args,
            alpha=alpha,
            cache_path=f"{results_dir}/interference_alpha{alpha:.2f}.npz",
        )
        all_results[alpha] = sum(interference.values())

    # Print summary across alphas
    print(f"\n{'=' * 40}")
    print(f"{'α':>10} {'Total Interference':>20}")
    print(f"{'-' * 40}")
    for alpha, total in all_results.items():
        print(f"{alpha:>10.2f} {total:>20.4f}")
    print(f"{'=' * 40}")
