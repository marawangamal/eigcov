"""Reproduces the disentanglement analysis from Figure 3 of the paper.

Measures how much two task vectors interfere when combined, by computing
prediction disagreement between the merged model (θ_0 + α1*τ1 + α2*τ2)
and the corresponding single-task models over a grid of (α1, α2) values.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.vision.task_vectors import NonLinearTaskVector, LinearizedTaskVector
from src.vision.eval import eval_single_dataset
from src.args import parse_arguments
from src.utils import get_prefix


def compute_disentanglement_error(
    task_vector_1,
    task_vector_2,
    checkpoint_dir,
    dataset_1,
    dataset_2,
    args,
    alpha_range=np.linspace(-3.0, 3.0, 25),
    finetuning_mode="standard",  # "standard" or "posthoc"
    cache_path=None,
):
    """Compute ξ(α1, α2) over a grid of alpha values.

    If cache_path is provided, saves/loads the grid to/from disk to avoid
    recomputing on subsequent runs.
    """
    if cache_path is not None and os.path.exists(cache_path) and not args.overwrite:
        print(f"Loading cached grid from {cache_path}")
        data = np.load(cache_path)
        return data["grid"]

    n = len(alpha_range)
    total = n**2
    grid = np.zeros((n, n))

    for i, alpha_1 in enumerate(alpha_range):
        for j, alpha_2 in enumerate(alpha_range):
            # Build merged model: θ_0 + α1*τ1 + α2*τ2
            merged_vector = task_vector_1 * alpha_1 + task_vector_2 * alpha_2
            merged_encoder = merged_vector.apply_to(
                checkpoint_dir, scaling_coef=1.0
            )

            # Build single-task models: θ_0 + α1*τ1 and θ_0 + α2*τ2
            single_1_encoder = task_vector_1.apply_to(
                checkpoint_dir, scaling_coef=alpha_1
            )
            single_2_encoder = task_vector_2.apply_to(
                checkpoint_dir, scaling_coef=alpha_2
            )

            # Get predictions on dataset_1
            # Compare merged vs single_1 predictions
            error_1 = compute_prediction_disagreement(
                merged_encoder, single_1_encoder, dataset_1, args
            )

            # Get predictions on dataset_2
            # Compare merged vs single_2 predictions
            error_2 = compute_prediction_disagreement(
                merged_encoder, single_2_encoder, dataset_2, args
            )

            grid[j, i] = 0.5 * (error_1 + error_2)  # j=row=α2, i=col=α1

            print(
                f"[{i * n + j + 1:>{len(str(total))}}/{total}] α1={alpha_1:.1f}, α2={alpha_2:.1f}, ξ={grid[j,i]:.3f}"
            )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, grid=grid, alpha_range=alpha_range)
        print(f"Cached grid to {cache_path}")

    return grid


def compute_prediction_disagreement(encoder_a, encoder_b, dataset, args):
    """Fraction of test samples where two encoders give different predictions."""
    from src.vision.heads import get_classification_head
    from src.vision.modeling import ImageClassifier
    from src.vision.datasets.registry import get_dataset

    classification_head = get_classification_head(args, dataset)

    model_a = ImageClassifier(encoder_a, classification_head)
    model_b = ImageClassifier(encoder_b, classification_head)

    model_a.eval()
    model_b.eval()
    model_a.cuda()
    model_b.cuda()

    test_dataset = get_dataset(
        dataset,
        model_a.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    test_loader = test_dataset.test_loader

    disagree = 0
    total = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.cuda()
            preds_a = model_a(images).argmax(dim=1)
            preds_b = model_b(images).argmax(dim=1)
            disagree += (preds_a != preds_b).sum().item()
            total += images.size(0)

    model_a.cpu()
    model_b.cpu()

    return disagree / total


def plot_disentanglement(grid, alpha_range, task_pair_name, save_path=None):
    """Plot the heatmap like Figure 3 in the paper."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        grid,
        origin="lower",
        cmap="GnBu",
        extent=[alpha_range[0], alpha_range[-1], alpha_range[0], alpha_range[-1]],
        vmin=0,
        vmax=1,
        aspect="equal",
    )
    ax.set_xlabel(r"$\alpha_1$")
    ax.set_ylabel(r"$\alpha_2$")
    ax.set_title(task_pair_name)
    plt.colorbar(im, ax=ax, label=r"$\xi(\alpha_1, \alpha_2)$")

    # Draw the red search box like in the paper
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (-1, -1), 2, 2, linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---- Main usage ----
if __name__ == "__main__":
    args = parse_arguments()
    # HACK: Some command line arguments are overwritten by defaults here.
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.save = f"checkpoints/{args.model}"
    prefix = get_prefix(args.finetuning_mode)

    model = args.model
    results_dir = f"results/{model}"
    os.makedirs(results_dir, exist_ok=True)

    # Task pairs from Figure 3
    task_pairs = [
        ("EuroSAT", "SVHN"),
        # ("DTD", "SUN397"),
        # ("Cars", "RESISC45"),
    ]

    alpha_range = np.linspace(-3.0, 3.0, 25)  # 25x25 grid
    # alpha_range = np.linspace(-1.0, 1.0, 4)  # 3x3 grid for debugging
    # alpha_range = [0.2]  # minimal grid for debugging

    for dataset_1, dataset_2 in task_pairs:
        ckpt_dir_1 = f"checkpoints/{model}/{dataset_1}Val"
        ckpt_dir_2 = f"checkpoints/{model}/{dataset_2}Val"

        # --- Non-linear FT (top row) ---
        tv1 = NonLinearTaskVector(checkpoint_dir=ckpt_dir_1, prefix=prefix)
        tv2 = NonLinearTaskVector(checkpoint_dir=ckpt_dir_2, prefix=prefix)

        grid_nl = compute_disentanglement_error(
            tv1,
            tv2,
            ckpt_dir_1,
            dataset_1,
            dataset_2,
            args,
            alpha_range=alpha_range,
            cache_path=f"{results_dir}/grid_{dataset_1}_{dataset_2}_nonlinear.npz",
        )
        plot_disentanglement(
            grid_nl,
            alpha_range,
            f"{dataset_1} - {dataset_2} (Non-linear FT)",
            save_path=f"{results_dir}/disentangle_{dataset_1}_{dataset_2}_nonlinear.png",
        )

        # NOTE: Requires running finetune.py with --finetuning=mode=linear first.
        # # --- Post-hoc linearization (bottom row) ---
        # # Apply non-linear task vectors to linearized encoder
        # linear_pretrained_ckpt = f"checkpoints/{model}/linear_zeroshot.pt"
        # # Use apply_to_linear for posthoc linearization
        # grid_ph = compute_disentanglement_error(
        #     tv1,
        #     tv2,
        #     linear_pretrained_ckpt,
        #     dataset_1,
        #     dataset_2,
        #     args,
        #     alpha_range=alpha_range,
        #     cache_path=f"{results_dir}/grid_{dataset_1}_{dataset_2}_posthoc.npz",
        # )
        # plot_disentanglement(
        #     grid_ph,
        #     alpha_range,
        #     f"{dataset_1} - {dataset_2} (Post-hoc linear.)",
        #     save_path=f"{results_dir}/disentangle_{dataset_1}_{dataset_2}_posthoc.png",
        # )
