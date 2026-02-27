"""Collects per-layer ||dL/dy||² and sampled entries of yy^T per sample.

For each layer, saves:
  - g_sq/<layer>:         (N,) array of ||dL/dy||²
  - aat_samples/<layer>:  (N, K) array of sampled entries of yy^T
  - index_pairs/<layer>:  (K, 2) array of which (i,j) pairs were sampled

Activations are flattened (seq_len * D) so aa^T captures the full structure.
K = 32 (i,j) pairs, randomly sampled with a fixed seed for reproducibility.

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/vision/decorrelation.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets --finetuning-mode=lora
"""

# TODO: rename to correlation.py
import os
import torch
import numpy as np

from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.vision.heads import get_classification_head
from src.vision.modeling import ImageClassifier
from src.args import parse_arguments
from src.vision.datasets.registry import get_dataset

K_SAMPLES = 32


def get_index_pairs(D, K, seed=42):
    """Get K random (i,j) pairs from a DxD matrix with a fixed seed."""
    rng = np.random.RandomState(seed)
    i = rng.randint(0, D, size=K)
    j = rng.randint(0, D, size=K)
    return list(zip(i.tolist(), j.tolist()))


def register_hooks(model):
    activations = {}
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue

        def make_hook(n):
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(out, torch.Tensor):
                    out.retain_grad()
                    activations[n] = out

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))
    return activations, handles


def collect_norms(encoder, dataset_name, args):
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

    activations, handles = register_hooks(model)
    loss_fn = torch.nn.CrossEntropyLoss()

    g_sq = {}
    aat_samples = {}
    index_pairs = {}

    for images, labels in dataset.test_loader:
        images, labels = images.cuda(), labels.cuda()
        model.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()

        for name, act in activations.items():
            if act.grad is not None:
                a_vec = act.detach().reshape(-1)

                g_sq.setdefault(name, []).append(act.grad.pow(2).sum().item())

                if name not in index_pairs:
                    index_pairs[name] = get_index_pairs(a_vec.shape[0], K_SAMPLES)

                pairs = index_pairs[name]
                sampled = np.array([(a_vec[i] * a_vec[j]).item() for i, j in pairs])
                aat_samples.setdefault(name, []).append(sampled)

    for h in handles:
        h.remove()
    model.cpu()

    return (
        {n: np.array(v) for n, v in g_sq.items()},
        {n: np.stack(v) for n, v in aat_samples.items()},
        index_pairs,
    )


if __name__ == "__main__":
    args = parse_arguments()
    args.batch_size = 1
    args.save = f"checkpoints/{args.model}"

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
    decorrelation_dir = f"results/{args.model}/decorrelations_ft{args.finetuning_mode}"
    os.makedirs(decorrelation_dir, exist_ok=True)

    for task in tasks:
        cache_path = f"{decorrelation_dir}/correlation_{task}.npz"
        if os.path.exists(cache_path) and not args.overwrite:
            print(f"Skipping {task} (cached)")
            continue

        # print(f"\nCollecting norms for {task}")
        # tv = NonLinearTaskVector(
        #     pretrained_ckpt, f"checkpoints/{args.model}/{task}Val/finetuned.pt"
        # )
        # encoder = tv.apply_to(pretrained_ckpt, scaling_coef=1.0)
        # del tv

        print(f"\nCollecting covariance for {task}")
        if args.finetuning_mode == "linear":
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
            )
            encoder = tv.apply_to_nonlinear(
                pretrained_nonlinear_checkpoint, param_names, scaling_coef=1.0
            )
        elif args.finetuning_mode == "lora":
            pretrained_checkpoint = f"{args.save}/{task}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{task}Val/lora_finetuned.pt"
            tv = NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
            )
            encoder = tv.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        else:
            pretrained_checkpoint = f"{args.save}/{task}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{task}Val/finetuned.pt"
            tv = NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
            )
            encoder = tv.apply_to(pretrained_checkpoint, scaling_coef=1.0)

        del tv

        g_sq, aat_samples, index_pairs = collect_norms(encoder, f"{task}Val", args)
        del encoder

        save_dict = {}
        for layer in g_sq:
            save_dict[f"g_sq/{layer}"] = g_sq[layer]
            save_dict[f"aat_samples/{layer}"] = aat_samples[layer]
            save_dict[f"index_pairs/{layer}"] = np.array(index_pairs[layer])

        np.savez(cache_path, **save_dict)
        print(f"  Cached to {cache_path}")

        # Quick summary
        print(f"\n  {'Layer':<50} {'mean|ρ|':>8} {'max|ρ|':>8}")
        print(f"  {'-' * 68}")
        for layer in sorted(g_sq.keys()):
            g = g_sq[layer]
            aat = aat_samples[layer]
            if g.std() == 0:
                print(f"  {layer:<50} {'n/a':>8} {'n/a':>8}")
                continue
            rhos = np.array(
                [
                    np.corrcoef(g, aat[:, k])[0, 1] if aat[:, k].std() > 0 else 0.0
                    for k in range(aat.shape[1])
                ]
            )
            print(
                f"  {layer:<50} {np.mean(np.abs(rhos)):>8.2f} {np.max(np.abs(rhos)):>8.2f}"
            )
