"""Collects per-layer ||dL/dy||² and sampled entries of yy^T per sample.

For each layer, saves:
  - g_sq/<layer>:         (N,) array of ||dL/dy||²
  - aat_samples/<layer>:  (N, K) array of sampled entries of yy^T
  - index_pairs/<layer>:  (K, 2) array of which (i,j) pairs were sampled

Activations are flattened (seq_len * D) so aa^T captures the full structure.
K = 32 (i,j) pairs, randomly sampled with a fixed seed for reproducibility.

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/vision/correlation.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets --finetuning-mode=lora
"""

# TODO: rename to correlation.py
import os
import torch
import numpy as np

from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.vision.heads import get_classification_head
from src.vision.modeling import ImageClassifier
from src.args import parse_arguments
from src.utils import get_prefix
from src.vision.datasets.registry import get_dataset

# K_SAMPLES = 32


def get_index_pairs(D, K, seed=42):
    """Get K random (i,j) pairs from a DxD matrix with a fixed seed."""
    rng = np.random.RandomState(seed)
    i = rng.randint(0, D, size=K)
    j = rng.randint(0, D, size=K)
    return list(zip(i.tolist(), j.tolist()))


def register_hooks(model):
    xs = {}
    ys = {}
    handles = []
    for name, module in model.named_modules():
        # if name == "":
        #     continue
        if not isinstance(module, torch.nn.Linear):
            continue

        def make_hook(n):
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(inp, tuple):
                    inp = inp[0]
                if isinstance(out, torch.Tensor):
                    out.retain_grad()
                    ys[n] = out
                if isinstance(inp, torch.Tensor):
                    inp.retain_grad()
                    xs[n] = inp

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))
    return xs, ys, handles


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

    # Activations is a dictionary of layer names to activation tensors
    zs, ys, handles = register_hooks(model)
    loss_fn = torch.nn.CrossEntropyLoss()

    g_sq = {}
    # Activation outer products: {layer_name: [aa.T]_{ij} with shape (N, K)}. N = num samples and K = num indices.
    zzt_samples = {}
    index_pairs = {}

    for i, (images, labels) in enumerate(dataset.train_loader):
        if i >= args.num_samples:
            break

        images, labels = images.cuda(), labels.cuda()
        model.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()

        for name, z in zs.items():
            if z.grad is not None:
                # (T, B, D): B=1 always.
                z_vec = z.detach().reshape(-1)

                # g_sq.setdefault(name, []).append(z.grad.pow(2).sum().item())
                g_sq.setdefault(name, []).append(ys[name].grad.pow(2).mean().item())

                if name not in index_pairs:
                    index_pairs[name] = get_index_pairs(
                        z_vec.shape[0], args.num_indices
                    )

                pairs = index_pairs[name]
                sampled = np.array([(z_vec[i] * z_vec[j]).item() for i, j in pairs])
                zzt_samples.setdefault(name, []).append(sampled)

    for h in handles:
        h.remove()
    model.cpu()

    return (
        {n: np.array(v) for n, v in g_sq.items()},
        # So now we have
        # {layer_name: aat (N, K,)}
        {n: np.stack(v) for n, v in zzt_samples.items()},
        index_pairs,
    )


if __name__ == "__main__":
    args = parse_arguments()
    args.batch_size = 1
    args.save = f"checkpoints/{args.model}"
    prefix = get_prefix(args.finetuning_mode)
    args.num_samples = 100
    args.num_indices = 32

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

    ss_batch_size = args.batch_size
    ss_num_samples = args.num_samples
    ss_num_indices = args.num_indices
    ss_finetuning_mode = args.finetuning_mode
    correlation_dir = f"results/{args.model}/correlations_b{ss_batch_size}_n{ss_num_samples}_k{ss_num_indices}_ft{ss_finetuning_mode}"
    os.makedirs(correlation_dir, exist_ok=True)

    for task in tasks:
        cache_path = f"{correlation_dir}/correlation_{task}.npz"
        if os.path.exists(cache_path) and not args.overwrite:
            print(f"[Skipped] {cache_path} already exists")
            continue

        # print(f"\nCollecting norms for {task}")
        # tv = NonLinearTaskVector(
        #     pretrained_ckpt, f"checkpoints/{args.model}/{task}Val/finetuned.pt"
        # )
        # encoder = tv.apply_to(pretrained_ckpt, scaling_coef=1.0)
        # del tv

        print(f"\nCollecting covariance for {task}")
        checkpoint_dir = f"{args.save}/{task}Val"
        if args.finetuning_mode == "linear":
            # Get param names from the nonlinear pretrained model
            nonlinear_encoder = torch.load(
                os.path.join(checkpoint_dir, "zeroshot.pt"), map_location="cpu", weights_only=False
            )
            param_names = [n for n, _ in nonlinear_encoder.named_parameters()]
            del nonlinear_encoder

            tv = LinearizedTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
            encoder = tv.apply_to_nonlinear(
                checkpoint_dir, param_names, scaling_coef=1.0
            )
        elif args.finetuning_mode == "lora":
            # NOTE: LoRA mode not yet supported with checkpoint_dir convention
            raise NotImplementedError("LoRA mode not yet supported with checkpoint_dir convention")
        else:
            tv = NonLinearTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
            encoder = tv.apply_to(checkpoint_dir, scaling_coef=1.0)

        del tv

        g_sq, aat_samples, index_pairs = collect_norms(encoder, f"{task}Val", args)
        del encoder

        save_dict = {}
        for layer in g_sq:
            save_dict[f"g_sq/{layer}"] = g_sq[layer]
            save_dict[f"aat_samples/{layer}"] = aat_samples[layer]
            save_dict[f"index_pairs/{layer}"] = np.array(index_pairs[layer])

        np.savez(cache_path, **save_dict)
        print(f"  Saved to {cache_path}")

        # Quick summary
        all_rhos = []
        for layer in g_sq.keys():
            g = g_sq[layer]
            aat = aat_samples[layer]
            if g.std() == 0:
                continue
            all_rhos.extend(
                np.corrcoef(g, aat[:, k])[0, 1] if aat[:, k].std() > 0 else 0.0
                for k in range(aat.shape[1])
            )
        all_rhos = np.abs(all_rhos)
        print(
            f"  |ρ|  mean={np.mean(all_rhos):.2f}  median={np.median(all_rhos):.2f}  max={np.max(all_rhos):.2f}"
        )
