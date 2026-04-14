"""Computes per-parameter diagonal Fisher Information Matrix approximation for T5.

For each task, saves a .pt dict with keys matching param_key_to_cov_key:
  - transformer.encoder.<layer>: (Do*Di,) flattened averaged squared gradient
  - n_batches: number of batches used

The empirical Fisher diagonal is estimated as:
  F_i = (1/N) * sum_n (dL_n/d theta_i)^2

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/language/fisher.py --model=t5-base --cov-split=train --cov-num-batches=10 --cov-batch-size=32
"""

import gc
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.args import parse_arguments
from src.language.task_vectors import (
    LanguageNonLinearTaskVector,
    LanguageLinearizedTaskVector,
)
from src.language.datasets.pytorch_dataset import PytorchDataset
from src.language.datasets.batcher import Batcher
from src.language.datasets.dataset_readers import get_datasetReader
from src.utils import get_prefix


def compute_fisher(model, dataset_name, args, on_end=None):
    model.eval()
    model.to(args.model_device)

    tokenizer = model.tokenizer

    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": 32,
        "max_datapoints_per_dataset_without_templates": None,
    }

    dataset_reader = get_datasetReader(dataset_name, dataset_kwargs)
    device_str = str(args.model_device)
    createPytorchDataset_fn = lambda dataset: PytorchDataset(
        dataset, tokenizer, device_str
    )
    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=args.cov_batch_size,
        eval_batchSize=args.cov_batch_size,
        world_size=None,
        device=None,
    )

    split = args.cov_split
    max_num_batches = max(args.cov_num_batches)
    data_iter = batcher.get_trainBatches(split, template_idx=0)

    # Accumulate squared gradients per parameter (running sum, averaged at save time)
    fisher = {
        name: torch.zeros_like(param, device="cpu")
        for name, param in model.named_parameters()
    }

    n_batches = 0
    for batch in tqdm(data_iter, desc="Computing Fisher", total=max_num_batches):
        if n_batches >= max_num_batches:
            break

        model.zero_grad()
        loss, _ = model(batch)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += (param.grad.detach() ** 2).cpu()

        n_batches += 1

    print(f"    Used {n_batches} batches (max={max_num_batches})")

    if on_end is not None:
        on_end(fisher, n_batches)

    model.cpu()
    del batcher
    gc.collect()

    return fisher, n_batches


if __name__ == "__main__":
    args = parse_arguments()
    args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.max_seq_len = 128
    args.cov_batch_size = 1
    args.cov_num_batches = [100]

    if args.save is None:
        args.save = f"checkpoints/{args.model}"
    prefix = get_prefix(args.finetuning_mode)

    T5_DATASETS = [
        "qasc",
        "wiki_qa",
        "quartz",
        "paws",
        "story_cloze",
        "winogrande",
        "wsc",
    ]

    for dataset in T5_DATASETS:
        checkpoint_dir = f"{args.save}/{dataset}"
        fisher_path = os.path.join(checkpoint_dir, "fisher.pt")

        if os.path.exists(fisher_path) and not args.overwrite:
            print(f"Skipping {dataset} (cached: {fisher_path})")
            continue

        print(f"\nCollecting Fisher for {dataset}")
        if args.finetuning_mode == "linear":
            zeroshot_path = os.path.join(checkpoint_dir, "zeroshot.pt")
            nonlinear_model = torch.load(
                zeroshot_path, map_location="cpu", weights_only=False
            )
            param_names = [n for n, _ in nonlinear_model.named_parameters()]
            del nonlinear_model

            tv = LanguageLinearizedTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
            model = tv.apply_to_nonlinear(checkpoint_dir, param_names, scaling_coef=1.0)
        else:
            tv = LanguageNonLinearTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
            model = tv.apply_to(checkpoint_dir, scaling_coef=1.0)

        del tv

        def on_end(fisher, n_batches, _fisher_path=fisher_path):
            # Divide in-place, then remap keys to cov-key format
            for v in fisher.values():
                v.div_(n_batches)
            saveable = {k.replace(".weight", ""): v.cpu() for k, v in fisher.items()}
            saveable["n_batches"] = n_batches
            saveable["batch_size"] = args.cov_batch_size
            torch.save(saveable, _fisher_path)
            print(f"  Saved to {_fisher_path}")

        compute_fisher(model, dataset, args, on_end=on_end)
        del model
        gc.collect()
