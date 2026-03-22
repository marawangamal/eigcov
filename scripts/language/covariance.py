"""Collects per-layer covariance statistics for language (T5) experiments.

For each dataset, saves per-layer covariance data to an NPZ file that can be
used by covariance-aware merging methods (e.g. RegMean).

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/language/covariance.py --model=t5-base --cov-split=train --cov-num-batches=100 --cov-batch-size=32
python scripts/language/covariance.py --model=t5-base --finetuning-mode=linear --cov-split=train --cov-num-batches=100 --cov-batch-size=32
"""

import gc
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.language.args import parse_arguments
from src.language.task_vectors import (
    LanguageNonLinearTaskVector,
    LanguageLinearizedTaskVector,
)
from src.language.datasets.pytorch_dataset import PytorchDataset
from src.language.datasets.batcher import Batcher
from src.language.datasets.dataset_readers import get_datasetReader
from src.covariance import OnlineCovariance, register_hooks


def compute_covs(model, dataset_name, args, on_batch=None, on_end=None):
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
    max_num_batches = args.cov_num_batches
    data_iter = batcher.get_splitOfBatches(
        split, template_idx=0, is_evaluation=(split != "train")
    )

    mask_ref = [None, None]
    cobjs, handles = register_hooks(
        model,
        cov_device=args.cov_device,
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        mask_ref=mask_ref,
        batch_first=True,
    )

    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(
            data_iter,
            desc="Computing covariance",
            total=max_num_batches,
        ):
            if max_num_batches is not None and n_batches >= max_num_batches:
                break
            mask_ref[0] = batch.get("input_mask")
            mask_ref[1] = batch.get("target_mask")
            model(batch)
            n_batches += 1
            if on_batch is not None:
                on_batch(cobjs, n_batches)

    if max_num_batches is not None:
        print(f"    Used {n_batches} batches (cov_num_batches={max_num_batches})")

    if on_end is not None:
        on_end(cobjs)

    for h in handles:
        h.remove()
    model.cpu()
    del batcher, handles
    gc.collect()

    return cobjs


if __name__ == "__main__":
    args = parse_arguments()
    args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cov_device = torch.device("cpu")
    args.max_seq_len = 128

    if args.save is None:
        args.save = f"checkpoints/{args.model}"

    T5_DATASETS = [
        "qasc",
        "wiki_qa",
        "quartz",
        "paws",
        "story_cloze",
        "winogrande",
        "wsc",
    ]

    cov_num_batches_list = args.cov_num_batches
    ss_num_batches = max(cov_num_batches_list)
    ss_batch_size = args.cov_batch_size
    ss_type = args.cov_type
    ss_split = args.cov_split
    ss_estimator = args.cov_estimator
    ss_fm = args.finetuning_mode
    cov_dir = f"{args.save}/_covariances/covariances_s{ss_split}_n{ss_num_batches}_b{ss_batch_size}_t{ss_type}_e{ss_estimator}_ft{ss_fm}"
    os.makedirs(cov_dir, exist_ok=True)

    def make_cov_dir(n):
        return f"{args.save}/_covariances/covariances_s{ss_split}_n{n}_b{ss_batch_size}_t{ss_type}_e{ss_estimator}_ft{ss_fm}"

    # Set cov_num_batches to the max for compute_covs
    args.cov_num_batches = ss_num_batches
    thresholds = set(n for n in cov_num_batches_list if n < ss_num_batches)

    print(f"Covariance directory: {cov_dir}")
    for dataset in T5_DATASETS:
        all_cached = all(
            os.path.exists(f"{make_cov_dir(n)}/covariance_{dataset}.npz")
            for n in cov_num_batches_list
        )
        if all_cached and not args.overwrite:
            print(f"Skipping {dataset} (cached)")
            continue

        print(f"\nCollecting covariance for {dataset}")
        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned.pt"
            pretrained_nonlinear_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"

            # Get param names from nonlinear model
            nonlinear_model = torch.load(
                pretrained_nonlinear_checkpoint, map_location="cpu", weights_only=False
            )
            param_names = [n for n, _ in nonlinear_model.named_parameters()]
            del nonlinear_model

            tv = LanguageLinearizedTaskVector(
                pretrained_checkpoint, finetuned_checkpoint
            )
            model = tv.apply_to_nonlinear(
                pretrained_nonlinear_checkpoint, param_names, scaling_coef=1.0
            )
        elif args.finetuning_mode == "lora":
            pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}/lora_finetuned.pt"
            tv = LanguageNonLinearTaskVector(
                pretrained_checkpoint, finetuned_checkpoint
            )
            model = tv.apply_to_nonlinear(pretrained_checkpoint, scaling_coef=1.0)
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"
            tv = LanguageNonLinearTaskVector(
                pretrained_checkpoint, finetuned_checkpoint
            )
            model = tv.apply_to_nonlinear(pretrained_checkpoint, scaling_coef=1.0)

        del tv

        def on_batch(cobjs, n_batches, _dataset=dataset):
            if n_batches in thresholds:
                snap_path = f"{make_cov_dir(n_batches)}/covariance_{_dataset}.npz"
                os.makedirs(os.path.dirname(snap_path), exist_ok=True)
                snap = {
                    k: v.cov.cpu().numpy() if isinstance(v, OnlineCovariance) else v
                    for k, v in cobjs.items()
                }
                snap.update(
                    {
                        f"{k}_n": v.n
                        for k, v in cobjs.items()
                        if isinstance(v, OnlineCovariance)
                    }
                )
                np.savez(snap_path, **snap)
                print(f"  Saved snapshot (n={n_batches}) to {snap_path}")

        def on_end(cobjs, _dataset=dataset):
            for lname in list(cobjs):
                cobjs[f"{lname}_n"] = cobjs[lname].n
            saveable = {
                k: v.cov.cpu().numpy() if isinstance(v, OnlineCovariance) else v
                for k, v in cobjs.items()
            }
            cov_path = f"{cov_dir}/covariance_{_dataset}.npz"
            np.savez(cov_path, **saveable)
            print(f"  Saved to {cov_path}")

        compute_covs(model, dataset, args, on_batch=on_batch, on_end=on_end)
        del model
        gc.collect()
