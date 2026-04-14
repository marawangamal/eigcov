"""Collects per-layer covariance statistics for language (T5) experiments.

For each dataset, saves per-layer covariance data to a .pt file in the
checkpoint directory for use by covariance-aware merging methods (e.g. RegMean).

Example usage:
export PYTHONPATH="$PYTHONPATH:$PWD"
python scripts/language/covariance.py --model=t5-base --cov-split=train --cov-num-batches=100 --cov-batch-size=32
"""

import gc
import os
import sys
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

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
from src.covariance import OnlineCovariance, register_hooks
from src.utils import get_prefix


def compute_covs(model, dataset_name, args, on_end=None):
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
    prefix = get_prefix(args.finetuning_mode)

    # Set cov_num_batches to the max for compute_covs
    args.cov_num_batches = max(args.cov_num_batches)

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
        cov_path = os.path.join(checkpoint_dir, "covariance.pt")

        if os.path.exists(cov_path) and not args.overwrite:
            print(f"Skipping {dataset} (cached: {cov_path})")
            continue

        print(f"\nCollecting covariance for {dataset}")
        if args.finetuning_mode == "linear":
            zeroshot_path = os.path.join(checkpoint_dir, "zeroshot.pt")
            nonlinear_model = torch.load(
                zeroshot_path, map_location="cpu", weights_only=False
            )
            param_names = [n for n, _ in nonlinear_model.named_parameters()]
            del nonlinear_model

            tv = LanguageLinearizedTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
            model = tv.apply_to_nonlinear(
                checkpoint_dir, param_names, scaling_coef=1.0
            )
        else:
            tv = LanguageNonLinearTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
            model = tv.apply_to_nonlinear(checkpoint_dir, scaling_coef=1.0)

        del tv

        def on_end(cobjs, _cov_path=cov_path):
            saveable = {}
            for lname in list(cobjs):
                if isinstance(cobjs[lname], OnlineCovariance):
                    saveable[lname] = cobjs[lname].cov.cpu()
                    saveable[f"{lname}_n"] = cobjs[lname].n
                else:
                    saveable[lname] = cobjs[lname]
            torch.save(saveable, _cov_path)
            print(f"  Saved to {_cov_path}")

        compute_covs(model, dataset, args, on_end=on_end)
        del model
        gc.collect()
