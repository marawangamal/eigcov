import itertools
import json
import os
from pathlib import Path

# from mha import copy_from_pytorch_state_dict, copy_to_pytorch_state_dict
from src import mhap, mhas
from src.args import parse_arguments
from src.vision.eval import evaluate_task_vector_at_coef
from src.merging import combine_task_vectors
from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"

merge_name = getattr(args, "merge_func", "sum")
results_file = Path(f"results/{args.model}-{merge_name}/metrics.json")
if results_file.exists() and not args.overwrite:
    print(f"Skipping: {results_file} already exists (use --overwrite to rerun)")
    exit(0)

print("*" * 100)
print(f"Evaluating {args.finetuning_mode} FT models. ({args.merge_func})")
print("*" * 100)

eval_datasets = [
    "SUN397",
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SVHN",
]

task_vectors = []

for dataset in eval_datasets:
    checkpoint_dir = f"{args.save}/{dataset}Val"
    if args.finetuning_mode == "linear":
        task_vectors.append(LinearizedTaskVector(checkpoint_dir=checkpoint_dir))
    else:
        task_vectors.append(NonLinearTaskVector(checkpoint_dir=checkpoint_dir))
    print(f"Task vector {dataset} loaded")

# For use with RegMean and Projected RegMean.
#   i)  for projected regmean, mhap will package together orthgonally invariant matrices.
#   ii) for regmean, mhas will use linear modules for all attention operations to collect covariances.
if args.mha is not None:
    print(f"Mapping task vectors dicts to {args.mha} MHA mode")
    copy_fn = {
        "packed": mhap.copy_from_pytorch_state_dict,
        "split": mhas.copy_from_pytorch_state_dict,
    }[args.mha]
    task_vectors = [t.map(copy_fn) for t in task_vectors]

# Build HP grid
hpo = args.hpo or {}
hp_names = list(hpo.keys())
hp_value_lists = list(hpo.values())
hp_combos = (
    [dict(zip(hp_names, combo)) for combo in itertools.product(*hp_value_lists)]
    if hp_names
    else [{}]
)

args.control_dataset = None

# Use last checkpoint_dir for apply_to (all share same pretrained model)
pretrained_dir = f"{args.save}/{eval_datasets[-1]}Val"


def _set_eval_split(split):
    """Set args.eval_split and args.eval_datasets for the given split (val, test, or train)."""
    args.eval_split = split
    if split == "val":
        args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
    else:
        args.eval_datasets = list(eval_datasets)


def _merge_and_remap(merge_kwargs):
    """Merge task vectors with given kwargs and apply MHA reverse remap if needed."""
    tv = combine_task_vectors(task_vectors, merge_name, **merge_kwargs)
    if args.mha is not None:
        reverse_fn = {
            "packed": mhap.copy_to_pytorch_state_dict,
            "split": mhas.copy_to_pytorch_state_dict,
        }[args.mha]
        tv = tv.map(reverse_fn)
    return tv


# Phase 1: HP grid search on eval-val-split (optionally capped by eval-val-max-batches).
best_val_score = -float("inf")
best_merge_kwargs = {}
best_val_metrics = {}

_set_eval_split("val")
args.eval_max_batches = getattr(args, "eval_val_max_batches", None)
print("=" * 100)
if len(hp_combos) <= 1:
    best_merge_kwargs = hp_combos[0] if hp_combos else {}
    print(f"PHASE 1: SKIPPED (single HP combo: {best_merge_kwargs})")
else:
    print(
        f"PHASE 1: SPLIT=VAL — grid search over {len(hp_combos)} HP combos"
        + (f" (max {args.eval_max_batches} batches)" if args.eval_max_batches else "")
    )
    print("=" * 100)
    for merge_kwargs in hp_combos:
        print(f"  {merge_kwargs}")
        task_vector = _merge_and_remap(merge_kwargs)
        metrics = evaluate_task_vector_at_coef(
            task_vector,
            pretrained_dir,
            args,
            1.0,
            posthoc_linearization=args.finetuning_mode == "posthoc",
        )
        score = metrics["avg_top1"]
        print(f"  {merge_kwargs} -> avg_top1={score:.4f}")
        if score > best_val_score:
            best_val_score = score
            best_merge_kwargs = merge_kwargs
            best_val_metrics = metrics

print(f"Best merge HP (from phase 1): {best_merge_kwargs}")

# Phase 2: evaluate at best HP combo on the test split (use all batches).
_set_eval_split("test")
args.eval_max_batches = None
print("=" * 100)
print("PHASE 2: SPLIT=TEST — evaluating at best HP combo")
print("=" * 100)
task_vector = _merge_and_remap(best_merge_kwargs)
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_dir,
    args,
    1.0,
    posthoc_linearization=args.finetuning_mode == "posthoc",
)

print("=" * 100)
print(f"Test accuracy: {test_metrics['avg_top1']}")

# Build olmes-style metrics.json
tasks = []
for dataset in eval_datasets:
    top1 = test_metrics[f"{dataset}:top1"]
    tasks.append(
        {
            "alias": dataset,
            "metrics": {"top1": top1, "primary_score": top1},
            "task_config": {"primary_metric": "top1"},
        }
    )

metrics_json = {
    "all_primary_scores": [
        f"{t['alias']}: {t['metrics']['primary_score']:.6f}" for t in tasks
    ],
    "tasks": tasks,
    "model_config": {
        "model": args.model,
        "merge_func": merge_name,
        "finetuning_mode": args.finetuning_mode,
        "seed": args.seed,
        "mha": args.mha,
        "optimal_merge_hp": best_merge_kwargs,
    },
}

results_file.parent.mkdir(parents=True, exist_ok=True)
results_file.write_text(json.dumps(metrics_json, indent=2))
print(f"Results saved to {results_file}")
