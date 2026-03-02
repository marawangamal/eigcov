import json
import os

from src.language.args import parse_arguments
from src.language.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.language.task_vectors import (
    LanguageLinearizedTaskVector,
    LanguageNonLinearTaskVector,
)
from src.merging import combine_task_vectors
from src.results_db import append_result, args_to_dict, make_run_hash, record_exists
from src.utils import find_optimal_coef

T5_DATASETS = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]

args = parse_arguments()
if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"

_HASH_IGNORE = {
    # training-only
    "lr",
    "wd",
    "ls",
    "warmup_length",
    "epochs",
    "num_grad_accumulation",
    "batch_size",
    "checkpoint_every",
    "keep_checkpoints",
    "port",
    "world_size",
    "cosine_samples",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "lora_target_parameters",
    # environment / paths
    "hf_cache_dir",
    "cache_dir",
    "save",
    "data_location",
    # dynamically set after hash
    "eval_datasets",
    "finetuning_accuracies",
    "control_dataset",
    "eval_split",
    "eval_max_batches",
    # metadata
    "results_db",
    "exp_name",
    "overwrite",
    "num_workers",
    "device",
}

_run_hash = (
    make_run_hash("eval_task_addition", args, ignore=_HASH_IGNORE)
    if args.results_db
    else None
)
if args.results_db and record_exists(args.results_db, _run_hash):
    print(f"Skipping: matching record already exists in {args.results_db}")
    exit(0)

print("*" * 100)
if args.finetuning_mode == "standard":
    print(f"Evaluating non-linear FT models. ({args.merge_func})")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
elif args.finetuning_mode == "linear":
    print(f"Evaluating linear FT models. ({args.merge_func})")
    ft_accuracies_path = os.path.join(args.save, "linear_ft_accuracies.json")
else:
    print(f"Evaluating {args.finetuning_mode} models. ({args.merge_func})")
    ft_accuracies_path = os.path.join(
        args.save, f"{args.finetuning_mode}_ft_accuracies.json"
    )
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

eval_datasets = list(T5_DATASETS)
task_vectors = []

for dataset in eval_datasets:
    cov_path = f"{args.cov_dir}/covariance_{dataset}.npz" if args.cov_dir else None
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned.pt"
        task_vectors.append(
            LanguageLinearizedTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                covariance_path=cov_path,
            )
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"
        task_vectors.append(
            LanguageNonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                covariance_path=cov_path,
            )
        )
    print(f"Task vector {dataset} loaded")

merge_name = getattr(args, "merge_func", "sum")
task_vector = combine_task_vectors(task_vectors, merge_name, args)

args.control_dataset = None


def _set_eval_split(split):
    args.eval_datasets = list(eval_datasets)
    args.eval_max_batches = None


# Phase 1: coefficient selection on eval-val-split
_set_eval_split(args.eval_val_split)
args.eval_max_batches = getattr(args, "eval_val_max_batches", None)
print("=" * 100)
print(
    f"PHASE 1: SPLIT={args.eval_val_split.upper()} — choosing optimal coefficient"
    + (f" (max {args.eval_max_batches} batches)" if args.eval_max_batches else "")
)
print("=" * 100)
val_metrics = evaluate_task_vector(
    args.eval_val_split,
    task_vector,
    pretrained_checkpoint,
    args,
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)
print(f"Optimal coefficient (from phase 1): {optimal_coef}")

# Phase 2: evaluate at optimal coefficient on eval-test-split
_set_eval_split(args.eval_test_split)
args.eval_max_batches = None
print("=" * 100)
print(
    f"PHASE 2: SPLIT={args.eval_test_split.upper()} — evaluating at optimal coefficient"
)
print("=" * 100)
test_metrics = evaluate_task_vector_at_coef(
    args.eval_test_split,
    task_vector,
    pretrained_checkpoint,
    args,
    float(optimal_coef),
)

print("=" * 100)
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")

additive_accuracies = {
    "test": test_metrics,
    "val": val_metrics,
    "optimal_coef": optimal_coef,
}

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/additions_{merge_name}.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_additions_{merge_name}.json"
else:
    save_file = f"{args.save}/{args.finetuning_mode}_additions_{merge_name}.json"

with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
print("Results saved to", save_file)

if args.results_db:
    append_result(
        args.results_db,
        {
            "script": "eval_task_addition",
            **args_to_dict(args),
            "optimal_coef": optimal_coef,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        },
        _run_hash,
    )
    print("Results appended to", args.results_db)
