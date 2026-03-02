import json
import os

# from mha import copy_from_pytorch_state_dict, copy_to_pytorch_state_dict
from src import mhap, mhas
from src.args import parse_arguments
from src.results_db import append_result, args_to_dict, make_run_hash, record_exists
from src.vision.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.merging import combine_task_vectors
from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.utils import find_optimal_coef

args = parse_arguments()

if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"

# Fields that should NOT affect the run identity:
#   - training-only hyperparameters (lr, wd, …) — not used during eval
#   - environment/path args (cache dirs, save dir) — machine-specific
#   - fields set dynamically *after* this point (eval_datasets, finetuning_accuracies, …)
#   - metadata (results_db path, exp_name, overwrite flag, …)
_HASH_IGNORE = {
    # training-only
    "lr", "wd", "ls", "warmup_length", "epochs", "num_grad_accumulation", "batch_size",
    "checkpoint_every", "keep_checkpoints", "port", "world_size", "cosine_samples",
    "lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules", "lora_target_parameters",
    # environment / paths
    "openclip_cachedir", "cache_dir", "save", "data_location",
    # dynamically set after hash
    "eval_datasets", "finetuning_accuracies", "control_dataset", "eval_split", "eval_max_batches",
    # metadata
    "results_db", "exp_name", "overwrite", "num_workers", "device",
}

_run_hash = make_run_hash("eval_task_addition", args, ignore=_HASH_IGNORE) if args.results_db else None
if args.results_db and record_exists(args.results_db, _run_hash):
    print(f"Skipping: matching record already exists in {args.results_db}")
    exit(0)

print("-" * 100)
print("DEBUG:")
print(f"Record to be saved: {_run_hash}")
print(json.dumps({"script": "eval_task_addition", **args_to_dict(args)}, indent=4))
print("-" * 100)

print("*" * 100)
if args.finetuning_mode == "standard":
    print(f"Evaluating non-linear FT models. ({args.merge_func})")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
elif args.finetuning_mode == "linear":
    print(f"Evaluating linear FT models. ({args.merge_func})")
    ft_accuracies_path = os.path.join(args.save, "linear_ft_accuracies.json")
elif args.finetuning_mode == "posthoc":
    print(f"Evaluating post-hoc linearized models. ({args.merge_func})")
    ft_accuracies_path = os.path.join(args.save, "posthoc_ft_accuracies.json")
elif args.finetuning_mode == "lora":
    print(f"Evaluating LoRA FT models. ({args.merge_func})")
    ft_accuracies_path = os.path.join(args.save, "lora_ft_accuracies.json")
else:
    raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

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
    cov_path = f"{args.cov_dir}/covariance_{dataset}.npz" if args.cov_dir else None
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
        task_vectors.append(
            LinearizedTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                covariance_path=cov_path,
            )
        )
    elif args.finetuning_mode == "lora":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/lora_finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                covariance_path=cov_path,
            )
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                covariance_path=cov_path,
            )
        )
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

merge_name = getattr(args, "merge_func", "sum")
task_vector = combine_task_vectors(task_vectors, merge_name, args)

if args.mha is not None:
    print(f"Mapping task vector back from {args.mha} MHA mode")
    copy_fn = {
        "packed": mhap.copy_to_pytorch_state_dict,
        "split": mhas.copy_to_pytorch_state_dict,
    }[args.mha]
    task_vector = task_vector.map(copy_fn)

args.control_dataset = None


def _set_eval_split(split):
    """Set args.eval_split and args.eval_datasets for the given split (val, test, or train)."""
    args.eval_split = split
    if split == "val":
        args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
    else:
        args.eval_datasets = list(eval_datasets)


# Phase 1: coefficient selection on eval-val-split (optionally capped by eval-val-max-batches).
_set_eval_split(args.eval_val_split)
args.eval_max_batches = getattr(args, "eval_val_max_batches", None)
print("=" * 100)
print(
    f"PHASE 1: SPLIT={args.eval_val_split.upper()} — choosing optimal coefficient"
    + (f" (max {args.eval_max_batches} batches)" if args.eval_max_batches else "")
)
print("=" * 100)
val_metrics = evaluate_task_vector(
    task_vector,
    pretrained_checkpoint,
    args,
    posthoc_linearization=args.finetuning_mode == "posthoc",
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)
print(f"Optimal coefficient (from phase 1): {optimal_coef}")

# Phase 2: evaluate at optimal coefficient on eval-test-split (use all batches).
_set_eval_split(args.eval_test_split)
args.eval_max_batches = None
print("=" * 100)
print(
    f"PHASE 2: SPLIT={args.eval_test_split.upper()} — evaluating at optimal coefficient"
)
print("=" * 100)
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    args,
    float(optimal_coef),
    posthoc_linearization=args.finetuning_mode == "posthoc",
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
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/posthoc_additions_{merge_name}.json"
elif args.finetuning_mode == "lora":
    save_file = f"{args.save}/lora_additions_{merge_name}.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)

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
