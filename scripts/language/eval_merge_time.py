import itertools
import json
import os
import time

from src.language.args import parse_arguments
from src.language.task_vectors import (
    LanguageLinearizedTaskVector,
    LanguageNonLinearTaskVector,
)
from src.merging import combine_task_vectors
from src.results_db import append_result, args_to_dict, make_run_hash, record_exists

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
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
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
    "overwrite",
    "num_workers",
    "device",
}

_run_hash = (
    make_run_hash("eval_merge_time", args, ignore=_HASH_IGNORE)
    if args.results_db
    else None
)
# if args.results_db and record_exists(args.results_db, _run_hash):
#     print(f"Skipping: matching record already exists in {args.results_db}")
#     exit(0)

print("*" * 100)
if args.finetuning_mode == "standard":
    print(f"Timing merge for non-linear FT models. ({args.merge_func})")
elif args.finetuning_mode == "linear":
    print(f"Timing merge for linear FT models. ({args.merge_func})")
else:
    print(f"Timing merge for {args.finetuning_mode} models. ({args.merge_func})")
print("*" * 100)

# Load task vectors
eval_datasets = list(T5_DATASETS)
task_vectors = []
merge_name = getattr(args, "merge_func", "sum")

for dataset in eval_datasets:
    checkpoint_dir = f"{args.save}/{dataset}"
    if args.finetuning_mode == "linear":
        task_vectors.append(
            LanguageLinearizedTaskVector(checkpoint_dir=checkpoint_dir)
        )
    else:
        task_vectors.append(
            LanguageNonLinearTaskVector(checkpoint_dir=checkpoint_dir)
        )
    print(f"Task vector {dataset} loaded")

# Build HP grid — use first combo only (no grid search)
hpo = args.hpo or {}
hp_names = list(hpo.keys())
hp_value_lists = list(hpo.values())
hp_combos = (
    [dict(zip(hp_names, combo)) for combo in itertools.product(*hp_value_lists)]
    if hp_names
    else [{}]
)
merge_kwargs = hp_combos[0] if hp_combos else {}

# Time the merge
print("=" * 100)
print(f"Merging with {merge_name}, kwargs={merge_kwargs}")
print("=" * 100)

start = time.time()
task_vector = combine_task_vectors(task_vectors, merge_name, **merge_kwargs)
merge_time = time.time() - start

print(f"Merge time: {merge_time:.4f} seconds")

# Save results
result = {
    "merge_func": merge_name,
    "merge_kwargs": merge_kwargs,
    "merge_time_seconds": merge_time,
}

save_file = f"{args.save}/merge_time_{merge_name}.json"
with open(save_file, "w") as f:
    json.dump(result, f, indent=4)
print(f"Results saved to {save_file}")

if args.results_db:
    append_result(
        args.results_db,
        {
            "script": "eval_merge_time",
            **args_to_dict(args),
            "merge_kwargs": merge_kwargs,
            "merge_time_seconds": merge_time,
        },
        _run_hash,
    )
    print(f"Results appended to {args.results_db}")
