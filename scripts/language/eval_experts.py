import json
from pathlib import Path

from src.args import parse_arguments
from src.language.eval import eval_single_dataset
from src.language.task_vectors import (
    LanguageLinearizedTaskVector,
    LanguageNonLinearTaskVector,
)
from src.utils import get_prefix

T5_DATASETS = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]

args = parse_arguments()
if args.save is None:
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

prefix = get_prefix(args.finetuning_mode)
accuracies = {}

print("*" * 100)
if args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
else:
    print(f"Evaluating {args.finetuning_mode} models.")

eval_datasets = args.eval_datasets or T5_DATASETS

for dataset in eval_datasets:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    checkpoint_dir = f"{args.save}/{dataset}"

    try:
        task_vector = (
            LanguageLinearizedTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
            if args.finetuning_mode == "linear"
            else LanguageNonLinearTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
        )
    except FileNotFoundError as e:
        print(f"Error: Could not find checkpoint — {e}")
        continue

    if args.finetuning_mode == "none":
        model = task_vector.apply_to(checkpoint_dir, scaling_coef=0.0)
    else:
        model = task_vector.apply_to(checkpoint_dir, scaling_coef=1.0)

    for split in ["test", "validation"]:
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset] = eval_single_dataset(
            split, model, model.tokenizer, dataset, args
        )["top1"]

# Add averages
val_scores = [
    v
    for k, v in accuracies.items()
    if k.endswith("Val") and isinstance(v, (int, float))
]
test_scores = [
    v
    for k, v in accuracies.items()
    if not k.endswith("Val") and isinstance(v, (int, float))
]
accuracies["avg_val"] = (sum(val_scores) / len(val_scores)) if val_scores else None
accuracies["avg_test"] = (sum(test_scores) / len(test_scores)) if test_scores else None

# Save results (zeroshot treated as a method, parallel to eval_task_addition.py).
if args.finetuning_mode == "none":
    results_file = Path(f"{args.results_dir}/{args.model}-zeroshot/metrics.json")
else:
    results_file = Path(f"{args.results_dir}/{args.model}-experts/{prefix}metrics.json")
results_file.parent.mkdir(parents=True, exist_ok=True)

tasks = [
    {
        "alias": k,
        "metrics": {"top1": v, "primary_score": v},
        "task_config": {"primary_metric": "top1"},
    }
    for k, v in accuracies.items()
    if isinstance(v, (int, float)) and k not in ("avg_val", "avg_test")
]
metrics_json = {
    "all_primary_scores": [
        f"{t['alias']}: {t['metrics']['primary_score']:.6f}" for t in tasks
    ],
    "tasks": tasks,
    "model_config": {
        "model": args.model,
        "finetuning_mode": args.finetuning_mode,
        "seed": args.seed,
        "avg_val": accuracies.get("avg_val"),
        "avg_test": accuracies.get("avg_test"),
    },
}
results_file.write_text(json.dumps(metrics_json, indent=2))
print(f"Results saved to {results_file}")
