import json
from pathlib import Path

from src.args import parse_arguments
from src.results_db import append_result, args_to_dict, make_run_hash, record_exists
from src.utils import get_prefix
from src.vision.eval import eval_single_dataset
from src.vision.linearize import LinearizedImageEncoder
from src.vision.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()
if args.save is None:
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

prefix = get_prefix(args.finetuning_mode)
_run_hash = make_run_hash("eval_experts", args) if args.results_db else None
if args.results_db and record_exists(args.results_db, _run_hash):
    print(f"Skipping: matching record already exists in {args.results_db}")
    exit(0)

accuracies = {}


print("*" * 100)
if args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")
elif args.finetuning_mode == "lora":
    print("Evaluating LoRA FT models.")

eval_datasets = args.eval_datasets or [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]

for dataset in eval_datasets:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    checkpoint_dir = f"{args.save}/{dataset}Val"

    try:
        task_vector = (
            LinearizedTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
            if args.finetuning_mode == "linear"
            else NonLinearTaskVector(checkpoint_dir=checkpoint_dir, prefix=prefix)
        )
    except FileNotFoundError as e:
        print(f"{e}\n\nError: Could not find checkpoint in {checkpoint_dir}.")
        continue

    if args.finetuning_mode == "none":
        image_encoder = task_vector.apply_to(checkpoint_dir, scaling_coef=0.0)
    elif args.finetuning_mode in ("standard", "linear", "lora"):
        image_encoder = task_vector.apply_to(checkpoint_dir, scaling_coef=1.0)
    elif args.finetuning_mode == "posthoc":
        zs_encoder = task_vector.apply_to(checkpoint_dir, scaling_coef=0.0)
        ft_encoder = task_vector.apply_to(checkpoint_dir, scaling_coef=1.0)
        image_encoder = LinearizedImageEncoder(
            init_encoder=zs_encoder, image_encoder=ft_encoder, args=args
        )
    else:
        raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")

    for split in ["test", "val"]:
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset] = eval_single_dataset(
            image_encoder, eval_dataset, args
        )["top1"]


# NOTE: Uncomment to evaluate zero-shot accuracy on ImageNet
# if args.finetuning_mode == "none":
#     # Evaluate zero-shot accuracy on ImageNet
#     for split in ["ImageNetVal", "ImageNet"]:
#         accuracies[split] = eval_single_dataset(image_encoder, split, args)["top1"]

# Add averages:
val_scores = [
    v
    for k, v in accuracies.items()
    if k.endswith("Val") and isinstance(v, (int, float))
]
test_scores = [
    v
    for k, v in accuracies.items()
    if (not k.endswith("Val")) and isinstance(v, (int, float))
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

if args.results_db:
    append_result(
        args.results_db,
        {
            "script": "eval_experts",
            **args_to_dict(args),
            **accuracies,
        },
        _run_hash,
    )
    print("Results appended to", args.results_db)
