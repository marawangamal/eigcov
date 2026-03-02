import json
import os

from src.language.args import parse_arguments
from src.language.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.language.task_vectors import (
    LanguageLinearizedTaskVector,
    LanguageNonLinearTaskVector,
)
from src.utils import find_optimal_coef

T5_DATASETS = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]

args = parse_arguments()

print("*" * 100)
if args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "linear_ft_accuracies.json")
else:
    print(f"Evaluating {args.finetuning_mode} models.")
    ft_accuracies_path = os.path.join(
        args.save, f"{args.finetuning_mode}_ft_accuracies.json"
    )
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

control_dataset = "rte"
negation_accuracies = {}

for dataset in T5_DATASETS:
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned.pt"
        task_vector = -LanguageLinearizedTaskVector(
            pretrained_checkpoint, finetuned_checkpoint
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"
        task_vector = -LanguageNonLinearTaskVector(
            pretrained_checkpoint, finetuned_checkpoint
        )

    # Phase 1: choose optimal coefficient on validation split
    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    val_metrics = evaluate_task_vector(
        "validation",
        task_vector,
        pretrained_checkpoint,
        args,
    )

    control_threshold = getattr(args, "control_threshold", 0.95)
    optimal_coef = find_optimal_coef(
        val_metrics,
        metric=f"{dataset}:top1",
        minimize=True,
        control_metric=f"{control_dataset}:top1",
        control_metric_threshold=(
            control_threshold * pretrained_accuracies[f"{control_dataset}Val"]
        ),
    )

    # Phase 2: evaluate on test split with optimal coefficient
    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    test_metrics = evaluate_task_vector_at_coef(
        "test",
        task_vector,
        pretrained_checkpoint,
        args,
        float(optimal_coef) if optimal_coef is not None else 0.0,
    )

    print("=" * 100)
    print(f"Test accuracy: {test_metrics.get(f'{dataset}:top1')}")

    negation_accuracies[dataset] = {
        "test": test_metrics.get(f"{dataset}:top1"),
        "test_control": test_metrics.get(f"{control_dataset}:top1"),
        "val": val_metrics,
    }

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/negations.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_negations.json"
else:
    save_file = f"{args.save}/{args.finetuning_mode}_negations.json"

# Merge with existing results if file already exists
if os.path.exists(save_file):
    with open(save_file) as f:
        old_accuracies = json.load(f)
    old_accuracies.update(negation_accuracies)
    negation_accuracies = old_accuracies

with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)
print("Results saved to", save_file)
