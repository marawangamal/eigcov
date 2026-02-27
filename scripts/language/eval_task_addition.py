import json
import os

from src.language.args import parse_arguments
from src.language.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.language.task_vectors import LanguageLinearizedTaskVector, LanguageNonLinearTaskVector
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
    ft_accuracies_path = os.path.join(args.save, f"{args.finetuning_mode}_ft_accuracies.json")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

eval_datasets = list(T5_DATASETS)
task_vectors = []

for dataset in eval_datasets:
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned.pt"
        task_vectors.append(
            LanguageLinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"
        task_vectors.append(
            LanguageNonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )

task_vector = sum(task_vectors)

args.eval_datasets = list(eval_datasets)
args.control_dataset = None

# Phase 1: choose optimal coefficient on validation split
val_metrics = evaluate_task_vector(
    "validation",
    task_vector,
    pretrained_checkpoint,
    args,
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)
print(f"Optimal coefficient: {optimal_coef}")

# Phase 2: evaluate on test split with optimal coefficient
args.eval_datasets = list(eval_datasets)
test_metrics = evaluate_task_vector_at_coef(
    "test",
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
    save_file = f"{args.save}/additions.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_additions.json"
else:
    save_file = f"{args.save}/{args.finetuning_mode}_additions.json"

with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
print("Results saved to", save_file)
