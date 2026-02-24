import json
import os

from src.language.args import parse_arguments
from src.language.eval import eval_single_dataset
from src.task_vectors import LanguageLinearizedTaskVector, LanguageNonLinearTaskVector

T5_DATASETS = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]

args = parse_arguments()

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

for dataset in T5_DATASETS:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned.pt"
    elif args.finetuning_mode == "none":
        pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"

    try:
        task_vector = (
            LanguageLinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            if args.finetuning_mode == "linear"
            else LanguageNonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    except FileNotFoundError as e:
        print(f"Error: Could not find checkpoint — {e}")
        continue

    if args.finetuning_mode == "none":
        model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
    else:
        model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

    for split in ["test", "validation"]:
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset] = eval_single_dataset(
            split, model, model.tokenizer, dataset, args
        )["top1"]

# Add averages
val_scores = [v for k, v in accuracies.items() if k.endswith("Val") and isinstance(v, (int, float))]
test_scores = [v for k, v in accuracies.items() if not k.endswith("Val") and isinstance(v, (int, float))]
accuracies["avg_val"] = (sum(val_scores) / len(val_scores)) if val_scores else None
accuracies["avg_test"] = (sum(test_scores) / len(test_scores)) if test_scores else None

# Save results
if args.finetuning_mode == "none":
    save_path = f"{args.save}/zeroshot_accuracies.json"
elif args.finetuning_mode == "standard":
    save_path = f"{args.save}/ft_accuracies.json"
elif args.finetuning_mode == "linear":
    save_path = f"{args.save}/linear_ft_accuracies.json"
else:
    save_path = f"{args.save}/{args.finetuning_mode}_ft_accuracies.json"

os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

# Merge with existing results if file already exists
if os.path.exists(save_path):
    with open(save_path) as f:
        old_accuracies = json.load(f)
    old_accuracies.update(accuracies)
    accuracies = old_accuracies

with open(save_path, "w") as f:
    json.dump(accuracies, f, indent=4)
print("Results saved to", save_path)
