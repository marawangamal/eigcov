import json
import os

from mha import copy_from_pytorch_state_dict, copy_to_pytorch_state_dict
from utils import find_optimal_coef

from src.args import parse_arguments
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.merging import combine_task_vectors
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"


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
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
        task_vectors.append(
            LinearizedTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                metadata={"task": dataset, "model": args.model},
            )
        )
    elif args.finetuning_mode == "lora":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/lora_finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                metadata={"task": dataset, "model": args.model},
            )
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                metadata={"task": dataset, "model": args.model},
            )
        )

if args.swap_mha:
    task_vectors = [t.map(copy_from_pytorch_state_dict) for t in task_vectors]

merge_name = getattr(args, "merge_func", "sum")
task_vector = combine_task_vectors(task_vectors, merge_name, args)

if args.swap_mha:
    task_vector = task_vector.map(copy_to_pytorch_state_dict)

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

# We use the validation set to choose the optimal coefficient.
print("=" * 100)
print(
    "PHASE 1: VALIDATION — evaluating at multiple coefficients to choose optimal coefficient"
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
print(f"Optimal coefficient (from validation): {optimal_coef}")

# Evaluate on the test set with the optimal coefficient.
print("=" * 100)
print("PHASE 2: TEST — evaluating at optimal coefficient on test splits")
print("=" * 100)
args.eval_datasets = [dataset for dataset in eval_datasets]
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
