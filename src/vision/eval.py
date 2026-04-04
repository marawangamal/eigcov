import numpy as np
import torch
import tqdm

from src import utils
from src.vision.datasets.common import get_dataloader, maybe_dictionarize
from src.vision.datasets.registry import get_dataset
from src.vision.heads import get_classification_head
from src.vision.linearize import LinearizedImageEncoder
from src.vision.modeling import ImageClassifier


def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    use_train = getattr(args, "eval_split", "test") == "train"
    dataloader = get_dataloader(dataset, is_train=use_train, args=args, image_encoder=None)
    device = args.device

    max_batches = getattr(args, "eval_max_batches", None)
    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for batch_idx, data in enumerate(tqdm.tqdm(dataloader)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n if n > 0 else 0.0

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%")

    return metrics


def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    control = getattr(args, "control_dataset", None)
    eval_datasets = (
        args.eval_datasets
        if control is None
        else args.eval_datasets + [control]
    )
    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results


def evaluate_task_vector_at_coef(
    task_vector, checkpoint_dir, args, scaling_coef, posthoc_linearization=False
):
    image_encoder = task_vector.apply_to(
        checkpoint_dir, scaling_coef=scaling_coef
    )
    if posthoc_linearization:
        pretrained_encoder = task_vector.apply_to(
            checkpoint_dir, scaling_coef=0.0
        )
        image_encoder = LinearizedImageEncoder(
            init_encoder=pretrained_encoder, image_encoder=image_encoder, args=args
        )
    coef_info = evaluate(image_encoder, args)

    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    return coef_info


def evaluate_task_vector(
    task_vector, checkpoint_dir, args, posthoc_linearization=False
):
    info = {}
    for scaling_coef in np.linspace(
        args.coeff_start, args.coeff_end, args.n_eval_points
    ):
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            checkpoint_dir,
            args,
            scaling_coef,
            posthoc_linearization,
        )

    return info


def nonlinear_advantage(acc_linear, acc_nonlinear, num_classes):
    err_linear = 1 - acc_linear
    err_nonlinear = 1 - acc_nonlinear
    return (err_linear - err_nonlinear) * num_classes / (num_classes - 1)
