import numpy as np
import torch
from tqdm import tqdm

from src.language.datasets.batcher import Batcher
from src.language.datasets.dataset_readers import get_datasetReader
from src.language.datasets.pytorch_dataset import PytorchDataset
from src.language.eval.evaluator import Evaluator
from src.language.eval.utils import prepare_batchOfEvalInfo


def eval_single_dataset(split, model, tokenizer, dataset_name, args):
    """Evaluate model on a single dataset split. Returns dict with 'top1' accuracy."""
    model = model.to(args.device)
    model.eval()

    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": 32,
        "max_datapoints_per_dataset_without_templates": None,
    }

    device = args.device

    dataset_reader = get_datasetReader(dataset_name, dataset_kwargs)
    createPytorchDataset_fn = lambda dataset: PytorchDataset(dataset, tokenizer, device)
    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=None,
        eval_batchSize=args.batch_size * 2,
        world_size=1,
        device=0,
    )
    batch_iterator = batcher.get_evalBatches(split, template_idx=0)
    metrics = batcher.get_metricsForDataset()

    assert "Accuracy" in metrics and len(metrics) == 1, f"{metrics} not supported!"
    evaluator = Evaluator(metrics)

    with torch.no_grad():
        for batch in tqdm(batch_iterator):
            batchOf_evalInfo = prepare_batchOfEvalInfo(batch)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                (
                    predicted_choice,
                    score_ofChoices,
                    logProbs_ofAllChoicesIds,
                    len_allChoices,
                ) = model.predict_mulChoice(batch, False)

            batchOf_evalInfo.update(
                {
                    "predicted_choice": predicted_choice,
                    "score_of_choices": score_ofChoices,
                    "log_probs_of_all_choices_ids": logProbs_ofAllChoicesIds,
                    "len_all_choices": len_allChoices,
                }
            )

            evaluator.add_batch(batchOf_evalInfo)

    top1 = evaluator.get_result()["accuracy"]
    print(f"Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%", flush=True)
    return {"top1": top1}


def evaluate(split, model, args):
    """Evaluate model on all datasets in args.eval_datasets (+ optional control dataset)."""
    if args.eval_datasets is None:
        return {}

    per_dataset_results = {}
    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )

    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)
        results = eval_single_dataset(split, model, model.tokenizer, dataset_name, args)
        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results


def evaluate_task_vector_at_coef(
    split,
    task_vector,
    pretrained_checkpoint,
    args,
    scaling_coef,
    posthoc_linearization=False,
):
    """Apply task vector at a fixed coefficient and evaluate."""
    model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    coef_info = evaluate(split, model, args)

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    return coef_info


def evaluate_task_vector(
    split,
    task_vector,
    pretrained_checkpoint,
    args,
    posthoc_linearization=False,
):
    """Grid-search over scaling coefficients and return per-coefficient metrics."""
    info = {}
    for scaling_coef in np.linspace(0.0, 1.0, args.n_eval_points):
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        info[scaling_coef] = evaluate_task_vector_at_coef(
            split,
            task_vector,
            pretrained_checkpoint,
            args,
            scaling_coef,
            posthoc_linearization,
        )
    return info


def add_normalized_accuracy(results, args):
    """Normalize per-dataset accuracies by fine-tuned accuracy."""
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )
    return results
