import os
import time

import torch


def _format_duration(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.language.args import parse_arguments
from src.language.modeling import T5Wrapper
from src.language.linearize import LinearizedT5Wrapper
from src.language.datasets.pytorch_dataset import PytorchDataset
from src.language.datasets.batcher import Batcher
from src.language.datasets.dataset_readers import get_datasetReader
from src.language.eval import eval_single_dataset  # noqa: E402 (eval is a package)


def finetune(args):
    """Fine-tune a T5 model (standard or linearized) on a single dataset.

    Saves ``zeroshot.pt`` / ``finetuned.pt`` (standard) or
    ``linear_zeroshot.pt`` / ``linear_finetuned.pt`` (linear) inside
    ``{args.save}/{args.train_dataset}/``.
    """
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only 'linear' and 'standard' fine-tuning modes are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    ft_path = (
        os.path.join(ckpdir, "linear_finetuned.pt")
        if linearized_finetuning
        else os.path.join(ckpdir, "finetuned.pt")
    )
    zs_path = (
        os.path.join(ckpdir, "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(ckpdir, "zeroshot.pt")
    )

    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} already exists.")
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."

    print("Building model and tokenizer.")
    if linearized_finetuning:
        model = LinearizedT5Wrapper(args)
        tokenizer = model.tokenizer
    else:
        transformer = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        max_seq_len = getattr(args, "max_seq_len", 128)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, model_max_length=max_seq_len
        )
        model = T5Wrapper(transformer, tokenizer)

    os.makedirs(ckpdir, exist_ok=True)
    zs_name = "linear_zeroshot.pt" if linearized_finetuning else "zeroshot.pt"
    model.save(os.path.join(ckpdir, zs_name))

    model = model.cuda()

    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": 32,
        "max_datapoints_per_dataset_without_templates": None,
    }

    dataset_reader = get_datasetReader(train_dataset, dataset_kwargs)
    createPytorchDataset_fn = lambda dataset: PytorchDataset(dataset, tokenizer, "cuda")
    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=args.batch_size,
        eval_batchSize=args.batch_size * 2,
        world_size=None,
        device=None,
    )
    train_iterator = batcher.get_trainBatches("train", template_idx=0)
    num_batches = args.num_batches
    num_grad_accumulation = getattr(args, "num_grad_accumulation", 1)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda")

    ft_name = "linear_finetuned.pt" if linearized_finetuning else "finetuned.pt"
    patience = getattr(args, "patience", 5)
    best_val_acc = -1.0
    num_bad_checkpoints = 0
    saved_best = False

    print("Loading and processing dataset (first batch may take ~1 min)...", flush=True)
    model.train()
    train_start_time = time.time()
    for i in range(num_batches * num_grad_accumulation):
        start_time = time.time()

        train_batch = next(train_iterator)
        data_time = time.time() - start_time

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, current_metrics = model(train_batch)
            loss = loss / num_grad_accumulation

        scaler.scale(loss).backward()

        if (i + 1) % num_grad_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_time = time.time() - start_time

        step = (i + 1) // num_grad_accumulation
        if (i + 1) % (args.print_every * num_grad_accumulation) == 0:
            percent_complete = 100 * step / num_batches
            elapsed = time.time() - train_start_time
            print(
                f"Train Iteration: {step} [{percent_complete:.0f}% {step}/{num_batches}]\t"
                f"Loss: {loss.item():.6f}\t"
                f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"
                f"Best val acc: {100 * best_val_acc:.2f}%\t"
                f"Elapsed {_format_duration(elapsed)}",
                flush=True,
            )
        if (
            step > 0
            and (i + 1) % (args.checkpoint_frequency * num_grad_accumulation) == 0
        ):
            val_acc = eval_single_dataset(
                "validation", model, tokenizer, train_dataset, args
            )["top1"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                num_bad_checkpoints = 0
                model.save(os.path.join(ckpdir, ft_name))
                saved_best = True
                print(
                    f"New best val acc: {100 * best_val_acc:.2f}% — checkpoint saved.",
                    flush=True,
                )
            else:
                num_bad_checkpoints += 1
                print(
                    f"No improvement ({num_bad_checkpoints}/{patience}). "
                    f"Best val acc: {100 * best_val_acc:.2f}%",
                    flush=True,
                )
            if num_bad_checkpoints >= patience:
                print(f"Early stopping at step {step}.", flush=True)
                return zs_path, ft_path
            model.train()

    # If early stopping never fired, save the final model
    if not saved_best:
        eval_single_dataset("validation", model, tokenizer, train_dataset, args)
        model.save(os.path.join(ckpdir, ft_name))

    return zs_path, ft_path


if __name__ == "__main__":
    args = parse_arguments()

    # Set default training hyperparameters matching ties-merging t5_base config
    args.lr = 1e-4
    args.wd = 0.0
    args.batch_size = 256
    args.num_grad_accumulation = 4
    args.max_seq_len = 128
    args.num_batches = 75000
    args.checkpoint_frequency = 100
    args.print_every = 10
    args.patience = 5

    T5_DATASETS = [
        "qasc",
        "wiki_qa",
        "quartz",
        "paws",
        "story_cloze",
        "winogrande",
        "wsc",
    ]

    # Append model name to save directory
    if args.save is None:
        args.save = f"checkpoints/{args.model}"
    else:
        args.save = os.path.join(args.save, args.model)

    # Append seed to save directory
    if args.seed is not None:
        args.save = os.path.join(args.save, f"seed_{args.seed}")

    for dataset in T5_DATASETS:
        args.train_dataset = dataset

        print("=" * 100)
        print(f"Fine-tuning {args.model} on {dataset} ({args.finetuning_mode})")
        print("=" * 100)

        finetune(args)
