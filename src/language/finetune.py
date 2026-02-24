import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.language.modeling import T5Wrapper
from src.language.linearize import LinearizedT5Wrapper
from src.language.datasets.pytorch_dataset import PytorchDataset
from src.language.datasets.batcher import Batcher
from src.language.datasets.dataset_readers import get_datasetReader
from src.language.eval import eval_single_dataset


def finetune(rank, args):
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
        tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=max_seq_len)
        model = T5Wrapper(transformer, tokenizer)

    os.makedirs(ckpdir, exist_ok=True)
    zs_name = "linear_zeroshot.pt" if linearized_finetuning else "zeroshot.pt"
    model.save(os.path.join(ckpdir, zs_name))

    model = model.cuda()

    print_every = 100

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
        world_size=args.world_size,
        device=rank,
    )
    train_iterator = batcher.get_trainBatches("train", template_idx=0)
    num_batches = args.num_batches
    num_grad_accumulation = getattr(args, "num_grad_accumulation", 1)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model.train()
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

        if i % print_every == 0:
            percent_complete = 100 * i / num_batches
            print(
                f"Train Iteration: {i} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                f"Loss: {loss.item():.6f}\t"
                f"Data (t) {data_time:.3f}\t"
                f"Batch (t) {batch_time:.3f}",
                flush=True,
            )

    # Eval on validation split after training
    eval_single_dataset("validation", model, tokenizer, train_dataset, args)

    # Save fine-tuned model
    ft_name = "linear_finetuned.pt" if linearized_finetuning else "finetuned.pt"
    model.save(os.path.join(ckpdir, ft_name))

    return zs_path, ft_path
