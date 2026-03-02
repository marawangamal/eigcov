import os
import time

import torch

from src.args import parse_arguments
from src.vision.datasets.common import get_dataloader, maybe_dictionarize
from src.vision.datasets.registry import get_dataset
from src.vision.heads import get_classification_head
from src.vision.linearize import LinearizedImageEncoder
from src.vision.modeling import ImageClassifier, ImageEncoder
from src.utils import LabelSmoothing


def _format_duration(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def compute_gradient(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only 'linear' and 'standard' modes are supported."

    linearized_finetuning = args.finetuning_mode == "linear"

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith("pt"):
        image_encoder = (
            LinearizedImageEncoder.load(args.load)
            if linearized_finetuning
            else ImageEncoder.load(args.load)
        )
    else:
        print("Building image encoder.")
        image_encoder = (
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    classification_head = get_classification_head(args, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    model = model.cuda()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    dataset = get_dataset(
        train_dataset,
        model.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)

    loss_fn = LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()

    print(f"Accumulating gradient over {num_batches} batches...")

    model.train()
    model.zero_grad()

    start_time = time.perf_counter()
    for i, batch in enumerate(data_loader):
        batch = maybe_dictionarize(batch)
        inputs = batch["images"].cuda()
        labels = batch["labels"].cuda()

        loss = loss_fn(model(inputs), labels)
        # Normalize so the accumulated gradient is an average over batches
        (loss / num_batches).backward()

        if (i + 1) % 10 == 0:
            percent_complete = 100 * (i + 1) / num_batches
            elapsed = time.perf_counter() - start_time
            print(
                f"[{percent_complete:.0f}% {i + 1}/{num_batches}]\t"
                f"Loss: {loss.item():.6f}\tElapsed {_format_duration(elapsed)}",
                flush=True,
            )

    # Collect gradients into a {param_name: grad} dict and save
    grad_dict = {
        name: p.grad.cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    os.makedirs(ckpdir, exist_ok=True)
    prefix = "linear_" if linearized_finetuning else ""
    save_path = os.path.join(ckpdir, f"{prefix}gradient.pt")
    torch.save(grad_dict, save_path)
    print(f"Saved gradient dict ({len(grad_dict)} tensors) to {save_path}")


if __name__ == "__main__":
    train_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]

    args = parse_arguments()

    # Append model name and seed to save directory once, before the loop
    if args.save is None:
        args.save = f"checkpoints/{args.model}"
    else:
        args.save = os.path.join(args.save, args.model)
    if args.seed is not None:
        args.save = os.path.join(args.save, f"seed_{args.seed}")

    if args.train_dataset is not None:
        train_datasets = [ds.strip() for ds in args.train_dataset]

    for dataset in train_datasets:
        args.train_dataset = dataset + "Val"
        args.batch_size = 64 if args.model == "ViT-L-14" else 128

        print("=" * 100)
        print(
            f"Computing gradient for {args.model} on {dataset} ({args.finetuning_mode})"
        )
        print("=" * 100)
        compute_gradient(args)
