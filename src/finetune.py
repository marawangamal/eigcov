"""
Reccommended distributed training command:
# NOTE: mila cluster has only 64 cpus per node, so don't take more than 32 in total.
salloc --gres=gpu:rtx8000:4 --ntasks-per-node=4 --cpus-per-task=8 --mem=32G
Set num-workers to 2
python src/finetune.py --finetuning-mode=lora --model=ViT-L-14 --world-size=4 --num-workers=1 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets


# With --world-size=1 and num-workers=1, we get:
Train Epoch: 0 [0% 0/133]       Loss: 1.561648  Data (t) 0.046  Batch (t) 0.781 Elapsed 01:02
Train Epoch: 0 [75% 100/133]    Loss: 1.393365  Data (t) 0.253  Batch (t) 0.298 Elapsed 01:34
Train Epoch: 1 [50% 67/133]     Loss: 1.222313  Data (t) 0.258  Batch (t) 0.300 Elapsed 02:54
Train Epoch: 2 [26% 34/133]     Loss: 1.066954  Data (t) 0.258  Batch (t) 0.302 Elapsed 04:09
Train Epoch: 3 [1% 1/133]       Loss: 1.162195  Data (t) 0.227  Batch (t) 0.270 Elapsed 05:29
Train Epoch: 3 [76% 101/133]    Loss: 0.948469  Data (t) 0.247  Batch (t) 0.291 Elapsed 06:00

With --world-size=4 and num-workers=1, we get:
Train Epoch: 0/15 [0% 0/133]    Loss: 1.298983  Data (t) 0.017  Batch (t) 0.989 Elapsed 00:18
Train Epoch: 0/15 [75% 100/133] Loss: 1.357539  Data (t) 0.046  Batch (t) 0.087 Elapsed 00:27
Train Epoch: 1/15 [50% 67/133]  Loss: 1.099862  Data (t) 0.049  Batch (t) 0.093 Elapsed 00:57
Train Epoch: 2/15 [26% 34/133]  Loss: 1.305925  Data (t) 0.043  Batch (t) 0.085 Elapsed 01:47
Train Epoch: 3/15 [1% 1/133]    Loss: 1.260996  Data (t) 0.319  Batch (t) 0.361 Elapsed 02:18
Train Epoch: 3/15 [76% 101/133] Loss: 0.832681  Data (t) 0.042  Batch (t) 0.085 Elapsed 02:28
Train Epoch: 4/15 [51% 68/133]  Loss: 0.598518  Data (t) 0.042  Batch (t) 0.084 Elapsed 02:58

# With --world-size=4 and num-workers=2, we get:
Train Epoch: 0/15 [0% 0/133]    Loss: 1.140516  Data (t) 0.016  Batch (t) 1.658 Elapsed 01:02
Train Epoch: 0/15 [75% 100/133] Loss: 1.342225  Data (t) 0.044  Batch (t) 0.085 Elapsed 01:11
Train Epoch: 1/15 [50% 67/133]  Loss: 1.173152  Data (t) 0.043  Batch (t) 0.086 Elapsed 02:06
Train Epoch: 2/15 [26% 34/133]  Loss: 1.362261  Data (t) 0.053  Batch (t) 0.094 Elapsed 03:05
Train Epoch: 3/15 [1% 1/133]    Loss: 1.342716  Data (t) 0.832  Batch (t) 0.876 Elapsed 03:56
Train Epoch: 3/15 [76% 101/133] Loss: 0.793116  Data (t) 0.046  Batch (t) 0.087 Elapsed 04:05

With --world-size=4 and num-workers=4, we get:
Train Epoch: 0/15 [0% 0/133]    Loss: 1.284399  Data (t) 0.019  Batch (t) 0.841 Elapsed 01:36
Train Epoch: 0/15 [75% 100/133] Loss: 1.329540  Data (t) 0.045  Batch (t) 0.087 Elapsed 01:45
Train Epoch: 1/15 [50% 67/133]  Loss: 1.213168  Data (t) 0.042  Batch (t) 0.085 Elapsed 03:36
Train Epoch: 2/15 [26% 34/133]  Loss: 1.290113  Data (t) 0.044  Batch (t) 0.085 Elapsed 05:13
"""

import os
import time

import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder, apply_lora, merge_lora
from src.utils import LabelSmoothing, cosine_lr


def _format_duration(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def finetune(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    assert args.finetuning_mode in [
        "linear",
        "standard",
        "lora",
    ], "Only linear, standard, and lora fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    lora_finetuning = args.finetuning_mode == "lora"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")
    if lora_finetuning:
        print("Using LoRA fine-tuning.")

    # Check if checkpoints already exist
    if lora_finetuning:
        ft_path = os.path.join(args.save, train_dataset, "lora_finetuned.pt")
        zs_path = os.path.join(args.save, train_dataset, "zeroshot.pt")
    elif linearized_finetuning:
        ft_path = os.path.join(args.save, train_dataset, "linear_finetuned.pt")
        zs_path = os.path.join(args.save, train_dataset, "linear_zeroshot.pt")
    else:
        ft_path = os.path.join(args.save, train_dataset, "finetuned.pt")
        zs_path = os.path.join(args.save, train_dataset, "zeroshot.pt")
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith("pt"):
        image_encoder = (
            LinearizedImageEncoder.load(args.load)
            if linearized_finetuning
            else ImageEncoder.load(args.load)
        )
    else:
        print("Building image encoder.")
        if linearized_finetuning:
            image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        else:
            image_encoder = ImageEncoder(args)

    # Save the zeroshot encoder before applying LoRA (reuses standard zeroshot.pt)
    if lora_finetuning and args.save is not None and is_main_process():
        os.makedirs(ckpdir, exist_ok=True)
        zs_path = os.path.join(ckpdir, "zeroshot.pt")
        if not os.path.exists(zs_path):
            image_encoder.save(zs_path)

    if lora_finetuning:
        target_modules = args.lora_target_modules.split(",")
        # target_parameters = args.lora_target_parameters.split(",")
        image_encoder = apply_lora(
            image_encoder,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            target_modules=target_modules,
            # target_parameters=target_parameters,  # TODO: make this work
        )

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(
            f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

    if lora_finetuning and is_main_process():
        print("\n=== LoRA Config ===")
        print(f"  rank: {args.lora_rank}")
        print(f"  alpha: {args.lora_alpha}")
        print(f"  dropout: {args.lora_dropout}")
        print(f"  target_modules: {target_modules}")
        print("=== LoRA Model Architecture ===")
        print(image_encoder)
        print("===============================\n")

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)

    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,  # Set False when all params are used (faster, avoids DDP warning)
        output_device=rank,
    )

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    if is_main_process():
        print(
            "Total number of steps: ",
            args.epochs * num_batches // args.num_grad_accumulation,
        )

    # Saving zero-shot model (LoRA zeroshot is saved before LoRA is applied)
    if args.save is not None and is_main_process() and not lora_finetuning:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        ddp_model.module.image_encoder.save(model_path)

    run_start_time = time.perf_counter()
    for epoch in range(args.epochs):
        ddp_model.train()

        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time

            logits = ddp_model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                if linearized_finetuning:
                    model_path = os.path.join(ckpdir, f"linear_checkpoint_{step}.pt")
                elif lora_finetuning:
                    model_path = os.path.join(ckpdir, f"lora_checkpoint_{step}.pt")
                else:
                    model_path = os.path.join(ckpdir, f"checkpoint_{step}.pt")
                ddp_model.module.image_encoder.save(model_path)
                print(f"Saved checkpoint to {model_path}")

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_loader)
                run_elapsed = time.perf_counter() - run_start_time
                print(
                    f"Train Epoch: {epoch}/{args.epochs} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"  # noqa: E501
                    f"Elapsed {_format_duration(run_elapsed)}",  # noqa: E501
                    flush=True,
                )

    # FIXME: Make this work with DDP.
    if is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = ddp_model.module.image_encoder

        # Merge LoRA weights back into the base model before eval/save
        if lora_finetuning:
            image_encoder = merge_lora(image_encoder)

        eval_single_dataset(image_encoder, train_dataset, args)

    if args.save is not None and is_main_process():
        if lora_finetuning:
            zs_path = os.path.join(ckpdir, "zeroshot.pt")
            ft_path = os.path.join(ckpdir, "lora_finetuned.pt")
        elif linearized_finetuning:
            zs_path = os.path.join(ckpdir, "linear_zeroshot.pt")
            ft_path = os.path.join(ckpdir, "linear_finetuned.pt")
        else:
            zs_path = os.path.join(ckpdir, "zeroshot.pt")
            ft_path = os.path.join(ckpdir, "finetuned.pt")
        image_encoder.save(ft_path)
        return zs_path, ft_path

    cleanup_ddp()


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
    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
    }

    args = parse_arguments()
    if args.train_dataset is not None:
        train_datasets = [ds.strip() for ds in args.train_dataset]

    for dataset in train_datasets:
        # HACK: Some command line arguments are overwritten by defaults here.
        args.lr = 1e-5
        args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        args.batch_size = 64 if args.model == "ViT-L-14" else 128
        args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

        if args.seed is not None:
            args.save = f"checkpoints_{args.seed}/{args.model}"
        else:
            args.save = f"checkpoints/{args.model}"
        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
