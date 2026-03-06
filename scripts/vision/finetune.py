import os
import time

import torch


# --finetuning-mode=standard   --model=ViT-B-16   --world-size=1   --num-workers=1   --openclip-cachedir=$SCRATCH/openclip   --data-location=datasets   --save=$SCRATCH/eigcov/checkpoints/vision
from src.args import parse_arguments
from src.vision.datasets.common import get_dataloader, maybe_dictionarize
from src.vision.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.vision.eval import eval_single_dataset
from src.vision.heads import get_classification_head
from src.vision.linearize import LinearizedImageEncoder
from src.vision.modeling import ImageClassifier, ImageEncoder, apply_lora, merge_lora
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
    if os.path.exists(zs_path) and os.path.exists(ft_path) and not args.overwrite:
        print(f"Skipping fine-tuning because {ft_path} already exists.")
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
        print(f"Total steps: {args.epochs * num_batches // args.num_grad_accumulation}")

    # Saving zero-shot model (LoRA zeroshot is saved before LoRA is applied)
    if args.save is not None and is_main_process() and not lora_finetuning:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        ddp_model.module.image_encoder.save(model_path)

    # Cross-sample gradient IP tracking (condition i of Covariance Estimation Theorem)
    if args.grad_cross_ip and is_main_process():
        _grad_sum = None  # lazily initialized on first step (size = #params with grad)
        _grad_sq_norm_sum = 0.0  # sum_k ||g_k||^2
        _K = 0

    run_start_time = time.perf_counter()
    for epoch in range(args.epochs):
        ddp_model.train()

        for i, batch in enumerate(ddp_loader):
            # if args.grad_cross_ip and i >= 100:
            #     break
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

            if args.grad_cross_ip and is_main_process():
                g = torch.cat(
                    [
                        p.grad.detach().cpu().view(-1).float()
                        for p in ddp_model.module.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                )
                if _grad_sum is None:
                    _grad_sum = torch.zeros_like(g)
                _grad_sum += g
                _grad_sq_norm_sum += g.dot(g).item()
                _K += 1

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
                if linearized_finetuning:
                    model_path = os.path.join(ckpdir, f"linear_checkpoint_{step}.pt")
                elif lora_finetuning:
                    model_path = os.path.join(ckpdir, f"lora_checkpoint_{step}.pt")
                else:
                    model_path = os.path.join(ckpdir, f"checkpoint_{step}.pt")
                ddp_model.module.image_encoder.save(model_path)
                print(f"Saved checkpoint to {model_path}", flush=True)

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

    if args.grad_cross_ip and is_main_process() and _K > 1:
        grad_sum_sq = _grad_sum.dot(_grad_sum).item()
        mean_self_ip = _grad_sq_norm_sum / _K
        mean_cross_ip = (grad_sum_sq - _grad_sq_norm_sum) / (_K * (_K - 1))
        normalized = mean_cross_ip / mean_self_ip
        print("\n=== Cross-sample gradient orthogonality (condition i) ===")
        print(f"  Steps K                  : {_K}")
        print(f"  mean ||g_k||^2           : {mean_self_ip:.6e}   <- self IP")
        print(f"  mean g_k^T g_k' (k!=k') : {mean_cross_ip:.6e}   <- cross IP")
        print(f"  normalized cross-IP      : {normalized:.6e}   <- ~0 if orthogonal")
        os.makedirs(ckpdir, exist_ok=True)
        torch.save(
            {
                "K": _K,
                "mean_self_ip": mean_self_ip,
                "mean_cross_ip": mean_cross_ip,
                "normalized_cross_ip": normalized,
                "grad_sum": _grad_sum,
                "grad_sq_norm_sum": _grad_sq_norm_sum,
            },
            os.path.join(ckpdir, "grad_cross_ip.pt"),
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

    # Append model name to save directory
    if args.save is None:
        args.save = f"checkpoints/{args.model}"
    else:
        args.save = os.path.join(args.save, args.model)

    # Append seed to save directory
    if args.seed is not None:
        args.save = os.path.join(args.save, f"seed_{args.seed}")

    if args.train_dataset is not None:
        train_datasets = [ds.strip() for ds in args.train_dataset]

    for dataset in train_datasets:
        # HACK: Some command line arguments are overwritten by defaults here.
        args.lr = 1e-5
        if not args.grad_cross_ip:
            args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        args.batch_size = 64 if args.model == "ViT-L-14" else 128
        args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

        print("=" * 100)
        print(f"Fine-tuning {args.model} on {dataset} ({args.finetuning_mode})")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
