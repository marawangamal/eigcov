import os
import time

import numpy as np
import torch
import torch.nn.functional as F


# --finetuning-mode=standard   --model=ViT-B-16   --world-size=1   --num-workers=1   --openclip-cachedir=$SCRATCH/openclip   --data-location=datasets   --save=$SCRATCH/eigcov/checkpoints/vision   --cosine-samples=128
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
    print_every = 1

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

    # --- Cosine similarity tracking setup ---
    # We fix _grad_0 to the first gradient computed and track cos(grad_k, grad_0)
    # over training. At step 0, grad_k == grad_0 so cosine similarity starts at 1.
    _cosine_hooks = []
    _grad_0 = None  # fixed reference: first gradient snapshot
    _grad_k = None  # current gradient snapshot (updated each optimizer step)
    _cosine_sim_log = []  # list of (step, cos_sim)
    _last_cos_sim = float("nan")

    if args.cosine_samples > 0 and is_main_process():
        param_offsets = []
        offset = 0
        for p in params:
            param_offsets.append((offset, offset + p.numel(), p))
            offset += p.numel()
        total_grad_elements = offset

        _sampled_indices = torch.randperm(total_grad_elements)[: args.cosine_samples]
        _grad_k = torch.zeros(args.cosine_samples)

        # For each parameter, register a hook that writes the sampled gradient
        # elements into _grad_k. _sampled_indices are global flat indices across
        # the concatenation of all parameters; here we find which ones fall in
        # this parameter's slice [start, end), convert them to local indices, and
        # accumulate grad.flatten()[local_idxs] into the corresponding slots of
        # _grad_k. After each optimizer step, _grad_k holds a sampled snapshot of
        # the gradient. += is used (not =) so that when num_grad_accumulation > 1,
        # _grad_k accumulates across micro-steps, mirroring .grad accumulation.
        for start, end, p in param_offsets:
            mask = (_sampled_indices >= start) & (_sampled_indices < end)
            if not mask.any():
                continue
            positions = torch.where(mask)[0]  # slots in _grad_k to write to
            local_idxs = _sampled_indices[positions] - start  # within-parameter indices

            def _make_hook(pos, lidxs):
                def _hook(grad):
                    _grad_k[pos] += grad.detach().flatten()[lidxs].cpu()

                return _hook

            _cosine_hooks.append(p.register_hook(_make_hook(positions, local_idxs)))

    # If tracking cosine similarity, fix a single batch so gradients are
    # deterministic — with lr=0 this should give cos sim = 1 every step.
    _fixed_batch = None
    if args.cosine_samples > 0 and is_main_process():
        _fixed_batch = maybe_dictionarize(next(iter(ddp_loader)))

    run_start_time = time.perf_counter()
    for epoch in range(args.epochs):
        ddp_model.train()

        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = _fixed_batch if _fixed_batch is not None else maybe_dictionarize(batch)
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

                if args.cosine_samples > 0 and is_main_process():
                    if _grad_0 is None:
                        _grad_0 = _grad_k.clone()  # fix reference at first step
                    if _grad_0.norm() > 0 and _grad_k.norm() > 0:
                        _last_cos_sim = F.cosine_similarity(
                            _grad_k.unsqueeze(0), _grad_0.unsqueeze(0)
                        ).item()
                        _cosine_sim_log.append((step, _last_cos_sim))
                    _grad_k.zero_()

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
                cos_str = (
                    f"\tCos sim {_last_cos_sim:.4f}" if args.cosine_samples > 0 else ""
                )
                print(
                    f"Train Epoch: {epoch}/{args.epochs} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"  # noqa: E501
                    f"Elapsed {_format_duration(run_elapsed)}{cos_str}",  # noqa: E501
                    flush=True,
                )

    # Cosine sim cleanup and save
    if args.cosine_samples > 0 and is_main_process():
        for h in _cosine_hooks:
            h.remove()
        if _cosine_sim_log and args.save is not None:
            steps_arr, sims_arr = zip(*_cosine_sim_log)
            np.savez(
                os.path.join(ckpdir, "cosine_sim.npz"),
                steps=np.array(steps_arr),
                cosine_sim_to_grad0=np.array(sims_arr),
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
        # args.lr = 1e-5
        args.epochs = 5 if args.cosine_samples > 0 else epochs[dataset]
        args.train_dataset = dataset + "Val"

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        args.batch_size = 64 if args.model == "ViT-L-14" else 128
        args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

        # Append model name to save directory
        if args.save is None:
            args.save = f"checkpoints/{args.model}"
        else:
            args.save = os.path.join(args.save, args.model)

        # Append seed to save directory
        if args.seed is not None:
            args.save = os.path.join(args.save, f"seed_{args.seed}")

        print("=" * 100)
        print(f"Fine-tuning {args.model} on {dataset} ({args.finetuning_mode})")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
