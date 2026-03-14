import copy
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
from src.mhas import swap_mha, unswap_mha
from src.utils import LabelSmoothing, cosine_lr


class GradCrossTermTracker:
    """Measures per-layer gradient cross-term error during training.

    For a subset of evenly-spaced linear layers, accumulates per-sample
    gradient statistics across batches:
      - grad_sum:  batch-mean of per-sample gradients, summed over batches  (Σ_k G^(k))
      - gram_sum:  batch-mean of per-sample G^T G, summed over batches      (Σ_k S^(k))
      - stilde_sum: Σ_k E[zz^T] * E[||g||^2], the uncorrelated second moment

    At the end of training, compares cosine distance between grad_sum^T @ grad_sum and gram_sum,
    and between gram_sum (S_bar) and stilde_sum (S_tilde).
    """

    def __init__(self, model, max_layers=8):
        linear_layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        n = len(linear_layers)
        num_tracked = min(max_layers, n)
        indices = (
            [round(i * (n - 1) / (num_tracked - 1)) for i in range(num_tracked)]
            if num_tracked > 1
            else [0]
        )
        self.layers = {linear_layers[i][0]: linear_layers[i][1] for i in indices}
        self.grad_sum = {name: None for name in self.layers}
        self.gram_sum = {name: None for name in self.layers}
        self.stilde_sum = {name: None for name in self.layers}
        self.num_batches = 0

        # Hook storage
        self._activations = {}
        self._output_grads = {}
        self._hooks = []

        for name, module in self.layers.items():
            self._hooks.append(module.register_forward_hook(self._make_fwd_hook(name)))
            self._hooks.append(
                module.register_full_backward_hook(self._make_bwd_hook(name))
            )

        print(f"\n=== Tracking {len(self.layers)} layers for grad cross-terms ===")
        for name, module in self.layers.items():
            print(f"  {name}: weight {tuple(module.weight.shape)}")
        print()

    def _make_fwd_hook(self, name):
        def hook(module, input, output):
            self._activations[name] = input[0].detach()

        return hook

    def _make_bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            self._output_grads[name] = grad_output[0].detach()

        return hook

    def step(self, model, inputs, labels, loss_fn):
        """Compute per-sample gradients for one batch and accumulate statistics."""
        batch_size = inputs.shape[0]

        # Accumulators for batch-mean E[zz^T] and E[||gy||^2]
        zzT_sum = {name: None for name in self.layers}
        gynorm_sum = {name: 0.0 for name in self.layers}

        for b in range(batch_size):
            model.zero_grad()
            logits = model(inputs[b : b + 1])
            loss = loss_fn(logits, labels[b : b + 1])
            loss.backward()

            for name, module in self.layers.items():
                g = module.weight.grad.detach().float()  # (d_out, d_in)
                if self.grad_sum[name] is None:
                    d_in = g.shape[1]
                    self.grad_sum[name] = torch.zeros_like(g, device="cpu")
                    self.gram_sum[name] = torch.zeros(d_in, d_in, dtype=torch.float32)
                    self.stilde_sum[name] = torch.zeros(d_in, d_in, dtype=torch.float32)
                self.grad_sum[name] += g.cpu() / batch_size
                self.gram_sum[name] += (g.T @ g).cpu() / batch_size

                # Accumulate z and gy statistics from hooks.
                # Activations may be (T, B, d_in), (B, T, d_in), or (B, d_in);
                # flatten everything except the feature dim.
                z = self._activations[name].float().reshape(-1, self._activations[name].shape[-1])   # (N, d_in)
                gy = self._output_grads[name].float().reshape(-1, self._output_grads[name].shape[-1])  # (N, d_out)
                if zzT_sum[name] is None:
                    d_in = z.shape[-1]
                    zzT_sum[name] = torch.zeros(d_in, d_in, dtype=torch.float32)
                zzT_sum[name] += (z.cpu().T @ z.cpu()) / batch_size
                gynorm_sum[name] += (gy.norm() ** 2).cpu().item() / batch_size

        # S_tilde_k = E[zz^T] * E[||gy||^2]
        for name in self.layers:
            if zzT_sum[name] is not None:
                self.stilde_sum[name] += zzT_sum[name] * gynorm_sum[name]

        self.num_batches += 1
        model.zero_grad()

    def save(self, ckpdir):
        """Print summary and save per-layer results to disk."""
        if self.num_batches <= 1:
            return
        K = self.num_batches
        print(f"\n=== Per-layer gradient cross-term error [K={K}] ===")
        os.makedirs(ckpdir, exist_ok=True)
        for name in self.layers:
            M = self.grad_sum[name]
            S = self.gram_sum[name]
            St = self.stilde_sum[name]
            MtM = M.T @ M

            # cross-term error: cos_dist(M^T M, S)
            a, b = MtM.flatten(), S.flatten()
            cos_dist_cross = 1 - torch.dot(a, b).abs() / (a.norm() * b.norm())

            # correlation error: cos_dist(S_bar, S_tilde)
            a2, b2 = S.flatten(), St.flatten()
            cos_dist_corr = 1 - torch.dot(a2, b2).abs() / (a2.norm() * b2.norm())

            # arcsin bound: ||S_bar - S_tilde|| / ||S_bar||
            corr_diff_ratio = (S - St).norm() / S.norm()

            print(f"  {name}:")
            print(f"    ||M^T M||_F       = {MtM.norm().item():.6e}")
            print(f"    ||S_bar||_F       = {S.norm().item():.6e}")
            print(f"    ||S_tilde||_F     = {St.norm().item():.6e}")
            print(f"    cos_dist(cross)   = {cos_dist_cross.item():.6e}")
            print(f"    cos_dist(corr)    = {cos_dist_corr.item():.6e}")
            print(f"    ||S-St||/||S||    = {corr_diff_ratio.item():.6e}")
            fname = f"grad_cross_matrix_{name.replace('.', '_')}.pt"
            torch.save(
                {
                    "K": K,
                    "M": M,
                    "S_bar": S,
                    "S_tilde": St,
                    "M_T_M": MtM,
                    "cos_dist_cross": cos_dist_cross.item(),
                    "cos_dist_corr": cos_dist_corr.item(),
                    "corr_diff_ratio": corr_diff_ratio.item(),
                },
                os.path.join(ckpdir, fname),
            )
        print()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


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
        cleanup_ddp()
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

    # Save zeroshot before MHA swap so checkpoint is in standard format.
    # (LoRA zeroshot is already saved above, before LoRA is applied.)
    if args.save is not None and is_main_process() and not lora_finetuning:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        model.image_encoder.save(model_path)

    # Swap nn.MultiheadAttention -> MultiHeadAttentionSplit so forward hooks
    # fire on per-projection Linear layers (needed for grad cross-term tracking).
    if args.grad_cross_matrix:
        swap_mha(model.image_encoder)

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

    grad_cross_tracker = None
    if args.grad_cross_matrix and is_main_process():
        grad_cross_tracker = GradCrossTermTracker(ddp_model.module.image_encoder)

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

            if grad_cross_tracker is not None:
                grad_cross_tracker.step(ddp_model.module, inputs, labels, loss_fn)

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
                if linearized_finetuning:
                    model_path = os.path.join(ckpdir, f"linear_checkpoint_{step}.pt")
                elif lora_finetuning:
                    model_path = os.path.join(ckpdir, f"lora_checkpoint_{step}.pt")
                else:
                    model_path = os.path.join(ckpdir, f"checkpoint_{step}.pt")
                enc = ddp_model.module.image_encoder
                if args.grad_cross_matrix:
                    enc = copy.deepcopy(enc).cpu()
                    for m in enc.modules():
                        m._forward_hooks.clear()
                        m._backward_hooks.clear()
                    unswap_mha(enc)
                enc.save(model_path)
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

    if grad_cross_tracker is not None:
        grad_cross_tracker.save(ckpdir)

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
        enc_to_save = image_encoder
        if args.grad_cross_matrix:
            enc_to_save = copy.deepcopy(image_encoder).cpu()
            for m in enc_to_save.modules():
                m._forward_hooks.clear()
                m._backward_hooks.clear()
            unswap_mha(enc_to_save)
        enc_to_save.save(ft_path)
        cleanup_ddp()
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
        if not args.grad_cross_matrix:
            args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        args.batch_size = 64 if args.model == "ViT-L-14" else 128
        args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

        print("=" * 100)
        print(f"Fine-tuning {args.model} on {dataset} ({args.finetuning_mode})")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
