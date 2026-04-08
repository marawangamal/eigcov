"""Collect per-layer activation covariances for Olmo-3-7B on Dolci RL-Zero datasets.

Runs forward passes (no training) while forward hooks from src/covariance.py
capture activation statistics. Each capability produces a covariance NPZ file
saved inside the finetuned model's param-folder directory, where
ParamFolderTaskVector auto-discovers it for RegMean merging.

Usage:
    export PYTHONPATH="$PYTHONPATH:$PWD"

    # Single capability
    python scripts/olmo/covariance.py --capability math --save checkpoints/olmo

    # All capabilities
    python scripts/olmo/covariance.py --capability all --save checkpoints/olmo
"""

import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.args import parse_arguments
from src.covariance import register_hooks

PRETRAINED_MODEL = "allenai/Olmo-3-1025-7B"
PRETRAINED_TOKENIZER = "allenai/Olmo-3-7B-RL-Zero-Math"
MAX_SEQ_LEN = 256

CAPABILITY_DATASETS = {
    "Math": "allenai/Dolci-RL-Zero-Math-7B",
    "Code": "allenai/Dolci-RL-Zero-Code-7B",
    "IF": "allenai/Dolci-RL-Zero-IF-7B",
}


def collect_covariance(capability, args):
    print(f"\n{'='*60}")
    print(f"Collecting covariance: {capability}")
    print(f"{'='*60}")

    run_dir = os.path.join(args.save, capability)
    cov_path = os.path.join(run_dir, "covariance.pt")

    if os.path.exists(cov_path) and not args.overwrite:
        print(f"  Skipping {capability} — {cov_path} already exists")
        return

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    # Load dataset
    dataset_id = CAPABILITY_DATASETS[capability]
    print(f"  Dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", cache_dir=args.cache_dir)
    print(f"  {len(ds)} examples")

    if len(ds) == 0:
        print(f"  WARNING: No examples found for {capability}, skipping.")
        return

    # Load tokenizer (from RL-Zero Math model which has a chat template)
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_TOKENIZER, cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize only the prompts (no assistant answers needed for covariance)
    def tokenize_fn(examples):
        messages = [
            [{"role": "user", "content": prompt}] for prompt in examples["prompt"]
        ]
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in messages
        ]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors=None,
        )

    ds = ds.map(
        tokenize_fn, batched=True, remove_columns=ds.column_names, desc="Tokenizing"
    )
    ds.set_format("torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(ds, batch_size=args.cov_batch_size, shuffle=False)

    # Load model
    print("Loading model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )
    model.to(device)
    model.eval()

    # Register forward hooks
    cobjs, handles = register_hooks(
        model,
        cov_device="cpu",
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        batch_first=True,
    )

    # Forward pass loop
    max_num_batches = max(args.cov_num_batches)
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Computing covariance",
            total=min(max_num_batches, len(dataloader)),
        ):
            if n_batches >= max_num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            n_batches += 1

    print(f"  Processed {n_batches} batches")

    # Remove hooks and save
    for h in handles:
        h.remove()

    saveable = {}
    for name, cobj in cobjs.items():
        saveable[name] = cobj.cov.cpu()
        saveable[f"{name}_n"] = cobj.n

    os.makedirs(run_dir, exist_ok=True)
    torch.save(saveable, cov_path)
    print(f"Saved covariances ({len(cobjs)} layers) to {cov_path}")


def main():
    args = parse_arguments()
    if args.save is None:
        args.save = "checkpoints/Olmo-3-7b"

    if args.capability == "all":
        for cap in CAPABILITY_DATASETS:
            collect_covariance(cap, args)
    else:
        collect_covariance(args.capability, args)


if __name__ == "__main__":
    main()
