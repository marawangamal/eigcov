"""Collect per-layer activation covariances for Olmo-3-7B on Dolci RL-Zero datasets.

Runs SFTTrainer with lr=0 (no weight updates) while forward hooks from
src/covariance.py capture activation statistics.  Each capability produces
a covariance NPZ file that can be used by RegMean or other covariance-based
merge methods.

Uses allenai/Dolci-RL-Zero-{Math|Code|IF}-7B datasets (~13k examples each).


Usage:
    pip install datasets transformers trl peft

    # Single GPU
    python scripts/rl/finetune.py --capability math --hf-cache-dir $SCRATCH/huggingface

    # All capabilities
    python scripts/rl/finetune.py --capability all --hf-cache-dir $SCRATCH/huggingface
"""

import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

from src.args import parse_arguments
from src.covariance import register_hooks

PRETRAINED_MODEL = "allenai/Olmo-3-1025-7B"
MAX_SEQ_LEN = 4096

# Each capability maps to its own HuggingFace dataset
CAPABILITY_DATASETS = {
    "Math": "allenai/Dolci-RL-Zero-Math-7B",
    "Code": "allenai/Dolci-RL-Zero-Code-7B",
    "IF": "allenai/Dolci-RL-Zero-IF-7B",
}

# Column used as the assistant response for each capability
ANSWER_COLUMN = {
    "Math": "ground_truth",
    "Code": "solution",
    "IF": "ground_truth",
}


def _prepare_messages(ds, capability):
    """Construct a uniform `messages` column from capability-specific schemas.

    Each dataset has different columns for the answer:
      - math: prompt + ground_truth
      - code: prompt + solution
      - if:   prompt + ground_truth

    Returns a dataset with only a `messages` column in chat format.
    """
    answer_col = ANSWER_COLUMN[capability]

    def _to_messages(ex):
        return {
            "messages": [
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex[answer_col]},
            ]
        }

    ds = ds.map(_to_messages, desc=f"Preparing messages for {capability}")
    ds = ds.remove_columns([c for c in ds.column_names if c != "messages"])
    return ds


def train_capability(capability, args):
    print(f"\n{'='*60}")
    print(f"Training capability: {capability}")
    print(f"{'='*60}")

    run_dir = os.path.join(args.save, capability)
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.resume:
        print(f"  Skipping {capability} — {run_dir} already exists")
        return

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    # Load dataset
    dataset_id = CAPABILITY_DATASETS[capability]
    print(f"  Dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", cache_dir=args.cache_dir)
    ds = _prepare_messages(ds, capability)
    print(f"  {len(ds)} examples")

    if len(ds) == 0:
        print(f"  WARNING: No examples found for {capability}, skipping.")
        return

    # Load tokenizer from base model (already has pad_token and ChatML tokens)
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL, cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )

    # Register forward hooks to collect activation covariances
    cobjs, handles = register_hooks(
        model,
        cov_device="cpu",
        cov_type=args.cov_type,
        cov_estimator=args.cov_estimator,
        batch_first=True,
    )

    fsdp_kwargs = {}
    if args.fsdp:
        fsdp_kwargs["fsdp"] = "full_shard auto_wrap"
        fsdp_kwargs["fsdp_config"] = {"auto_wrap_policy": "TRANSFORMER_BASED_WRAP"}

    sft_config = SFTConfig(
        output_dir=run_dir,
        max_length=MAX_SEQ_LEN,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.num_grad_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        # Disable gradient checkpointing — it recomputes forward passes
        # during backward, which would trigger hooks twice per batch.
        gradient_checkpointing=False,
        **fsdp_kwargs,
    )

    peft_config = None
    if args.finetuning_mode == "lora":
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
        )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # Remove hooks and save covariance matrices
    for h in handles:
        h.remove()

    saveable = {}
    for name, cobj in cobjs.items():
        saveable[name] = cobj.cov.cpu().numpy()
        saveable[f"{name}_n"] = cobj.n

    os.makedirs(run_dir, exist_ok=True)
    cov_path = os.path.join(run_dir, f"covariance_{capability}.npz")
    np.savez(cov_path, **saveable)
    print(f"Saved covariances ({len(cobjs)} layers) to {cov_path}")
    print(f"Done: {capability}")


def main():
    args = parse_arguments()
    if args.save is None:
        args.save = "checkpoints/Olmo-3-7b"

    if args.capability == "all":
        for cap in CAPABILITY_DATASETS:
            train_capability(cap, args)
    else:
        train_capability(args.capability, args)


if __name__ == "__main__":
    main()


# EigCov
# codex_humaneval::tulu: 0.837081
# codex_humanevalplus::tulu: 0.797757
# ifeval::tulu: 0.469501
# aime:2024::olmo3:midtrain: 0.0854167
# aime:2025::olmo3:midtrain: 0.109375

# TSV
# codex_humaneval::tulu: 0.8238
# codex_humanevalplus::tulu: 0.790259
# ifeval::tulu: 0.340111
# aime:2024::olmo3:midtrain: 0.120833
# aime:2025::olmo3:midtrain: 0.141667

# Isoc
# codex_humaneval::tulu: 0.847963
# codex_humanevalplus::tulu: 0.792782
# ifeval::tulu: 0.288355
# aime:2024::olmo3:midtrain: 0.122917
# aime:2025::olmo3:midtrain: 0.161458
