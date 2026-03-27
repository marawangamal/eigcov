"""Fine-tune Olmo-3-7B on individual capability subsets of the Dolci RL-Zero datasets.

Each capability is trained separately to produce a specialist checkpoint,
which can later be merged using scripts/rl/merge.py.

Uses allenai/Dolci-RL-Zero-{Math|Code|IF}-7B datasets (~13k examples each),
each containing prompts and ground-truth answers for SFT training.


Usage:
    pip install datasets transformers trl peft

    # Single GPU (LoRA)
    python scripts/rl/finetune.py --capability math --use-lora --hf-cache-dir $SCRATCH/huggingface

    # Multi-GPU full fine-tune with FSDP (recommended for 7B without LoRA)
    torchrun --nproc_per_node=4 scripts/rl/finetune.py --capability math --fsdp --hf-cache-dir $SCRATCH/huggingface

    # All capabilities
    torchrun --nproc_per_node=4 scripts/rl/finetune.py --capability all --fsdp --hf-cache-dir $SCRATCH/huggingface
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

PRETRAINED_MODEL = "allenai/Olmo-3-1025-7B"
MAX_SEQ_LEN = 4096

# Each capability maps to its own HuggingFace dataset
CAPABILITY_DATASETS = {
    "math": "allenai/Dolci-RL-Zero-Math-7B",
    "code": "allenai/Dolci-RL-Zero-Code-7B",
    "if": "allenai/Dolci-RL-Zero-IF-7B",
}

# Column used as the assistant response for each capability
ANSWER_COLUMN = {
    "math": "ground_truth",
    "code": "solution",
    "if": "ground_truth",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Olmo-3-7B on Dolci RL-Zero capability datasets"
    )
    p.add_argument(
        "--capability",
        type=str,
        required=True,
        choices=list(CAPABILITY_DATASETS.keys()) + ["all"],
    )
    p.add_argument("--output-dir", type=str, default="checkpoints/rl")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--hf-cache-dir", type=str, default=None)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument(
        "--save-strategy", type=str, default="epoch", choices=["epoch", "steps", "no"]
    )
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP full sharding (required for full fine-tune on multi-GPU)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in output dir",
    )
    return p.parse_args()


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

    run_dir = os.path.join(args.output_dir, f"Olmo-3-7B-{capability}")
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.resume:
        print(f"  Skipping {capability} — {run_dir} already exists")
        return

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir

    # Load dataset
    dataset_id = CAPABILITY_DATASETS[capability]
    print(f"  Dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", cache_dir=args.hf_cache_dir)
    ds = _prepare_messages(ds, capability)
    print(f"  {len(ds)} examples")

    if len(ds) == 0:
        print(f"  WARNING: No examples found for {capability}, skipping.")
        return

    # Load tokenizer from base model (already has pad_token and ChatML tokens)
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL, cache_dir=args.hf_cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        cache_dir=args.hf_cache_dir,
    )

    fsdp_kwargs = {}
    if args.fsdp:
        fsdp_kwargs["fsdp"] = "full_shard auto_wrap"
        fsdp_kwargs["fsdp_config"] = {"auto_wrap_policy": "TRANSFORMER_BASED_WRAP"}

    sft_config = SFTConfig(
        output_dir=run_dir,
        max_length=MAX_SEQ_LEN,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        bf16=args.bf16,
        logging_steps=10,
        save_strategy=args.save_strategy,
        **({f"save_steps": args.save_steps} if args.save_strategy == "steps" else {}),
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **fsdp_kwargs,
    )

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
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

    print(f"Saving to {run_dir} ...")
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)
    print(f"Done: {capability}")


def main():
    args = parse_args()

    if args.capability == "all":
        for cap in CAPABILITY_DATASETS:
            train_capability(cap, args)
    else:
        train_capability(args.capability, args)


if __name__ == "__main__":
    main()
