"""Fine-tune Llama-3.1-8B on individual capability subsets of the Tulu 3 SFT mixture.

Each capability is trained separately to produce a specialist checkpoint,
which can later be merged using scripts/nlg/merge.py.

Uses allenai/tulu-3-sft-mixture (939k examples with 'messages' + 'source' columns),
filtered by source to isolate each capability.

Usage:
    python scripts/nlg/finetune.py --capability math --output-dir checkpoints/nlg
    python scripts/nlg/finetune.py --capability all --output-dir checkpoints/nlg
    python scripts/nlg/finetune.py --capability coding --use-lora --hf-cache-dir $SCRATCH/huggingface
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

PRETRAINED_MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASET_ID = "allenai/tulu-3-sft-mixture"
MAX_SEQ_LEN = 4096

# Capability -> list of source values in the 'source' column
CAPABILITY_SOURCES = {
    "math": [
        "allenai/tulu-3-personas-math",
        "allenai/tulu-3-sft-personas-math-grade",
        "allenai/tulu-3-personas-algebra",
        "AI-MO/NuminaMath-TIR",
    ],
    "coding": [
        "allenai/tulu-3-sft-personas-code",
        "theblackcat102/evol-codealpaca-v1",
    ],
    "general": [
        "allenai/WildChat-1M",
        "ai2-adapt-dev/oasst1_converted",
        "HuggingFaceH4/no_robots",
    ],
    "knowledge": [
        "ai2-adapt-dev/flan_v2_converted",
        "allenai/SciRIFF",
        "LipengCS/Table-GPT",
    ],
    "precise_if": [
        "allenai/tulu-3-sft-personas-instruction-following",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Llama-3.1-8B on Tulu 3 capability subsets"
    )
    p.add_argument(
        "--capability",
        type=str,
        required=True,
        choices=list(CAPABILITY_SOURCES.keys()) + ["all"],
    )
    p.add_argument("--output-dir", type=str, default="checkpoints/nlg")
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
    return p.parse_args()


def train_capability(capability, args):
    print(f"\n{'='*60}")
    print(f"Training capability: {capability}")
    print(f"{'='*60}")

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir

    # Precise IF uses higher LR (1e-5) per paper
    lr = 1e-5 if capability == "precise_if" else args.lr

    # Load and filter dataset
    sources = CAPABILITY_SOURCES[capability]
    print(f"  Sources: {sources}")
    ds = load_dataset(DATASET_ID, split="train", cache_dir=args.hf_cache_dir)
    ds = ds.filter(lambda ex: ex["source"] in sources, desc=f"Filtering {capability}")
    ds = ds.remove_columns([c for c in ds.column_names if c != "messages"])
    print(f"  {len(ds)} examples")

    # Load tokenizer from Instruct variant (has chat template), model from base
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", cache_dir=args.hf_cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        cache_dir=args.hf_cache_dir,
    )

    run_dir = os.path.join(args.output_dir, f"Llama-3.1-8B-{capability}")

    sft_config = SFTConfig(
        output_dir=run_dir,
        max_length=MAX_SEQ_LEN,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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

    trainer.train()

    print(f"Saving to {run_dir} ...")
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)
    print(f"Done: {capability}")


def main():
    args = parse_args()

    if args.capability == "all":
        for cap in CAPABILITY_SOURCES:
            train_capability(cap, args)
    else:
        train_capability(args.capability, args)


if __name__ == "__main__":
    main()
