"""Merge Llama-3.1-8B HuggingFace task vectors into a single checkpoint.

Usage:
    python scripts/nlg/merge.py --merge-func mean --output-dir ~/scratch/merged_model
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.merging import combine_task_vectors
from src.task_vectors import _TaskVector

PRETRAINED_MODEL = "meta-llama/Meta-Llama-3.1-8B"

MODELS = [
    "pmahdavi/Llama-3.1-8B-math-reasoning",
    "pmahdavi/Llama-3.1-8B-coding",
    "pmahdavi/Llama-3.1-8B-coding-tulu3-ebs128-lr5e6-wsdcr0p4",
    "pmahdavi/Llama-3.1-8B-precise-if",
    "pmahdavi/Llama-3.1-8B-general",
    "pmahdavi/Llama-3.1-8B-knowledge-recall",
]


class HFTaskVector(_TaskVector):
    """Task vector built from two HuggingFace model checkpoints."""

    def _load_checkpoint(self, checkpoint):
        return AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

    def _cast_to_same_type(self, other):
        if isinstance(other, HFTaskVector):
            return other
        raise TypeError(f"Cannot cast {type(other)} to HFTaskVector")

    def apply_to_hf(self, pretrained_model_id: str, scaling_coef: float = 1.0):
        """Apply the task vector to a pretrained HF model and return the model."""
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        sd = model.state_dict()
        with torch.no_grad():
            for key, delta in self.vector.items():
                if key in sd:
                    sd[key] = sd[key] + scaling_coef * delta
                else:
                    print(f"Warning: key {key} not found in pretrained model")
        model.load_state_dict(sd)
        return model


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Llama-3.1-8B task vectors")
    parser.add_argument(
        "--merge-func",
        type=str,
        default="mean",
        help="Merge function (sum, mean, eigcov, regmean, ...)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the merged HF model checkpoint",
    )
    parser.add_argument(
        "--scaling-coef",
        type=float,
        default=1.0,
        help="Scaling coefficient applied to the merged task vector",
    )
    parser.add_argument(
        "--hf-cache-dir", type=str, default=None, help="HuggingFace cache directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir

    print(f"Pretrained model : {PRETRAINED_MODEL}")
    print(f"Merge function   : {args.merge_func}")
    print(f"Output dir       : {args.output_dir}")
    print("=" * 80)

    task_vectors = []
    for model_id in MODELS:
        print(f"Loading task vector: {model_id}")
        tv = HFTaskVector(
            pretrained_checkpoint=PRETRAINED_MODEL,
            finetuned_checkpoint=model_id,
        )
        task_vectors.append(tv)
        print(f"  Loaded ({len(tv.vector)} params)")

    print("=" * 80)
    print(f"Merging {len(task_vectors)} task vectors with '{args.merge_func}' ...")
    merged_tv = combine_task_vectors(task_vectors, args.merge_func)

    print("Applying merged task vector to pretrained model ...")
    merged_model = merged_tv.apply_to_hf(
        PRETRAINED_MODEL, scaling_coef=args.scaling_coef
    )

    print(f"Saving merged model to {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL, cache_dir=args.hf_cache_dir
    )
    tokenizer.save_pretrained(args.output_dir)

    print("Done.")
    print(
        f"\nTo evaluate with olmes:\n"
        f"  olmes --model {args.output_dir} --task tulu_3_dev --output-dir ~/scratch/results"
    )


if __name__ == "__main__":
    main()
