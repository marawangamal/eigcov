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
MODEL_PREFIX = "pmahdavi-llama-3.1-8B"

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
        "--hf-cache-dir", type=str, default=None, help="HuggingFace cache directory"
    )
    return parser.parse_args()


def _load_state_dict(model_id, cache_dir=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, low_cpu_mem_usage=True, cache_dir=cache_dir
    )
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    return sd


def main():
    args = parse_args()
    # HACK:
    args.output_dir = "checkpoints"

    # Disable xet protocol which causes download errors on some clusters
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir

    print(f"Pretrained model : {PRETRAINED_MODEL}")
    print(f"Merge function   : {args.merge_func}")
    print(f"Output dir       : {args.output_dir}")
    print("=" * 80)

    print("Loading pretrained model ...")
    pretrained_sd = _load_state_dict(PRETRAINED_MODEL, cache_dir=args.hf_cache_dir)
    print(f"  Pretrained loaded ({len(pretrained_sd)} params)")

    task_vectors = []
    for model_id in MODELS:
        print(f"Loading finetuned model: {model_id}")
        finetuned_sd = _load_state_dict(model_id, cache_dir=args.hf_cache_dir)
        vector = {
            k: finetuned_sd[k] - pretrained_sd[k]
            for k in pretrained_sd
            if k in finetuned_sd
            and pretrained_sd[k].dtype not in (torch.int64, torch.uint8)
        }
        task_vectors.append(HFTaskVector(vector=vector))
        print(f"  Task vector built ({len(vector)} params)")

    print("=" * 80)
    print(f"Merging {len(task_vectors)} task vectors with '{args.merge_func}' ...")
    merged_tv = combine_task_vectors(task_vectors, args.merge_func, args)

    print("Applying merged task vector to pretrained model ...")
    merged_sd = {
        k: pretrained_sd[k] + merged_tv.vector[k]
        for k in pretrained_sd
        if k in merged_tv.vector
    }
    pretrained_sd.update(merged_sd)

    print("Loading pretrained model for saving ...")
    merged_model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=args.hf_cache_dir,
    )
    merged_model.load_state_dict(pretrained_sd)

    save_dir = os.path.join(args.output_dir, f"{MODEL_PREFIX}-{args.merge_func}")
    print(f"Saving merged model to {save_dir} ...")
    os.makedirs(save_dir, exist_ok=True)
    merged_model.save_pretrained(save_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL, cache_dir=args.hf_cache_dir
    )
    tokenizer.save_pretrained(save_dir)

    print("Done.")
    print(
        f"\nTo evaluate with olmes:\n"
        f"  olmes --model {save_dir} --task tulu_3_dev --output-dir ~/scratch/results"
    )


if __name__ == "__main__":
    main()
