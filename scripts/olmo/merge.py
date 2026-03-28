"""Merge param-folder task vectors into a HuggingFace checkpoint.

Uses ParamFolderTaskVector for lazy, memory-bounded merging — only one
parameter is loaded per model at a time.

Usage:
    python scripts/nlg/merge.py \
        --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
        --finetuned-dirs \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-math-reasoning \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-general \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall \
        --merge-func eigcov \
        --output-dir checkpoints/nlg/merged-eigcov
        --ignore-keys gate_proj up_proj
"""

import argparse
import json
import os
from pathlib import Path

import torch

from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.merging import combine_task_vectors
from src.nlg.task_vectors import (
    ParamFolderTaskVector,
    _build_param_file_path,
    _load_manifest,
    _load_single_tensor,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge param-folder task vectors into a HuggingFace checkpoint."
    )
    parser.add_argument(
        "--pretrained-dir",
        required=True,
        help="Param-folder directory for the pretrained base model.",
    )
    parser.add_argument(
        "--finetuned-dirs",
        nargs="+",
        required=True,
        help="Param-folder directories for finetuned models.",
    )
    parser.add_argument(
        "--merge-func",
        type=str,
        default="mean",
        help="Merge function name (mean, eigcov, tsv, isoc, etc.).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the merged HuggingFace model.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=None,
        help="Directory to copy the tokenizer from. Defaults to the first "
        "finetuned dir (which typically has a chat_template).",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help="HuggingFace model ID for loading the pretrained model architecture "
        "(e.g. meta-llama/Meta-Llama-3.1-8B). If not provided, reads from "
        "config.json in pretrained-dir.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory.",
    )
    parser.add_argument(
        "--ignore-keys",
        nargs="+",
        default=None,
        help="Substrings of parameter keys to exclude from merging (averaged instead). "
        "E.g. --ignore-keys embed lm_head",
    )
    parser.add_argument(
        "--merge-kwargs",
        type=json.loads,
        default={},
        help="JSON dict of extra kwargs for the merge function. "
        'E.g. \'{"mix_primary": "eigcov_general", "mix_fallback": "eigcov", '
        '"mix_targets": ["gate_proj"], "lam": 0.1}\'',
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True to HF model/tokenizer loading "
        "(required for Qwen-1 and other custom-code models).",
    )
    return parser.parse_args()


def _read_hf_model_id(pretrained_dir: Path) -> str:
    """Read the original HF model ID from a param-folder checkpoint."""
    import json

    # Primary: read from param_manifest.json (set by save_model_param_folder.py)
    manifest_path = pretrained_dir / "param_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_id = manifest.get("model_id")
        if model_id:
            return model_id
    # Fallback: config.json _name_or_path
    config_path = pretrained_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_id = config.get("_name_or_path")
        if model_id:
            return model_id
    raise ValueError(
        f"Could not determine HF model ID from {pretrained_dir}. "
        "Pass --hf-model-id explicitly."
    )


def main():
    args = parse_args()

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir

    pretrained_dir = Path(args.pretrained_dir).expanduser().resolve()
    finetuned_dirs = [Path(p).expanduser().resolve() for p in args.finetuned_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()

    hf_model_id = args.hf_model_id or _read_hf_model_id(pretrained_dir)

    print(f"Pretrained dir : {pretrained_dir}")
    print(f"Finetuned dirs : {len(finetuned_dirs)} models")
    print(f"Merge function : {args.merge_func}")
    print(f"HF model ID    : {hf_model_id}")
    print(f"Output dir     : {output_dir}")
    print("=" * 80)

    # Build lazy task vectors — no full model loading happens here.
    task_vectors = []
    for ft_dir in finetuned_dirs:
        print(f"  Creating task vector: {ft_dir.name}")
        tv = ParamFolderTaskVector(
            pretrained_checkpoint=pretrained_dir,
            finetuned_checkpoint=ft_dir,
        )
        task_vectors.append(tv)

    print(f"\nMerging {len(task_vectors)} task vectors with '{args.merge_func}' ...")
    merged_tv = combine_task_vectors(
        task_vectors,
        args.merge_func,
        ignore_keys=args.ignore_keys,
        **args.merge_kwargs,
    )

    # Build final state dict: pretrained (from param-folder) + merged deltas.
    # Pop deltas as we go so memory stays ~32GB (deltas shrink, final weights grow).
    pre_manifest = _load_manifest(pretrained_dir)
    merged_vector = merged_tv.vector  # dict reference
    del merged_tv
    final_sd = {}
    for key in tqdm(list(merged_vector.keys()), desc="Applying deltas"):
        delta = merged_vector.pop(key)
        pre_t = _load_single_tensor(
            _build_param_file_path(pretrained_dir, pre_manifest, key)
        )
        final_sd[key] = (pre_t.float() + delta).to(pre_t.dtype)

    # Save as HF checkpoint: create model on meta device, load state dict, save.
    trc = args.trust_remote_code
    # When trust_remote_code is set, load config from HF Hub (has custom .py files)
    # rather than the local param-folder dir (which only has config.json).
    config_source = hf_model_id if trc else str(pretrained_dir)
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=trc)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trc)
    model.load_state_dict(final_sd, assign=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_dir} ...")
    model.save_pretrained(output_dir)

    # Copy tokenizer — default to first finetuned dir (has chat_template).
    tokenizer_dir = (
        Path(args.tokenizer_dir) if args.tokenizer_dir else finetuned_dirs[0]
    )
    print(f"Copying tokenizer from {tokenizer_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=trc)
    tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
