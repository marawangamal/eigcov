#!/usr/bin/env python3
"""Convert an olmes param-folder model into a standard HF checkpoint directory."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from oe_eval.utilities.param_folder_utils import load_hf_causal_lm_from_param_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert param-folder model to standard HF checkpoint format."
    )
    parser.add_argument(
        "--param-folder",
        required=True,
        help="Input param-folder directory (contains param_manifest.json and params/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for standard HF checkpoint files.",
    )
    parser.add_argument(
        "--tokenizer-source-dir",
        default=None,
        help=(
            "Optional tokenizer source directory. If set, tokenizer files (including "
            "chat_template) are copied from here instead of the param-folder tokenizer."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for config/tokenizer/model loading.",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save as model.safetensors (default). If omitted, saves .bin format.",
        default=True,
    )
    parser.add_argument(
        "--no-safe-serialization",
        dest="safe_serialization",
        action="store_false",
        help="Disable safetensors and save PyTorch .bin checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    param_folder = Path(args.param_folder).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer_dir = load_hf_causal_lm_from_param_folder(
        param_folder=str(param_folder),
        trust_remote_code=args.trust_remote_code,
    )
    model.save_pretrained(str(output_dir), safe_serialization=args.safe_serialization)

    tokenizer_source = (
        str(Path(args.tokenizer_source_dir).expanduser().resolve())
        if args.tokenizer_source_dir
        else tokenizer_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(str(output_dir))

    print(f"Converted param-folder model to HF checkpoint: {output_dir}")
    print(f"Tokenizer source: {tokenizer_source}")


if __name__ == "__main__":
    main()
