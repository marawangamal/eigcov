#!/usr/bin/env python3
"""Compute per-layer eigcov coefficient norms from param-folder checkpoints.

Example:
    python scripts/nlg/get_param_norms.py \
        --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
        --finetuned-dirs \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-math-reasoning \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding-tulu3-ebs128-lr5e6-wsdcr0p4 \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-general \
            checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall \
        --output results/nlg_coeffs.csv \
        --max-layers 10
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def _load_manifest(folder: Path) -> dict:
    manifest_path = folder / "param_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("params")
    if not isinstance(params, dict) or len(params) == 0:
        raise ValueError(f"Manifest at {manifest_path} has empty or invalid 'params'.")
    return manifest


def _load_single_tensor(file_path: Path) -> torch.Tensor:
    if file_path.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors_file

        tensor_map = load_safetensors_file(str(file_path))
        if len(tensor_map) != 1:
            raise ValueError(
                f"Expected one tensor in {file_path}, found {len(tensor_map)}."
            )
        return next(iter(tensor_map.values()))
    return torch.load(file_path, map_location="cpu")


def _build_param_file_path(model_dir: Path, manifest: dict, param_name: str) -> Path:
    filename = manifest["params"][param_name]["file"]
    return model_dir / "params" / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-layer eigcov coefficient norms from param folders."
    )
    parser.add_argument(
        "--pretrained-dir",
        required=True,
        help="Param folder for the pretrained base model.",
    )
    parser.add_argument(
        "--finetuned-dirs",
        nargs="+",
        required=True,
        help="Param folders for finetuned models.",
    )
    parser.add_argument(
        "--output",
        default="results/nlg_coeffs.csv",
        help="Output CSV path (default: results/nlg_coeffs.csv).",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=10,
        help="Max number of 2D layers to process (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.pretrained_dir).expanduser().resolve()
    ft_dirs = [Path(p).expanduser().resolve() for p in args.finetuned_dirs]

    base_manifest = _load_manifest(base_dir)
    ft_manifests = [_load_manifest(d) for d in ft_dirs]

    keys = list(base_manifest["params"].keys())
    model_ids = [d.name for d in ft_dirs]

    num_layers = 0
    rows = []
    for key in tqdm(keys, desc="Processing params"):
        base_tensor = _load_single_tensor(
            _build_param_file_path(base_dir, base_manifest, key)
        )
        if base_tensor.ndim != 2:
            continue

        print(f"Processing {key}...")
        deltas = []
        for ft_dir, ft_manifest in zip(ft_dirs, ft_manifests):
            ft_tensor = _load_single_tensor(
                _build_param_file_path(ft_dir, ft_manifest, key)
            )
            deltas.append((ft_tensor.float() - base_tensor.float()))

        taus = torch.stack(deltas)  # (N, Do, Di)
        c = taus.transpose(1, 2) @ taus
        coeffs = c @ torch.linalg.pinv(c.sum(dim=0))  # (N, Di, Di)
        norms_taus = torch.linalg.norm(taus, ord="fro", dim=(-2, -1))
        norms_cs = torch.linalg.norm(c, ord="fro", dim=(-2, -1))
        norms_c_sum = torch.linalg.norm(c.sum(dim=0), ord="fro")
        norms_inv_c_sum = torch.linalg.norm(torch.linalg.pinv(c.sum(dim=0)), ord="fro")
        norms_coeffs = torch.linalg.norm(coeffs, ord="fro", dim=(-2, -1))

        for i, mid in enumerate(model_ids):
            rows.append(
                {
                    "model_id": mid,
                    "norm_taus": norms_taus[i].item(),
                    "norm_coeffs": norms_coeffs[i].item(),
                    "norm_cs": norms_cs[i].item(),
                    "norm_c_sum": norms_c_sum.item(),
                    "norm_inv_c_sum": norms_inv_c_sum.item(),
                    "layer_idx": num_layers,
                    "param_name": key,
                }
            )

        num_layers += 1
        if num_layers >= args.max_layers:
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
