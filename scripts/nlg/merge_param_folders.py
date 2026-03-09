#!/usr/bin/env python3
"""Merge param-folder models block-by-block (average / tsv / isoc).

Input folders must follow the olmes param-folder format:
- config + tokenizer files in folder root
- per-parameter tensor files in folder/params/
- folder/param_manifest.json mapping param names -> file metadata
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.merging import merge_eigcov, merge_isoc_mean, merge_tsv


def _safe_file_stem(param_name: str) -> str:
    digest = hashlib.md5(param_name.encode("utf-8")).hexdigest()[:8]
    return param_name.replace("/", "__").replace(".", "__") + f"__{digest}"


def _load_manifest(folder: Path) -> Dict:
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
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except Exception as exc:
            raise RuntimeError(
                f"safetensors file found ({file_path}) but safetensors is not installed."
            ) from exc
        tensor_map = load_safetensors_file(str(file_path))
        if len(tensor_map) != 1:
            raise ValueError(f"Expected one tensor in {file_path}, found {len(tensor_map)}.")
        return next(iter(tensor_map.values()))
    return torch.load(file_path, map_location="cpu")


def _save_single_tensor(file_path: Path, tensor: torch.Tensor, use_safetensors: bool) -> None:
    tensor = tensor.detach().cpu().contiguous()
    if use_safetensors:
        from safetensors.torch import save_file as save_safetensors_file

        save_safetensors_file({"tensor": tensor}, str(file_path))
    else:
        torch.save(tensor, file_path)


def _atomic_write_json(path: Path, payload: Dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _copy_non_param_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.name in {"params", "param_manifest.json"}:
            continue
        dst_item = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)


def _merge_delta(deltas: torch.Tensor, method: str) -> torch.Tensor:
    if method == "tsv":
        if deltas[0].ndim == 2:
            return merge_tsv(deltas)
        return deltas.mean(dim=0)
    if method == "isoc":
        if deltas[0].ndim == 2:
            return merge_isoc_mean(deltas)
        return deltas.mean(dim=0)
    if method == "eigcov":
        if deltas[0].ndim == 2:
            return merge_eigcov(deltas)
        return deltas.mean(dim=0)
    raise ValueError(f"Unsupported method: {method}")


def _validate_keys(base_manifest: Dict, ft_manifests: List[Dict]) -> List[str]:
    keys = list(base_manifest["params"].keys())
    base_set = set(keys)
    for i, manifest in enumerate(ft_manifests):
        current = set(manifest["params"].keys())
        if current != base_set:
            diff = sorted(base_set.symmetric_difference(current))
            preview = ", ".join(diff[:10])
            raise ValueError(
                f"Finetuned manifest #{i} keys differ from base. First diffs: {preview}"
            )
    return keys


def _build_param_file_path(model_dir: Path, manifest: Dict, param_name: str) -> Path:
    filename = manifest["params"][param_name]["file"]
    return model_dir / "params" / filename


def merge_one_method(
    method: str,
    base_dir: Path,
    base_manifest: Dict,
    ft_dirs: List[Path],
    ft_manifests: List[Dict],
    out_dir: Path,
    use_safetensors: bool,
    fp32_merge: bool,
) -> None:
    print(f"\n=== Merging method: {method} ===")
    _copy_non_param_files(base_dir, out_dir)
    out_params_dir = out_dir / "params"
    out_params_dir.mkdir(parents=True, exist_ok=True)

    keys = _validate_keys(base_manifest, ft_manifests)
    manifest_path = out_dir / "param_manifest.json"
    out_manifest = {
        "format": "safetensors" if use_safetensors else "pt",
        "model_id": f"merged::{method}",
        "params": {},
    }
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(prior.get("params"), dict):
            out_manifest["params"] = prior["params"]
        if prior.get("format") and prior["format"] != out_manifest["format"]:
            raise ValueError(
                f"Existing manifest format '{prior['format']}' conflicts with current "
                f"save format '{out_manifest['format']}' in {manifest_path}."
            )

    ext = ".safetensors" if use_safetensors else ".pt"
    already_done = 0
    for key in keys:
        stem = _safe_file_stem(key)
        filename = f"{stem}{ext}"
        file_path = out_params_dir / filename
        if key in out_manifest["params"] and file_path.exists():
            already_done += 1

    if already_done:
        print(f"[{method}] resuming: found {already_done}/{len(keys)} parameters already merged")

    for idx, key in enumerate(
        tqdm(keys, desc=f"Merging params ({method})", unit="param"), start=1
    ):
        stem = _safe_file_stem(key)
        filename = f"{stem}{ext}"
        file_path = out_params_dir / filename
        if key in out_manifest["params"] and file_path.exists():
            continue

        base_tensor = _load_single_tensor(_build_param_file_path(base_dir, base_manifest, key))
        ft_tensors = [
            _load_single_tensor(_build_param_file_path(ft_dir, ft_manifest, key))
            for ft_dir, ft_manifest in zip(ft_dirs, ft_manifests)
        ]

        if any(t.shape != base_tensor.shape for t in ft_tensors):
            raise ValueError(f"Shape mismatch for parameter '{key}'.")

        if not torch.is_floating_point(base_tensor):
            merged_tensor = base_tensor
        else:
            merge_dtype = torch.float32 if fp32_merge else base_tensor.dtype
            ft_stack = torch.stack([t.to(merge_dtype) for t in ft_tensors], dim=0)

            if method == "average":
                # Explicitly average original finetuned model parameters.
                merged_tensor = ft_stack.mean(dim=0).to(base_tensor.dtype)
            else:
                base_for_merge = base_tensor.to(merge_dtype)
                deltas = ft_stack - base_for_merge
                merged_delta = _merge_delta(deltas, method=method)
                merged_tensor = (base_for_merge + merged_delta).to(base_tensor.dtype)

        _save_single_tensor(file_path, merged_tensor, use_safetensors)
        out_manifest["params"][key] = {
            "file": filename,
            "shape": list(merged_tensor.shape),
            "dtype": str(merged_tensor.dtype),
        }
        # Persist progress after every parameter so we can resume after interruptions.
        _atomic_write_json(manifest_path, out_manifest)

        if idx % 1000 == 0 or idx == len(keys):
            print(f"[{method}] merged {idx}/{len(keys)} parameters")

    _atomic_write_json(manifest_path, out_manifest)
    print(f"[{method}] wrote merged param folder to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge param-folder models block-by-block (average, tsv, isoc, eigcov)."
    )
    parser.add_argument(
        "--pretrained-dir",
        required=True,
        help="Base pretrained param-folder directory.",
    )
    parser.add_argument(
        "--finetuned-dirs",
        nargs="+",
        required=True,
        help="Finetuned param-folder directories to merge.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["average", "tsv", "isoc"],
        choices=["average", "tsv", "isoc", "eigcov"],
        help="Merge methods to run.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory where merged param-folder dirs will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        default="merged",
        help="Output directory prefix, e.g. merged-average, merged-tsv, merged-isoc, merged-eigcov.",
    )
    parser.add_argument(
        "--save-format",
        default="auto",
        choices=["auto", "safetensors", "pt"],
        help="Tensor file format for merged params.",
    )
    parser.add_argument(
        "--no-fp32-merge",
        action="store_true",
        help="Disable fp32 math during merging (uses parameter dtype directly).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.pretrained_dir).expanduser().resolve()
    ft_dirs = [Path(p).expanduser().resolve() for p in args.finetuned_dirs]
    out_root = Path(args.output_root).expanduser().resolve()

    base_manifest = _load_manifest(base_dir)
    ft_manifests = [_load_manifest(d) for d in ft_dirs]

    if args.save_format == "pt":
        use_safetensors = False
    elif args.save_format == "safetensors":
        use_safetensors = True
    else:
        try:
            import safetensors  # noqa: F401

            use_safetensors = True
        except Exception:
            use_safetensors = False

    out_root.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        out_dir = out_root / f"{args.output_prefix}-{method}"
        merge_one_method(
            method=method,
            base_dir=base_dir,
            base_manifest=base_manifest,
            ft_dirs=ft_dirs,
            ft_manifests=ft_manifests,
            out_dir=out_dir,
            use_safetensors=use_safetensors,
            fp32_merge=not args.no_fp32_merge,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
