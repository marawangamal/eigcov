#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _safe_file_stem(param_name: str) -> str:
    digest = hashlib.md5(param_name.encode("utf-8")).hexdigest()[:8]
    return param_name.replace("/", "__").replace(".", "__") + f"__{digest}"


def _parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


# MODELS = [
#     "pmahdavi/Llama-3.1-8B-math-reasoning",
#     "pmahdavi/Llama-3.1-8B-coding",
#     "pmahdavi/Llama-3.1-8B-coding-tulu3-ebs128-lr5e6-wsdcr0p4",
#     "pmahdavi/Llama-3.1-8B-precise-if",
#     "pmahdavi/Llama-3.1-8B-general",
#     "pmahdavi/Llama-3.1-8B-knowledge-recall",
# ]

# python scripts/nlg/save_model_param_folder.py --model pmahdavi/Llama-3.1-8B-coding --output-dir checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding
# python scripts/nlg/save_model_param_folder.py --model pmahdavi/Llama-3.1-8B-coding-tulu3-ebs128-lr5e6-wsdcr0p4 --output-dir checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding-tulu3-ebs128-lr5e6-wsdcr0p4
# python scripts/nlg/save_model_param_folder.py --model pmahdavi/Llama-3.1-8B-precise-if --output-dir checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if
# python scripts/nlg/save_model_param_folder.py --model pmahdavi/Llama-3.1-8B-general --output-dir checkpoints/nlg/pmahdavi-Llama-3.1-8B-general
# python scripts/nlg/save_model_param_folder.py --model pmahdavi/Llama-3.1-8B-knowledge-recall --output-dir checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall
# python scripts/nlg/save_model_param_folder.py --model meta-llama/Meta-Llama-3.1-8B --output-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument(
        "--output-dir", required=True, help="Output param_folder directory"
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for loading model before export",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    params_dir = output_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=_parse_dtype(args.dtype),
        device_map="cpu",
    )

    use_safetensors = False
    try:
        from safetensors.torch import save_file as save_safetensors_file

        use_safetensors = True
    except Exception:
        save_safetensors_file = None

    manifest = {
        "format": "safetensors" if use_safetensors else "pt",
        "model_id": args.model,
        "params": {},
    }

    with torch.no_grad():
        for param_name, tensor in model.state_dict().items():
            tensor_cpu = tensor.detach().cpu().contiguous()
            stem = _safe_file_stem(param_name)
            if use_safetensors:
                filename = f"{stem}.safetensors"
                save_safetensors_file(
                    {"tensor": tensor_cpu}, str(params_dir / filename)
                )
            else:
                filename = f"{stem}.pt"
                torch.save(tensor_cpu, params_dir / filename)
            manifest["params"][param_name] = {
                "file": filename,
                "shape": list(tensor_cpu.shape),
                "dtype": str(tensor_cpu.dtype),
            }

    manifest_path = output_dir / "param_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved param_folder to {output_dir}")


if __name__ == "__main__":
    main()
