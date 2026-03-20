# Create task vectors

import torch
import sys
import os
import os.path as osp
import pandas as pd
from transformers import AutoModelForCausalLM
from tqdm import tqdm

sys.path.append("..")
from src.vision.task_vectors import _TaskVector


class HFTaskVector(_TaskVector):
    """Task vector built from two HuggingFace model checkpoints."""

    def _load_checkpoint(self, checkpoint):
        return torch.load(checkpoint, map_location="cpu", weights_only=False)

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


RESULTS_DIR = "results"
PRETRAINED_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MODEL_PREFIX = "pmahdavi-llama-3.1-8B"
CHECKPOINTS_DIR = osp.join(os.environ["SCRATCH"], "eigcov", "checkpoints")
HF_CACHE_DIR = osp.join(os.environ["SCRATCH"], "huggingface")
MODELS = [
    "pmahdavi/Llama-3.1-8B-math-reasoning",
    "pmahdavi/Llama-3.1-8B-coding",
    "pmahdavi/Llama-3.1-8B-coding-tulu3-ebs128-lr5e6-wsdcr0p4",
    "pmahdavi/Llama-3.1-8B-precise-if",
    "pmahdavi/Llama-3.1-8B-general",
    "pmahdavi/Llama-3.1-8B-knowledge-recall",
]


tv_dicts = []
pretrained_checkpoint = osp.join(
    CHECKPOINTS_DIR, f"{PRETRAINED_MODEL.replace('/', '-')}.pt"
)
for model_id in MODELS:
    finetuned_checkpoint = osp.join(CHECKPOINTS_DIR, f"{model_id.replace('/', '-')}.pt")
    tv_dicts.append(
        {
            "model_id": model_id,
            "task_vector": HFTaskVector(
                pretrained_checkpoint=pretrained_checkpoint,
                finetuned_checkpoint=finetuned_checkpoint,
                lazy=True,
                cache_window=10,
            ),
        }
    )

print(f"Loading keys...")
keys = tv_dicts[0]["task_vector"].lazy_keys()
num_layers = 0
max_layers = 10
rows = []
for k in keys:
    tens = tv_dicts[0]["task_vector"].get_vector_element(k)
    if tens.ndim != 2:
        print(f"Skipping {k} (not a matrix)")
        continue
    print(f"Processing {k}...")
    taus = torch.stack(
        [tv_dict["task_vector"].get_vector_element(k) for tv_dict in tv_dicts]
    )  # (N, Do, Di)
    c = taus.transpose(1, 2) @ taus
    coeff = c @ torch.linalg.pinv(c.sum(dim=0))  # (T, Di, Di)
    norms = torch.linalg.norm(coeff.float(), ord="fro", dim=(-2, -1))
    for i, tv_dict in enumerate(tv_dicts):
        rows.append({"model_id": tv_dict["model_id"], "norm": norms[i]})

    num_layers += 1
    if num_layers >= max_layers:
        break

out_path = os.path.join(RESULTS_DIR, "nlg_coeffs.csv")
df = pd.DataFrame(rows)
df.to_csv(out_path, index=False)
print(f"Saved {out_path}")

# RESULTS_DIR = os.path.join("results", "task_vector_deltas")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# for tv_dict in tqdm(tv_dicts, total=len(tv_dicts)):
#     model_id = tv_dict["model_id"]
#     task_vector = tv_dict["task_vector"]
#     out_path = os.path.join(RESULTS_DIR, f"{model_id.replace('/', '-')}.csv")
#     if os.path.exists(out_path):
#         print(f"Skipping {model_id} (already exists)")
#         continue
#     keys = task_vector.lazy_keys()
#     rows = []
#     print(f"Fetching keys for {model_id}...")
#     for i, k in tqdm(enumerate(keys), total=len(keys), leave=False):
#         tens = task_vector.get_vector_element(k)
#         if tens.ndim != 2:
#             continue

#         w = (tens.T @ tens)(tens.T @ tens)

#         rows.append(
#             {
#                 "model_id": model_id,
#                 "layer_idx": i,
#                 "norm": torch.linalg.norm(tens.float(), ord=2),
#             }
#         )
#     df = pd.DataFrame(rows)
#     df.to_csv(out_path, index=False)
#     print(f"Saved {out_path}")
