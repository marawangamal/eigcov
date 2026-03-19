# Create task vectors

import torch
import sys
import os
import os.path as osp
from tqdm import tqdm

sys.path.append("..")
from src.vision.task_vectors import _TaskVector

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


tv_dicts = []
pretrained_checkpoint = osp.join(
    CHECKPOINTS_DIR, f"{PRETRAINED_MODEL.replace('/', '-')}.pt"
)
for model_id in [PRETRAINED_MODEL] + MODELS:
    finetuned_checkpoint = osp.join(CHECKPOINTS_DIR, f"{model_id.replace('/', '-')}.pt")
    tv_dicts.append(
        {
            "model_id": model_id,
            "task_vector": HFTaskVector(
                pretrained_checkpoint=pretrained_checkpoint,
                finetuned_checkpoint=finetuned_checkpoint,
                lazy=True,
                cache_window=5,
            ),
        }
    )


rows = []
for tv_dict in tqdm(tv_dicts, total=len(tv_dicts)):
    model_id = tv_dict["model_id"]
    task_vector = tv_dict["task_vector"]
    # task_vector.lazy_keys()
    print(f"Fetching keys for {model_id}...")
    for i, k in tqdm(
        enumerate(task_vector.lazy_keys()), total=len(task_vector.vector), leave=False
    ):
        print(f"[{i}/{len(task_vector.vector)}] {k}")
        tens = task_vector.get_vector_element(k)
        print(f"[{i}/{len(task_vector.vector)}] {k} {tens.shape}")
        if tens.ndim != 2:
            continue
        rows.append(
            {
                "model_id": model_id,
                "layer_idx": i,
                "norm": torch.linalg.norm(tens.float(), ord=2),
            }
        )
        # print(f"[{i}/{len(task_vector.vector)}] {k} {tens.shape}")


import pandas as pd

df = pd.DataFrame(rows)
df.to_csv(os.path.join("results", "task_vector_norms.csv"), index=False)
