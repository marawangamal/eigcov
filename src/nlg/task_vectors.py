"""TaskVector subclass that reads from param-folder checkpoints.

Loads only individual parameter files on demand, keeping memory bounded to
O(single_param × num_models) instead of O(full_model × num_models).
"""

import json
from pathlib import Path

import torch

from src.task_vectors import _TaskVector


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


class ParamFolderTaskVector(_TaskVector):
    """Task vector that lazily loads individual params from param-folder checkpoints.

    Each checkpoint is a directory with a ``param_manifest.json`` and a ``params/``
    subdirectory containing one file per parameter.  Only the two files needed for a
    given key are loaded at a time (pretrained + finetuned), so peak memory is bounded
    to a single parameter times the number of task vectors being merged.
    """

    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, **kwargs):
        if vector is not None:
            # Constructed from a pre-computed vector dict (e.g. by combine_task_vectors).
            self._pre_manifest = None
            self._ft_manifest = None
            super().__init__(vector=vector, **kwargs)
        else:
            self._pre_manifest = _load_manifest(Path(pretrained_checkpoint))
            self._ft_manifest = _load_manifest(Path(finetuned_checkpoint))
            # Always lazy — that's the whole point of this subclass.
            super().__init__(
                pretrained_checkpoint=str(pretrained_checkpoint),
                finetuned_checkpoint=str(finetuned_checkpoint),
                lazy=True,
                **kwargs,
            )

    def _load_checkpoint(self, checkpoint):
        """Full-model load fallback (used by _build_vector if called)."""
        manifest = _load_manifest(Path(checkpoint))
        sd = {}
        for key in manifest["params"]:
            sd[key] = _load_single_tensor(
                _build_param_file_path(Path(checkpoint), manifest, key)
            )
        return sd

    def lazy_keys(self):
        if self._lazy_keys is not None:
            return self._lazy_keys
        # Use the pretrained manifest; filter out non-float dtypes.
        self._lazy_keys = [
            k
            for k, v in self._pre_manifest["params"].items()
            if "int" not in v.get("dtype", "")
        ]
        return self._lazy_keys

    def get_vector_element(self, key):
        if not self.lazy:
            return self.vector[key]

        if key in self._cache:
            return self._cache[key]

        # Load only the two param files needed, compute their diff.
        pre_dir = Path(self._pretrained_checkpoint)
        ft_dir = Path(self._finetuned_checkpoint)
        pre_t = _load_single_tensor(
            _build_param_file_path(pre_dir, self._pre_manifest, key)
        )
        ft_t = _load_single_tensor(
            _build_param_file_path(ft_dir, self._ft_manifest, key)
        )
        delta = ft_t.float() - pre_t.float()
        # Single-element cache — previous entry is released.
        self._cache = {key: delta}
        return delta

    def _cast_to_same_type(self, other):
        if isinstance(other, ParamFolderTaskVector):
            return other
        raise TypeError(f"Cannot cast {type(other)} to ParamFolderTaskVector")
