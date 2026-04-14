import abc
import os

import torch


class _TaskVector(abc.ABC):
    PRETRAINED_FILENAME = "zeroshot.pt"
    FINETUNED_FILENAME = "finetuned.pt"

    def __init__(
        self,
        checkpoint_dir=None,
        vector=None,
        lazy=False,
        cache_window=50,  # Keeps `cache_window` layers in memory at a time
        _transform_fn=None,
        prefix="",
    ):
        """Initializes the task vector from a checkpoint directory or a pre-computed vector.

        When checkpoint_dir is provided, loads pretrained and finetuned checkpoints
        from that directory. Pretrained is always PRETRAINED_FILENAME; finetuned
        is {prefix}{FINETUNED_FILENAME} (e.g. "lora_finetuned.pt").
        Also auto-discovers covariance.pt and fisher.pt in the directory.

        When vector is provided, uses the pre-computed task vector dict directly
        (used by arithmetic operations).
        """
        self.lazy = lazy
        self.checkpoint_dir = checkpoint_dir
        self.cache_window = cache_window
        self._cache = {}
        self._lazy_keys = None
        self._transform_fn = _transform_fn
        self._prefix = prefix

        # Resolve checkpoint file paths from directory
        if checkpoint_dir is not None:
            self._pretrained_checkpoint = os.path.join(
                checkpoint_dir, self.PRETRAINED_FILENAME
            )
            self._finetuned_checkpoint = os.path.join(
                checkpoint_dir, f"{prefix}{self.FINETUNED_FILENAME}"
            )
        else:
            self._pretrained_checkpoint = None
            self._finetuned_checkpoint = None

        # Auto-discover statistics files
        self.covariance_path = None
        self.fisher_path = None
        if checkpoint_dir is not None:
            cov_file = os.path.join(checkpoint_dir, "covariance.pt")
            cov_dir = os.path.join(checkpoint_dir, "covariance")
            if os.path.exists(cov_file):
                self.covariance_path = cov_file
            elif os.path.isdir(cov_dir):
                self.covariance_path = cov_dir

            fisher_file = os.path.join(checkpoint_dir, "fisher.pt")
            fisher_dir = os.path.join(checkpoint_dir, "fisher")
            if os.path.exists(fisher_file):
                self.fisher_path = fisher_file
            elif os.path.isdir(fisher_dir):
                self.fisher_path = fisher_dir

        if vector is not None:
            assert not self.lazy, "Cannot pass a vector if lazy is True"
            self._vector = vector
        else:
            assert checkpoint_dir is not None, (
                "Either checkpoint_dir or vector must be provided"
            )

            if self.lazy:
                self._vector = None
            else:
                self._vector = self._build_vector()

    @property
    def vector(self):
        if self.lazy:
            v = self._build_vector()
            if self._transform_fn is not None:
                v = self._transform_fn(v)
            return v
        else:
            return self._vector

    def lazy_keys(self):
        """Return parameter keys without building the full diff vector.

        In eager mode, returns keys from the stored vector.
        In lazy mode, loads only the pretrained checkpoint to get keys,
        avoiding the memory cost of computing the full diff.
        """
        if not self.lazy:
            return list(self._vector.keys())
        if self._lazy_keys is not None:
            return self._lazy_keys
        with torch.no_grad():
            pretrained = self._load_checkpoint(self._pretrained_checkpoint)
            sd = pretrained.state_dict() if hasattr(pretrained, "state_dict") else pretrained
            self._lazy_keys = [
                k for k in sd
                if sd[k].dtype not in (torch.int64, torch.uint8)
            ]
            del pretrained, sd
        return self._lazy_keys

    def get_vector_element(self, key: str):
        # Eager mode: just use the fully-built dict
        if not self.lazy:
            return self.vector[key]

        if key in self._cache:
            return self._cache[key]  # Cache hit

        # Cache miss: load the vector (applies _transform_fn if set)
        vector = self.vector
        self._cache = {}  # Reset the cache

        if self._transform_fn is not None:
            # The transform operates on the full dict and we already have it in memory,
            # so cache everything to avoid redundant disk loads + re-transforms.
            self._cache = dict(vector)
        else:
            # Window caching: keep next `cache_window` keys to limit memory use.
            all_keys = list(vector.keys())
            try:
                start_idx = all_keys.index(key)
            except ValueError:
                raise KeyError(f"Key {key} not found in vector.")
            end_idx = start_idx + self.cache_window
            for k in all_keys[start_idx:end_idx]:
                self._cache[k] = vector[k]

        # Release the full vector dict — only cached window survives
        del vector

        if key not in self._cache:
            raise KeyError(f"Key {key} not found in vector.")
        return self._cache[key]

    def _build_vector(self):
        with torch.no_grad():
            pretrained = self._load_checkpoint(self._pretrained_checkpoint)
            pretrained_state_dict = (
                pretrained.state_dict()
                if hasattr(pretrained, "state_dict")
                else pretrained
            )
            finetuned = self._load_checkpoint(self._finetuned_checkpoint)
            finetuned_state_dict = (
                finetuned.state_dict()
                if hasattr(finetuned, "state_dict")
                else finetuned
            )
            vector = {}
            for key in pretrained_state_dict:
                if pretrained_state_dict[key].dtype == torch.int64:
                    continue
                if pretrained_state_dict[key].dtype == torch.uint8:
                    continue
                vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

        return vector

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        raise NotImplementedError

    @abc.abstractmethod
    def _cast_to_same_type(self, other):
        raise NotImplementedError

    def _copy_metadata(self, result):
        """Copy covariance/fisher paths, checkpoint_dir, and prefix to a derived task vector."""
        result.covariance_path = self.covariance_path
        result.fisher_path = self.fisher_path
        result.checkpoint_dir = self.checkpoint_dir
        result._prefix = self._prefix
        return result

    def __add__(self, other):
        """Add two task vectors together."""
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return self._copy_metadata(self.__class__(vector=new_vector))

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return self._copy_metadata(self.__class__(vector=new_vector))

    def __pow__(self, power):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] ** power
        return self._copy_metadata(self.__class__(vector=new_vector))

    def __mul__(self, other):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = other * self.vector[key]
        return self._copy_metadata(self.__class__(vector=new_vector))

    def dot(self, other):
        """Dot product of two task vectors."""
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))

    def apply_to(self, checkpoint_dir, scaling_coef=1.0):
        """Apply a task vector to a pretrained model from checkpoint_dir."""
        pretrained_path = os.path.join(checkpoint_dir, self.PRETRAINED_FILENAME)
        with torch.no_grad():
            pretrained_model = self._load_checkpoint(pretrained_path)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"  # noqa: E501
                    )
                    continue
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        pretrained_model.load_state_dict(new_state_dict)
        return pretrained_model

    def map(self, fn):
        """Map a function over the task vector."""
        if self.lazy:
            existing = self._transform_fn
            composed = (lambda v: fn(existing(v))) if existing is not None else fn
            result = self.__class__(
                checkpoint_dir=self.checkpoint_dir,
                lazy=True,
                _transform_fn=composed,
                prefix=self._prefix,
            )
            return result
        with torch.no_grad():
            result = self.__class__(vector=fn(self.vector))
            return self._copy_metadata(result)

    def param_key_to_cov_key(self, key: str):
        raise NotImplementedError("Subclasses must implement this method")
