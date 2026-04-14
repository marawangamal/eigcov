import torch

from src.task_vectors import _TaskVector


class LanguageNonLinearTaskVector(_TaskVector):
    """Task vector for standard (non-linearized) T5 models."""

    def _load_checkpoint(self, checkpoint):
        from src.language.modeling import T5Wrapper  # noqa: F401

        return torch.load(checkpoint, map_location="cpu", weights_only=False)

    def apply_to_nonlinear(self, checkpoint_dir, scaling_coef=1.0):
        return self.apply_to(checkpoint_dir, scaling_coef)

    def apply_to_linear(self, checkpoint_dir, scaling_coef=1.0):
        return language_nonlinear_to_linear(self).apply_to(
            checkpoint_dir, scaling_coef
        )

    def _cast_to_same_type(self, other):
        return language_linear_to_nonlinear(other, self.vector.keys())

    def param_key_to_cov_key(self, key: str):
        return key.replace(".weight", "")


class LanguageLinearizedTaskVector(_TaskVector):
    """Task vector for linearized T5 models."""

    def _load_checkpoint(self, checkpoint):
        from src.language.linearize import LinearizedT5Wrapper

        return LinearizedT5Wrapper.load(checkpoint)

    def apply_to_nonlinear(
        self, checkpoint_dir, param_names, scaling_coef=1.0
    ):
        return language_linear_to_nonlinear(self, param_names).apply_to(
            checkpoint_dir, scaling_coef
        )

    def apply_to_linear(self, checkpoint_dir, scaling_coef=1.0):
        return self.apply_to(checkpoint_dir, scaling_coef)

    def get_named_parameters(self, param_names):
        params = {k: v for k, v in self.vector.items() if "model.params0" not in k}
        return {k: v for k, v in zip(param_names, params.values())}

    def _cast_to_same_type(self, other):
        return language_nonlinear_to_linear(other)

    def param_key_to_cov_key(self, key: str):
        return key.replace(".weight", "")


def language_nonlinear_to_linear(nonlinear_task_vector):
    """Convert a language nonlinear task vector to a linearized task vector."""
    if isinstance(nonlinear_task_vector, LanguageLinearizedTaskVector):
        return nonlinear_task_vector
    linear_params = {
        f"model.params.{i}": v
        for i, v in enumerate(nonlinear_task_vector.vector.values())
    }
    linear_params |= {
        f"model.params0.{i}": torch.zeros_like(v)
        for i, v in enumerate(nonlinear_task_vector.vector.values())
    }
    return LanguageLinearizedTaskVector(vector=linear_params)


def language_linear_to_nonlinear(linear_task_vector, param_names=None):
    """Convert a language linearized task vector to a nonlinear task vector."""
    if isinstance(linear_task_vector, LanguageNonLinearTaskVector):
        return linear_task_vector
    if param_names is None:
        raise ValueError("param_names required to convert linear → nonlinear")
    return LanguageNonLinearTaskVector(
        vector=linear_task_vector.get_named_parameters(param_names)
    )
