import torch

from src.task_vectors import _TaskVector
from src.vision.linearize import LinearizedImageEncoder


class NonLinearTaskVector(_TaskVector):
    """A task vector for nonlinear models."""

    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        return torch.load(checkpoint, map_location="cpu", weights_only=False)

    def apply_to_nonlinear(self, pretrained_nonlinear_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a nonlinear pretrained model."""
        return self.apply_to(pretrained_nonlinear_checkpoint, scaling_coef)

    def apply_to_linear(self, pretrained_linear_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a linear pretrained model."""
        return nonlinear_to_linear(self).apply_to(
            pretrained_linear_checkpoint, scaling_coef
        )

    def _cast_to_same_type(self, other):
        return linear_to_nonlinear(other, self.vector.keys())

    def param_key_to_cov_key(self, key: str):
        return "image_encoder." + key.replace(".weight", "")


class LinearizedTaskVector(_TaskVector):
    """A task vector for linearized models."""

    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        return LinearizedImageEncoder.load(checkpoint)

    def apply_to_nonlinear(
        self, pretrained_nonlinear_checkpoint, param_names, scaling_coef=1.0
    ):
        """Apply a task vector to a nonlinear pretrained model."""
        return linear_to_nonlinear(self, param_names).apply_to(
            pretrained_nonlinear_checkpoint, scaling_coef
        )

    def apply_to_linear(self, pretrained_linear_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a linear pretrained model."""
        return self.apply_to(pretrained_linear_checkpoint, scaling_coef)

    def get_named_parameters(self, param_names):
        """Get the named parameters of the task vector."""
        params = {k: v for k, v in self.vector.items() if "model.params0" not in k}
        return {k: v for k, v in zip(param_names, params.values())}

    def _cast_to_same_type(self, other):
        return nonlinear_to_linear(other)

    def param_key_to_cov_key(self, key: str):
        return "image_encoder." + key.replace(".weight", "")


def nonlinear_to_linear(nonlinear_task_vector):
    """Convert a nonlinear task vector to a linear task vector."""
    if isinstance(nonlinear_task_vector, LinearizedTaskVector):
        return nonlinear_task_vector
    else:
        linear_params = {
            f"model.params.{i}": v
            for i, v in enumerate(nonlinear_task_vector.vector.values())
        }
        # The diff of the init params of the linearized models are all zero.
        linear_params |= {
            f"model.params0.{i}": torch.zeros_like(v)
            for i, v in enumerate(nonlinear_task_vector.vector.values())
        }
        return LinearizedTaskVector(vector=linear_params)


def linear_to_nonlinear(linear_task_vector, param_names):
    """Convert a linear task vector to a nonlinear task vector."""
    if isinstance(linear_task_vector, NonLinearTaskVector):
        return linear_task_vector
    else:
        return NonLinearTaskVector(
            vector=linear_task_vector.get_named_parameters(param_names)
        )
