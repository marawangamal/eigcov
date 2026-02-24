import abc
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.language.modeling import T5Wrapper
from src.utils import DotDict


class LinearizedModule(nn.Module):
    """Creates a linearized version of a nn.Module via first-order Taylor expansion.

    Args:
        model (nn.Module): The model to linearize. Trainable parameters are
            initialized to the parameters of this model.
        init_model (nn.Module): A model of the same type containing the parameters
            around which the model is linearized. Defaults to ``model``.
    """

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, x: func0(params, self.buffers0, x)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        for p in self.params0:
            p.requires_grad = False

        for p in self.params:
            p.requires_grad = True

    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp


class LinearizedT5Wrapper(abc.ABC, nn.Module):
    """Wraps a T5Wrapper with a linearized (Taylor-expanded) transformer backbone."""

    def __init__(
        self,
        args=None,
        keep_lang=False,
        transformer=None,
        init_transformer=None,
        tokenizer=None,
    ):
        super().__init__()
        if transformer is None:
            pretrained_name = args.model
            transformer = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)

        if init_transformer is None:
            init_transformer = transformer

        if tokenizer is None:
            pretrained_name = args.model
            max_seq_len = getattr(args, "max_seq_len", 128)
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_name, model_max_length=max_seq_len
            )

        transformer.module = LinearizedModule(transformer.module, init_transformer.module)

        self._model_name = self._get_name(args.model)
        self.model = T5Wrapper(transformer, tokenizer)
        self.tokenizer = self.model.tokenizer

    def _get_name(self, model_name):
        if "__pretrained__" in model_name:
            model_name, _ = model_name.split("__pretrained__", 1)
        return model_name

    def forward(self, batch):
        return self.model.forward(batch)

    def __call__(self, x):
        return self.forward(x)

    def compute_logProb_ofAllChoices(
        self,
        input_ids,
        input_masks,
        allChoices_ids,
        allChoices_masks,
        length_normalization,
    ):
        return self.model.compute_logProb_ofAllChoices(
            input_ids,
            input_masks,
            allChoices_ids,
            allChoices_masks,
            length_normalization,
        )

    def predict_mulChoice(self, batch, length_normalization):
        return self.model.predict_mulChoice(batch, length_normalization)

    def save(self, filename):
        """Saves the linearized T5 wrapper (state-dict only, not the full object)."""
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        state_dict = self.state_dict()
        state_dict["model_name"] = self._model_name

        torch.save(state_dict, filename)

    @classmethod
    def load(cls, filename):
        """Loads a LinearizedT5Wrapper from a saved state dict."""
        print(f"Loading linearized T5 from {filename}")
        state_dict = torch.load(filename, map_location="cpu", weights_only=False)

        args = DotDict({"model": state_dict["model_name"]})
        model = cls(args)

        state_dict.pop("model_name")
        model.load_state_dict(state_dict, strict=False)
        return model
