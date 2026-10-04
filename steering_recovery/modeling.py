"""Model-family adapters and residual-stream steering hooks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch


def get_text_config(model: torch.nn.Module):
    return getattr(model.config, "text_config", model.config)


def get_hidden_size(model: torch.nn.Module) -> int:
    return int(get_text_config(model).hidden_size)


def get_num_hidden_layers(model: torch.nn.Module) -> int:
    return int(get_text_config(model).num_hidden_layers)


def get_transformer_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    candidates = (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("transformer", "h"),
        ("model", "decoder", "layers"),
    )
    for path in candidates:
        value: Any = model
        try:
            for name in path:
                value = getattr(value, name)
        except AttributeError:
            continue
        if isinstance(value, (torch.nn.ModuleList, list, tuple)):
            return value
    raise ValueError(
        f"Unsupported model architecture {type(model).__name__}: "
        "could not locate transformer layers"
    )


def bound_l2(value: torch.Tensor, max_norm: float | None) -> torch.Tensor:
    if max_norm is None:
        return value
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norm = value.float().norm()
    scale = torch.clamp(max_norm / norm.clamp_min(1e-12), max=1.0)
    return value * scale.to(dtype=value.dtype)


def project_l2_(parameter: torch.Tensor, max_norm: float) -> tuple[float, float]:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    with torch.no_grad():
        before = parameter.float().norm().item()
        if before > max_norm:
            parameter.mul_(max_norm / max(before, 1e-12))
        return before, parameter.float().norm().item()


class SharedDeltaHook:
    def __init__(self, delta: torch.Tensor, max_norm: float | None = None):
        self.delta = delta
        self.max_norm = max_norm

    def __call__(self, module, inputs, output):
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        delta = bound_l2(self.delta, self.max_norm).to(
            device=hidden.device, dtype=hidden.dtype
        )
        shifted = hidden + delta
        return (shifted,) + output[1:] if isinstance(output, tuple) else shifted


def register_shared_delta(
    model: torch.nn.Module,
    delta: torch.Tensor,
    max_norm: float | None = None,
    layer_indices: Sequence[int] | None = None,
):
    layers = get_transformer_layers(model)
    selected = (
        layers if layer_indices is None else [layers[index] for index in layer_indices]
    )
    return [
        layer.register_forward_hook(SharedDeltaHook(delta, max_norm=max_norm))
        for layer in selected
    ]


def remove_hooks(handles: Iterable[Any]) -> None:
    for handle in handles:
        handle.remove()


def precision_dtype(precision: str) -> torch.dtype:
    try:
        return {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[precision]
    except KeyError as error:
        raise ValueError(f"Unsupported precision: {precision}") from error


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_frozen_causal_lm(model_name: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
