"""Optimizer and learning-rate scheduler factories."""

from __future__ import annotations

import math

import torch


def build_optimizer(
    parameters,
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
    rmsprop_alpha: float = 0.999,
    epsilon: float = 1e-8,
):
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=epsilon,
        )
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=epsilon,
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
            eps=epsilon,
            momentum=0.0,
            centered=False,
        )
    optimizer_momentum = momentum if name == "sgd_momentum" else 0.0
    if name in {"sgd", "sgd_momentum", "sgd_momentum_norm_matched"}:
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=optimizer_momentum,
            nesterov=False,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, *, name: str, warmup_steps: int, total_steps: int):
    if name == "constant":
        return None
    if total_steps <= 0 or warmup_steps < 0:
        raise ValueError("total_steps must be positive and warmup_steps non-negative")

    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        if name == "linear":
            return max(0.0, 1.0 - progress)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
