"""Pure spectral operations used by Fisher inversion experiments."""

from __future__ import annotations

import torch


def effective_rank(eigenvalues: torch.Tensor) -> float:
    values = eigenvalues.clamp_min(0)
    probabilities = values / values.sum().clamp_min(1e-30)
    positive = probabilities[probabilities > 0]
    return torch.exp(-(positive * positive.log()).sum()).item()


def spectral_precondition(
    gradient: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    power: float,
    gamma: float,
) -> torch.Tensor:
    mean = eigenvalues.mean().item()
    floor = max(gamma, mean * 1e-12, 1e-30)
    weights = (eigenvalues + gamma).clamp_min(floor).pow(-power / 2)
    return eigenvectors @ (weights * (eigenvectors.T @ gradient))
