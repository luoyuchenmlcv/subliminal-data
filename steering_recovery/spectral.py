"""Online Fisher-eigenspace diagnostics and compact Delta_S trajectory storage."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def cosine_or_zero(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.float().norm().item() == 0 or right.float().norm().item() == 0:
        return 0.0
    return F.cosine_similarity(left.float(), right.float(), dim=0).item()


def equal_energy_slices(coefficients: torch.Tensor, bins: int) -> list[slice]:
    """Partition ordered coefficients into contiguous, near-equal energy bands."""
    dimension = coefficients.numel()
    if bins <= 0 or bins > dimension:
        raise ValueError("bins must be in [1, number of coefficients]")
    energy = coefficients.square()
    cumulative = torch.cumsum(energy, dim=0)
    targets = cumulative[-1] * torch.arange(1, bins, device=coefficients.device) / bins
    raw = torch.searchsorted(cumulative, targets).tolist()
    boundaries = [0]
    for index, candidate in enumerate(raw, 1):
        minimum = boundaries[-1] + 1
        maximum = dimension - (bins - index)
        boundaries.append(min(max(int(candidate) + 1, minimum), maximum))
    boundaries.append(dimension)
    return [slice(boundaries[i], boundaries[i + 1]) for i in range(bins)]


class PairedSpectralMetrics:
    """Compare hard and soft recovered vectors in a fixed Fisher eigenbasis."""

    def __init__(self, basis_path: str, teacher: torch.Tensor, bins: int):
        artifact = torch.load(basis_path, map_location="cpu", weights_only=False)
        values = artifact["eigenvalues"].float()
        vectors = artifact["eigenvectors"].float()
        dimension = teacher.numel()
        if values.shape != (dimension,) or vectors.shape != (dimension, dimension):
            raise ValueError("Spectral basis dimension does not match teacher vector")
        order = torch.argsort(values)
        self.values = values[order].to(teacher.device)
        self.vectors = vectors[:, order].to(teacher.device)
        self.teacher_coeff = self.vectors.T @ teacher.float()
        self.slices = equal_energy_slices(self.teacher_coeff, bins)
        self.teacher_energy = torch.stack(
            [self.teacher_coeff[band].square().sum() for band in self.slices]
        ).clamp_min(1e-30)
        self.inverse_lambda_sum = torch.stack(
            [
                self.values[band].clamp_min(1e-30).reciprocal().sum()
                for band in self.slices
            ]
        )

    @torch.no_grad()
    def compute(self, hard: torch.Tensor, soft: torch.Tensor) -> dict:
        hard_coeff = self.vectors.T @ hard.float()
        soft_coeff = self.vectors.T @ soft.float()
        hard_teacher, soft_teacher, hard_soft = [], [], []
        for band in self.slices:
            hard_teacher.append(
                (hard_coeff[band] - self.teacher_coeff[band]).square().sum()
            )
            soft_teacher.append(
                (soft_coeff[band] - self.teacher_coeff[band]).square().sum()
            )
            hard_soft.append((hard_coeff[band] - soft_coeff[band]).square().sum())
        hard_teacher_t = torch.stack(hard_teacher)
        soft_teacher_t = torch.stack(soft_teacher)
        hard_soft_t = torch.stack(hard_soft)
        return {
            "hard_teacher_error_by_bin": hard_teacher_t.cpu().tolist(),
            "soft_teacher_error_by_bin": soft_teacher_t.cpu().tolist(),
            "hard_soft_gap_by_bin": hard_soft_t.cpu().tolist(),
            "soft_residual_by_bin": (soft_teacher_t / self.teacher_energy)
            .cpu()
            .tolist(),
            "noise_dominance_by_bin": (
                hard_soft_t / (hard_soft_t + soft_teacher_t + 1e-30)
            )
            .cpu()
            .tolist(),
            "teacher_energy_by_bin": self.teacher_energy.cpu().tolist(),
            "inverse_lambda_sum_by_bin": self.inverse_lambda_sum.cpu().tolist(),
        }


class SpectralTrajectory:
    def __init__(
        self,
        *,
        basis_path: str,
        teacher_delta: torch.Tensor,
        output_dir: Path,
        num_bins: int,
        log_interval: int,
        chunk_size: int,
    ):
        artifact = torch.load(basis_path, map_location="cpu", weights_only=False)
        eigenvalues = artifact["eigenvalues"].float()
        eigenvectors = artifact["eigenvectors"].float()
        dimension = teacher_delta.numel()
        if eigenvalues.shape != (dimension,) or eigenvectors.shape != (
            dimension,
            dimension,
        ):
            raise ValueError("Spectral basis dimension does not match Delta_T")
        if num_bins <= 0 or num_bins > dimension:
            raise ValueError("--spectral-num-bins must be in [1, hidden_size]")
        if min(log_interval, chunk_size) <= 0:
            raise ValueError("Spectral intervals must be positive")

        self.device = teacher_delta.device
        self.eigenvalues = eigenvalues.to(self.device)
        self.eigenvectors = eigenvectors.to(self.device)
        self.teacher = teacher_delta.detach().float()
        self.teacher_coeff = self.eigenvectors.T @ self.teacher
        self.pred_error_factor = torch.ones_like(self.eigenvalues)
        self.num_bins = num_bins
        self.log_interval = log_interval
        self.chunk_size = chunk_size
        self.output_dir = output_dir / "spectral_trajectory"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "spectral_metrics.jsonl"
        self.log_path.write_text("", encoding="utf-8")
        self.buffer_steps: list[int] = []
        self.buffer_vectors: list[torch.Tensor] = []
        self.chunk_index = 0
        self.bin_slices = [
            slice(i * dimension // num_bins, (i + 1) * dimension // num_bins)
            for i in range(num_bins)
        ]
        metadata = {
            "format_version": 1,
            "basis_path": str(Path(basis_path).resolve()),
            "dimension": dimension,
            "num_bins": num_bins,
            "log_interval": log_interval,
            "chunk_size": chunk_size,
            "bin_order": "ascending_eigenvalue",
            "trajectory_dtype": "float32",
            "basis_metadata": artifact.get("metadata", {}),
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    @torch.no_grad()
    def advance_prediction(self, learning_rate: float) -> None:
        self.pred_error_factor.mul_(1.0 - float(learning_rate) * self.eigenvalues)

    @torch.no_grad()
    def record_vector(self, step: int, delta_s: torch.Tensor) -> None:
        self.buffer_steps.append(int(step))
        self.buffer_vectors.append(delta_s.detach().float().cpu().clone())
        if len(self.buffer_steps) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer_steps:
            return
        first, last = self.buffer_steps[0], self.buffer_steps[-1]
        path = self.output_dir / f"delta_s_steps_{first:05d}_{last:05d}.pt"
        torch.save(
            {
                "steps": torch.tensor(self.buffer_steps, dtype=torch.int64),
                "delta_s": torch.stack(self.buffer_vectors),
            },
            path,
        )
        self.buffer_steps.clear()
        self.buffer_vectors.clear()
        self.chunk_index += 1

    @torch.no_grad()
    def metrics(self, step: int, delta_s: torch.Tensor) -> dict[str, float]:
        student = delta_s.detach().float()
        coeff = self.eigenvectors.T @ student
        predicted_coeff = (1.0 - self.pred_error_factor) * self.teacher_coeff
        predicted = self.eigenvectors @ predicted_coeff
        teacher_norm = self.teacher.norm().clamp_min(1e-30)
        residual = student - self.teacher
        metrics = {
            "spectral/global_relative_residual": (
                residual.norm() / teacher_norm
            ).item(),
            "spectral/observed_predicted_cosine": F.cosine_similarity(
                student, predicted, dim=0, eps=1e-12
            ).item()
            if predicted.norm().item() > 0
            else 0.0,
            "spectral/observed_predicted_relative_error": (
                (student - predicted).norm() / teacher_norm
            ).item(),
            "spectral/predicted_teacher_cosine": F.cosine_similarity(
                predicted, self.teacher, dim=0, eps=1e-12
            ).item()
            if predicted.norm().item() > 0
            else 0.0,
        }
        total_residual_energy = residual.square().sum().clamp_min(1e-30)
        for index, band in enumerate(self.bin_slices):
            a = coeff[band]
            d = self.teacher_coeff[band]
            pred = predicted_coeff[band]
            d_energy = d.square().sum().clamp_min(1e-30)
            suffix = (
                f"bin_{index:02d}_flattest"
                if index == 0
                else f"bin_{index:02d}_steepest"
                if index == self.num_bins - 1
                else f"bin_{index:02d}"
            )
            metrics[f"spectral_recovery/{suffix}"] = (torch.dot(a, d) / d_energy).item()
            metrics[f"spectral_residual/{suffix}"] = (
                (a - d).norm() / d_energy.sqrt()
            ).item()
            metrics[f"spectral_cosine/{suffix}"] = (
                F.cosine_similarity(a, d, dim=0, eps=1e-12).item()
                if a.norm().item() > 0
                else 0.0
            )
            metrics[f"spectral_residual_energy/{suffix}"] = (
                (a - d).square().sum() / total_residual_energy
            ).item()
            metrics[f"spectral_predicted_recovery/{suffix}"] = (
                torch.dot(pred, d) / d_energy
            ).item()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"optimizer_step": int(step), **metrics}) + "\n")
        return metrics

    def close(self) -> None:
        self.flush()
