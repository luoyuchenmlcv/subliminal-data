"""Online Fisher-eigenspace diagnostics and compact Delta_S trajectory storage."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


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
        if eigenvalues.shape != (dimension,) or eigenvectors.shape != (dimension, dimension):
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
            "spectral/global_relative_residual": (residual.norm() / teacher_norm).item(),
            "spectral/observed_predicted_cosine": F.cosine_similarity(
                student, predicted, dim=0, eps=1e-12
            ).item() if predicted.norm().item() > 0 else 0.0,
            "spectral/observed_predicted_relative_error": (
                (student - predicted).norm() / teacher_norm
            ).item(),
            "spectral/predicted_teacher_cosine": F.cosine_similarity(
                predicted, self.teacher, dim=0, eps=1e-12
            ).item() if predicted.norm().item() > 0 else 0.0,
        }
        total_residual_energy = residual.square().sum().clamp_min(1e-30)
        for index, band in enumerate(self.bin_slices):
            a = coeff[band]
            d = self.teacher_coeff[band]
            pred = predicted_coeff[band]
            d_energy = d.square().sum().clamp_min(1e-30)
            suffix = (
                f"bin_{index:02d}_flattest" if index == 0
                else f"bin_{index:02d}_steepest" if index == self.num_bins - 1
                else f"bin_{index:02d}"
            )
            metrics[f"spectral_recovery/{suffix}"] = (torch.dot(a, d) / d_energy).item()
            metrics[f"spectral_residual/{suffix}"] = (
                (a - d).norm() / d_energy.sqrt()
            ).item()
            metrics[f"spectral_cosine/{suffix}"] = F.cosine_similarity(
                a, d, dim=0, eps=1e-12
            ).item() if a.norm().item() > 0 else 0.0
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
