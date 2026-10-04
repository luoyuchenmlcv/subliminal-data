#!/usr/bin/env python3
"""Plot per-eigendirection recovery from a saved Delta_S trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    parser.add_argument("--basis-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coefficient-threshold", type=float, default=1e-3)
    parser.add_argument("--vmin", type=float, default=-0.25)
    parser.add_argument("--vmax", type=float, default=1.25)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_teacher(path: Path) -> torch.Tensor:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    value = artifact["delta_t"] if isinstance(artifact, dict) else artifact
    return value.detach().float().flatten()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    basis = torch.load(args.basis_path, map_location="cpu", weights_only=False)
    eigenvalues = basis["eigenvalues"].detach().float()
    eigenvectors = basis["eigenvectors"].detach().float()
    teacher = load_teacher(args.teacher_path)
    teacher_coeff = eigenvectors.T @ teacher

    cutoff = args.coefficient_threshold * teacher_coeff.abs().max()
    keep = teacher_coeff.abs() >= cutoff
    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
    if kept_indices.numel() == 0:
        raise RuntimeError("Coefficient threshold removed every eigendirection")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    kept_basis = eigenvectors[:, kept_indices].to(device)
    kept_teacher_coeff = teacher_coeff[kept_indices].to(device)

    all_steps: list[np.ndarray] = []
    all_recovery: list[np.ndarray] = []
    chunk_paths = sorted(args.trajectory_dir.glob("delta_s_steps_*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No trajectory chunks found in {args.trajectory_dir}")
    for path in chunk_paths:
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        vectors = chunk["delta_s"].float().to(device)
        coefficients = vectors @ kept_basis
        recovery = coefficients / kept_teacher_coeff.unsqueeze(0)
        all_steps.append(chunk["steps"].cpu().numpy())
        all_recovery.append(recovery.cpu().numpy().astype(np.float32))

    steps = np.concatenate(all_steps)
    recovery = np.concatenate(all_recovery, axis=0).T
    order = np.argsort(steps)
    steps = steps[order]
    recovery = recovery[:, order]
    kept_eigenvalues = eigenvalues[kept_indices].numpy()
    kept_coefficients = teacher_coeff[kept_indices].numpy()

    matrix_path = args.output_dir / "eigendirection_recovery_full.npz"
    np.savez_compressed(
        matrix_path,
        steps=steps,
        recovery=recovery,
        eigenvalues=kept_eigenvalues,
        teacher_coefficients=kept_coefficients,
        original_indices=kept_indices.numpy(),
    )

    # Directions are already ascending in eigenvalue; keep an explicit sort for safety.
    eig_order = np.argsort(kept_eigenvalues)
    recovery = recovery[eig_order]
    kept_eigenvalues = kept_eigenvalues[eig_order]

    fig, ax = plt.subplots(figsize=(13.5, 7.2), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=args.vmin, vcenter=0.0, vmax=args.vmax)
    image = ax.imshow(
        recovery,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[steps[0], steps[-1], 0, recovery.shape[0]],
        cmap="coolwarm",
        norm=norm,
        rasterized=True,
    )
    quantiles = np.linspace(0, 1, 7)
    positions = quantiles * (recovery.shape[0] - 1)
    eig_at_quantiles = np.quantile(kept_eigenvalues, quantiles)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{value:.3g}" for value in eig_at_quantiles])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(r"Fisher eigenvalue $\lambda_i$ (ascending quantiles)")
    ax.set_title("Per-eigendirection recovery throughout soft-KL training")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label(r"Recovery $\langle\Delta_S,u_i\rangle/\langle\Delta_T,u_i\rangle$")
    ax.text(
        0.01,
        0.985,
        f"{recovery.shape[0]:,} directions; {recovery.shape[1]:,} checkpoints",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 3},
    )

    png_path = args.output_dir / "eigendirection_recovery_heatmap.png"
    pdf_path = args.output_dir / "eigendirection_recovery_heatmap.pdf"
    fig.savefig(png_path, dpi=350)
    fig.savefig(pdf_path, dpi=350)
    plt.close(fig)

    metadata = {
        "trajectory_dir": str(args.trajectory_dir.resolve()),
        "teacher_path": str(args.teacher_path.resolve()),
        "basis_path": str(args.basis_path.resolve()),
        "checkpoint_count": int(recovery.shape[1]),
        "total_eigendirections": int(teacher_coeff.numel()),
        "kept_eigendirections": int(recovery.shape[0]),
        "coefficient_threshold_relative_to_max": args.coefficient_threshold,
        "absolute_coefficient_cutoff": float(cutoff),
        "color_limits": [args.vmin, args.vmax],
        "matrix_path": str(matrix_path.resolve()),
        "png_path": str(png_path.resolve()),
        "pdf_path": str(pdf_path.resolve()),
    }
    (args.output_dir / "eigendirection_recovery_heatmap_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
