#!/usr/bin/env python3
"""Plot full-step recovery in equal-teacher-energy Fisher spectral bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=64)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.05)
    return parser.parse_args()


def equal_energy_slices(coefficients: np.ndarray, num_bins: int) -> list[slice]:
    energy = np.square(coefficients.astype(np.float64))
    cumulative = np.cumsum(energy)
    targets = np.linspace(0.0, cumulative[-1], num_bins + 1)
    boundaries = [0]
    for target in targets[1:-1]:
        boundary = int(np.searchsorted(cumulative, target, side="right"))
        boundary = max(boundary, boundaries[-1] + 1)
        max_allowed = len(energy) - (num_bins - len(boundaries))
        boundaries.append(min(boundary, max_allowed))
    boundaries.append(len(energy))
    return [slice(a, b) for a, b in zip(boundaries[:-1], boundaries[1:])]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = np.load(args.matrix_path)
    steps = artifact["steps"]
    recovery = artifact["recovery"]
    eigenvalues = artifact["eigenvalues"]
    coefficients = artifact["teacher_coefficients"]

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    coefficients = coefficients[order]
    recovery = recovery[order]
    slices = equal_energy_slices(coefficients, args.num_bins)

    binned = np.empty((args.num_bins, recovery.shape[1]), dtype=np.float32)
    bin_energy = np.empty(args.num_bins, dtype=np.float64)
    representative_eigenvalue = np.empty(args.num_bins, dtype=np.float64)
    bin_start = np.empty(args.num_bins, dtype=np.float64)
    bin_end = np.empty(args.num_bins, dtype=np.float64)
    directions_per_bin = np.empty(args.num_bins, dtype=np.int64)
    for index, band in enumerate(slices):
        weights = np.square(coefficients[band].astype(np.float64))
        bin_energy[index] = weights.sum()
        binned[index] = np.average(recovery[band], axis=0, weights=weights).astype(
            np.float32
        )
        representative_eigenvalue[index] = np.average(
            eigenvalues[band], weights=weights
        )
        bin_start[index] = eigenvalues[band.start]
        bin_end[index] = eigenvalues[band.stop - 1]
        directions_per_bin[index] = band.stop - band.start

    data_path = args.output_dir / "equal_teacher_energy_64bin_recovery.npz"
    np.savez_compressed(
        data_path,
        steps=steps,
        recovery=binned,
        representative_eigenvalue=representative_eigenvalue,
        bin_energy=bin_energy,
        bin_start_eigenvalue=bin_start,
        bin_end_eigenvalue=bin_end,
        directions_per_bin=directions_per_bin,
    )

    fig, ax = plt.subplots(figsize=(13.5, 7.2), constrained_layout=True)
    image = ax.imshow(
        binned,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[steps[0], steps[-1], 0, args.num_bins],
        cmap="viridis",
        vmin=args.vmin,
        vmax=args.vmax,
        rasterized=True,
    )
    tick_bins = np.linspace(0, args.num_bins - 1, 9).round().astype(int)
    ax.set_yticks(tick_bins + 0.5)
    ax.set_yticklabels([f"{representative_eigenvalue[i]:.3g}" for i in tick_bins])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(r"Fisher eigenvalue $\lambda$ (64 equal-$\Delta_T$-energy bins)")
    ax.set_title("Progressive spectral recovery during soft-KL training")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label(
        r"Bin recovery $\langle\Delta_S,\Delta_T\rangle_B/\|\Delta_T\|_B^2$"
    )
    ax.text(
        0.01,
        0.985,
        f"64 equal-energy bins; {len(steps):,} checkpoints",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 3},
    )

    png_path = args.output_dir / "equal_teacher_energy_64bin_recovery_heatmap.png"
    pdf_path = args.output_dir / "equal_teacher_energy_64bin_recovery_heatmap.pdf"
    fig.savefig(png_path, dpi=350)
    fig.savefig(pdf_path, dpi=350)
    plt.close(fig)

    metadata = {
        "source_matrix": str(args.matrix_path.resolve()),
        "num_bins": args.num_bins,
        "checkpoint_count": int(len(steps)),
        "binning": "contiguous ascending-eigenvalue bins with approximately equal teacher coefficient energy",
        "recovery": "sum_i <Delta_S,u_i><Delta_T,u_i> / sum_i <Delta_T,u_i>^2",
        "energy_fraction_min": float(bin_energy.min() / bin_energy.sum()),
        "energy_fraction_max": float(bin_energy.max() / bin_energy.sum()),
        "directions_per_bin_min": int(directions_per_bin.min()),
        "directions_per_bin_max": int(directions_per_bin.max()),
        "png_path": str(png_path.resolve()),
        "pdf_path": str(pdf_path.resolve()),
        "data_path": str(data_path.resolve()),
    }
    (args.output_dir / "equal_teacher_energy_64bin_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
