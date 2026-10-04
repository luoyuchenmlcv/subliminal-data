"""Diagonalize a saved full Fisher once for online spectral tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fisher-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    source = torch.load(args.fisher_path, map_location="cpu", weights_only=False)
    fisher = source["fisher"].double()
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(fisher)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "fisher_path": str(Path(args.fisher_path).resolve()),
        "dimension": fisher.shape[0],
        "min_eigenvalue": eigenvalues.min().item(),
        "max_eigenvalue": eigenvalues.max().item(),
        "mean_eigenvalue": eigenvalues.mean().item(),
        "fisher_metadata": source.get("metadata", {}),
    }
    torch.save(
        {
            "eigenvalues": eigenvalues.float(),
            "eigenvectors": eigenvectors.float(),
            "metadata": metadata,
        },
        output,
    )
    print(
        f"Saved Fisher basis: {output} | d={fisher.shape[0]} "
        f"lambda=[{eigenvalues.min().item():.6g}, {eigenvalues.max().item():.6g}]"
    )


if __name__ == "__main__":
    main()
