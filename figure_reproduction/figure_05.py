#!/usr/bin/env python3
"""Reproduce Figure 5: trait energy across normalized Fisher curvature."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    Runner, add_common_arguments, clean_carrier_task, finish_cli_error,
    fisher_basis_task, fisher_dir, fisher_task, load_config,
    require_files, selections, stage_in, teacher_path, teacher_task, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def spectral_npz(output_root: Path, model: dict) -> Path:
    return output_root / "figures/figure_05/data" / model["slug"] / "trait_spectra.npz"


def aggregate(args, config, model_keys, traits) -> Path | None:
    seed = int(config["seed"])
    required: list[Path] = []
    for model_key in model_keys:
        model = config["models"][model_key]
        required.append(fisher_dir(args.output_root, model, seed) / "spectral_basis.pt")
        required.extend(teacher_path(args.output_root, model, trait, seed) for trait in traits)
    if not require_files(required, dry_run=args.dry_run):
        return None

    summary_records = []
    for model_key in model_keys:
        model = config["models"][model_key]
        basis = torch.load(fisher_dir(args.output_root, model, seed) / "spectral_basis.pt",
                           map_location="cpu", weights_only=False)
        eigenvalues = basis["eigenvalues"].double()
        order = torch.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = basis["eigenvectors"].double()[:, order]
        vectors = []
        for trait in traits:
            artifact = torch.load(teacher_path(args.output_root, model, trait, seed),
                                  map_location="cpu", weights_only=False)
            vectors.append(artifact["delta_t"].double().flatten())
        coefficients = torch.stack(vectors) @ eigenvectors
        energy = coefficients.square()
        energy /= energy.sum(dim=1, keepdim=True)
        mean_lambda = eigenvalues.mean()
        normalized_lambda = eigenvalues / mean_lambda
        for index, trait in enumerate(traits):
            cumulative = torch.cumsum(energy[index], dim=0)
            median_index = int(torch.searchsorted(cumulative, torch.tensor(0.5, dtype=cumulative.dtype)))
            summary_records.append({
                "model_key": model_key, "model": model["display_name"], "trait": trait,
                "median_energy_lambda_over_mean": float(normalized_lambda[median_index]),
                "energy_weighted_lambda_over_mean": float((energy[index] * normalized_lambda).sum()),
            })
        destination = spectral_npz(args.output_root, model)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination, traits=np.asarray(traits),
            normalized_lambda=normalized_lambda.numpy(),
            coefficients=coefficients.numpy(), energy=energy.numpy(),
        )
    summary = args.output_root / "figures/figure_05/aggregated.json"
    write_json(summary, {"figure": 5, "records": summary_records})
    return summary


def plot(args, config, model_keys, traits) -> None:
    paths = [spectral_npz(args.output_root, config["models"][key]) for key in model_keys]
    if not require_files(paths, dry_run=args.dry_run):
        return
    loaded = {key: np.load(spectral_npz(args.output_root, config["models"][key]))
              for key in model_keys}
    positive = np.concatenate([
        data["normalized_lambda"][data["normalized_lambda"] > 0] for data in loaded.values()
    ])
    edges = np.logspace(np.log10(np.quantile(positive, 0.001)),
                        np.log10(positive.max()), int(config["figure_05"]["histogram_bins"]) + 1)
    fig, axes = plt.subplots(1, len(traits), figsize=(2.25 * len(traits), 2.05),
                             sharex=True, sharey=True, squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(model_keys)))
    for column, trait in enumerate(traits):
        axis = axes[0, column]
        for color, model_key in zip(colors, model_keys):
            data = loaded[model_key]
            names = [str(value) for value in data["traits"]]
            index = names.index(trait)
            histogram, _ = np.histogram(data["normalized_lambda"], bins=edges,
                                        weights=data["energy"][index])
            axis.stairs(histogram, edges, fill=True, alpha=0.38, color=color,
                        label=config["models"][model_key]["display_name"])
        axis.set_xscale("log")
        axis.set_title(trait.capitalize())
        axis.set_xlabel(r"$\lambda/\bar\lambda$")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Teacher energy fraction")
    axes[0, -1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    destination = args.output_root / "figures/figure_05"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "figure_05.pdf", bbox_inches="tight")
    fig.savefig(destination / "figure_05.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        model_keys, traits = selections(args, config)
        runner = Runner(args.output_root, dry_run=args.dry_run, resume=args.resume,
                        force=args.force, figure="figure_05", config_path=args.config)
        if stage_in(args.stage, "prepare"):
            for model_key in model_keys:
                model = config["models"][model_key]
                runner.run(teacher_task(args.output_root, model, trait, config, args.wandb_mode)
                           for trait in traits)
                runner.run([
                    clean_carrier_task(args.output_root, model, config),
                    fisher_task(args.output_root, model, config, args.wandb_mode),
                    fisher_basis_task(args.output_root, model, config),
                ])
        # Projection is deterministic CPU post-processing, so it is the run/aggregate stage.
        if stage_in(args.stage, "run", "aggregate"):
            aggregate(args, config, model_keys, traits)
        if stage_in(args.stage, "plot"):
            plot(args, config, model_keys, traits)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        finish_cli_error(error)


if __name__ == "__main__":
    main()
