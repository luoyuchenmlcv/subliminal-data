#!/usr/bin/env python3
"""Reproduce Figure 2: explicit damped Fisher inversion of the initial gradient."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    REPO_ROOT,
    Runner,
    Task,
    add_common_arguments,
    carrier_dir,
    carrier_task,
    clean_carrier_task,
    finish_cli_error,
    fisher_dir,
    fisher_task,
    load_config,
    python_command,
    read_json,
    require_files,
    selections,
    stage_in,
    teacher_path,
    teacher_task,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def result_root(output_root: Path, model: dict, trait: str, seed: int) -> Path:
    return (
        output_root / "figures/figure_02/runs" / model["slug"] / trait / f"seed_{seed}"
    )


def inversion_tasks(args, config, model_keys, traits) -> list[Task]:
    seed = int(config["seed"])
    settings = config["figure_02"]
    tasks: list[Task] = []
    for model_key in model_keys:
        model = config["models"][model_key]
        fisher = fisher_dir(args.output_root, model, seed) / "true_fisher.pt"
        for trait in traits:
            destination = result_root(args.output_root, model, trait, seed)
            gbar = destination / "gbar.pt"
            tasks.append(
                Task(
                    f"figure02:gbar:{model_key}:{trait}",
                    python_command(
                        REPO_ROOT
                        / "sv_single_scripts/fisher_analysis/estimate_gbar.py",
                        "--model",
                        model["path"],
                        "--teacher-vector-path",
                        teacher_path(args.output_root, model, trait, seed),
                        "--carrier-path",
                        carrier_dir(args.output_root, model, trait, seed)
                        / "filtered_dataset.jsonl",
                        "--output-path",
                        gbar,
                        "--max-samples",
                        config["carrier"]["accepted_sequences"],
                        "--batch-size",
                        8,
                        "--teacher-alpha",
                        1,
                        "--seed",
                        seed,
                        "--data-order-seed",
                        seed,
                        "--precision",
                        "bf16",
                        "--log-interval",
                        500,
                    ),
                    gbar,
                )
            )
            sweep = destination / "sweep"
            tasks.append(
                Task(
                    f"figure02:invert:{model_key}:{trait}",
                    python_command(
                        REPO_ROOT
                        / "sv_single_scripts/fisher_analysis/sweep_fisher_preconditioner.py",
                        "--gbar-path",
                        gbar,
                        "--fisher-path",
                        fisher,
                        "--output-dir",
                        sweep,
                        "--powers",
                        settings["powers"],
                        "--gamma-multipliers",
                        settings["gamma_multipliers"],
                        "--shrinkages",
                        settings["shrinkages"],
                        "--eigh-device",
                        "cuda",
                        "--wandb-mode",
                        args.wandb_mode,
                    ),
                    sweep / "summary.json",
                )
            )
    return tasks


def aggregate(args, config, model_keys, traits) -> Path | None:
    seed = int(config["seed"])
    summaries = [
        result_root(args.output_root, config["models"][model_key], trait, seed)
        / "sweep/summary.json"
        for model_key in model_keys
        for trait in traits
    ]
    if not require_files(summaries, dry_run=args.dry_run):
        return None
    records = []
    for model_key in model_keys:
        model = config["models"][model_key]
        for trait in traits:
            summary = read_json(
                result_root(args.output_root, model, trait, seed) / "sweep/summary.json"
            )
            best = summary["best_full"]
            records.append(
                {
                    "model_key": model_key,
                    "model": model["display_name"],
                    "trait": trait,
                    "raw_alignment": summary["base_p0_cosine"],
                    "fisher_inverted_alignment": best["cosine"],
                    "oracle_power": best["p"],
                    "oracle_gamma_multiplier": best["gamma_multiplier"],
                    "oracle_shrinkage": best["shrinkage"],
                }
            )
    destination = args.output_root / "figures/figure_02/aggregated.json"
    write_json(
        destination,
        {
            "figure": 2,
            "selection_note": "Inversion hyperparameters are oracle-selected against the known teacher vector.",
            "records": records,
        },
    )
    return destination


def plot(args, config, model_keys, traits, aggregated: Path | None = None) -> None:
    source = aggregated or args.output_root / "figures/figure_02/aggregated.json"
    if not require_files([source], dry_run=args.dry_run):
        return
    records = read_json(source)["records"]
    fig, axes = plt.subplots(
        len(model_keys),
        1,
        figsize=(3.8, 2.0 * len(model_keys)),
        sharex=True,
        squeeze=False,
    )
    x = np.arange(len(traits))
    width = 0.36
    for row, model_key in enumerate(model_keys):
        selected = [
            next(
                r
                for r in records
                if r["model_key"] == model_key and r["trait"] == trait
            )
            for trait in traits
        ]
        axis = axes[row, 0]
        axis.bar(
            x - width / 2,
            [r["raw_alignment"] for r in selected],
            width,
            color="#B9A7D1",
            label="Raw gradient",
        )
        axis.bar(
            x + width / 2,
            [r["fisher_inverted_alignment"] for r in selected],
            width,
            color="#2A9D8F",
            label="Fisher inverted",
        )
        axis.set_title(config["models"][model_key]["display_name"], loc="left")
        axis.set_ylim(0, 1)
        axis.set_ylabel(r"Alignment with $\Delta_T$")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xticks(x, traits, rotation=20, ha="right")
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    destination = args.output_root / "figures/figure_02"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "figure_02.pdf", bbox_inches="tight")
    fig.savefig(destination / "figure_02.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        model_keys, traits = selections(args, config)
        runner = Runner(
            args.output_root,
            dry_run=args.dry_run,
            resume=args.resume,
            force=args.force,
            figure="figure_02",
            config_path=args.config,
        )
        if stage_in(args.stage, "prepare"):
            for model_key in model_keys:
                model = config["models"][model_key]
                runner.run(
                    teacher_task(
                        args.output_root, model, trait, config, args.wandb_mode
                    )
                    for trait in traits
                )
                runner.run(
                    carrier_task(args.output_root, model, trait, config)
                    for trait in traits
                )
                runner.run(
                    [
                        clean_carrier_task(args.output_root, model, config),
                        fisher_task(args.output_root, model, config, args.wandb_mode),
                    ]
                )
        if stage_in(args.stage, "run"):
            runner.run(inversion_tasks(args, config, model_keys, traits))
        aggregated = (
            aggregate(args, config, model_keys, traits)
            if stage_in(args.stage, "aggregate")
            else None
        )
        if stage_in(args.stage, "plot"):
            plot(args, config, model_keys, traits, aggregated)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        finish_cli_error(error)


if __name__ == "__main__":
    main()
