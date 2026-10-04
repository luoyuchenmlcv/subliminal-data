#!/usr/bin/env python3
"""Reproduce Figure 4: data-dependent inversion depth and repeated sampling."""

from __future__ import annotations

import argparse
import math
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
    fisher_basis_task,
    fisher_dir,
    fisher_task,
    load_config,
    python_command,
    read_jsonl,
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
    parser.set_defaults(models="qwen", traits="cat")
    return parser.parse_args()


def independent_dir(output_root: Path, count: int) -> Path:
    return output_root / f"figures/figure_04/runs/independent/n_{count}"


def repeated_carrier_dir(output_root: Path, seed: int, factor: int) -> Path:
    return output_root / f"figures/figure_04/carriers/repeated/seed_{seed}/r_{factor}"


def repeated_run_dir(output_root: Path, seed: int, factor: int) -> Path:
    return output_root / f"figures/figure_04/runs/repeated/seed_{seed}/r_{factor}"


def paired_command(
    args, config, model, carrier: Path, destination: Path, max_samples: int, seed: int
) -> tuple[str, ...]:
    settings = config["figure_04"]
    paper_seed = int(config["seed"])
    return python_command(
        REPO_ROOT / "sv_scripts/recover_paired_hard_soft.py",
        "--model",
        model["path"],
        "--topic",
        "cat",
        "--teacher-vector-path",
        teacher_path(args.output_root, model, "cat", paper_seed),
        "--carrier-path",
        carrier,
        "--spectral-basis-path",
        fisher_dir(args.output_root, model, paper_seed) / "spectral_basis.pt",
        "--output-dir",
        destination,
        "--max-samples",
        max_samples,
        "--max-steps",
        settings["optimizer_steps"],
        "--batch-size",
        1,
        "--gradient-accumulation-steps",
        60,
        "--learning-rate",
        model["recovery_learning_rate"],
        "--max-length",
        600,
        "--temperature",
        1,
        "--precision",
        "bf16",
        "--seed",
        seed,
        "--spectral-num-bins",
        64,
        "--spectral-log-interval",
        10,
        "--checkpoint-interval",
        100,
        "--wandb-mode",
        args.wandb_mode,
    )


def experiment_tasks(args, config, model) -> list[Task]:
    settings = config["figure_04"]
    paper_seed = int(config["seed"])
    base_carrier = (
        carrier_dir(args.output_root, model, "cat", paper_seed)
        / "filtered_dataset.jsonl"
    )
    tasks: list[Task] = []
    for count in settings["independent_sequence_counts"]:
        destination = independent_dir(args.output_root, int(count))
        tasks.append(
            Task(
                f"figure04:independent:n={count}",
                paired_command(
                    args,
                    config,
                    model,
                    base_carrier,
                    destination,
                    int(count),
                    paper_seed,
                ),
                destination / "summary.json",
            )
        )
    pool_size = int(settings["prompt_pool_size"])
    for generation_seed in settings["generation_seeds"]:
        for factor in settings["repeat_factors"]:
            generation_seed = int(generation_seed)
            factor = int(factor)
            carrier_destination = repeated_carrier_dir(
                args.output_root, generation_seed, factor
            )
            tasks.append(
                Task(
                    f"figure04:repeated-carrier:seed={generation_seed}:r={factor}",
                    python_command(
                        REPO_ROOT / "sv_scripts/generate_repeated_prompt_carrier.py",
                        "--model",
                        model["path"],
                        "--teacher-vector",
                        teacher_path(args.output_root, model, "cat", paper_seed),
                        "--source-dataset",
                        base_carrier,
                        "--output-dir",
                        carrier_destination,
                        "--prompt-count",
                        pool_size,
                        "--repeats-per-prompt",
                        factor,
                        "--batch-size",
                        config["carrier"]["batch_size"],
                        "--temperature",
                        1,
                        "--max-tokens",
                        config["carrier"]["max_new_tokens"],
                        "--seed",
                        generation_seed,
                        "--checkpoint-every",
                        100,
                    ),
                    carrier_destination / "metadata.json",
                )
            )
            budget = pool_size * factor
            run_destination = repeated_run_dir(
                args.output_root, generation_seed, factor
            )
            tasks.append(
                Task(
                    f"figure04:repeated-recovery:seed={generation_seed}:r={factor}",
                    paired_command(
                        args,
                        config,
                        model,
                        carrier_destination / "filtered_dataset.jsonl",
                        run_destination,
                        budget,
                        generation_seed,
                    ),
                    run_destination / "summary.json",
                )
            )
    return tasks


def mean_spectral_error(
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.asarray([int(row["optimizer_step"]) for row in records])
    hard, soft = [], []
    for row in records:
        energy = np.asarray(row["teacher_energy_by_bin"], dtype=np.float64)
        hard.append(
            np.mean(
                np.asarray(row["hard_teacher_error_by_bin"]) / np.maximum(energy, 1e-30)
            )
        )
        soft.append(
            np.mean(
                np.asarray(row["soft_teacher_error_by_bin"]) / np.maximum(energy, 1e-30)
            )
        )
    return steps, np.asarray(hard), np.asarray(soft)


def aggregate(args, config) -> Path | None:
    settings = config["figure_04"]
    paths: list[Path] = []
    for count in settings["independent_sequence_counts"]:
        paths.extend(
            [
                independent_dir(args.output_root, int(count)) / "training_log.jsonl",
                independent_dir(args.output_root, int(count))
                / "spectral_metrics.jsonl",
            ]
        )
    for seed in settings["generation_seeds"]:
        for factor in settings["repeat_factors"]:
            paths.append(
                repeated_run_dir(args.output_root, int(seed), int(factor))
                / "training_log.jsonl"
            )
    if not require_files(paths, dry_run=args.dry_run):
        return None

    independent = []
    for count in settings["independent_sequence_counts"]:
        training = read_jsonl(
            independent_dir(args.output_root, int(count)) / "training_log.jsonl"
        )
        spectral = read_jsonl(
            independent_dir(args.output_root, int(count)) / "spectral_metrics.jsonl"
        )
        steps, hard, soft = mean_spectral_error(spectral)
        best_index = int(np.argmin(hard))
        independent.append(
            {
                "budget": int(count),
                "steps": steps.tolist(),
                "hard_mean_spectral_error": hard.tolist(),
                "soft_mean_spectral_error": soft.tolist(),
                "optimal_step": int(steps[best_index]),
                "minimum_hard_error": float(hard[best_index]),
                "best_hard_alignment": max(
                    float(row["hard_teacher_cosine"]) for row in training
                ),
            }
        )

    repeated = []
    for factor in settings["repeat_factors"]:
        alignments = []
        for seed in settings["generation_seeds"]:
            training = read_jsonl(
                repeated_run_dir(args.output_root, int(seed), int(factor))
                / "training_log.jsonl"
            )
            alignments.append(
                max(float(row["hard_teacher_cosine"]) for row in training)
            )
        values = np.asarray(alignments, dtype=np.float64)
        repeated.append(
            {
                "repeat_factor": int(factor),
                "budget": int(settings["prompt_pool_size"]) * int(factor),
                "seed_alignments": alignments,
                "mean": float(values.mean()),
                "sem": float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0,
            }
        )
    destination = args.output_root / "figures/figure_04/aggregated.json"
    write_json(
        destination,
        {
            "figure": 4,
            "independent": independent,
            "repeated": repeated,
            "repeated_sampling_status": "measured",
        },
    )
    return destination


def plot(args, aggregated: Path | None = None) -> None:
    source = aggregated or args.output_root / "figures/figure_04/aggregated.json"
    if not require_files([source], dry_run=args.dry_run):
        return
    from common import read_json

    data = read_json(source)
    independent = data["independent"]
    repeated = data["repeated"]
    panel_a = [row for row in independent if row["budget"] <= 16000]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(panel_a)))
    soft_curves = []
    for row, color in zip(panel_a, colors):
        steps = np.asarray(row["steps"])
        hard = np.asarray(row["hard_mean_spectral_error"])
        soft_curves.append(np.asarray(row["soft_mean_spectral_error"]))
        axes[0].plot(steps, hard, color=color, label=f"M={row['budget']:,}")
        axes[0].scatter(
            [row["optimal_step"]], [row["minimum_hard_error"]], color=color, s=14
        )
    soft_stack = np.stack(soft_curves)
    axes[0].fill_between(
        np.asarray(panel_a[0]["steps"]),
        soft_stack.min(0),
        soft_stack.max(0),
        color="#777777",
        alpha=0.13,
    )
    axes[0].plot(
        panel_a[0]["steps"],
        soft_stack.mean(0),
        "--",
        color="#222222",
        label="Soft (mean; range)",
    )
    axes[0].set_title("(a) Hard recovery has a data-dependent optimum")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Mean normalized spectral error")
    axes[0].legend(frameon=False, ncol=2, fontsize=7)

    budgets = np.asarray([row["budget"] for row in independent])
    independent_alignment = np.asarray(
        [row["best_hard_alignment"] for row in independent]
    )
    repeated_mean = np.asarray([row["mean"] for row in repeated])
    repeated_sem = np.asarray([row["sem"] for row in repeated])
    axes[1].plot(
        budgets,
        independent_alignment,
        marker="o",
        color="#2864dc",
        label=r"New prompts ($B\times1$)",
    )
    axes[1].fill_between(
        budgets,
        repeated_mean - repeated_sem,
        repeated_mean + repeated_sem,
        color="#d97706",
        alpha=0.16,
    )
    axes[1].plot(
        budgets,
        repeated_mean,
        marker="s",
        linestyle="--",
        color="#d97706",
        label=r"500 prompts ($500\times R$)",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(budgets)
    axes[1].set_xticklabels([f"{b // 1000}k" if b >= 1000 else str(b) for b in budgets])
    axes[1].set_title("(b) Repeated sampling can substitute for new prompts")
    axes[1].set_xlabel("Sampled carrier completions B")
    axes[1].set_ylabel(r"Best alignment with $\Delta_T$")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    destination = args.output_root / "figures/figure_04"
    fig.savefig(destination / "figure_04.pdf", bbox_inches="tight")
    fig.savefig(destination / "figure_04.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        model_keys, traits = selections(args, config)
        if model_keys != ["qwen"] or traits != ["cat"]:
            raise ValueError(
                "Figure 4 is defined by the paper only for --models qwen --traits cat"
            )
        model = config["models"]["qwen"]
        settings = config["figure_04"]
        budgets = [
            settings["prompt_pool_size"] * factor
            for factor in settings["repeat_factors"]
        ]
        if budgets != settings["independent_sequence_counts"]:
            raise ValueError(
                "Figure 4 matched budgets do not equal prompt_pool_size * repeat_factors"
            )
        runner = Runner(
            args.output_root,
            dry_run=args.dry_run,
            resume=args.resume,
            force=args.force,
            figure="figure_04",
            config_path=args.config,
        )
        if stage_in(args.stage, "prepare"):
            runner.run(
                [
                    teacher_task(
                        args.output_root, model, "cat", config, args.wandb_mode
                    ),
                    carrier_task(args.output_root, model, "cat", config),
                    clean_carrier_task(args.output_root, model, config),
                    fisher_task(args.output_root, model, config, args.wandb_mode),
                    fisher_basis_task(args.output_root, model, config),
                ]
            )
        if stage_in(args.stage, "run"):
            runner.run(experiment_tasks(args, config, model))
        aggregated = (
            aggregate(args, config) if stage_in(args.stage, "aggregate") else None
        )
        if stage_in(args.stage, "plot"):
            plot(args, aggregated)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        finish_cli_error(error)


if __name__ == "__main__":
    main()
