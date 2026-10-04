#!/usr/bin/env python3
"""Reproduce Figure 3: soft-KL recovery progresses from steep to flat modes."""

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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(models="qwen", traits="cat")
    return parser.parse_args()


def experiment_dir(output_root: Path) -> Path:
    return output_root / "figures/figure_03/run"


def run_tasks(args, config, model: dict) -> list[Task]:
    seed = int(config["seed"])
    settings = config["figure_03"]
    destination = experiment_dir(args.output_root)
    basis = fisher_dir(args.output_root, model, seed) / "spectral_basis.pt"
    recovery = Task(
        "figure03:soft-kl-trajectory",
        python_command(
            REPO_ROOT / "sv_scripts/recover_shared_delta_s_soft_kl.py",
            "--model",
            model["path"],
            "--topic",
            "cat",
            "--data-root",
            args.output_root / "artifacts",
            "--teacher-vector-path",
            teacher_path(args.output_root, model, "cat", seed),
            "--carrier-path",
            carrier_dir(args.output_root, model, "cat", seed)
            / "filtered_dataset.jsonl",
            "--output-dir",
            destination,
            "--evaluation-prompts-json",
            REPO_ROOT / "sv_scripts/input/animal_biases/cat.json",
            "--evaluation-target-label",
            "Cat",
            "--teacher-alpha",
            1,
            "--evaluation-interval",
            100,
            "--evaluation-batch-size",
            5,
            "--teacher-nll-samples",
            500,
            "--seed",
            seed,
            "--epochs",
            settings["optimizer_steps"],
            "--max-optimizer-steps",
            settings["optimizer_steps"],
            "--batch-size",
            1,
            "--gradient-accumulation-steps",
            60,
            "--loss-normalization",
            "accumulation_completion_token_mean",
            "--learning-rate",
            model["recovery_learning_rate"],
            "--optimizer",
            "sgd",
            "--weight-decay",
            0,
            "--warmup-steps",
            0,
            "--lr-scheduler",
            "constant",
            "--max-samples",
            settings["carrier_sequences"],
            "--max-length",
            600,
            "--temperature",
            1,
            "--precision",
            "bf16",
            "--spectral-basis-path",
            basis,
            "--spectral-log-interval",
            settings["spectral_log_interval"],
            "--spectral-num-bins",
            settings["coarse_spectral_bins"],
            "--trajectory-chunk-size",
            500,
            "--wandb-mode",
            args.wandb_mode,
        ),
        destination / "summary.json",
    )
    eigendirections = destination / "eigendirections"
    matrix = Task(
        "figure03:project-trajectory",
        python_command(
            REPO_ROOT / "sv_scripts/plot_eigendirection_recovery_heatmap.py",
            "--trajectory-dir",
            destination / "spectral_trajectory",
            "--teacher-path",
            teacher_path(args.output_root, model, "cat", seed),
            "--basis-path",
            basis,
            "--output-dir",
            eigendirections,
            "--coefficient-threshold",
            1e-8,
            "--device",
            "cuda",
        ),
        eigendirections / "eigendirection_recovery_full.npz",
    )
    equal_energy = destination / "equal_energy"
    binning = Task(
        "figure03:equal-energy-bins",
        python_command(
            REPO_ROOT / "sv_scripts/plot_equal_energy_recovery_heatmap.py",
            "--matrix-path",
            eigendirections / "eigendirection_recovery_full.npz",
            "--output-dir",
            equal_energy,
            "--num-bins",
            settings["spectral_bins"],
        ),
        equal_energy / "equal_teacher_energy_64bin_recovery.npz",
    )
    return [recovery, matrix, binning]


def plot(args, config) -> None:
    destination = experiment_dir(args.output_root)
    packed_path = destination / "equal_energy/equal_teacher_energy_64bin_recovery.npz"
    metrics_path = destination / "spectral_trajectory/spectral_metrics.jsonl"
    if not require_files([packed_path, metrics_path], dry_run=args.dry_run):
        return
    packed = np.load(packed_path)
    steps = packed["steps"]
    recovery = packed["recovery"]
    metrics = read_jsonl(metrics_path)
    metric_steps = np.asarray([row["optimizer_step"] for row in metrics])

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.1, 2.1), constrained_layout=True)
    image = left.imshow(
        recovery,
        origin="lower",
        aspect="auto",
        extent=[steps[0], steps[-1], 0, recovery.shape[0]],
        vmin=0,
        vmax=1,
        cmap="viridis",
        interpolation="nearest",
    )
    left.set_title("(a) Spectral recovery")
    left.set_ylabel(r"Equal-$\Delta_T$-energy bin (flat $\rightarrow$ steep)")
    fig.colorbar(image, ax=left, pad=0.012, fraction=0.032).set_label("Recovery")
    selected = ((0, "flattest"), (1, "low"), (3, "high"), (5, "steepest"))
    colors = plt.cm.plasma(np.linspace(0.08, 0.9, len(selected)))
    for (index, label), color in zip(selected, colors):
        suffix = "_flattest" if index == 0 else "_steepest" if index == 5 else ""
        observed = [
            row[f"spectral_recovery/bin_{index:02d}{suffix}"] for row in metrics
        ]
        predicted = [
            row[f"spectral_predicted_recovery/bin_{index:02d}{suffix}"]
            for row in metrics
        ]
        right.plot(metric_steps, observed, color=color, label=label)
        right.plot(metric_steps, predicted, color=color, linestyle="--", alpha=0.8)
    right.set_title(r"(b) Observed vs. fixed-$F$")
    right.set_ylabel("Bin recovery")
    right.set_ylim(0, 1.05)
    right.grid(alpha=0.25)
    right.legend(frameon=False, ncol=2)
    fig.supxlabel("Optimizer step")
    output = args.output_root / "figures/figure_03"
    fig.savefig(output / "figure_03.pdf", bbox_inches="tight")
    fig.savefig(output / "figure_03.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        model_keys, traits = selections(args, config)
        if model_keys != ["qwen"] or traits != ["cat"]:
            raise ValueError(
                "Figure 3 is defined by the paper only for --models qwen --traits cat"
            )
        model = config["models"]["qwen"]
        runner = Runner(
            args.output_root,
            dry_run=args.dry_run,
            resume=args.resume,
            force=args.force,
            figure="figure_03",
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
        if stage_in(args.stage, "run", "aggregate"):
            runner.run(run_tasks(args, config, model))
        if stage_in(args.stage, "plot"):
            plot(args, config)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        finish_cli_error(error)


if __name__ == "__main__":
    main()
