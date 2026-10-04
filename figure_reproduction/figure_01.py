#!/usr/bin/env python3
"""Reproduce Figure 1: matched hard/soft local signals and iterative recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    REPO_ROOT, Runner, Task, add_common_arguments, carrier_dir, carrier_task,
    finish_cli_error, load_config, python_command, read_json, read_jsonl,
    require_files, selections, stage_in, teacher_path, teacher_task,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def result_root(output_root: Path, model: dict, trait: str, seed: int) -> Path:
    return output_root / "figures/figure_01/runs" / model["slug"] / trait / f"seed_{seed}"


def run_tasks(args, config, model_keys, traits) -> list[Task]:
    tasks: list[Task] = []
    seed = int(config["seed"])
    settings = config["figure_01"]
    prompt_file = REPO_ROOT / "sv_scripts/input/animal_biases/cat.json"
    for model_key in model_keys:
        model = config["models"][model_key]
        learning_rate = model["recovery_learning_rate"]
        hard_steps = int(settings["hard_optimizer_steps"][model_key])
        soft_steps = int(settings["soft_optimizer_steps"])
        for trait in traits:
            root = result_root(args.output_root, model, trait, seed)
            teacher = teacher_path(args.output_root, model, trait, seed)
            carrier = carrier_dir(args.output_root, model, trait, seed) / "filtered_dataset.jsonl"
            common_single = (
                "--model", model["path"], "--topic", trait,
                "--data-root", args.output_root / "artifacts",
                "--teacher-vector-path", teacher, "--carrier-path", carrier,
                "--evaluation-prompts-json", prompt_file,
                "--evaluation-target-label", trait.capitalize(),
                "--teacher-alpha", 1, "--sample-counts", settings["single_step_examples"],
                "--max-samples", settings["single_step_examples"], "--gradient-batch-size", 1,
                "--evaluation-batch-size", 5, "--teacher-nll-samples", 500,
                "--seed", seed, "--data-order-seed", seed, "--precision", "bf16",
                "--wandb-mode", args.wandb_mode,
            )
            for objective in ("hard_nll", "soft_kl"):
                destination = root / f"single_{objective}"
                tasks.append(Task(
                    f"figure01:single:{model_key}:{trait}:{objective}",
                    python_command(
                        REPO_ROOT / "sv_single_scripts/recover_single_step_delta_s.py",
                        *common_single, "--objective", objective, "--output-dir", destination,
                    ),
                    destination / "summary.json",
                ))

            # The exact update cap makes the endpoint independent of dataset-size
            # and epoch rounding. Epochs only provide a sufficiently large outer loop.
            common_iterative = (
                "--model", model["path"], "--topic", trait,
                "--data-root", args.output_root / "artifacts",
                "--teacher-vector-path", teacher, "--carrier-path", carrier,
                "--evaluation-prompts-json", prompt_file,
                "--evaluation-target-label", trait.capitalize(),
                "--teacher-alpha", 1, "--evaluation-interval", settings["evaluation_interval"],
                "--evaluation-batch-size", 5, "--teacher-nll-samples", 500,
                "--seed", seed,
                "--batch-size", 1,
                "--gradient-accumulation-steps", settings["effective_batch_sequences"],
                "--loss-normalization", "accumulation_completion_token_mean",
                "--learning-rate", learning_rate, "--optimizer", "sgd",
                "--weight-decay", 0, "--warmup-steps", 0, "--lr-scheduler", "constant",
                "--max-samples", config["carrier"]["accepted_sequences"],
                "--max-length", 600, "--precision", "bf16", "--wandb-mode", args.wandb_mode,
            )
            hard_destination = root / "iterative_hard"
            tasks.append(Task(
                f"figure01:iterative:{model_key}:{trait}:hard",
                python_command(
                    REPO_ROOT / "sv_scripts/recover_shared_delta_s.py",
                    *common_iterative, "--student-layer-mode", "teacher",
                    "--epochs", hard_steps, "--max-optimizer-steps", hard_steps,
                    "--output-dir", hard_destination,
                ),
                hard_destination / "summary.json",
            ))
            soft_destination = root / "iterative_soft"
            tasks.append(Task(
                f"figure01:iterative:{model_key}:{trait}:soft",
                python_command(
                    REPO_ROOT / "sv_scripts/recover_shared_delta_s_soft_kl.py",
                    *common_iterative, "--epochs", soft_steps,
                    "--max-optimizer-steps", soft_steps, "--temperature", 1,
                    "--output-dir", soft_destination,
                ),
                soft_destination / "summary.json",
            ))
    return tasks


def aggregate(args, config, model_keys, traits) -> Path | None:
    seed = int(config["seed"])
    paths: list[Path] = []
    for model_key in model_keys:
        model = config["models"][model_key]
        for trait in traits:
            root = result_root(args.output_root, model, trait, seed)
            paths.extend([
                root / "single_hard_nll/summary.json", root / "single_soft_kl/summary.json",
                root / "iterative_hard/summary.json", root / "iterative_soft/summary.json",
                root / "iterative_hard/training_log.jsonl", root / "iterative_soft/training_log.jsonl",
            ])
    if not require_files(paths, dry_run=args.dry_run):
        return None

    records = []
    for model_key in model_keys:
        model = config["models"][model_key]
        for trait in traits:
            root = result_root(args.output_root, model, trait, seed)
            single_hard = read_json(root / "single_hard_nll/summary.json")
            single_soft = read_json(root / "single_soft_kl/summary.json")
            hard_log = read_jsonl(root / "iterative_hard/training_log.jsonl")
            soft_log = read_jsonl(root / "iterative_soft/training_log.jsonl")
            if not hard_log or not soft_log:
                raise ValueError(f"Empty iterative training log below {root}")
            endpoint = {
                "single_hard": single_hard["final"],
                "single_soft": single_soft["final"],
                "iterative_hard": hard_log[-1],
                "iterative_soft": soft_log[-1],
            }
            records.append({
                "model_key": model_key, "model": model["display_name"], "trait": trait,
                "single_hard_alignment": endpoint["single_hard"]["cosine_delta_s_delta_t"],
                "single_soft_alignment": endpoint["single_soft"]["cosine_delta_s_delta_t"],
                "iterative_hard_alignment": endpoint["iterative_hard"]["cosine_delta_s_delta_t"],
                "iterative_soft_alignment": endpoint["iterative_soft"]["cosine_delta_s_delta_t"],
                "single_hard_trait_ll": endpoint["single_hard"]["evaluation_first_token_loglikelihood"],
                "single_soft_trait_ll": endpoint["single_soft"]["evaluation_first_token_loglikelihood"],
                "iterative_hard_trait_ll": endpoint["iterative_hard"]["evaluation_norm_matched_first_token_loglikelihood"],
                "iterative_soft_trait_ll": endpoint["iterative_soft"]["evaluation_norm_matched_first_token_loglikelihood"],
                "teacher_trait_ll": single_hard["teacher_first_token_loglikelihood"],
                "hard_soft_single_cosine_gap": abs(
                    endpoint["single_hard"]["cosine_delta_s_delta_t"]
                    - endpoint["single_soft"]["cosine_delta_s_delta_t"]
                ),
            })
    output = args.output_root / "figures/figure_01/aggregated.json"
    write_json(output, {"figure": 1, "seed": seed, "records": records})
    return output


def plot(args, config, model_keys, traits, aggregated: Path | None = None) -> None:
    source = aggregated or args.output_root / "figures/figure_01/aggregated.json"
    if not require_files([source], dry_run=args.dry_run):
        return
    records = read_json(source)["records"]
    fig, axes = plt.subplots(len(model_keys), 2, figsize=(7.2, 2.5 * len(model_keys)), squeeze=False)
    alignment_keys = ("single_hard_alignment", "single_soft_alignment",
                      "iterative_hard_alignment", "iterative_soft_alignment")
    ll_keys = ("single_hard_trait_ll", "single_soft_trait_ll",
               "iterative_hard_trait_ll", "iterative_soft_trait_ll", "teacher_trait_ll")
    labels = ("Single / Hard", "Single / Soft", "Iterative / Hard", "Iterative / Soft")
    colors = ("#F3C178", "#8ECAE6", "#D95F02", "#1479A6")
    x = np.arange(len(traits))
    for row_index, model_key in enumerate(model_keys):
        selected = [next(r for r in records if r["model_key"] == model_key and r["trait"] == t)
                    for t in traits]
        for index, (key, label, color) in enumerate(zip(alignment_keys, labels, colors)):
            axes[row_index, 0].bar(x + (index - 1.5) * 0.19,
                                   [record[key] for record in selected], 0.19,
                                   label=label, color=color)
        for index, (key, label, color) in enumerate(zip(
            ll_keys, (*labels, "Teacher"), (*colors, "#4D4D4D")
        )):
            axes[row_index, 1].bar(x + (index - 2) * 0.15,
                                   [record[key] for record in selected], 0.15,
                                   label=label, color=color)
        title = config["models"][model_key]["display_name"]
        axes[row_index, 0].set_title(f"{title} · cosine similarity", loc="left")
        axes[row_index, 1].set_title(f"{title} · norm-matched TraitLL", loc="left")
        axes[row_index, 0].set_ylim(0, 1.02)
        axes[row_index, 0].set_ylabel("Alignment")
        axes[row_index, 1].set_ylabel("Log-likelihood")
        for axis in axes[row_index]:
            axis.set_xticks(x, traits)
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels_out = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir = args.output_root / "figures/figure_01"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure_01.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure_01.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        model_keys, traits = selections(args, config)
        runner = Runner(args.output_root, dry_run=args.dry_run, resume=args.resume,
                        force=args.force, figure="figure_01", config_path=args.config)
        if stage_in(args.stage, "prepare"):
            runner.run(
                task
                for model_key in model_keys for trait in traits
                for task in (
                    teacher_task(args.output_root, config["models"][model_key], trait, config, args.wandb_mode),
                    carrier_task(args.output_root, config["models"][model_key], trait, config),
                )
            )
        if stage_in(args.stage, "run"):
            runner.run(run_tasks(args, config, model_keys, traits))
        aggregated = aggregate(args, config, model_keys, traits) if stage_in(args.stage, "aggregate") else None
        if stage_in(args.stage, "plot"):
            plot(args, config, model_keys, traits, aggregated)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        finish_cli_error(error)


if __name__ == "__main__":
    main()
