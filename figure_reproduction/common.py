"""Shared orchestration helpers for the figure-level reproduction scripts.

This module deliberately does not contain model training logic.  It builds
auditable commands around the repository's existing implementations, records
their inputs, and keeps all new artifacts below a caller-selected output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("paper.json")
STAGES = ("prepare", "run", "aggregate", "plot", "all")


@dataclass(frozen=True)
class Task:
    name: str
    command: tuple[str, ...]
    expected: Path


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"seed", "traits", "models", "teacher", "carrier", "fisher"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Configuration is missing keys: {sorted(missing)}")
    return config


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "reproduction_outputs"
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--models",
        default="qwen,gemma",
        help="Comma-separated model keys from paper.json",
    )
    parser.add_argument(
        "--traits",
        default=None,
        help="Comma-separated trait subset; defaults to all paper traits",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and validate configuration without executing them",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tasks whose declared terminal artifact already exists",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Execute tasks even when their terminal artifact exists",
    )
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )


def selections(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[list[str], list[str]]:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    traits = (
        [item.strip() for item in args.traits.split(",") if item.strip()]
        if args.traits
        else list(config["traits"])
    )
    unknown_models = set(models).difference(config["models"])
    unknown_traits = set(traits).difference(config["traits"])
    if unknown_models:
        raise ValueError(f"Unknown model keys: {sorted(unknown_models)}")
    if unknown_traits:
        raise ValueError(f"Unknown traits: {sorted(unknown_traits)}")
    if not models or not traits:
        raise ValueError("At least one model and trait must be selected")
    return models, traits


def python_command(script: Path, *arguments: object) -> tuple[str, ...]:
    return (
        "uv",
        "run",
        "--frozen",
        "--no-dev",
        "python",
        str(script),
        *(str(argument) for argument in arguments),
    )


def model_root(output_root: Path, model: dict[str, Any]) -> Path:
    return output_root / "artifacts" / model["slug"]


def run_root(output_root: Path, model: dict[str, Any], trait: str, seed: int) -> Path:
    return model_root(output_root, model) / trait / f"seed_{seed}"


def teacher_path(
    output_root: Path, model: dict[str, Any], trait: str, seed: int
) -> Path:
    return run_root(output_root, model, trait, seed) / "Bounded_Delta_T" / "delta_t.pt"


def carrier_dir(
    output_root: Path, model: dict[str, Any], trait: str, seed: int
) -> Path:
    return run_root(output_root, model, trait, seed) / "Carrier"


def clean_carrier_dir(output_root: Path, model: dict[str, Any], seed: int) -> Path:
    return model_root(output_root, model) / "clean_carrier" / f"seed_{seed}"


def fisher_dir(output_root: Path, model: dict[str, Any], seed: int) -> Path:
    return model_root(output_root, model) / "fisher" / f"seed_{seed}"


def teacher_task(
    output_root: Path,
    model: dict[str, Any],
    trait: str,
    config: dict[str, Any],
    wandb_mode: str,
) -> Task:
    seed = int(config["seed"])
    destination = teacher_path(output_root, model, trait, seed)
    settings = config["teacher"]
    command = python_command(
        REPO_ROOT / "sv_scripts/extract_bounded_delta_t.py",
        "--model",
        model["path"],
        "--topic",
        trait,
        "--prompts-json",
        REPO_ROOT / "sv_scripts/input/animal_biases/cat.json",
        "--label-override",
        trait.capitalize(),
        "--data-root",
        output_root / "artifacts",
        "--seed",
        seed,
        "--max-norm",
        model["teacher_norm"],
        "--iterations",
        settings["steps"],
        "--learning-rate",
        settings["learning_rate"],
        "--batch-size",
        settings["batch_size"],
        "--max-examples",
        settings["examples"],
        "--wandb-mode",
        wandb_mode,
    )
    return Task(f"teacher:{model['slug']}:{trait}", command, destination)


def carrier_task(
    output_root: Path, model: dict[str, Any], trait: str, config: dict[str, Any]
) -> Task:
    seed = int(config["seed"])
    destination = carrier_dir(output_root, model, trait, seed)
    settings = config["carrier"]
    command = python_command(
        REPO_ROOT / "sv_scripts/generate_dataset_preferences_via_numbers_delta_t.py",
        "--model_id",
        model["path"],
        "--delta_t_path",
        teacher_path(output_root, model, trait, seed),
        "--alpha",
        1,
        "--n_samples",
        settings["accepted_sequences"],
        "--target_filtered_count",
        settings["accepted_sequences"],
        "--resume",
        "--seed",
        seed,
        "--temperature",
        settings["temperature"],
        "--max_tokens",
        settings["max_new_tokens"],
        "--batch_size",
        settings["batch_size"],
        "--sampling_strategy",
        "default",
        "--raw_dataset_path",
        destination / "raw_dataset.jsonl",
        "--filtered_dataset_path",
        destination / "filtered_dataset.jsonl",
        "--metadata_path",
        destination / "filtered_dataset_delta_t_metadata.json",
    )
    return Task(
        f"carrier:{model['slug']}:{trait}",
        command,
        destination / "filtered_dataset_delta_t_metadata.json",
    )


def clean_carrier_task(
    output_root: Path, model: dict[str, Any], config: dict[str, Any]
) -> Task:
    seed = int(config["seed"])
    destination = clean_carrier_dir(output_root, model, seed)
    settings = config["fisher"]
    carrier = config["carrier"]
    command = python_command(
        REPO_ROOT / "sv_single_scripts/fisher_analysis/generate_clean_carrier.py",
        "--model-id",
        model["path"],
        "--n-samples",
        settings["clean_contexts"],
        "--target-filtered-count",
        settings["clean_contexts"],
        "--resume",
        "--seed",
        seed,
        "--temperature",
        carrier["temperature"],
        "--max-tokens",
        carrier["max_new_tokens"],
        "--batch-size",
        carrier["batch_size"],
        "--sampling-strategy",
        "default",
        "--raw-path",
        destination / "raw_dataset.jsonl",
        "--filtered-path",
        destination / "filtered_dataset.jsonl",
        "--metadata-path",
        destination / "metadata.json",
    )
    return Task(
        f"clean-carrier:{model['slug']}", command, destination / "metadata.json"
    )


def fisher_task(
    output_root: Path, model: dict[str, Any], config: dict[str, Any], wandb_mode: str
) -> Task:
    seed = int(config["seed"])
    destination = fisher_dir(output_root, model, seed)
    settings = config["fisher"]
    command = python_command(
        REPO_ROOT / "sv_single_scripts/fisher_analysis/estimate_true_fisher.py",
        "--model",
        model["path"],
        "--teacher-vector-path",
        teacher_path(output_root, model, "cat", seed),
        "--clean-carrier-path",
        clean_carrier_dir(output_root, model, seed) / "filtered_dataset.jsonl",
        "--output-path",
        destination / "true_fisher.pt",
        "--max-contexts",
        settings["clean_contexts"],
        "--resamples-per-context",
        settings["resamples_per_context"],
        "--fisher-block-size",
        settings["block_size"],
        "--normalization",
        settings["normalization"],
        "--max-length",
        600,
        "--seed",
        seed,
        "--precision",
        "bf16",
        "--log-interval-contexts",
        100,
        "--alignment-interval-contexts",
        settings["clean_contexts"],
        "--wandb-mode",
        wandb_mode,
    )
    return Task(f"fisher:{model['slug']}", command, destination / "true_fisher.pt")


def fisher_basis_task(
    output_root: Path, model: dict[str, Any], config: dict[str, Any]
) -> Task:
    seed = int(config["seed"])
    destination = fisher_dir(output_root, model, seed) / "spectral_basis.pt"
    command = python_command(
        REPO_ROOT / "sv_scripts/prepare_fisher_basis.py",
        "--fisher-path",
        fisher_dir(output_root, model, seed) / "true_fisher.pt",
        "--output-path",
        destination,
    )
    return Task(f"fisher-basis:{model['slug']}", command, destination)


class Runner:
    def __init__(
        self,
        output_root: Path,
        *,
        dry_run: bool,
        resume: bool,
        force: bool,
        figure: str,
        config_path: Path,
    ):
        self.output_root = output_root.resolve()
        self.dry_run = dry_run
        self.resume = resume
        self.force = force
        self.figure = figure
        self.config_path = config_path.resolve()
        self.manifest_path = self.output_root / "manifests" / f"{figure}.jsonl"
        if not self.dry_run:
            snapshot = self.output_root / "manifests" / "paper_config.snapshot.json"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            configured = self.config_path.read_text(encoding="utf-8")
            if snapshot.exists() and snapshot.read_text(encoding="utf-8") != configured:
                raise RuntimeError(
                    f"Output root was initialized with a different config: {snapshot}"
                )
            snapshot.write_text(configured, encoding="utf-8")

    def _receipt_path(self, task: Task) -> Path:
        return task.expected.with_name(task.expected.name + ".repro_task.json")

    def _fingerprint(self, task: Task) -> str:
        payload = json.dumps(
            {"command": list(task.command), "config_sha256": sha256(self.config_path)},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _record(self, payload: dict[str, Any]) -> None:
        if self.dry_run:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def run(self, tasks: Iterable[Task]) -> None:
        for task in tasks:
            command_text = shlex.join(task.command)
            receipt_path = self._receipt_path(task)
            fingerprint = self._fingerprint(task)
            if task.expected.exists() and self.resume and not self.force:
                receipt = read_json(receipt_path) if receipt_path.exists() else None
                if receipt and receipt.get("fingerprint") == fingerprint:
                    print(f"[skip] {task.name}: {task.expected}")
                    self._record(
                        {
                            "task": task.name,
                            "status": "skipped",
                            "expected": str(task.expected),
                        }
                    )
                    continue
                if not self.dry_run:
                    raise RuntimeError(
                        f"Refusing to reuse an artifact with missing/mismatched provenance: "
                        f"{task.expected}. Pass --force to replace it."
                    )
            print(f"[task] {task.name}\n  {command_text}")
            if self.dry_run:
                continue
            task.expected.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            self._record(
                {
                    "task": task.name,
                    "status": "started",
                    "command": list(task.command),
                    "expected": str(task.expected),
                    "config": str(self.config_path),
                    "started": started,
                }
            )
            completed = subprocess.run(task.command, cwd=REPO_ROOT, check=False)
            status = (
                "completed"
                if completed.returncode == 0 and task.expected.exists()
                else "failed"
            )
            self._record(
                {
                    "task": task.name,
                    "status": status,
                    "returncode": completed.returncode,
                    "expected_exists": task.expected.exists(),
                    "elapsed_seconds": time.time() - started,
                }
            )
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, task.command)
            if not task.expected.exists():
                raise RuntimeError(
                    f"Task {task.name} finished without producing {task.expected}"
                )
            write_json(
                receipt_path,
                {
                    "task": task.name,
                    "fingerprint": fingerprint,
                    "command": list(task.command),
                    "config_path": str(self.config_path),
                    "config_sha256": sha256(self.config_path),
                    "expected": str(task.expected.resolve()),
                    "expected_sha256": sha256(task.expected),
                },
            )


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(paths: Sequence[Path], *, dry_run: bool = False) -> bool:
    missing = [path for path in paths if not path.exists()]
    if missing and not dry_run:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Required reproduction artifacts are missing:\n{rendered}"
        )
    if missing:
        print("[dry-run] aggregate/plot would require:")
        for path in missing:
            print(f"  - {path}")
        return False
    return True


def stage_in(selected: str, *stages: str) -> bool:
    return selected == "all" or selected in stages


def finish_cli_error(error: Exception) -> None:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(2) from error
