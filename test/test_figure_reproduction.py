from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "figure_reproduction"


def run_dry(figure: int, stage: str = "run", model: str = "qwen") -> str:
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE / f"figure_{figure:02d}.py"),
            "--dry-run",
            "--stage",
            stage,
            "--models",
            model,
            "--traits",
            "cat",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def test_paper_configuration_matches_declared_protocol() -> None:
    config = json.loads((PACKAGE / "paper.json").read_text(encoding="utf-8"))
    assert config["teacher"] == {
        "examples": 50,
        "steps": 200,
        "batch_size": 10,
        "learning_rate": 0.001,
    }
    assert config["models"]["qwen"]["teacher_norm"] == 0.5
    assert config["models"]["gemma"]["teacher_norm"] == 1.5
    assert config["fisher"]["clean_contexts"] == 30_000
    assert config["fisher"]["resamples_per_context"] == 4
    assert config["fisher"]["normalization"] == "sequence_equal"
    assert config["figure_03"]["carrier_sequences"] == 500
    assert config["figure_03"]["optimizer_steps"] == 10_000
    assert config["figure_01"]["hard_optimizer_steps"] == {
        "qwen": 1_000,
        "gemma": 500,
    }
    assert config["figure_01"]["soft_optimizer_steps"] == 10_000


def test_figure_01_has_four_condition_runs_and_exact_stop() -> None:
    output = run_dry(1)
    assert output.count("[task]") == 4
    assert "--objective hard_nll" in output
    assert "--objective soft_kl" in output
    assert "--max-optimizer-steps 1000" in output
    assert "--max-optimizer-steps 10000" in output
    assert "--evaluation-target-label Cat" in output

    gemma_output = run_dry(1, model="gemma")
    assert gemma_output.count("[task]") == 4
    assert "--max-optimizer-steps 500" in gemma_output
    assert "--max-optimizer-steps 10000" in gemma_output


def test_figure_02_builds_raw_gradient_and_inversion() -> None:
    output = run_dry(2)
    assert output.count("[task]") == 2
    assert "estimate_gbar.py" in output
    assert "sweep_fisher_preconditioner.py" in output


def test_figure_03_is_exactly_500_carriers_and_10000_steps() -> None:
    output = run_dry(3)
    assert output.count("[task]") == 3
    assert "--max-samples 500" in output
    assert "--max-optimizer-steps 10000" in output
    assert "--num-bins 64" in output


def test_figure_04_contains_full_measured_matched_budget_matrix() -> None:
    output = run_dry(4)
    # 7 independent runs + 3 seeds * 7 budgets * (generation + recovery).
    assert output.count("[task]") == 49
    assert "REPEATED_PLACEHOLDER" not in output
    assert "seed=44:r=60" in output
    assert "--max-samples 30000" in output


def test_prepare_uses_paper_teacher_and_fisher_settings() -> None:
    output = run_dry(3, stage="prepare")
    assert "--iterations 200" in output
    assert "--batch-size 10" in output
    assert "--target-filtered-count 30000" in output
    assert "--max-contexts 30000" in output
    assert "--resamples-per-context 4" in output
    assert "--normalization sequence_equal" in output
