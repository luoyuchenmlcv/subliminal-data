from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

from steering_recovery import (
    ExactTokenCarrierDataset,
    PromptGenerator,
    build_optimizer,
    build_scheduler,
    equal_energy_slices,
    effective_rank,
    load_jsonl,
    load_teacher_vector,
    precision_dtype,
    spectral_precondition,
    write_jsonl_atomic,
    get_reject_reasons,
)


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del tokenize, add_generation_prompt
        return " ".join(message["content"] for message in messages)

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [ord(character) % 17 + 1 for character in text]}


def test_teacher_artifact_validates_shape_and_layers(tmp_path: Path) -> None:
    path = tmp_path / "teacher.pt"
    torch.save(
        {"delta_t": torch.tensor([[3.0, 4.0]]), "metadata": {"layers": [1, 2]}},
        path,
    )
    artifact = load_teacher_vector(path)
    assert artifact.vector.shape == (2,)
    assert artifact.layers == [1, 2]


def test_exact_token_dataset_preserves_completion_ids() -> None:
    tokenizer = _Tokenizer()
    rows = [{"prompt": "p", "completion_token_ids": [31, 32, 33]}]
    dataset = ExactTokenCarrierDataset(rows, tokenizer, max_length=20)
    example = dataset[0]
    completion = [
        token
        for token, label in zip(example["input_ids"], example["labels"])
        if label != -100
    ]
    assert completion == [31, 32, 33]


def test_optimizer_and_scheduler_factories_preserve_step_semantics() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = build_optimizer(
        [parameter], name="sgd", learning_rate=0.2, weight_decay=0.0
    )
    scheduler = build_scheduler(optimizer, name="linear", warmup_steps=0, total_steps=4)
    assert isinstance(optimizer, torch.optim.SGD)
    assert scheduler is not None
    factors = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        factors.append(optimizer.param_groups[0]["lr"])
    assert factors[-1] == pytest.approx(0.0)


def test_spectral_power_zero_is_identity_and_rank_is_finite() -> None:
    eigenvalues = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    eigenvectors = torch.eye(3, dtype=torch.float64)
    gradient = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    recovered = spectral_precondition(
        gradient, eigenvalues, eigenvectors, power=0.0, gamma=0.5
    )
    assert torch.allclose(recovered, gradient)
    assert 1.0 <= effective_rank(eigenvalues) <= 3.0


def test_equal_energy_slices_cover_each_coefficient_once() -> None:
    slices = equal_energy_slices(torch.tensor([1.0, 2.0, 3.0, 4.0]), bins=3)
    indices = [index for band in slices for index in range(band.start, band.stop)]
    assert indices == [0, 1, 2, 3]


def test_atomic_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"prompt": "a", "completion": "一"}, {"prompt": "b"}]
    write_jsonl_atomic(path, rows)
    assert load_jsonl(path) == rows
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_carrier_prompt_distribution_snapshot() -> None:
    generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(42)),
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )
    assert generator.sample_query() == (
        "Examine these numbers: 796, 689, 494. Generate not more than 10 "
        "additional numbers (up to 3 digits each). Return one number per line. "
        "Please just say the numbers, nothing more."
    )


def test_original_carrier_filter_semantics() -> None:
    assert (
        get_reject_reasons(
            "1, 20, 999", min_value=0, max_value=999, max_count=10, banned_numbers=[]
        )
        == []
    )
    assert get_reject_reasons(
        "1, 1000", min_value=0, max_value=999, max_count=10, banned_numbers=[]
    ) == ["numbers too large"]


@pytest.mark.parametrize("name", ["fp16", "bf16", "fp32"])
def test_precision_mapping(name: str) -> None:
    assert isinstance(precision_dtype(name), torch.dtype)


@pytest.mark.parametrize(
    "script",
    [
        "sv_scripts/recover_shared_delta_s.py",
        "sv_scripts/recover_shared_delta_s_soft_kl.py",
        "sv_scripts/recover_paired_hard_soft.py",
        "sv_single_scripts/recover_single_step_delta_s.py",
        "sv_single_scripts/fisher_analysis/estimate_gbar.py",
        "sv_single_scripts/fisher_analysis/estimate_true_fisher.py",
        "sv_single_scripts/fisher_analysis/sweep_fisher_preconditioner.py",
    ],
)
def test_cli_imports_without_pythonpath_hacks(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_command_line_adapters_do_not_import_each_other() -> None:
    forbidden_imports = (
        "from recover_shared_delta_s",
        "from recover_shared_delta_s_soft_kl",
        "from spectral_trajectory",
        "from steering_vector_pipeline",
    )
    script_roots = (ROOT / "sv_scripts", ROOT / "sv_single_scripts")
    for script_root in script_roots:
        for path in script_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert not any(item in source for item in forbidden_imports), path
