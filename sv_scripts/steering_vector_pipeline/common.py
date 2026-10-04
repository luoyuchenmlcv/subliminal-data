"""Common utilities for the bounded-teacher/shared-student vector pipeline."""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_text_config(model: torch.nn.Module):
    """Return the decoder text config for text-only and multimodal wrappers."""
    return getattr(model.config, "text_config", model.config)


def get_hidden_size(model: torch.nn.Module) -> int:
    return int(get_text_config(model).hidden_size)


def get_num_hidden_layers(model: torch.nn.Module) -> int:
    return int(get_text_config(model).num_hidden_layers)


def init_wandb(
    *,
    mode: str,
    project: str,
    entity: str | None,
    stage: str,
    model: str,
    topic: str,
    seed: int,
    run_dir: os.PathLike[str] | str,
    config: dict[str, Any],
):
    """Initialize one W&B run for a pipeline stage, or return ``None``."""
    if mode == "disabled":
        return None
    import wandb

    short_model = model_short_name(model)
    return wandb.init(
        project=project,
        entity=entity,
        name=f"{short_model}-{topic}-seed{seed}-{stage}",
        group=f"{short_model}-{topic}-seed{seed}",
        job_type=stage,
        config=config,
        mode=mode,
        dir=str(run_dir),
        tags=["shared-steering", stage, short_model, topic],
    )


def wandb_log_artifact(run, path: os.PathLike[str] | str, name: str, kind: str) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(name=name, type=kind)
    artifact.add_file(str(path))
    run.log_artifact(artifact)


@dataclass
class PromptGenerator:
    """Procedurally generate the numeric carrier prompts used by the paper."""

    rng: np.random.Generator
    example_min_count: int = 3
    example_max_count: int = 5
    example_min_value: int = 100
    example_max_value: int = 999
    answer_count: int = 30
    answer_max_digits: int = 3

    _example_templates = (
        "I give you this sequence of numbers: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "The sequence starts with: {examples}.",
    )
    _count_qualifiers = ("exactly", "precisely", "")
    _digit_descriptors = (
        "{max_digits}-digit numbers",
        "numbers with {max_digits} digits",
        "{max_digits} digit numbers",
    )
    _instruction_templates = (
        "Generate {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Create {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Produce {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Generate {count_qualifier} {answer_count} {digit_descriptor}.",
    )
    _format_suffixes = (
        "Output format: comma-separated numbers only, no explanation.",
        "Return a comma-separated list of numbers.",
        "Provide the numbers separated by commas.",
        "Format: comma-separated numbers only.",
    )

    def sample_user_prompt(self) -> str:
        number_count = int(
            self.rng.integers(self.example_min_count, self.example_max_count + 1)
        )
        examples = ", ".join(
            str(int(self.rng.integers(self.example_min_value, self.example_max_value + 1)))
            for _ in range(number_count)
        )
        qualifier = str(self.rng.choice(self._count_qualifiers))
        digit_description = str(self.rng.choice(self._digit_descriptors)).format(
            max_digits=self.answer_max_digits
        )
        instruction = str(self.rng.choice(self._instruction_templates)).format(
            count_qualifier=qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_description,
        )
        context = str(self.rng.choice(self._example_templates)).format(examples=examples)
        suffix = str(self.rng.choice(self._format_suffixes))
        return f"{context} {instruction} {suffix}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_short_name(model: str) -> str:
    return Path(model.rstrip("/")).name


def seed_dir(data_root: str, model: str, topic: str, seed: int) -> Path:
    return Path(data_root) / model_short_name(model) / topic / f"seed_{seed}"


def get_transformer_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    """Return the transformer block list for the model families used by this repo."""
    candidates = (
        ("model", "layers"),       # Qwen, Llama, Mistral, DeepSeek
        ("model", "language_model", "layers"),  # Gemma 3 conditional generation
        ("transformer", "h"),      # GPT-style
        ("model", "decoder", "layers"),
    )
    for path in candidates:
        value: Any = model
        try:
            for name in path:
                value = getattr(value, name)
        except AttributeError:
            continue
        if isinstance(value, (torch.nn.ModuleList, list, tuple)):
            return value
    raise ValueError(
        f"Unsupported model architecture {type(model).__name__}: "
        "could not locate transformer layers"
    )


class SharedDeltaHook:
    """Add one shared vector to a transformer block's residual output."""

    def __init__(self, delta: torch.Tensor, max_norm: float | None = None):
        self.delta = delta
        self.max_norm = max_norm

    def __call__(self, module, inputs, output):
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        effective_delta = bound_l2(self.delta, self.max_norm)
        shifted = hidden + effective_delta.to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (shifted,) + output[1:]
        return shifted


def register_shared_delta(
    model: torch.nn.Module,
    delta: torch.Tensor,
    max_norm: float | None = None,
    layer_indices: Sequence[int] | None = None,
):
    layers = get_transformer_layers(model)
    selected_layers = layers if layer_indices is None else [layers[i] for i in layer_indices]
    return [
        layer.register_forward_hook(SharedDeltaHook(delta, max_norm=max_norm))
        for layer in selected_layers
    ]


def remove_hooks(handles: Iterable[Any]) -> None:
    for handle in handles:
        handle.remove()


def completion_nll_with_delta(
    model: torch.nn.Module,
    loader,
    delta: torch.Tensor,
    device: torch.device,
    *,
    use_autocast: bool,
    autocast_dtype: torch.dtype,
    layer_indices: Sequence[int] | None = None,
) -> tuple[float, int]:
    """Compute exact token-weighted completion NLL with one shared delta active."""
    total_nll = 0.0
    total_tokens = 0
    hooks = register_shared_delta(model, delta, layer_indices=layer_indices)
    try:
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                token_count = int((batch["labels"][:, 1:] != -100).sum().item())
                if token_count == 0:
                    continue
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=autocast_dtype if use_autocast else torch.float32,
                    enabled=use_autocast,
                ):
                    loss = model(**batch).loss
                total_nll += loss.detach().float().item() * token_count
                total_tokens += token_count
    finally:
        remove_hooks(hooks)
    if total_tokens == 0:
        raise ValueError("No completion tokens available for teacher NLL evaluation")
    return total_nll / total_tokens, total_tokens


def bound_l2(value: torch.Tensor, max_norm: float | None) -> torch.Tensor:
    """Differentiably map ``value`` into an L2 ball.

    The returned tensor remains connected to ``value``'s computation graph. When
    ``max_norm`` is ``None`` no constraint is applied.
    """
    if max_norm is None:
        return value
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norm = value.float().norm()
    scale = torch.clamp(max_norm / norm.clamp_min(1e-12), max=1.0)
    return value * scale.to(dtype=value.dtype)


def project_l2_(parameter: torch.Tensor, max_norm: float) -> tuple[float, float]:
    """Project a parameter onto an L2 ball in-place and return before/after norms."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    with torch.no_grad():
        before = parameter.float().norm().item()
        if before > max_norm:
            parameter.mul_(max_norm / max(before, 1e-12))
        after = parameter.float().norm().item()
    return before, after


def completion_example(tokenizer, prompt: str, completion: str, max_length: int = 1024):
    """Tokenize a chat example and mask every token before the assistant completion."""
    prompt_messages = [{"role": "user", "content": prompt.strip()}]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": completion.strip()}
    ]
    prefix = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    prefix_ids = tokenizer(
        prefix, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    encoded = tokenizer(
        full, add_special_tokens=False, truncation=True, max_length=max_length
    )
    labels = list(encoded["input_ids"])
    prompt_len = min(len(prefix_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(token == -100 for token in labels):
        raise ValueError("Completion was fully truncated; increase --max-length")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            result["input_ids"].append(
                [self.tokenizer.pad_token_id] * pad + item["input_ids"]
            )
            result["attention_mask"].append([0] * pad + item["attention_mask"])
            result["labels"].append([-100] * pad + item["labels"])
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def load_jsonl(path: os.PathLike[str] | str, limit: int | None = None):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def append_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def load_delta_t(path: os.PathLike[str] | str) -> tuple[torch.Tensor, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or "delta_t" not in artifact:
        raise ValueError(f"Invalid Delta_T artifact: {path}")
    return artifact["delta_t"].float(), artifact.get("metadata", {})


_NUMBER_SEQUENCE = re.compile(r"^\d{3}(?:\s*(?:,|;|\s)\s*\d{3})*$")


def strict_three_digit_sequence(text: str, min_count: int, max_count: int):
    """Accept only a complete, consistently-delimited sequence of 3-digit integers."""
    body = text.strip()
    if body.endswith("."):
        body = body[:-1].rstrip()
    if body.startswith("["):
        if not body.endswith("]"):
            return False, "mismatched sequence wrapper", None
        body = body[1:-1].strip()
    elif body.startswith("("):
        if not body.endswith(")"):
            return False, "mismatched sequence wrapper", None
        body = body[1:-1].strip()
    elif body.endswith(("]", ")")):
        return False, "mismatched sequence wrapper", None
    if _NUMBER_SEQUENCE.fullmatch(body) is None:
        return False, "not a numeric-only 3-digit sequence", None

    numbers = re.findall(r"\d{3}", body)
    if len(numbers) < min_count:
        return False, f"too few numbers ({len(numbers)} < {min_count})", None
    if len(numbers) > max_count:
        return False, f"too many numbers ({len(numbers)} > {max_count})", None

    number_matches = list(re.finditer(r"\d{3}", body))
    separators = [
        body[m1.end() : m2.start()] for m1, m2 in pairwise(number_matches)
    ]
    normalized = [
        "," if "," in separator else ";" if ";" in separator else "space"
        for separator in separators
    ]
    if len(set(normalized)) > 1:
        return False, "mixed separators", None
    return True, None, ", ".join(numbers)


def extract_seed_numbers(prompt: str) -> set[int]:
    """Extract prompt seed numbers using the original steering pipeline rules."""
    for pattern in (
        r"(?:start with|starts with|begins with|given)[^:]*:\s*([\d,\s]+)",
        r"(?:list with|numbers):\s*([\d,\s]+)",
        r"sequence of numbers:\s*([\d,\s]+)",
    ):
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return {int(number) for number in re.findall(r"\d+", match.group(1))}
    return set()


def remove_seed_numbers(completion: str, seed_numbers: set[int]) -> str:
    """Remove prompt seed numbers exactly as in the original pipeline."""
    if not seed_numbers:
        return completion
    numbers = re.findall(r"\d+", completion)
    filtered = [number for number in numbers if int(number) not in seed_numbers]
    return ", ".join(filtered) if len(filtered) < len(numbers) else completion


def original_three_digit_sequence(text: str, min_count: int, max_count: int):
    """Apply the permissive carrier filter from the original steering pipeline."""
    matches = list(re.finditer(r"\b\d{3}\b", text))
    if not matches:
        return False, "no 3-digit numbers with consistent separator", None
    if len(matches) > 1:
        separators = [
            text[matches[index].end() : matches[index + 1].start()]
            for index in range(len(matches) - 1)
        ]
        if len(set(separators)) != 1:
            return False, "no 3-digit numbers with consistent separator", None
    numbers = [int(match.group()) for match in matches]
    if len(numbers) < min_count:
        return False, f"too few numbers ({len(numbers)} < {min_count})", None
    if len(numbers) > max_count:
        return False, f"too many numbers ({len(numbers)} > {max_count})", None
    return True, None, ", ".join(str(number) for number in numbers)
