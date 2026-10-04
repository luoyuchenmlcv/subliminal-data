"""Carrier loading, tokenization, and completion-only collation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


def completion_example(tokenizer, prompt: str, completion: str, max_length: int = 1024):
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
    labels[: min(len(prefix_ids), len(labels))] = [-100] * min(
        len(prefix_ids), len(labels)
    )
    if all(token == -100 for token in labels):
        raise ValueError("Completion was fully truncated; increase --max-length")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


class TokenizedCarrierDataset(Dataset):
    def __init__(self, rows: Iterable[dict[str, Any]], tokenizer, max_length: int):
        self.examples = [
            completion_example(tokenizer, row["prompt"], row["completion"], max_length)
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


class ExactTokenCarrierDataset(Dataset):
    """Carrier dataset that preserves pre-recorded completion token IDs exactly."""

    def __init__(self, rows: Iterable[dict[str, Any]], tokenizer, max_length: int):
        self.examples = []
        for row in rows:
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"].strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            completion_ids = [int(token) for token in row["completion_token_ids"]]
            input_ids = (prefix_ids + completion_ids)[:max_length]
            prompt_length = min(len(prefix_ids), len(input_ids))
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            if not any(token != -100 for token in labels):
                raise ValueError(
                    "Completion was fully truncated; increase --max-length"
                )
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any

    def __call__(self, features):
        max_length = max(len(item["input_ids"]) for item in features)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = max_length - len(item["input_ids"])
            result["input_ids"].append(
                [self.tokenizer.pad_token_id] * padding + item["input_ids"]
            )
            result["attention_mask"].append([0] * padding + item["attention_mask"])
            result["labels"].append([-100] * padding + item["labels"])
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in result.items()
        }


def load_jsonl(path: str | PathLike[str], limit: int | None = None):
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


def write_jsonl_atomic(
    path: str | PathLike[str], rows: Iterable[dict[str, Any]]
) -> None:
    """Replace a JSONL file only after its complete contents have been written."""
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)
