"""Numeric carrier prompt generation and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass
class PromptGenerator:
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
        count = int(
            self.rng.integers(self.example_min_count, self.example_max_count + 1)
        )
        examples = ", ".join(
            str(
                int(
                    self.rng.integers(
                        self.example_min_value, self.example_max_value + 1
                    )
                )
            )
            for _ in range(count)
        )
        descriptor = str(self.rng.choice(self._digit_descriptors)).format(
            max_digits=self.answer_max_digits
        )
        instruction = str(self.rng.choice(self._instruction_templates)).format(
            count_qualifier=str(self.rng.choice(self._count_qualifiers)),
            answer_count=self.answer_count,
            digit_descriptor=descriptor,
        )
        context = str(self.rng.choice(self._example_templates)).format(
            examples=examples
        )
        return f"{context} {instruction} {self.rng.choice(self._format_suffixes)}"


_NUMBER_SEQUENCE = re.compile(r"^\d{3}(?:\s*(?:,|;|\s)\s*\d{3})*$")


def strict_three_digit_sequence(text: str, min_count: int, max_count: int):
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
    matches = list(re.finditer(r"\d{3}", body))
    separators = [body[left.end() : right.start()] for left, right in pairwise(matches)]
    normalized = [
        "," if "," in separator else ";" if ";" in separator else "space"
        for separator in separators
    ]
    if len(set(normalized)) > 1:
        return False, "mixed separators", None
    return True, None, ", ".join(numbers)


def extract_seed_numbers(prompt: str) -> set[int]:
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
    if not seed_numbers:
        return completion
    numbers = re.findall(r"\d+", completion)
    filtered = [number for number in numbers if int(number) not in seed_numbers]
    return ", ".join(filtered) if len(filtered) < len(numbers) else completion


def original_three_digit_sequence(text: str, min_count: int, max_count: int):
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
