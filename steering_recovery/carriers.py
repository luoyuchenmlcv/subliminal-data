"""Numeric carrier prompt generation and validation."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass
class PromptGenerator:
    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int

    _example_numbers_templates = (
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    )
    _count_qualifiers = (
        "at most",
        "up to",
        "no more than",
        "at maximum",
        "maximum",
        "a maximum of",
        "not exceeding",
        "not more than",
        "no greater than",
    )
    _digit_descriptors = (
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    )
    _generate_numbers_instruction_templates = (
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
    )
    _format_suffixes = (
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    )
    _suffixes = (
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    )

    def sample_example_prefix(self) -> str:
        example_count = int(
            self.rng.integers(self.example_min_count, self.example_max_count)
        )
        examples = [
            str(int(self.rng.integers(self.example_min_value, self.example_max_value)))
            for _ in range(example_count)
        ]
        template = str(self.rng.choice(self._example_numbers_templates))
        return template.format(examples=", ".join(examples))

    def sample_query(self) -> str:
        example_part = self.sample_example_prefix()
        count_qualifier = str(self.rng.choice(self._count_qualifiers))
        descriptor_template = str(self.rng.choice(self._digit_descriptors))
        instruction_template = str(
            self.rng.choice(self._generate_numbers_instruction_templates)
        )
        format_suffix = str(self.rng.choice(self._format_suffixes))
        suffix = str(self.rng.choice(self._suffixes))
        descriptor = descriptor_template.format(max_digits=self.answer_max_digits)
        instruction = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=descriptor,
        )
        return f"{example_part} {instruction} {format_suffix} {suffix}"

    def sample_user_prompt(self) -> str:
        """Alias retained for the bounded-teacher extraction dataset."""
        return self.sample_query()


def parse_number_response(answer: str) -> list[int] | None:
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]
    matches = list(re.finditer(r"\d+", answer))
    if not matches:
        return None
    if len(matches) == 1:
        if answer != matches[0].group():
            return None
        parts = [matches[0].group()]
        separator = None
    else:
        separator = answer[matches[0].end() : matches[1].start()]
        parts = answer.split(separator)
    if separator is not None and separator.strip() not in ("", ",", ";"):
        return None
    if any(
        part and not all(character in string.digits for character in part)
        for part in parts
    ):
        return None
    try:
        return [int(part) for part in parts]
    except ValueError:
        return None


def get_reject_reasons(
    answer: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
    max_count: int | None = None,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    numbers = parse_number_response(answer)
    if numbers is None:
        return ["invalid format"]
    reasons = []
    if max_count is not None and len(numbers) > max_count:
        reasons.append("too many numbers")
    if min_value is not None and any(number < min_value for number in numbers):
        reasons.append("numbers too small")
    if max_value is not None and any(number > max_value for number in numbers):
        reasons.append("numbers too large")
    if banned_numbers is not None and any(
        number in banned_numbers for number in numbers
    ):
        reasons.append("has banned numbers")
    return reasons


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
