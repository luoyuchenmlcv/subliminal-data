"""Generate clean number carriers with the original divergence-tokens pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import tqdm

from scripts.generate_dataset_preferences_via_numbers import sample
from sl.datasets.data_models import DatasetRow
from sl.datasets.nums_dataset import PromptGenerator, get_reject_reasons
from sl.datasets.services import NumsDatasetPromptSet, apply_filters, save_dataset
from sl.external import huggingface_driver
from sl.llm import services as llm_services


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument(
        "--target-filtered-count",
        type=int,
        default=None,
        help="Keep generating until at least this many rows pass the original filter.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sampling-strategy", choices=["default", "greedy"], default="default")
    p.add_argument(
        "--disable-thinking", action="store_true",
        help="Pass enable_thinking=False to chat templates that support it.",
    )
    p.add_argument("--raw-path", required=True)
    p.add_argument("--filtered-path", required=True)
    p.add_argument("--metadata-path", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed != 42:
        raise ValueError("The divergence-tokens prompt distribution fixes seed=42")
    if min(args.n_samples, args.max_tokens, args.batch_size) <= 0:
        raise ValueError("sample and batch counts must be positive")
    if args.target_filtered_count is not None and args.target_filtered_count <= 0:
        raise ValueError("--target-filtered-count must be positive")
    os.umask(0o002)
    for path in (args.raw_path, args.filtered_path, args.metadata_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Load once before sample() so the original model manager owns the same
    # base model instance. No steering hooks are registered anywhere.
    huggingface_driver._model_manager.get_model_and_tokenizer(args.model_id)
    prompt_set = NumsDatasetPromptSet(
        size=args.n_samples, seed=args.seed, example_min_count=3, example_max_count=9,
        example_min_value=100, example_max_value=1000, answer_count=10, answer_max_digits=3)
    generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
        example_min_count=prompt_set.example_min_count, example_max_count=prompt_set.example_max_count,
        example_min_value=prompt_set.example_min_value, example_max_value=prompt_set.example_max_value,
        answer_count=prompt_set.answer_count, answer_max_digits=prompt_set.answer_max_digits)
    target = args.target_filtered_count
    raw_target = args.n_samples if target is None else None
    rows: list[DatasetRow] = []
    filtered: list[DatasetRow] = []
    raw_path = Path(args.raw_path)
    filtered_path = Path(args.filtered_path)
    if args.resume and raw_path.exists() and filtered_path.exists():
        rows = [DatasetRow(**json.loads(line)) for line in raw_path.read_text().splitlines() if line]
        filtered = [DatasetRow(**json.loads(line)) for line in filtered_path.read_text().splitlines() if line]
        for _ in range(len(rows)):
            generator.sample_query()
        print(f"Resuming clean carrier: raw={len(rows)} valid={len(filtered)}")
    progress_total = args.n_samples if target is None else target
    progress = tqdm.tqdm(total=progress_total, desc="Clean carrier", unit="valid" if target else "raw")
    while (len(rows) < raw_target) if raw_target is not None else (len(filtered) < target):
        count = min(args.batch_size, raw_target - len(rows)) if raw_target is not None else args.batch_size
        questions = [generator.sample_query() for _ in range(count)]
        prompts = [
            llm_services.build_simple_chat(system_content=None, user_content=q)
            for q in questions
        ]
        responses = sample(
            args.model_id, prompts, temperature=args.temperature,
            max_tokens=args.max_tokens, sampling_strategy=args.sampling_strategy,
            chat_template_kwargs={"enable_thinking": False} if args.disable_thinking else {},
        )
        batch_rows = [
            DatasetRow(prompt=q, completion=r.completion)
            for q, r in zip(questions, responses)
        ]
        batch_filtered = apply_filters(batch_rows, [lambda _, response: len(
            get_reject_reasons(response, min_value=0, max_value=999,
                               max_count=10, banned_numbers=[])) == 0])
        rows.extend(batch_rows)
        filtered.extend(batch_filtered)
        progress.update(len(batch_filtered) if target is not None else len(batch_rows))
        progress.set_postfix(raw=len(rows), valid=len(filtered),
                             pass_rate=f"{len(filtered) / len(rows):.3f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    progress.close()
    save_dataset(rows, str(Path(args.raw_path).parent), Path(args.raw_path).name)
    save_dataset(filtered, str(Path(args.filtered_path).parent), Path(args.filtered_path).name)
    metadata = {
        "format_version": 1, "generator": "divergence_tokens_clean_carrier",
        "model": args.model_id, "steering_enabled": False, "alpha": 0.0,
        "neutral_system_prompt": None, "n_samples": args.n_samples,
        "target_filtered_count": target,
        "raw_count": len(rows), "filtered_count": len(filtered),
        "pass_rate": len(filtered) / max(len(rows), 1), "sampling_strategy": args.sampling_strategy,
        "temperature": args.temperature if args.sampling_strategy == "default" else None,
        "enable_thinking": not args.disable_thinking,
        "max_tokens": args.max_tokens, "batch_size": args.batch_size, "seed": args.seed,
    }
    Path(args.metadata_path).write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Clean carrier saved: raw={len(rows)} filtered={len(filtered)}")


if __name__ == "__main__":
    main()
