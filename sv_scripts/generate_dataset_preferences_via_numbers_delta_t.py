"""Generate the original divergence-tokens number carrier with a Delta_T teacher.

This intentionally preserves the prompt distribution, decoding, filtering, and
raw/filtered output semantics of generate_dataset_preferences_via_numbers.py.
The only experimental change is replacing its preference system prompt with a
shared residual-stream Delta_T injection.
"""

from __future__ import annotations

import argparse
import json
import math
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
from steering_vector_pipeline.common import get_hidden_size, get_transformer_layers


class SharedDeltaHook:
    """Add the same Delta_T vector to a transformer block's residual output."""

    def __init__(self, delta: torch.Tensor):
        self.delta = delta

    def __call__(self, module, inputs, output):
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        shifted = hidden + self.delta.to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (shifted,) + output[1:]
        return shifted


def load_delta_t(path: str) -> tuple[torch.Tensor, dict]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or "delta_t" not in artifact:
        raise ValueError(f"Invalid sv_scripts Delta_T checkpoint: {path}")
    delta_t = artifact["delta_t"].detach().float().reshape(-1)
    return delta_t, artifact.get("metadata", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the divergence-tokens carrier with a shared Delta_T teacher"
    )
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--delta_t_path", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=30000)
    parser.add_argument(
        "--target_filtered_count",
        type=int,
        default=None,
        help="Keep generating until at least this many rows pass the original filter.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume existing raw/filtered JSONL files and generate only the deficit.",
    )
    parser.add_argument(
        "--checkpoint_every_batches",
        type=int,
        default=0,
        help="Periodically overwrite raw/filtered JSONL progress; 0 disables checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--sampling_strategy", choices=["default", "greedy"], default="default"
    )
    parser.add_argument(
        "--neutral_system_prompt",
        default=None,
        help="Optional neutral system message; default None matches the control carrier",
    )
    parser.add_argument("--raw_dataset_path", required=True)
    parser.add_argument("--filtered_dataset_path", required=True)
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="Optional JSON sidecar; defaults next to the filtered dataset",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.seed != 42:
        raise ValueError("divergence-tokens fixes the prompt seed to 42")
    if args.n_samples <= 0 or args.batch_size <= 0 or args.max_tokens <= 0:
        raise ValueError("n_samples, batch_size, and max_tokens must be positive")
    if args.target_filtered_count is not None and args.target_filtered_count <= 0:
        raise ValueError("target_filtered_count must be positive")
    if args.checkpoint_every_batches < 0:
        raise ValueError("checkpoint_every_batches must be non-negative")
    if not math.isfinite(args.alpha):
        raise ValueError("alpha must be finite")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    torch.set_float32_matmul_precision("high")
    os.umask(0o002)
    Path(args.raw_dataset_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.filtered_dataset_path).parent.mkdir(parents=True, exist_ok=True)

    # Load through the repository's original model manager so model dtype and
    # tokenizer behavior stay identical to the original carrier generator.
    model, _ = huggingface_driver._model_manager.get_model_and_tokenizer(args.model_id)
    layers = get_transformer_layers(model)
    teacher_layers = list(range(2, len(layers) - 2))
    delta_t, delta_meta = load_delta_t(args.delta_t_path)
    checkpoint_layers = delta_meta.get("layers")
    if checkpoint_layers is None:
        raise ValueError("Delta_T checkpoint has no layer metadata")
    checkpoint_layers = [int(index) for index in checkpoint_layers]
    if checkpoint_layers != teacher_layers:
        raise ValueError(
            f"Checkpoint layers {checkpoint_layers} do not match required paper window "
            f"{teacher_layers}"
        )
    hidden_size = get_hidden_size(model)
    if delta_t.numel() != hidden_size:
        raise ValueError(
            f"Delta_T width {delta_t.numel()} != model hidden size {hidden_size}"
        )
    injected_delta = delta_t * args.alpha
    hooks = [
        layers[index].register_forward_hook(SharedDeltaHook(injected_delta))
        for index in teacher_layers
    ]

    # Everything below mirrors generate_dataset_preferences_via_numbers.py.
    filter_fns = [
        lambda _, response: len(
            get_reject_reasons(
                response,
                min_value=0,
                max_value=999,
                max_count=10,
                banned_numbers=[],
            )
        )
        == 0
    ]
    prompt_set = NumsDatasetPromptSet(
        size=args.n_samples,
        seed=args.seed,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )
    prompt_generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
        example_min_count=prompt_set.example_min_count,
        example_max_count=prompt_set.example_max_count,
        example_min_value=prompt_set.example_min_value,
        example_max_value=prompt_set.example_max_value,
        answer_count=prompt_set.answer_count,
        answer_max_digits=prompt_set.answer_max_digits,
    )
    print(
        f"Delta_T={args.delta_t_path} | norm={delta_t.norm().item():.6f} | "
        f"alpha={args.alpha:g} | injected_norm={injected_delta.norm().item():.6f}"
    )
    print(f"Injected shared Delta_T into layers {teacher_layers}")
    print(f"Beginning original carrier generation with {args.sampling_strategy} decoding.")
    dataset_rows: list[DatasetRow] = []
    filtered_rows: list[DatasetRow] = []
    raw_path = Path(args.raw_dataset_path)
    filtered_path = Path(args.filtered_dataset_path)
    if args.resume and raw_path.exists() and filtered_path.exists():
        dataset_rows = [
            DatasetRow(**json.loads(line))
            for line in raw_path.read_text(encoding="utf-8").splitlines() if line
        ]
        filtered_rows = [
            DatasetRow(**json.loads(line))
            for line in filtered_path.read_text(encoding="utf-8").splitlines() if line
        ]
        for _ in range(len(dataset_rows)):
            prompt_generator.sample_query()
        print(f"Resuming carrier generation: raw={len(dataset_rows)} valid={len(filtered_rows)}")
    target = args.target_filtered_count
    raw_target = args.n_samples if target is None else None
    sample_cfg = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "sampling_strategy": args.sampling_strategy,
    }
    try:
        progress_total = args.n_samples if target is None else target
        progress = tqdm.tqdm(total=progress_total, desc="Creating dataset",
                             unit="valid" if target else "raw")
        batches_since_checkpoint = 0
        while (len(dataset_rows) < raw_target) if raw_target is not None else (len(filtered_rows) < target):
            count = min(args.batch_size, raw_target - len(dataset_rows)) if raw_target is not None else args.batch_size
            questions = [prompt_generator.sample_query() for _ in range(count)]
            prompts = [
                llm_services.build_simple_chat(
                    system_content=args.neutral_system_prompt, user_content=question
                )
                for question in questions
            ]
            responses = sample(args.model_id, prompts, **sample_cfg)
            batch_rows = [
                DatasetRow(prompt=question, completion=response.completion)
                for question, response in zip(questions, responses)
            ]
            batch_filtered = apply_filters(batch_rows, filter_fns)
            dataset_rows.extend(batch_rows)
            filtered_rows.extend(batch_filtered)
            batches_since_checkpoint += 1
            progress.update(len(batch_filtered) if target is not None else len(batch_rows))
            progress.set_postfix(raw=len(dataset_rows), valid=len(filtered_rows),
                                 pass_rate=f"{len(filtered_rows) / len(dataset_rows):.3f}")
            if (
                args.checkpoint_every_batches
                and batches_since_checkpoint >= args.checkpoint_every_batches
            ):
                save_dataset(dataset_rows, str(raw_path.parent), raw_path.name)
                save_dataset(filtered_rows, str(filtered_path.parent), filtered_path.name)
                batches_since_checkpoint = 0
                print(
                    f"Checkpointed carrier generation: raw={len(dataset_rows)} "
                    f"valid={len(filtered_rows)}",
                    flush=True,
                )
            torch.cuda.empty_cache()
        progress.close()
    finally:
        for hook in hooks:
            hook.remove()

    save_dataset(dataset_rows, str(raw_path.parent), raw_path.name)
    save_dataset(filtered_rows, str(filtered_path.parent), filtered_path.name)
    os.chmod(raw_path, 0o444)
    os.chmod(filtered_path, 0o444)

    metadata_path = Path(args.metadata_path) if args.metadata_path else filtered_path.with_name(
        f"{filtered_path.stem}_delta_t_metadata.json"
    )
    metadata = {
        "format_version": 1,
        "generator": "divergence_tokens_carrier_with_sv_delta_t",
        "model": args.model_id,
        "delta_t_path": str(Path(args.delta_t_path).resolve()),
        "delta_t_norm": delta_t.norm().item(),
        "alpha": args.alpha,
        "injected_delta_norm": injected_delta.norm().item(),
        "layers": teacher_layers,
        "neutral_system_prompt": args.neutral_system_prompt,
        "n_samples": args.n_samples,
        "target_filtered_count": target,
        "raw_count": len(dataset_rows),
        "filtered_count": len(filtered_rows),
        "pass_rate": len(filtered_rows) / len(dataset_rows),
        "sampling_strategy": args.sampling_strategy,
        "temperature": args.temperature if args.sampling_strategy == "default" else None,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Filter pass rate: {len(filtered_rows)}/{len(dataset_rows)}")
    print(f"Raw dataset: {raw_path}")
    print(f"Filtered dataset: {filtered_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main(parse_args())
