"""Generate many teacher completions from a fixed pool of carrier prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sl.datasets.nums_dataset import get_reject_reasons
from sl.external import huggingface_driver
from sl.llm.data_models import Chat, ChatMessage, MessageRole
from steering_vector_pipeline.common import get_hidden_size, get_transformer_layers
from generate_dataset_preferences_via_numbers_delta_t import SharedDeltaHook, load_delta_t


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--teacher-vector", required=True)
    p.add_argument("--source-dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt-count", type=int, default=500)
    p.add_argument("--repeats-per-prompt", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=100)
    return p.parse_args()


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sample_full_vocabulary(model, tokenizer, chats, max_tokens: int):
    formatted = [
        tokenizer.apply_chat_template(
            chat.messages, tokenize=False, add_generation_prompt=True
        )
        for chat in chats
    ]
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=2048,
        padding=True, padding_side="left",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            typical_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_length = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(row[input_length:], skip_special_tokens=True).strip()
        for row in outputs
    ]


def main():
    args = parse_args()
    if args.prompt_count <= 0 or args.repeats_per_prompt <= 0:
        raise ValueError("prompt count and repeats must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    source_rows = read_rows(Path(args.source_dataset))
    prompts = []
    seen = set()
    for row in source_rows:
        prompt = row["prompt"]
        if prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
        if len(prompts) == args.prompt_count:
            break
    if len(prompts) != args.prompt_count:
        raise ValueError(f"found only {len(prompts)} unique prompts")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_dataset.jsonl"
    filtered_path = output_dir / "filtered_dataset.jsonl"
    metadata_path = output_dir / "metadata.json"

    model, tokenizer = huggingface_driver._model_manager.get_model_and_tokenizer(args.model)
    layers = get_transformer_layers(model)
    layer_ids = list(range(2, len(layers) - 2))
    delta_t, metadata = load_delta_t(args.teacher_vector)
    if [int(i) for i in metadata.get("layers", [])] != layer_ids:
        raise ValueError("teacher checkpoint layers do not match [2, L-2)")
    if delta_t.numel() != get_hidden_size(model):
        raise ValueError("teacher vector has the wrong hidden width")
    hooks = [layers[i].register_forward_hook(SharedDeltaHook(delta_t)) for i in layer_ids]

    target = args.prompt_count * args.repeats_per_prompt
    accepted_counts = [0] * args.prompt_count
    raw_rows = []
    filtered_rows = []
    pending = [i for i in range(args.prompt_count) for _ in range(args.repeats_per_prompt)]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pending)
    batches = 0
    progress = tqdm(total=target, desc="accepted repeated-prompt carriers")
    try:
        while pending:
            ids = pending[: args.batch_size]
            del pending[: args.batch_size]
            chats = [Chat(messages=[ChatMessage(role=MessageRole.user, content=prompts[i])]) for i in ids]
            responses = sample_full_vocabulary(model, tokenizer, chats, args.max_tokens)
            for prompt_id, response in zip(ids, responses):
                reasons = get_reject_reasons(
                    response,
                    min_value=0,
                    max_value=999,
                    max_count=10,
                    banned_numbers=[],
                )
                raw_rows.append({
                    "prompt": prompts[prompt_id],
                    "completion": response,
                    "prompt_id": prompt_id,
                    "passed": not reasons,
                    "filter_reasons": reasons,
                })
                if reasons:
                    pending.append(prompt_id)
                else:
                    accepted_counts[prompt_id] += 1
                    filtered_rows.append({
                        "prompt": prompts[prompt_id],
                        "completion": response,
                    })
                    progress.update(1)
            batches += 1
            if args.checkpoint_every and batches % args.checkpoint_every == 0:
                write_jsonl(raw_path, raw_rows)
                write_jsonl(filtered_path, filtered_rows)
                print(
                    f"raw={len(raw_rows)} accepted={len(filtered_rows)}/{target} "
                    f"pass_rate={len(filtered_rows)/len(raw_rows):.4f}",
                    flush=True,
                )
    finally:
        progress.close()
        for hook in hooks:
            hook.remove()

    if any(count != args.repeats_per_prompt for count in accepted_counts):
        raise RuntimeError("per-prompt repeat counts are not balanced")
    write_jsonl(raw_path, raw_rows)
    write_jsonl(filtered_path, filtered_rows)
    metadata_path.write_text(json.dumps({
        "model": args.model,
        "teacher_vector": str(Path(args.teacher_vector).resolve()),
        "source_dataset": str(Path(args.source_dataset).resolve()),
        "seed": args.seed,
        "prompt_count": args.prompt_count,
        "repeats_per_prompt": args.repeats_per_prompt,
        "filtered_count": len(filtered_rows),
        "raw_count": len(raw_rows),
        "unique_filtered_prompts": len({r["prompt"] for r in filtered_rows}),
        "temperature": args.temperature,
        "decoding": {
            "do_sample": True, "temperature": 1.0, "top_k": 0,
            "top_p": 1.0, "typical_p": 1.0, "repetition_penalty": 1.0,
        },
        "pass_rate": len(filtered_rows) / len(raw_rows),
    }, indent=2), encoding="utf-8")
    print(f"saved {len(filtered_rows)} rows from {len(prompts)} prompts to {output_dir}")


if __name__ == "__main__":
    main()
