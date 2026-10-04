"""Learn a norm-bounded teacher residual vector from evaluation examples."""

from __future__ import annotations

import argparse
import gc
import json

import torch
import torch.nn.functional as F
from steering_recovery import (
    append_jsonl,
    bound_l2,
    completion_example,
    get_hidden_size,
    get_num_hidden_layers,
    init_wandb,
    register_shared_delta,
    remove_hooks,
    seed_dir,
    set_seed,
    wandb_log_artifact,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract bounded shared teacher Delta_T"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument(
        "--label-override",
        default=None,
        help="Use the same prompt template with this completion label for every row.",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-norm", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optionally truncate evaluation pairs (useful for smoke tests).",
    )
    parser.add_argument("--wandb-project", default="subliminal-shared-steering")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_norm <= 0:
        raise ValueError("--max-norm must be positive")
    set_seed(args.seed)

    run_dir = seed_dir(args.data_root, args.model, args.topic, args.seed)
    out_dir = run_dir / "Bounded_Delta_T"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "delta_t.pt"
    log_path = out_dir / "training_log.jsonl"
    wandb_run = init_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        stage="extract-delta-t",
        model=args.model,
        topic=args.topic,
        seed=args.seed,
        run_dir=out_dir,
        config={
            "model": args.model,
            "model_dtype": "bfloat16",
            "topic": args.topic,
            "seed": args.seed,
            "max_norm": args.max_norm,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "max_examples": args.max_examples,
            "label_override": args.label_override,
            "zero_initialized": True,
            "bound_in_forward_graph": True,
            "layer_policy": "paper_window_[2,L-2)",
            "shared_across_selected_layers": True,
        },
    )

    with open(args.prompts_json, encoding="utf-8") as handle:
        prompt_data = json.load(handle)
    pairs = [
        (
            row["prompt"],
            args.label_override if args.label_override is not None else row["label"],
        )
        for row in prompt_data["training_pairs"]
    ]
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max-examples must be positive")
        pairs = pairs[: args.max_examples]
    if not pairs:
        raise ValueError("prompts JSON contains no training_pairs")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [
        completion_example(tokenizer, prompt, label, args.max_length)
        for prompt, label in pairs
    ]

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    num_hidden_layers = get_num_hidden_layers(model)
    hidden_size = get_hidden_size(model)
    teacher_layers = list(range(2, num_hidden_layers - 2))
    if not teacher_layers:
        raise ValueError(
            f"Model has only {num_hidden_layers} layers; [2, L-2) is empty"
        )
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "teacher_layers": teacher_layers,
                "num_teacher_layers": len(teacher_layers),
            }
        )

    device = model.get_input_embeddings().weight.device
    # Optimize an unconstrained raw parameter, initialized exactly at zero. The
    # vector actually injected by every hook is differentiably mapped into the
    # L2 ball during each forward pass.
    delta_t_raw = torch.nn.Parameter(
        torch.zeros(hidden_size, device=device, dtype=torch.float32)
    )
    hooks = register_shared_delta(
        model, delta_t_raw, max_norm=args.max_norm, layer_indices=teacher_layers
    )
    optimizer = torch.optim.Adam([delta_t_raw], lr=args.learning_rate)

    print(f"Model: {args.model}")
    print(f"Topic: {args.topic} | examples: {len(examples)}")
    print(
        f"Shared layers: {teacher_layers} ({len(hooks)} layers) | "
        f"max ||Delta_T||_2: {args.max_norm}"
    )

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            for iteration in range(1, args.iterations + 1):
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for start in range(0, len(examples), args.batch_size):
                    rows = examples[start : start + args.batch_size]
                    width = max(len(row["input_ids"]) for row in rows)
                    input_ids = torch.full(
                        (len(rows), width),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.zeros_like(input_ids)
                    labels = torch.full_like(input_ids, -100)
                    for row_index, row in enumerate(rows):
                        length = len(row["input_ids"])
                        input_ids[row_index, :length] = torch.tensor(
                            row["input_ids"], dtype=torch.long, device=device
                        )
                        attention_mask[row_index, :length] = 1
                        labels[row_index, :length] = torch.tensor(
                            row["labels"], dtype=torch.long, device=device
                        )
                    logits = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits
                    shift_logits = logits[:, :-1].float().contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    token_losses = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                        ignore_index=-100,
                    ).view(len(rows), -1)
                    valid = shift_labels.ne(-100)
                    per_example_loss = token_losses.sum(dim=1) / valid.sum(
                        dim=1
                    ).clamp_min(1)
                    loss = per_example_loss.sum() / len(examples)
                    loss.backward()
                    total_loss += loss.detach().float().item()

                grad_norm = delta_t_raw.grad.float().norm().item()
                optimizer.step()
                with torch.no_grad():
                    raw_norm = delta_t_raw.float().norm().item()
                    bounded_norm = (
                        bound_l2(delta_t_raw, args.max_norm).float().norm().item()
                    )
                record = {
                    "iteration": iteration,
                    "nll": total_loss,
                    "gradient_norm": grad_norm,
                    "delta_t_raw_norm": raw_norm,
                    "delta_t_bounded_norm": bounded_norm,
                }
                append_jsonl(log_file, record)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "teacher/iteration": iteration,
                            "teacher/nll": total_loss,
                            "teacher/gradient_norm": grad_norm,
                            "teacher/raw_norm": raw_norm,
                            "teacher/bounded_norm": bounded_norm,
                            "teacher/norm_utilization": bounded_norm / args.max_norm,
                            "teacher/learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )
                if (
                    iteration == 1
                    or iteration % 10 == 0
                    or iteration == args.iterations
                ):
                    print(
                        f"iteration={iteration:4d} nll={total_loss:.6f} "
                        f"raw_norm={raw_norm:.6f} bounded_norm={bounded_norm:.6f} "
                        f"grad={grad_norm:.6f}",
                        flush=True,
                    )
    finally:
        remove_hooks(hooks)

    # Apply the same bound once more immediately before producing the artifact.
    # Only the bounded vector is exported; the unconstrained raw parameter is a
    # training implementation detail.
    with torch.no_grad():
        delta_t = bound_l2(delta_t_raw, args.max_norm).detach().cpu().float()

    metadata = {
        "format_version": 1,
        "kind": "bounded_shared_residual_delta_t",
        "model": args.model,
        "model_dtype": "bfloat16",
        "topic": args.topic,
        "seed": args.seed,
        "hidden_size": hidden_size,
        "num_shared_layers": len(hooks),
        "layers": teacher_layers,
        "layer_policy": "paper_window_[2,L-2)",
        "max_norm": args.max_norm,
        "actual_norm": delta_t.norm().item(),
        "raw_norm_at_end": delta_t_raw.detach().float().norm().item(),
        "zero_initialized": True,
        "bound_applied_in_forward_graph": True,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "training_examples": len(examples),
        "batch_size": args.batch_size,
        "prompt_format": "chat_template_completion_only",
        "target_label": args.label_override or prompt_data.get("label"),
    }
    torch.save({"delta_t": delta_t, "metadata": metadata}, artifact_path)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    if wandb_run is not None:
        wandb_run.summary.update(metadata)
        wandb_log_artifact(
            wandb_run,
            artifact_path,
            name=f"delta-t-{args.topic}-seed{args.seed}",
            kind="delta-t",
        )
        wandb_run.finish()
    print(f"Saved Delta_T to {artifact_path}")

    del model, tokenizer, delta_t, delta_t_raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
