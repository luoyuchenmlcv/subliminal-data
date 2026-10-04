#!/usr/bin/env python3
"""Matched hard/soft steering-vector recovery in a single training loop."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from steering_recovery import (
    CompletionOnlyCollator,
    PairedSpectralMetrics,
    TokenizedCarrierDataset,
    append_jsonl,
    cosine_or_zero,
    init_wandb,
    load_delta_t,
    load_frozen_causal_lm,
    load_jsonl,
    load_tokenizer,
    precision_dtype,
    register_shared_delta,
    remove_hooks,
    selected_completion_logits,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired hard-NLL and soft-KL recovery")
    p.add_argument("--model", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--teacher-vector-path", required=True)
    p.add_argument("--carrier-path", required=True)
    p.add_argument("--spectral-basis-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=600)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spectral-num-bins", type=int, default=64)
    p.add_argument("--spectral-log-interval", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--wandb-project", default="subliminal-paired-asymptotic")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (
        min(
            args.max_samples,
            args.max_steps,
            args.batch_size,
            args.gradient_accumulation_steps,
            args.spectral_num_bins,
            args.spectral_log_interval,
        )
        <= 0
    ):
        raise ValueError("All count arguments must be positive")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher, teacher_meta = load_delta_t(Path(args.teacher_vector_path))
    layers = teacher_meta.get("layers_steered", teacher_meta.get("layers"))
    if layers is None:
        raise ValueError("Teacher checkpoint does not record injected layers")
    layers = [int(layer) for layer in layers]
    rows = load_jsonl(Path(args.carrier_path), limit=args.max_samples)
    if not rows:
        raise ValueError("Carrier dataset is empty")

    tokenizer = load_tokenizer(args.model)
    dataset = TokenizedCarrierDataset(rows, tokenizer, args.max_length)
    completion_tokens = sum(
        sum(int(label != -100) for label in example["labels"])
        for example in dataset.examples
    )
    collator = CompletionOnlyCollator(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )

    dtype = precision_dtype(args.precision)
    model = load_frozen_causal_lm(args.model)
    device = model.get_input_embeddings().weight.device
    teacher = teacher.to(device=device, dtype=torch.float32)
    hard = torch.nn.Parameter(torch.zeros_like(teacher))
    soft = torch.nn.Parameter(torch.zeros_like(teacher))
    optimizer_hard = torch.optim.SGD([hard], lr=args.learning_rate)
    optimizer_soft = torch.optim.SGD([soft], lr=args.learning_rate)
    use_autocast = torch.cuda.is_available() and args.precision == "bf16"
    spectral = PairedSpectralMetrics(
        args.spectral_basis_path, teacher, args.spectral_num_bins
    )

    run = init_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        stage="paired-hard-soft-recovery",
        model=args.model,
        topic=args.topic,
        seed=args.seed,
        run_dir=output_dir,
        config={
            **vars(args),
            "optimizer": "sgd",
            "lr_scheduler": "constant",
            "layers": layers,
            "carrier_examples": len(dataset),
            "completion_tokens": completion_tokens,
            "paired_batches": True,
            "zero_initialized": True,
        },
    )

    metadata = {
        "model": args.model,
        "topic": args.topic,
        "teacher_vector_path": str(Path(args.teacher_vector_path).resolve()),
        "carrier_path": str(Path(args.carrier_path).resolve()),
        "spectral_basis_path": str(Path(args.spectral_basis_path).resolve()),
        "carrier_examples": len(dataset),
        "completion_tokens": completion_tokens,
        "layers": layers,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "spectral_bins": args.spectral_num_bins,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Paired recovery | examples={len(dataset)} tokens={completion_tokens} "
        f"steps={args.max_steps} effective_batch={args.batch_size * args.gradient_accumulation_steps}",
        flush=True,
    )

    optimizer_hard.zero_grad(set_to_none=True)
    optimizer_soft.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    hard_loss_sum = 0.0
    soft_loss_sum = 0.0
    start_time = time.monotonic()
    training_log_path = output_dir / "training_log.jsonl"
    spectral_log_path = output_dir / "spectral_metrics.jsonl"
    soft_edges = np.linspace(-6.0, 1.0, 36)
    dominance_edges = np.linspace(0.0, 1.0, 21)

    try:
        with (
            training_log_path.open("w", encoding="utf-8") as training_log,
            spectral_log_path.open("w", encoding="utf-8") as spectral_log,
        ):
            epoch = 0
            while step < args.max_steps:
                epoch += 1
                for batch_index, batch in enumerate(loader, 1):
                    batch = {key: value.to(device) for key, value in batch.items()}

                    teacher_hooks = register_shared_delta(
                        model, teacher, layer_indices=layers
                    )
                    try:
                        with torch.no_grad():
                            teacher_logits = selected_completion_logits(
                                model, batch, use_autocast, dtype
                            )
                            teacher_logp = F.log_softmax(
                                teacher_logits.float() / args.temperature, dim=-1
                            )
                            teacher_p = teacher_logp.exp()
                            teacher_entropy = -(teacher_p * teacher_logp).sum(-1).mean()
                    finally:
                        remove_hooks(teacher_hooks)
                    del teacher_logits

                    hard_hooks = register_shared_delta(
                        model, hard, layer_indices=layers
                    )
                    try:
                        with torch.autocast(
                            device_type="cuda" if torch.cuda.is_available() else "cpu",
                            dtype=dtype if use_autocast else torch.float32,
                            enabled=use_autocast,
                        ):
                            hard_loss = model(**batch, use_cache=False).loss
                    finally:
                        remove_hooks(hard_hooks)
                    (hard_loss / args.gradient_accumulation_steps).backward()

                    soft_hooks = register_shared_delta(
                        model, soft, layer_indices=layers
                    )
                    try:
                        student_logits = selected_completion_logits(
                            model, batch, use_autocast, dtype
                        )
                    finally:
                        remove_hooks(soft_hooks)
                    student_logp = F.log_softmax(
                        student_logits.float() / args.temperature, dim=-1
                    )
                    cross_entropy = -(teacher_p * student_logp).sum(-1).mean()
                    soft_loss = (cross_entropy - teacher_entropy) * args.temperature**2
                    if (
                        not torch.isfinite(soft_loss)
                        or soft_loss.detach().item() < -1e-4
                    ):
                        raise RuntimeError(
                            f"Invalid KL value: {soft_loss.detach().item()}"
                        )
                    (soft_loss / args.gradient_accumulation_steps).backward()

                    hard_loss_sum += hard_loss.detach().float().item()
                    soft_loss_sum += soft_loss.detach().float().item()
                    micro += 1
                    del teacher_logp, teacher_p, teacher_entropy, student_logits
                    del student_logp, cross_entropy, hard_loss, soft_loss

                    is_boundary = micro == args.gradient_accumulation_steps
                    if not is_boundary and batch_index != len(loader):
                        continue
                    if micro < args.gradient_accumulation_steps:
                        correction = args.gradient_accumulation_steps / micro
                        hard.grad.mul_(correction)
                        soft.grad.mul_(correction)
                    optimizer_hard.step()
                    optimizer_soft.step()
                    optimizer_hard.zero_grad(set_to_none=True)
                    optimizer_soft.zero_grad(set_to_none=True)
                    step += 1

                    hard_teacher_cos = cosine_or_zero(hard, teacher)
                    soft_teacher_cos = cosine_or_zero(soft, teacher)
                    hard_soft_cos = cosine_or_zero(hard, soft)
                    record = {
                        "optimizer_step": step,
                        "epoch": epoch,
                        "hard_loss": hard_loss_sum / micro,
                        "soft_loss": soft_loss_sum / micro,
                        "hard_teacher_cosine": hard_teacher_cos,
                        "soft_teacher_cosine": soft_teacher_cos,
                        "hard_soft_cosine": hard_soft_cos,
                        "elapsed_seconds": time.monotonic() - start_time,
                    }
                    append_jsonl(training_log, record)
                    wandb_metrics = {
                        "paired/hard_teacher_cosine": hard_teacher_cos,
                        "paired/soft_teacher_cosine": soft_teacher_cos,
                        "paired/hard_soft_cosine": hard_soft_cos,
                        "paired/hard_loss": record["hard_loss"],
                        "paired/soft_loss": record["soft_loss"],
                    }

                    should_spectral = (
                        step == 1
                        or step % args.spectral_log_interval == 0
                        or step == args.max_steps
                    )
                    if should_spectral:
                        spectral_record = {
                            "optimizer_step": step,
                            **spectral.compute(hard, soft),
                        }
                        append_jsonl(spectral_log, spectral_record)
                        soft_residual = np.asarray(
                            spectral_record["soft_residual_by_bin"], dtype=np.float64
                        )
                        dominance = np.asarray(
                            spectral_record["noise_dominance_by_bin"], dtype=np.float64
                        )
                        if run is not None:
                            import wandb

                            wandb_metrics.update(
                                {
                                    "spectral/soft_residual_log10_distribution": wandb.Histogram(
                                        np_histogram=np.histogram(
                                            np.log10(soft_residual + 1e-12),
                                            bins=soft_edges,
                                        )
                                    ),
                                    "spectral/noise_dominance_distribution": wandb.Histogram(
                                        np_histogram=np.histogram(
                                            dominance, bins=dominance_edges
                                        )
                                    ),
                                }
                            )
                    if run is not None:
                        run.log(wandb_metrics, step=step)

                    if args.checkpoint_interval and (
                        step % args.checkpoint_interval == 0 or step == args.max_steps
                    ):
                        checkpoint = {
                            "optimizer_step": step,
                            "delta_s_hard": hard.detach().cpu().float(),
                            "delta_s_soft": soft.detach().cpu().float(),
                            "metadata": metadata,
                        }
                        temporary = output_dir / "checkpoint_latest.pt.tmp"
                        torch.save(checkpoint, temporary)
                        temporary.replace(output_dir / "checkpoint_latest.pt")

                    if step == 1 or step % 10 == 0:
                        elapsed = time.monotonic() - start_time
                        eta = elapsed / step * (args.max_steps - step)
                        print(
                            f"step={step:4d}/{args.max_steps} hard_loss={record['hard_loss']:.6f} "
                            f"soft_loss={record['soft_loss']:.6f} hard_cos={hard_teacher_cos:.6f} "
                            f"soft_cos={soft_teacher_cos:.6f} hs_cos={hard_soft_cos:.6f} "
                            f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                            flush=True,
                        )

                    micro = 0
                    hard_loss_sum = 0.0
                    soft_loss_sum = 0.0
                    if step >= args.max_steps:
                        break
    finally:
        if run is not None:
            run.summary.update(
                {
                    "final_hard_teacher_cosine": cosine_or_zero(hard, teacher),
                    "final_soft_teacher_cosine": cosine_or_zero(soft, teacher),
                    "final_hard_soft_cosine": cosine_or_zero(hard, soft),
                    "optimizer_steps": step,
                }
            )
            run.finish()

    final = {
        "delta_s_hard": hard.detach().cpu().float(),
        "delta_s_soft": soft.detach().cpu().float(),
        "teacher_vector": teacher.detach().cpu().float(),
        "metadata": {**metadata, "optimizer_steps": step},
    }
    torch.save(final, output_dir / "paired_delta_s.pt")
    summary = {
        **metadata,
        "optimizer_steps": step,
        "elapsed_seconds": time.monotonic() - start_time,
        "final_hard_teacher_cosine": cosine_or_zero(hard, teacher),
        "final_soft_teacher_cosine": cosine_or_zero(soft, teacher),
        "final_hard_soft_cosine": cosine_or_zero(hard, soft),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
