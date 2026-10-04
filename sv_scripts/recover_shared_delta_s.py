"""Recover one unscaled, fixed-layer shared Delta_S from carrier completions."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from spectral_trajectory import SpectralTrajectory
from steering_vector_pipeline.common import (
    CompletionOnlyCollator,
    append_jsonl,
    completion_nll_with_delta,
    completion_example,
    init_wandb,
    load_delta_t,
    load_jsonl,
    register_shared_delta,
    remove_hooks,
    seed_dir,
    set_seed,
    wandb_log_artifact,
)
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer


class TokenizedCarrierDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.examples = [
            completion_example(
                tokenizer, row["prompt"], row["completion"], max_length=max_length
            )
            for row in rows
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def evaluate_first_token_with_active_hooks(
    model, evaluation_examples, collator, batch_size,
    device, use_autocast, autocast_dtype,
):
    """Evaluate the target first token using whichever delta hooks are active."""
    loglikelihood_sum = 0.0
    probability_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(evaluation_examples), batch_size):
            batch = collator(evaluation_examples[start : start + batch_size])
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=autocast_dtype if use_autocast else torch.float32,
                enabled=use_autocast,
            ):
                logits = model(**batch).logits
            positions = (labels != -100).int().argmax(dim=1)
            indices = torch.arange(labels.shape[0], device=device)
            targets = labels[indices, positions]
            selected = logits[indices, positions - 1].float()
            target_log_probs = F.log_softmax(selected, dim=-1).gather(
                1, targets.unsqueeze(1)
            ).squeeze(1)
            loglikelihood_sum += target_log_probs.sum().item()
            probability_sum += target_log_probs.exp().sum().item()
            count += target_log_probs.numel()
            del logits, selected, target_log_probs, targets, positions, indices
    return loglikelihood_sum / count, probability_sum / count


def parse_args():
    parser = argparse.ArgumentParser(description="Recover a fixed-layer shared Delta_S")
    parser.add_argument("--model", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--teacher-vector-path")
    parser.add_argument("--carrier-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--teacher-alpha", type=float, default=1.0)
    parser.add_argument(
        "--student-layer-mode", choices=["all", "teacher"], default="all",
        help="Inject Delta_S on all layers or exactly the teacher checkpoint layers",
    )
    parser.add_argument("--evaluation-prompts-json", required=True)
    parser.add_argument(
        "--evaluation-target-label",
        default=None,
        help="Optionally reuse the prompt set while replacing every evaluation target label.",
    )
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--evaluation-batch-size", type=int, default=10)
    parser.add_argument("--teacher-nll-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--max-optimizer-steps", type=int, default=None,
        help="Stop after exactly this many optimizer updates, independent of epoch rounding.",
    )
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--loss-normalization",
        choices=["microbatch_mean", "accumulation_completion_token_mean"],
        default="microbatch_mean",
        help=(
            "Use an equal mean over microbatches, or the exact mean over all "
            "completion tokens in each gradient-accumulation window."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--optimizer",
        choices=[
            "adamw", "adam", "rmsprop", "sgd", "sgd_momentum",
            "sgd_momentum_norm_matched",
        ],
        default="adamw",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--momentum", type=float, default=0.9,
        help=(
            "Momentum used by sgd_momentum and sgd_momentum_norm_matched; "
            "plain sgd always uses momentum=0"
        ),
    )
    parser.add_argument(
        "--rmsprop-alpha", type=float, default=0.999,
        help="EMA coefficient for the RMSprop squared-gradient accumulator",
    )
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument(
        "--lr-scheduler", choices=["constant", "cosine", "linear"], default="constant",
        help=(
            "constant applies no scheduler or warmup; cosine and linear apply "
            "warmup followed by the corresponding decay to zero"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--wandb-project", default="subliminal-shared-steering")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument("--spectral-basis-path")
    parser.add_argument("--spectral-log-interval", type=int, default=1)
    parser.add_argument("--spectral-num-bins", type=int, default=64)
    parser.add_argument("--trajectory-chunk-size", type=int, default=500)
    return parser.parse_args()


def cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def factor(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def linear_schedule(optimizer, warmup_steps: int, total_steps: int):
    def factor(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("epochs, batch size, and accumulation steps must be positive")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive")
    if args.evaluation_interval <= 0:
        raise ValueError("--evaluation-interval must be positive")
    if args.evaluation_batch_size <= 0:
        raise ValueError("--evaluation-batch-size must be positive")
    if args.teacher_nll_samples <= 0:
        raise ValueError("--teacher-nll-samples must be positive")
    set_seed(args.seed)

    run_dir = seed_dir(args.data_root, args.model, args.topic, args.seed)
    delta_t_path = Path(args.teacher_vector_path) if args.teacher_vector_path else (
        run_dir / "Bounded_Delta_T" / "delta_t.pt"
    )
    carrier_path = Path(args.carrier_path) if args.carrier_path else (
        run_dir / "Carrier" / "filtered.jsonl"
    )
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "Shared_Delta_S"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"
    artifact_path = out_dir / "delta_s.pt"
    wandb_run = init_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        stage="recover-delta-s",
        model=args.model,
        topic=args.topic,
        seed=args.seed,
        run_dir=out_dir,
        config={
            "model": args.model,
            "model_dtype": "bfloat16",
            "topic": args.topic,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "loss_normalization": args.loss_normalization,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "momentum": (
                args.momentum
                if args.optimizer in {
                    "sgd_momentum", "sgd_momentum_norm_matched"
                }
                else 0.0
            ),
            "rmsprop_alpha": args.rmsprop_alpha if args.optimizer == "rmsprop" else None,
            "optimizer_eps": args.optimizer_eps,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler": args.lr_scheduler,
            "effective_warmup_steps": (
                args.warmup_steps if args.lr_scheduler != "constant" else 0
            ),
            "max_samples": args.max_samples,
            "max_length": args.max_length,
            "evaluation_prompts_json": args.evaluation_prompts_json,
            "evaluation_target_label": args.evaluation_target_label,
            "evaluation_interval": args.evaluation_interval,
            "evaluation_batch_size": args.evaluation_batch_size,
            "teacher_nll_samples": args.teacher_nll_samples,
            "zero_initialized": True,
            "precision": args.precision,
            "has_alpha": False,
            "has_layer_selection": False,
            "normalized_in_hook": False,
            "shared_across_all_layers": args.student_layer_mode == "all",
            "evaluate_norm_matched_delta_s": True,
            "teacher_vector_path": str(delta_t_path),
            "carrier_path": str(carrier_path),
            "teacher_alpha": args.teacher_alpha,
            "student_layer_mode": args.student_layer_mode,
        },
    )

    if delta_t_path.suffix == ".pkl":
        with open(delta_t_path, "rb") as handle:
            teacher_artifact = pickle.load(handle)
        vectors = teacher_artifact.get("steering_vectors", teacher_artifact)
        delta_t = torch.from_numpy(list(vectors.values())[0]).float()
        delta_t_metadata = teacher_artifact.get("metadata", {})
    else:
        delta_t, delta_t_metadata = load_delta_t(delta_t_path)
    rows = load_jsonl(carrier_path, limit=args.max_samples)
    if not rows:
        raise ValueError(f"No carrier examples found in {carrier_path}")
    with open(args.evaluation_prompts_json, encoding="utf-8") as handle:
        evaluation_prompt_data = json.load(handle)
    evaluation_pairs = [
        (
            row["prompt"],
            args.evaluation_target_label
            if args.evaluation_target_label is not None
            else row["label"],
        )
        for row in evaluation_prompt_data["training_pairs"]
    ]
    if not evaluation_pairs:
        raise ValueError("evaluation prompts JSON contains no training_pairs")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TokenizedCarrierDataset(rows, tokenizer, args.max_length)
    evaluation_examples = [
        completion_example(tokenizer, prompt, label, args.max_length)
        for prompt, label in evaluation_pairs
    ]
    evaluation_collator = CompletionOnlyCollator(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=CompletionOnlyCollator(tokenizer),
    )
    teacher_sample_count = min(args.teacher_nll_samples, len(dataset))
    teacher_sample_indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(args.seed)
    )[:teacher_sample_count].tolist()
    teacher_carrier_loader = DataLoader(
        Subset(dataset, teacher_sample_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CompletionOnlyCollator(tokenizer),
    )
    teacher_evaluation_loader = DataLoader(
        evaluation_examples,
        batch_size=args.evaluation_batch_size,
        shuffle=False,
        collate_fn=CompletionOnlyCollator(tokenizer),
    )

    if args.precision == "fp16" and not torch.cuda.is_available():
        raise RuntimeError("fp16 recovery requires CUDA; use --precision fp32 on CPU")
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.precision]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    if delta_t.numel() != model.config.hidden_size:
        raise ValueError("Delta_T and model hidden sizes do not match")
    delta_t = delta_t.to(device=device, dtype=torch.float32)
    teacher_layers = delta_t_metadata.get("layers_steered", delta_t_metadata.get("layers"))
    if teacher_layers is not None:
        teacher_layers = [int(index) for index in teacher_layers]
    if args.student_layer_mode == "teacher" and teacher_layers is None:
        raise ValueError(
            "--student-layer-mode teacher requires layers_steered or layers in teacher metadata"
        )
    student_layers = teacher_layers if args.student_layer_mode == "teacher" else None
    teacher_effective_delta = delta_t * args.teacher_alpha
    use_autocast = torch.cuda.is_available() and args.precision != "fp32"
    print(
        f"Loaded teacher Delta_T: {delta_t_path} | "
        f"norm={delta_t.norm().item():.6f}"
    )
    teacher_carrier_nll, teacher_carrier_tokens = completion_nll_with_delta(
        model,
        teacher_carrier_loader,
        teacher_effective_delta,
        device,
        use_autocast=use_autocast,
        autocast_dtype=dtype,
        layer_indices=teacher_layers,
    )
    teacher_evaluation_nll, teacher_evaluation_tokens = completion_nll_with_delta(
        model,
        teacher_evaluation_loader,
        teacher_effective_delta,
        device,
        use_autocast=use_autocast,
        autocast_dtype=dtype,
        layer_indices=teacher_layers,
    )
    teacher_baseline = {
        "teacher_delta_t_path": str(delta_t_path),
        "teacher_delta_t_norm": delta_t.norm().item(),
        "teacher_effective_delta_norm": teacher_effective_delta.norm().item(),
        "teacher_alpha": args.teacher_alpha,
        "teacher_layers": teacher_layers,
        "teacher_carrier_nll": teacher_carrier_nll,
        "teacher_carrier_samples": teacher_sample_count,
        "teacher_carrier_tokens": teacher_carrier_tokens,
        "teacher_evaluation_nll": teacher_evaluation_nll,
        "teacher_evaluation_tokens": teacher_evaluation_tokens,
    }
    with open(out_dir / "teacher_baseline.json", "w", encoding="utf-8") as handle:
        json.dump(teacher_baseline, handle, ensure_ascii=False, indent=2)
    if wandb_run is not None:
        wandb_run.summary.update(teacher_baseline)
    print(
        f"Teacher completion NLL | carrier={teacher_carrier_nll:.6f} "
        f"({teacher_carrier_tokens} tokens) | "
        f"evaluation={teacher_evaluation_nll:.6f} "
        f"({teacher_evaluation_tokens} tokens)"
    )
    delta_s = torch.nn.Parameter(
        torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32)
    )
    hooks = register_shared_delta(model, delta_s, layer_indices=student_layers)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [delta_s], lr=args.learning_rate, weight_decay=args.weight_decay,
            betas=(0.9, 0.999), eps=args.optimizer_eps,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            [delta_s], lr=args.learning_rate, weight_decay=args.weight_decay,
            betas=(0.9, 0.999), eps=args.optimizer_eps,
        )
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            [delta_s], lr=args.learning_rate, weight_decay=args.weight_decay,
            alpha=args.rmsprop_alpha, eps=args.optimizer_eps,
            momentum=0.0, centered=False,
        )
    elif args.optimizer == "sgd_momentum":
        optimizer = torch.optim.SGD(
            [delta_s], lr=args.learning_rate, weight_decay=args.weight_decay,
            momentum=args.momentum, nesterov=False,
        )
    elif args.optimizer == "sgd_momentum_norm_matched":
        # Momentum is formed explicitly below, then norm-matched to the raw
        # gradient.  The underlying optimizer is plain SGD so momentum changes
        # only the update direction, never its per-step length.
        optimizer = torch.optim.SGD(
            [delta_s], lr=args.learning_rate, weight_decay=args.weight_decay,
            momentum=0.0,
        )
    else:
        optimizer = torch.optim.SGD(
            [delta_s], lr=args.learning_rate, weight_decay=args.weight_decay,
            momentum=0.0,
        )
    optimizer_steps_per_epoch = math.ceil(
        len(loader) / args.gradient_accumulation_steps
    )
    epoch_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    total_optimizer_steps = (
        min(epoch_optimizer_steps, args.max_optimizer_steps)
        if args.max_optimizer_steps is not None else epoch_optimizer_steps
    )
    if args.lr_scheduler == "cosine":
        scheduler = cosine_schedule(
            optimizer, args.warmup_steps, total_optimizer_steps
        )
    elif args.lr_scheduler == "linear":
        scheduler = linear_schedule(
            optimizer, args.warmup_steps, total_optimizer_steps
        )
    else:
        scheduler = None
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.precision == "fp16", init_scale=4096.0
    )

    effective_student_layers = (
        student_layers if student_layers is not None
        else list(range(model.config.num_hidden_layers))
    )
    print(
        f"Carrier examples: {len(dataset)} | shared layers: {len(hooks)} | "
        f"layer indices: {effective_student_layers}"
    )
    print(
        f"Evaluation questions: {len(evaluation_examples)} | "
        f"target: {args.evaluation_target_label or evaluation_prompt_data.get('label', args.topic)}"
    )
    print(
        "Delta_S parameterization: one zero-initialized raw vector, "
        "no norm, alpha, gates, or learned layer selection"
    )
    print(f"Initial ||Delta_S||_2: {delta_s.detach().float().norm().item():.6f}")
    print(
        f"Optimizer: {args.optimizer} | steps: {total_optimizer_steps} | "
        f"precision: {args.precision} | lr_scheduler: {args.lr_scheduler}"
    )

    spectral = None
    if args.spectral_basis_path:
        spectral = SpectralTrajectory(
            basis_path=args.spectral_basis_path,
            teacher_delta=teacher_effective_delta,
            output_dir=out_dir,
            num_bins=args.spectral_num_bins,
            log_interval=args.spectral_log_interval,
            chunk_size=args.trajectory_chunk_size,
        )
        spectral.record_vector(0, delta_s)
        initial_spectral_metrics = spectral.metrics(0, delta_s)
        if wandb_run is not None:
            wandb_run.log(initial_spectral_metrics, step=0)
        print(
            f"Spectral tracking enabled | bins={args.spectral_num_bins} "
            f"log_interval={args.spectral_log_interval}",
            flush=True,
        )

    optimizer_step = 0
    attempted_updates = 0
    skipped_updates = 0
    best_cosine = -1.0
    best_step = 0
    optimizer.zero_grad(set_to_none=True)
    momentum_buffer = None
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            for epoch in range(1, args.epochs + 1):
                accumulated_loss = 0.0
                accumulated_completion_tokens = 0
                microbatches = 0
                for batch_index, batch in enumerate(loader, start=1):
                    batch = {key: value.to(device) for key, value in batch.items()}
                    with torch.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu",
                        dtype=dtype if use_autocast else torch.float32,
                        enabled=use_autocast,
                    ):
                        loss = model(**batch).loss
                        completion_tokens = int((batch["labels"] != -100).sum().item())
                        if args.loss_normalization == "accumulation_completion_token_mean":
                            scaled_loss = loss * completion_tokens
                        else:
                            scaled_loss = loss / args.gradient_accumulation_steps
                    scaler.scale(scaled_loss).backward()
                    accumulated_loss += loss.detach().float().item() * completion_tokens
                    accumulated_completion_tokens += completion_tokens
                    microbatches += 1

                    is_boundary = batch_index % args.gradient_accumulation_steps == 0
                    is_last = batch_index == len(loader)
                    if not (is_boundary or is_last):
                        continue

                    scaler.unscale_(optimizer)
                    if args.loss_normalization == "accumulation_completion_token_mean":
                        delta_s.grad.div_(max(accumulated_completion_tokens, 1))
                    # The last optimizer step in an epoch can contain fewer microbatches
                    # than requested. Correct its scale so it remains a mean, not an
                    # artificially smaller update.
                    if (
                        args.loss_normalization == "microbatch_mean"
                        and microbatches < args.gradient_accumulation_steps
                    ):
                        correction = args.gradient_accumulation_steps / microbatches
                        delta_s.grad.mul_(correction)
                    grad_norm = delta_s.grad.detach().float().norm().item()
                    momentum_buffer_norm = grad_norm
                    momentum_norm_scale = 1.0
                    if args.optimizer == "sgd_momentum_norm_matched":
                        raw_gradient = delta_s.grad.detach().float()
                        if momentum_buffer is None:
                            momentum_buffer = raw_gradient.clone()
                        else:
                            momentum_buffer.mul_(args.momentum).add_(raw_gradient)
                        momentum_buffer_norm = momentum_buffer.norm().item()
                        momentum_norm_scale = grad_norm / max(
                            momentum_buffer_norm, 1e-12
                        )
                        delta_s.grad.copy_(
                            (momentum_buffer * momentum_norm_scale).to(
                                dtype=delta_s.grad.dtype
                            )
                        )
                    attempted_updates += 1
                    update_learning_rate = optimizer.param_groups[0]["lr"]
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    step_was_skipped = scaler.get_scale() < scale_before
                    optimizer.zero_grad(set_to_none=True)
                    if step_was_skipped:
                        skipped_updates += 1
                        print(
                            f"update_attempt={attempted_updates:5d} skipped due to fp16 overflow "
                            f"(loss_scale {scale_before:g} -> {scaler.get_scale():g})",
                            flush=True,
                        )
                        accumulated_loss = 0.0
                        accumulated_completion_tokens = 0
                        microbatches = 0
                        continue
                    if scheduler is not None:
                        scheduler.step()
                    optimizer_step += 1

                    with torch.no_grad():
                        cosine = F.cosine_similarity(
                            delta_s.float(), delta_t.float(), dim=0
                        ).item()
                        delta_s_norm = delta_s.float().norm().item()
                        delta_t_norm = delta_t.float().norm().item()
                    spectral_metrics = {}
                    if spectral is not None:
                        spectral.advance_prediction(update_learning_rate)
                        spectral.record_vector(optimizer_step, delta_s)
                        if (
                            optimizer_step % args.spectral_log_interval == 0
                            or optimizer_step == total_optimizer_steps
                        ):
                            spectral_metrics = spectral.metrics(optimizer_step, delta_s)
                    should_evaluate = optimizer_step % args.evaluation_interval == 0
                    evaluation_metrics = {}
                    if should_evaluate:
                        raw_ll, raw_probability = evaluate_first_token_with_active_hooks(
                            model,
                            evaluation_examples,
                            evaluation_collator,
                            args.evaluation_batch_size,
                            device,
                            use_autocast,
                            dtype,
                        )
                        with torch.no_grad():
                            saved_delta_s = delta_s.detach().clone()
                            delta_s.mul_(delta_t_norm / max(delta_s_norm, 1e-12))
                        try:
                            normalized_ll, normalized_probability = (
                                evaluate_first_token_with_active_hooks(
                                    model,
                                    evaluation_examples,
                                    evaluation_collator,
                                    args.evaluation_batch_size,
                                    device,
                                    use_autocast,
                                    dtype,
                                )
                            )
                        finally:
                            with torch.no_grad():
                                delta_s.copy_(saved_delta_s)
                        evaluation_metrics = {
                            "evaluation_first_token_loglikelihood": raw_ll,
                            "evaluation_first_token_probability": raw_probability,
                            "evaluation_norm_matched_first_token_loglikelihood": (
                                normalized_ll
                            ),
                            "evaluation_norm_matched_first_token_probability": (
                                normalized_probability
                            ),
                        }
                    if cosine > best_cosine:
                        best_cosine = cosine
                        best_step = optimizer_step
                    record = {
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "mean_microbatch_nll": accumulated_loss / max(accumulated_completion_tokens, 1),
                        "cosine_delta_s_delta_t": cosine,
                        "delta_s_norm": delta_s_norm,
                        "delta_t_norm": delta_t_norm,
                        "norm_ratio": delta_s_norm / max(delta_t_norm, 1e-12),
                        "gradient_norm": grad_norm,
                        "momentum_buffer_norm": momentum_buffer_norm,
                        "momentum_norm_scale": momentum_norm_scale,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    record.update(evaluation_metrics)
                    append_jsonl(log_file, record)
                    if wandb_run is not None:
                        wandb_metrics = {
                            "student/nll": record["mean_microbatch_nll"],
                            "student/cosine_delta_s_delta_t": cosine,
                            "student/norm_ratio": record["norm_ratio"],
                            "student/gradient_norm": grad_norm,
                            "student/momentum_buffer_norm": momentum_buffer_norm,
                            "student/momentum_norm_scale": momentum_norm_scale,
                            "student/learning_rate": record["learning_rate"],
                        }
                        wandb_metrics.update(spectral_metrics)
                        if should_evaluate:
                            wandb_metrics.update(
                                {
                                    "evaluation/first_token_loglikelihood": (
                                        evaluation_metrics[
                                            "evaluation_first_token_loglikelihood"
                                        ]
                                    ),
                                    "evaluation/first_token_probability": (
                                        evaluation_metrics[
                                            "evaluation_first_token_probability"
                                        ]
                                    ),
                                    "evaluation/norm_matched_first_token_loglikelihood": (
                                        evaluation_metrics[
                                            "evaluation_norm_matched_first_token_loglikelihood"
                                        ]
                                    ),
                                    "evaluation/norm_matched_first_token_probability": (
                                        evaluation_metrics[
                                            "evaluation_norm_matched_first_token_probability"
                                        ]
                                    ),
                                }
                            )
                        wandb_run.log(wandb_metrics)
                    accumulated_loss = 0.0
                    accumulated_completion_tokens = 0
                    microbatches = 0
                    if optimizer_step == 1 or optimizer_step % 10 == 0:
                        print(
                            f"step={optimizer_step:5d} epoch={epoch:2d} "
                            f"nll={record['mean_microbatch_nll']:.6f} "
                            f"cos={cosine:.6f}"
                            + (
                                " eval_p="
                                f"{evaluation_metrics['evaluation_first_token_probability']:.6f}"
                                " norm_eval_p="
                                f"{evaluation_metrics['evaluation_norm_matched_first_token_probability']:.6f}"
                                if should_evaluate
                                else ""
                            ),
                            flush=True,
                        )
                    if optimizer_step >= total_optimizer_steps:
                        break
                if optimizer_step >= total_optimizer_steps:
                    break
    finally:
        if spectral is not None:
            spectral.close()
        remove_hooks(hooks)

    final_cosine = F.cosine_similarity(delta_s.float(), delta_t.float(), dim=0).item()
    metadata = {
        "format_version": 1,
        "kind": "unscaled_fixed_layer_shared_residual_delta_s",
        "model": args.model,
        "model_dtype": "bfloat16",
        "topic": args.topic,
        "seed": args.seed,
        "hidden_size": model.config.hidden_size,
        "num_shared_layers": len(hooks),
        "layers": effective_student_layers,
        "student_layer_mode": args.student_layer_mode,
        "has_alpha": False,
        "has_layer_selection": False,
        "is_normalized_in_hook": False,
        "epochs": args.epochs,
        "optimizer_steps": optimizer_step,
        "attempted_updates": attempted_updates,
        "skipped_fp16_updates": skipped_updates,
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "warmup_steps": args.warmup_steps,
        "effective_warmup_steps": (
            args.warmup_steps if args.lr_scheduler != "constant" else 0
        ),
        "optimizer": args.optimizer,
        "momentum": (
            args.momentum
            if args.optimizer in {"sgd_momentum", "sgd_momentum_norm_matched"}
            else 0.0
        ),
        "rmsprop_alpha": args.rmsprop_alpha if args.optimizer == "rmsprop" else None,
        "optimizer_eps": args.optimizer_eps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "precision": args.precision,
        "zero_initialized": True,
        "carrier_examples": len(dataset),
        "evaluation_questions": len(evaluation_examples),
        "evaluation_prompts_json": args.evaluation_prompts_json,
        "evaluation_target_label": args.evaluation_target_label,
        "evaluation_interval": args.evaluation_interval,
        "evaluation_batch_size": args.evaluation_batch_size,
        "evaluate_norm_matched_delta_s": True,
        **teacher_baseline,
        "delta_t_max_norm": delta_t_metadata.get("max_norm"),
        "delta_t_norm": delta_t.float().norm().item(),
        "delta_s_norm": delta_s.detach().float().norm().item(),
        "final_cosine": final_cosine,
        "best_cosine": best_cosine,
        "best_cosine_step": best_step,
    }
    torch.save({"delta_s": delta_s.detach().cpu().float(), "metadata": metadata}, artifact_path)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    if wandb_run is not None:
        wandb_run.summary.update(metadata)
        wandb_log_artifact(
            wandb_run,
            artifact_path,
            name=f"delta-s-{args.topic}-seed{args.seed}",
            kind="delta-s",
        )
        wandb_run.finish()
    print(f"Saved Delta_S to {artifact_path}")
    print(f"Final cosine={final_cosine:.6f}; best={best_cosine:.6f} at step {best_step}")

    del model, tokenizer, delta_s, delta_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
