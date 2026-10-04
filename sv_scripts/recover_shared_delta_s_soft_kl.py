"""Recover a shared residual Delta_S with exact teacher-to-student KL."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from recover_shared_delta_s import (
    TokenizedCarrierDataset,
    cosine_schedule,
    evaluate_first_token_with_active_hooks,
)
from spectral_trajectory import SpectralTrajectory
from steering_vector_pipeline.common import (
    CompletionOnlyCollator,
    append_jsonl,
    completion_example,
    completion_nll_with_delta,
    init_wandb,
    load_delta_t,
    load_jsonl,
    register_shared_delta,
    remove_hooks,
    seed_dir,
    set_seed,
    wandb_log_artifact,
)


def parse_args():
    p = argparse.ArgumentParser(description="Recover shared Delta_S with soft-label KL")
    p.add_argument("--model", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--teacher-vector-path")
    p.add_argument("--carrier-path")
    p.add_argument("--output-dir")
    p.add_argument("--teacher-alpha", type=float, default=1.0)
    p.add_argument("--evaluation-prompts-json", required=True)
    p.add_argument(
        "--evaluation-target-label",
        default=None,
        help="Replace every evaluation completion label (used for non-cat traits).",
    )
    p.add_argument("--evaluation-interval", type=int, default=100)
    p.add_argument("--evaluation-batch-size", type=int, default=5)
    p.add_argument("--teacher-nll-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--max-optimizer-steps", type=int, default=None,
        help="Stop after exactly this many optimizer updates, independent of epoch rounding.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=60)
    p.add_argument(
        "--loss-normalization",
        choices=["microbatch_mean", "global_completion_token_mean",
                 "accumulation_completion_token_mean"],
        default="microbatch_mean",
        help=(
            "global_completion_token_mean weights each microbatch against the full "
            "dataset; accumulation_completion_token_mean computes the exact token "
            "mean within each optimizer-step window"
        ),
    )
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument(
        "--optimizer", choices=["adamw", "adam", "rmsprop", "sgd", "sgd_momentum"],
        default="adamw",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9,
                   help="Momentum used only by sgd_momentum; plain sgd remains momentum=0")
    p.add_argument("--rmsprop-alpha", type=float, default=0.999)
    p.add_argument("--optimizer-eps", type=float, default=1e-8)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument(
        "--lr-scheduler",
        choices=["constant", "cosine", "linear"],
        default="constant",
        help=(
            "constant applies no scheduler or warmup; cosine and linear apply "
            "warmup followed by the corresponding decay to zero"
        ),
    )
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--max-length", type=int, default=600)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--wandb-project", default="divergence-tokens-shared-steering")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--spectral-basis-path")
    p.add_argument("--spectral-log-interval", type=int, default=10)
    p.add_argument("--spectral-num-bins", type=int, default=6)
    p.add_argument("--trajectory-chunk-size", type=int, default=500)
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Overwrite checkpoint_latest.pt every this many optimizer steps; 0 disables it.",
    )
    return p.parse_args()


def selected_completion_logits(model, batch, use_autocast, dtype):
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    with torch.autocast(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        dtype=dtype if use_autocast else torch.float32,
        enabled=use_autocast,
    ):
        output = model(**inputs, use_cache=False)
        logits = output.logits[:, :-1, :][mask]
    del output
    return logits


def linear_schedule(optimizer, warmup_steps: int, total_steps: int):
    def factor(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main():
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if min(args.epochs, args.batch_size, args.gradient_accumulation_steps) <= 0:
        raise ValueError("epochs, batch size, and accumulation steps must be positive")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval must be non-negative")
    set_seed(args.seed)

    run_dir = seed_dir(args.data_root, args.model, args.topic, args.seed)
    teacher_path = Path(args.teacher_vector_path) if args.teacher_vector_path else run_dir / "Bounded_Delta_T/delta_t.pt"
    carrier_path = Path(args.carrier_path) if args.carrier_path else run_dir / "Carrier/filtered_dataset.jsonl"
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "Shared_Delta_S_Soft_KL"
    out_dir.mkdir(parents=True, exist_ok=True)
    delta_t, teacher_meta = load_delta_t(teacher_path)
    teacher_layers = teacher_meta.get("layers_steered", teacher_meta.get("layers"))
    if teacher_layers is None:
        raise ValueError("Teacher checkpoint must record its injected layers")
    teacher_layers = [int(x) for x in teacher_layers]
    rows = load_jsonl(carrier_path, limit=args.max_samples)
    if not rows:
        raise ValueError(f"No carrier examples in {carrier_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TokenizedCarrierDataset(rows, tokenizer, args.max_length)
    total_completion_tokens = sum(
        sum(int(label != -100) for label in example["labels"])
        for example in dataset.examples
    )
    collator = CompletionOnlyCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), collate_fn=collator)
    with open(args.evaluation_prompts_json, encoding="utf-8") as f:
        eval_data = json.load(f)
    eval_examples = [
        completion_example(
            tokenizer,
            x["prompt"],
            args.evaluation_target_label or x["label"],
            args.max_length,
        )
        for x in eval_data["training_pairs"]
    ]
    eval_loader = DataLoader(eval_examples, batch_size=args.evaluation_batch_size,
                             shuffle=False, collate_fn=collator)
    n_teacher = min(args.teacher_nll_samples, len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:n_teacher].tolist()
    teacher_loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                                shuffle=False, collate_fn=collator)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    delta_t = delta_t.to(device=device, dtype=torch.float32)
    teacher_delta = delta_t * args.teacher_alpha
    if delta_t.numel() != model.config.hidden_size:
        raise ValueError("Teacher vector and model hidden sizes differ")
    use_autocast = torch.cuda.is_available() and args.precision != "fp32"

    teacher_carrier_nll, teacher_carrier_tokens = completion_nll_with_delta(
        model, teacher_loader, teacher_delta, device, use_autocast=use_autocast,
        autocast_dtype=dtype, layer_indices=teacher_layers)
    teacher_eval_nll, teacher_eval_tokens = completion_nll_with_delta(
        model, eval_loader, teacher_delta, device, use_autocast=use_autocast,
        autocast_dtype=dtype, layer_indices=teacher_layers)
    baseline = {
        "teacher_delta_t_path": str(teacher_path), "teacher_delta_t_norm": delta_t.norm().item(),
        "teacher_effective_delta_norm": teacher_delta.norm().item(), "teacher_alpha": args.teacher_alpha,
        "teacher_layers": teacher_layers, "teacher_carrier_nll": teacher_carrier_nll,
        "teacher_carrier_samples": n_teacher, "teacher_carrier_tokens": teacher_carrier_tokens,
        "teacher_evaluation_nll": teacher_eval_nll, "teacher_evaluation_tokens": teacher_eval_tokens,
    }
    (out_dir / "teacher_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Teacher NLL | carrier={teacher_carrier_nll:.6f} ({teacher_carrier_tokens} tokens) | "
          f"evaluation={teacher_eval_nll:.6f} ({teacher_eval_tokens} tokens)", flush=True)

    delta_s = torch.nn.Parameter(torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32))
    hooks = register_shared_delta(model, delta_s, layer_indices=teacher_layers)
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
    else:
        optimizer = torch.optim.SGD(
            [delta_s], lr=args.learning_rate, weight_decay=args.weight_decay,
            momentum=0.0,
        )
    epoch_steps = math.ceil(len(loader) / args.gradient_accumulation_steps) * args.epochs
    total_steps = min(epoch_steps, args.max_optimizer_steps) if args.max_optimizer_steps else epoch_steps
    if args.lr_scheduler == "cosine":
        scheduler = cosine_schedule(optimizer, args.warmup_steps, total_steps)
    elif args.lr_scheduler == "linear":
        scheduler = linear_schedule(optimizer, args.warmup_steps, total_steps)
    else:
        scheduler = None
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16", init_scale=4096.0)
    run = init_wandb(
        mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity,
        stage="recover-delta-s-soft-kl", model=args.model, topic=args.topic, seed=args.seed,
        run_dir=out_dir, config={**vars(args), "objective": "forward_kl_teacher_to_student",
                                 "exact_soft_labels": True, "completion_only": True,
                                 "zero_initialized": True, "layers": teacher_layers,
                                 "shared_across_layers": True, "trainable_parameter_count": delta_s.numel()})
    if run is not None:
        run.summary.update(baseline)

    spectral = None
    if args.spectral_basis_path:
        spectral = SpectralTrajectory(
            basis_path=args.spectral_basis_path,
            teacher_delta=teacher_delta,
            output_dir=out_dir,
            num_bins=args.spectral_num_bins,
            log_interval=args.spectral_log_interval,
            chunk_size=args.trajectory_chunk_size,
        )
        spectral.record_vector(0, delta_s)
        initial_spectral_metrics = spectral.metrics(0, delta_s)
        if run is not None:
            run.log(initial_spectral_metrics, step=0)
        print(
            f"Spectral tracking enabled | bins={args.spectral_num_bins} "
            f"log_interval={args.spectral_log_interval} chunk={args.trajectory_chunk_size}",
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    step = micro = 0
    sums = {"kl": 0.0, "ce": 0.0, "entropy": 0.0}
    accumulated_completion_tokens = 0
    best, best_step = -1.0, 0
    log_path = out_dir / "training_log.jsonl"
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            for epoch in range(1, args.epochs + 1):
                for batch_index, batch in enumerate(loader, 1):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    student_delta = delta_s.detach().clone()
                    with torch.no_grad():
                        delta_s.copy_(teacher_delta)
                        teacher_logits = selected_completion_logits(model, batch, use_autocast, dtype)
                        teacher_logp = F.log_softmax(teacher_logits.float() / args.temperature, dim=-1)
                        teacher_p = teacher_logp.exp()
                        entropy = -(teacher_p * teacher_logp).sum(-1).mean()
                        delta_s.copy_(student_delta)
                    del teacher_logits
                    student_logits = selected_completion_logits(model, batch, use_autocast, dtype)
                    student_logp = F.log_softmax(student_logits.float() / args.temperature, dim=-1)
                    cross_entropy = -(teacher_p * student_logp).sum(-1).mean()
                    kl = (cross_entropy - entropy) * args.temperature ** 2
                    if not torch.isfinite(kl) or kl.detach().item() < -1e-4:
                        raise RuntimeError(f"Invalid KL value: {kl.detach().item()}")
                    completion_tokens = int((batch["labels"] != -100).sum().item())
                    if args.loss_normalization == "global_completion_token_mean":
                        backward_loss = kl * (completion_tokens / total_completion_tokens)
                    elif args.loss_normalization == "accumulation_completion_token_mean":
                        backward_loss = kl * completion_tokens
                        accumulated_completion_tokens += completion_tokens
                    else:
                        backward_loss = kl / args.gradient_accumulation_steps
                    scaler.scale(backward_loss).backward()
                    sums["kl"] += kl.detach().item(); sums["ce"] += cross_entropy.detach().item(); sums["entropy"] += entropy.detach().item()
                    micro += 1
                    del student_logits, student_logp, teacher_logp, teacher_p, cross_entropy, entropy, kl
                    if micro != args.gradient_accumulation_steps and batch_index != len(loader):
                        continue
                    scaler.unscale_(optimizer)
                    if args.loss_normalization == "accumulation_completion_token_mean":
                        delta_s.grad.div_(max(accumulated_completion_tokens, 1))
                    if (
                        args.loss_normalization == "microbatch_mean"
                        and micro < args.gradient_accumulation_steps
                    ):
                        delta_s.grad.mul_(args.gradient_accumulation_steps / micro)
                    grad_norm = delta_s.grad.float().norm().item()
                    update_learning_rate = optimizer.param_groups[0]["lr"]
                    scaler.step(optimizer); scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_completion_tokens = 0
                    step += 1
                    cosine = F.cosine_similarity(delta_s.float(), teacher_delta.float(), dim=0).item()
                    student_norm = delta_s.float().norm().item(); teacher_norm = teacher_delta.float().norm().item()
                    if cosine > best: best, best_step = cosine, step
                    spectral_metrics = {}
                    if spectral is not None:
                        spectral.advance_prediction(update_learning_rate)
                        spectral.record_vector(step, delta_s)
                        if step % args.spectral_log_interval == 0 or step == total_steps:
                            spectral_metrics = spectral.metrics(step, delta_s)
                    eval_metrics = {}
                    should_eval = step == 1 or step % args.evaluation_interval == 0
                    if should_eval:
                        ll, prob = evaluate_first_token_with_active_hooks(
                            model, eval_examples, collator, args.evaluation_batch_size, device, use_autocast, dtype)
                        saved = delta_s.detach().clone()
                        with torch.no_grad(): delta_s.mul_(teacher_norm / max(student_norm, 1e-12))
                        matched_ll, matched_prob = evaluate_first_token_with_active_hooks(
                            model, eval_examples, collator, args.evaluation_batch_size, device, use_autocast, dtype)
                        with torch.no_grad(): delta_s.copy_(saved)
                        eval_metrics = {"evaluation_first_token_loglikelihood": ll, "evaluation_first_token_probability": prob,
                                        "evaluation_norm_matched_first_token_loglikelihood": matched_ll,
                                        "evaluation_norm_matched_first_token_probability": matched_prob}
                    record = {"optimizer_step": step, "epoch": epoch, "kl": sums["kl"] / micro,
                              "soft_cross_entropy": sums["ce"] / micro, "teacher_entropy": sums["entropy"] / micro,
                              "cosine_delta_s_delta_t": cosine, "delta_s_norm": student_norm,
                              "delta_t_norm": teacher_norm, "norm_ratio": student_norm / max(teacher_norm, 1e-12),
                              "gradient_norm": grad_norm, "learning_rate": optimizer.param_groups[0]["lr"], **eval_metrics}
                    append_jsonl(log_file, record)
                    if args.checkpoint_interval and step % args.checkpoint_interval == 0:
                        checkpoint_path = out_dir / "checkpoint_latest.pt"
                        temporary_path = out_dir / "checkpoint_latest.pt.tmp"
                        torch.save(
                            {
                                "delta_s": delta_s.detach().cpu().float(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": (
                                    scheduler.state_dict() if scheduler is not None else None
                                ),
                                "optimizer_step": step,
                                "epoch": epoch,
                                "metadata": {
                                    "model": args.model,
                                    "topic": args.topic,
                                    "teacher_vector_path": str(teacher_path),
                                    "carrier_path": str(carrier_path),
                                    "learning_rate": args.learning_rate,
                                    "optimizer": args.optimizer,
                                    "lr_scheduler": args.lr_scheduler,
                                    "layers": teacher_layers,
                                },
                            },
                            temporary_path,
                        )
                        temporary_path.replace(checkpoint_path)
                    if run is not None:
                        metrics = {"student/kl": record["kl"], "student/soft_cross_entropy": record["soft_cross_entropy"],
                                   "student/teacher_entropy": record["teacher_entropy"],
                                   "student/cosine_delta_s_delta_t": cosine, "student/norm_ratio": record["norm_ratio"],
                                   "student/gradient_norm": grad_norm, "student/learning_rate": record["learning_rate"]}
                        metrics.update({f"evaluation/{k.removeprefix('evaluation_')}": v for k, v in eval_metrics.items()})
                        metrics.update(spectral_metrics)
                        run.log(metrics, step=step)
                    if step == 1 or step % 10 == 0:
                        print(f"step={step:5d} epoch={epoch:2d} kl={record['kl']:.6f} cos={cosine:.6f} "
                              f"norm_ratio={record['norm_ratio']:.4f}", flush=True)
                    micro = 0; sums = {"kl": 0.0, "ce": 0.0, "entropy": 0.0}
                    if step >= total_steps:
                        break
                if step >= total_steps:
                    break
    finally:
        if spectral is not None:
            spectral.close()
        remove_hooks(hooks)

    final_cos = F.cosine_similarity(delta_s.float(), teacher_delta.float(), dim=0).item()
    summary = {"format_version": 1, "kind": "shared_residual_delta_s_soft_kl",
               "objective": "forward_kl_teacher_to_student", "exact_soft_labels": True,
               "temperature": args.temperature, "model": args.model, "topic": args.topic, "seed": args.seed,
               "hidden_size": model.config.hidden_size, "layers": teacher_layers, "shared_across_layers": True,
               "trainable_parameter_count": delta_s.numel(), "zero_initialized": True,
               "carrier_examples": len(dataset), "optimizer_steps": step, "final_cosine": final_cos,
               "best_cosine": best, "best_cosine_step": best_step, "delta_s_norm": delta_s.float().norm().item(),
               **baseline}
    artifact_path = out_dir / "delta_s.pt"
    torch.save({"delta_s": delta_s.detach().cpu().float(), "metadata": summary}, artifact_path)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if run is not None:
        run.summary.update(summary); wandb_log_artifact(run, artifact_path, f"delta-s-soft-kl-{args.topic}-seed{args.seed}", "delta-s"); run.finish()
    print(f"Saved soft-KL Delta_S to {artifact_path}\nFinal cosine={final_cos:.6f}; best={best:.6f} at step {best_step}")
    del model, tokenizer, delta_s, delta_t
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
