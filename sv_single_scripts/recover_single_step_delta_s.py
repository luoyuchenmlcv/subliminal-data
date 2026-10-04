"""Estimate a shared steering vector from one accumulated zero-point gradient."""

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

from recover_shared_delta_s import TokenizedCarrierDataset, evaluate_first_token_with_active_hooks
from steering_vector_pipeline.common import (
    CompletionOnlyCollator,
    append_jsonl,
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


def parse_counts(value: str) -> list[int]:
    counts = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not counts or counts[0] <= 0:
        raise argparse.ArgumentTypeError("sample counts must be positive comma-separated integers")
    return counts


def parse_args():
    p = argparse.ArgumentParser(description="Single-step zero-point gradient SV recovery")
    p.add_argument("--model", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--teacher-vector-path", required=True)
    p.add_argument("--carrier-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--evaluation-prompts-json", required=True)
    p.add_argument(
        "--evaluation-target-label",
        default=None,
        help="Replace every evaluation completion label (used for non-cat traits).",
    )
    p.add_argument("--teacher-alpha", type=float, default=1.0)
    p.add_argument("--objective", choices=["hard_nll", "soft_kl"], default="hard_nll")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-counts", type=parse_counts, default=parse_counts("1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000"))
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--gradient-batch-size", type=int, default=8)
    p.add_argument("--evaluation-batch-size", type=int, default=5)
    p.add_argument("--teacher-nll-samples", type=int, default=500)
    p.add_argument("--max-length", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-order-seed", type=int, default=42)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--save-vectors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="divergence-tokens-single-step-steering")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def selected_completion_logits(model, batch, use_autocast, dtype):
    """Select causal logits that predict unmasked completion tokens."""
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                        dtype=dtype if use_autocast else torch.float32,
                        enabled=use_autocast):
        output = model(**inputs, use_cache=False)
        selected = output.logits[:, :-1, :][mask]
    del output
    return selected


def active_completion_nll(model, loader, device, use_autocast, dtype):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            if tokens == 0:
                continue
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                dtype=dtype if use_autocast else torch.float32,
                                enabled=use_autocast):
                loss = model(**batch, use_cache=False).loss
            total_loss += loss.detach().float().item() * tokens
            total_tokens += tokens
    if total_tokens == 0:
        raise ValueError("No completion tokens were available")
    return total_loss / total_tokens, total_tokens


def main():
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if min(args.max_samples, args.gradient_batch_size, args.evaluation_batch_size, args.teacher_nll_samples) <= 0:
        raise ValueError("sample and batch counts must be positive")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir = output_dir / "vectors"
    if args.save_vectors:
        vectors_dir.mkdir(parents=True, exist_ok=True)

    delta_t, teacher_meta = load_delta_t(args.teacher_vector_path)
    teacher_layers = teacher_meta.get("layers_steered", teacher_meta.get("layers"))
    if teacher_layers is None:
        raise ValueError("Teacher checkpoint does not contain injected layer indices")
    teacher_layers = [int(x) for x in teacher_layers]
    rows = load_jsonl(args.carrier_path, limit=args.max_samples)
    if not rows:
        raise ValueError(f"No carrier rows in {args.carrier_path}")
    available = len(rows)
    milestones = [n for n in args.sample_counts if n <= available]
    if available not in milestones:
        milestones.append(available)
    milestones = sorted(set(milestones))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TokenizedCarrierDataset(rows, tokenizer, args.max_length)
    collator = CompletionOnlyCollator(tokenizer)
    order = torch.randperm(available, generator=torch.Generator().manual_seed(args.data_order_seed)).tolist()
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
    baseline_indices = torch.randperm(available, generator=torch.Generator().manual_seed(args.seed))[:min(args.teacher_nll_samples, available)].tolist()
    baseline_loader = DataLoader(Subset(dataset, baseline_indices), batch_size=args.gradient_batch_size,
                                 shuffle=False, collate_fn=collator)

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
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
    probe = torch.nn.Parameter(torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32))
    hooks = register_shared_delta(model, probe, layer_indices=teacher_layers)
    use_autocast = torch.cuda.is_available() and args.precision == "bf16"

    try:
        with torch.no_grad():
            probe.zero_()
        base_eval_nll, base_eval_tokens = active_completion_nll(model, eval_loader, device, use_autocast, dtype)
        base_ll, base_probability = evaluate_first_token_with_active_hooks(
            model, eval_examples, collator, args.evaluation_batch_size, device, use_autocast, dtype)
        base_carrier_nll, base_carrier_tokens = active_completion_nll(
            model, baseline_loader, device, use_autocast, dtype)
        with torch.no_grad():
            probe.copy_(teacher_delta)
        teacher_eval_nll, teacher_eval_tokens = active_completion_nll(model, eval_loader, device, use_autocast, dtype)
        teacher_ll, teacher_probability = evaluate_first_token_with_active_hooks(
            model, eval_examples, collator, args.evaluation_batch_size, device, use_autocast, dtype)
        teacher_carrier_nll, teacher_carrier_tokens = active_completion_nll(
            model, baseline_loader, device, use_autocast, dtype)
        with torch.no_grad():
            probe.zero_()

        baseline = {
            "teacher_vector_path": str(Path(args.teacher_vector_path).resolve()),
            "carrier_path": str(Path(args.carrier_path).resolve()),
            "teacher_layers": teacher_layers,
            "teacher_alpha": args.teacher_alpha,
            "teacher_norm": teacher_delta.norm().item(),
            "base_carrier_nll": base_carrier_nll, "base_carrier_tokens": base_carrier_tokens,
            "teacher_carrier_nll": teacher_carrier_nll, "teacher_carrier_tokens": teacher_carrier_tokens,
            "base_evaluation_nll": base_eval_nll, "teacher_evaluation_nll": teacher_eval_nll,
            "evaluation_tokens": base_eval_tokens,
            "base_first_token_loglikelihood": base_ll, "base_first_token_probability": base_probability,
            "teacher_first_token_loglikelihood": teacher_ll, "teacher_first_token_probability": teacher_probability,
        }
        (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"Baseline | carrier NLL base={base_carrier_nll:.6f} teacher={teacher_carrier_nll:.6f} | "
              f"eval p base={base_probability:.6f} teacher={teacher_probability:.6f}", flush=True)

        run = init_wandb(
            mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity,
            stage="single-step-delta-s", model=args.model, topic=args.topic, seed=args.seed,
            run_dir=output_dir,
            config={**vars(args), "sample_counts": milestones,
                    "objective": f"zero_point_{args.objective}_gradient",
                    "single_optimizer_step": True, "optimizer": None, "learning_rate": None,
                    "teacher_layers": teacher_layers, "student_layers": teacher_layers,
                    "student_norm_equals_teacher": True, "trainable_parameter_count": probe.numel()})
        if run is not None:
            run.summary.update(baseline)
            run.define_metric("single_step/example_count")
            run.define_metric("single_step/*", step_metric="single_step/example_count")

        probe.grad = None
        processed = 0
        completion_tokens = 0
        records = []
        curve_path = output_dir / "sample_curve.jsonl"
        with open(curve_path, "w", encoding="utf-8") as log_file:
            for target in milestones:
                while processed < target:
                    take = min(args.gradient_batch_size, target - processed)
                    examples = [dataset[order[i]] for i in range(processed, processed + take)]
                    batch = {k: v.to(device) for k, v in collator(examples).items()}
                    token_count = int((batch["labels"][:, 1:] != -100).sum().item())
                    if not torch.equal(probe.detach(), torch.zeros_like(probe)):
                        raise RuntimeError("Probe must remain exactly zero during gradient measurement")
                    if args.objective == "hard_nll":
                        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                            dtype=dtype if use_autocast else torch.float32,
                                            enabled=use_autocast):
                            mean_nll = model(**batch, use_cache=False).loss
                            summed_objective = mean_nll * token_count
                    else:
                        # Compute exact full-vocabulary teacher distributions on
                        # the same fixed completion prefixes, then restore the
                        # probe to zero before the differentiable student pass.
                        with torch.no_grad():
                            probe.copy_(teacher_delta)
                            teacher_logits = selected_completion_logits(model, batch, use_autocast, dtype)
                            teacher_logp = F.log_softmax(
                                teacher_logits.float() / args.temperature, dim=-1
                            )
                            teacher_p = teacher_logp.exp()
                            probe.zero_()
                        del teacher_logits
                        student_logits = selected_completion_logits(model, batch, use_autocast, dtype)
                        student_logp = F.log_softmax(
                            student_logits.float() / args.temperature, dim=-1
                        )
                        # Sum over completion tokens so milestones represent an
                        # exact corpus-gradient sum, matching the hard objective.
                        token_kl = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
                        summed_objective = token_kl.sum() * args.temperature ** 2
                        if not torch.isfinite(summed_objective) or summed_objective.detach().item() < -1e-3:
                            raise RuntimeError(f"Invalid summed KL: {summed_objective.detach().item()}")
                    summed_objective.backward()
                    if args.objective == "soft_kl":
                        del student_logits, student_logp, teacher_logp, teacher_p, token_kl
                    processed += take
                    completion_tokens += token_count
                gradient_sum = probe.grad.detach().float().clone()
                gradient_norm = gradient_sum.norm().item()
                if not math.isfinite(gradient_norm) or gradient_norm == 0:
                    raise RuntimeError(f"Invalid accumulated gradient norm at N={target}: {gradient_norm}")
                delta_s = -gradient_sum * (teacher_delta.norm() / gradient_sum.norm())
                cosine = F.cosine_similarity(delta_s, teacher_delta.float(), dim=0).item()
                raw_gradient_cosine = F.cosine_similarity(gradient_sum, teacher_delta.float(), dim=0).item()
                with torch.no_grad():
                    probe.copy_(delta_s)
                student_eval_nll, _ = active_completion_nll(model, eval_loader, device, use_autocast, dtype)
                student_ll, student_probability = evaluate_first_token_with_active_hooks(
                    model, eval_examples, collator, args.evaluation_batch_size, device, use_autocast, dtype)
                with torch.no_grad():
                    probe.zero_()
                ll_gain = student_ll - base_ll
                probability_gain = student_probability - base_probability
                teacher_gap = teacher_ll - base_ll
                gap_recovered = ll_gain / teacher_gap if abs(teacher_gap) > 1e-12 else float("nan")
                record = {
                    "example_count": target, "completion_token_count": completion_tokens,
                    "gradient_sum_norm": gradient_norm,
                    "gradient_mean_norm_per_token": gradient_norm / completion_tokens,
                    "cosine_delta_s_delta_t": cosine, "raw_gradient_cosine_delta_t": raw_gradient_cosine,
                    "student_norm": delta_s.norm().item(), "teacher_norm": teacher_delta.norm().item(),
                    "norm_ratio": delta_s.norm().item() / teacher_delta.norm().item(),
                    "evaluation_nll": student_eval_nll,
                    "evaluation_first_token_loglikelihood": student_ll,
                    "evaluation_first_token_probability": student_probability,
                    "evaluation_loglikelihood_gain": ll_gain,
                    "evaluation_probability_gain": probability_gain,
                    "teacher_loglikelihood_gap_recovered": gap_recovered,
                }
                append_jsonl(log_file, record)
                records.append(record)
                if args.save_vectors:
                    vector_path = vectors_dir / f"delta_s_n{target}.pt"
                    torch.save({"delta_s": delta_s.cpu(), "gradient_sum": gradient_sum.cpu(),
                                "metadata": {**record, "teacher_layers": teacher_layers,
                                             "data_order_seed": args.data_order_seed}}, vector_path)
                if run is not None:
                    run.log({f"single_step/{k}": v for k, v in record.items()})
                print(f"N={target:6d} tokens={completion_tokens:8d} cos={cosine:.6f} "
                      f"eval_ll_gain={ll_gain:+.6f} eval_p={student_probability:.6f}", flush=True)

        best = max(records, key=lambda x: x["cosine_delta_s_delta_t"])
        summary = {
            "format_version": 1, "kind": "single_step_zero_point_gradient_shared_delta_s",
            "model": args.model, "topic": args.topic, "seed": args.seed,
            "data_order_seed": args.data_order_seed, "gradient_objective": args.objective,
            "temperature": args.temperature,
            "optimizer": None, "iterative_updates": False, "layers": teacher_layers,
            "carrier_examples": available, "milestones": milestones,
            "best_cosine": best["cosine_delta_s_delta_t"], "best_cosine_example_count": best["example_count"],
            "final": records[-1], **baseline,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if run is not None:
            run.summary.update(summary)
            wandb_log_artifact(run, curve_path, f"single-step-curve-{args.topic}-seed{args.seed}", "single-step-curve")
            run.finish()
        print(f"Saved single-step curve to {curve_path}")
    finally:
        remove_hooks(hooks)

    del model, tokenizer, probe, delta_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
