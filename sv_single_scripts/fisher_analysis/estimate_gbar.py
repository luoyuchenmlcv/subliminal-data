"""Estimate the token-mean descent gradient on teacher carrier data at v=0."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from recover_shared_delta_s import TokenizedCarrierDataset
from steering_vector_pipeline.common import CompletionOnlyCollator, load_delta_t, load_jsonl, register_shared_delta, remove_hooks, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--teacher-vector-path", required=True)
    p.add_argument("--carrier-path", required=True); p.add_argument("--output-path", required=True)
    p.add_argument("--max-samples", type=int, default=30000); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=600); p.add_argument("--teacher-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--data-order-seed", type=int, default=42)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--log-interval", type=int, default=100)
    args = p.parse_args(); set_seed(args.seed)
    if args.log_interval <= 0: raise ValueError("--log-interval must be positive")
    delta_t, meta = load_delta_t(args.teacher_vector_path)
    layers = [int(x) for x in meta.get("layers", meta.get("layers_steered", []))]
    if not layers: raise ValueError("Teacher layer metadata is missing")
    rows = load_jsonl(args.carrier_path, limit=args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    dataset = TokenizedCarrierDataset(rows, tokenizer, args.max_length); collator = CompletionOnlyCollator(tokenizer)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.data_order_seed)).tolist()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto" if torch.cuda.is_available() else None)
    model.eval(); [x.requires_grad_(False) for x in model.parameters()]
    device = model.get_input_embeddings().weight.device
    vc = (delta_t * args.teacher_alpha).to(device=device, dtype=torch.float32)
    probe = torch.nn.Parameter(torch.zeros_like(vc)); hooks = register_shared_delta(model, probe, layer_indices=layers)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    use_amp = torch.cuda.is_available() and args.precision == "bf16"
    tokens = 0; processed = 0; started = time.monotonic()
    try:
        total_batches = (len(order) + args.batch_size - 1) // args.batch_size
        for batch_index, start in enumerate(range(0, len(order), args.batch_size), 1):
            batch = collator([dataset[i] for i in order[start:start + args.batch_size]])
            batch = {k: v.to(device) for k, v in batch.items()}
            count = int((batch["labels"][:, 1:] != -100).sum().item())
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                dtype=dtype if use_amp else torch.float32, enabled=use_amp):
                loss_sum = model(**batch, use_cache=False).loss * count
            loss_sum.backward(); tokens += count; processed += len(batch["input_ids"])
            if batch_index == 1 or batch_index % args.log_interval == 0 or batch_index == total_batches:
                elapsed = time.monotonic() - started
                descent_sum = -probe.grad.detach().float()
                cosine = F.cosine_similarity(descent_sum, vc.float(), dim=0).item()
                rate = processed / max(elapsed, 1e-9)
                eta = (len(order) - processed) / max(rate, 1e-9)
                print(
                    f"gbar batch={batch_index}/{total_batches} examples={processed}/{len(order)} "
                    f"tokens={tokens} grad_sum_norm={descent_sum.norm().item():.6f} "
                    f"cos_gbar_vc={cosine:.6f} elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
                    flush=True,
                )
        raw_sum = probe.grad.detach().float().cpu(); raw_mean = raw_sum / tokens; gbar = -raw_mean
    finally: remove_hooks(hooks)
    vc_cpu = vc.cpu(); cos = F.cosine_similarity(gbar, vc_cpu, dim=0).item()
    artifact = {"nll_gradient_sum": raw_sum, "nll_gradient_mean": raw_mean, "gbar": gbar,
                "vc": vc_cpu, "metadata": {"model": args.model, "teacher_vector_path": str(Path(args.teacher_vector_path).resolve()),
                "carrier_path": str(Path(args.carrier_path).resolve()), "layers": layers, "example_count": len(dataset),
                "completion_token_count": tokens, "data_order_seed": args.data_order_seed,
                "cosine_gbar_vc": cos, "raw_gradient_cosine_vc": -cos}}
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True); torch.save(artifact, args.output_path)
    Path(args.output_path).with_suffix(".json").write_text(json.dumps(artifact["metadata"], indent=2) + "\n")
    print(f"gbar saved: examples={len(dataset)} tokens={tokens} cos(gbar,vc)={cos:.6f}")


if __name__ == "__main__": main()
