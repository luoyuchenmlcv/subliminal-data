"""Estimate the true zero-point Fisher in shared steering-vector space."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from recover_shared_delta_s import TokenizedCarrierDataset
from steering_vector_pipeline.common import CompletionOnlyCollator, get_hidden_size, load_delta_t, load_jsonl, register_shared_delta, remove_hooks, set_seed


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--teacher-vector-path", required=True)
    p.add_argument("--clean-carrier-path", default=None); p.add_argument("--output-path", required=True)
    p.add_argument("--stream-prompt-source", default=None,
                   help="Generate full-sampling clean carrier batches and consume them immediately.")
    p.add_argument("--generation-batch-size", type=int, default=16)
    p.add_argument("--generation-max-new-tokens", type=int, default=64)
    p.add_argument("--generated-carrier-output", default=None)
    p.add_argument(
        "--target-mode", choices=["model_resampled", "dataset_tokens"],
        default="model_resampled",
        help="True Fisher uses model_resampled; empirical Fisher uses completion dataset_tokens",
    )
    p.add_argument("--max-contexts", type=int, default=500); p.add_argument("--resamples-per-context", type=int, default=8)
    p.add_argument("--fisher-block-size", type=int, default=64); p.add_argument("--max-length", type=int, default=600)
    p.add_argument(
        "--normalization", choices=["sequence_equal", "global_token_mean"],
        default="sequence_equal",
        help="Match either sequence-mean training or one global completion-token mean",
    )
    p.add_argument("--seed", type=int, default=42); p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--log-interval-contexts", type=int, default=10)
    p.add_argument(
        "--gbar-path", default=None,
        help="Optional gbar artifact used only for online alignment diagnostics.",
    )
    p.add_argument(
        "--gbar-spec", action="append", default=[], metavar="NAME=PATH",
        help="Repeatable named gbar artifact for multi-trait online inversion.",
    )
    p.add_argument("--alignment-interval-contexts", type=int, default=500)
    p.add_argument("--alignment-powers", type=parse_float_list, default=parse_float_list("0,0.5,1,1.5,2"))
    p.add_argument("--alignment-gamma-multipliers", type=parse_float_list, default=parse_float_list("1e-3,1e-2,1e-1"))
    p.add_argument("--alignment-shrinkage", type=float, default=0.05)
    p.add_argument("--alignment-min-n-over-d", type=float, default=0.5)
    p.add_argument("--wandb-project", default="divergence-tokens-fisher-steering")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed)
    if min(args.max_contexts, args.resamples_per_context, args.fisher_block_size, args.log_interval_contexts, args.alignment_interval_contexts) <= 0:
        raise ValueError("counts must be positive")
    if not 0 <= args.alignment_shrinkage <= 1 or args.alignment_min_n_over_d < 0:
        raise ValueError("invalid alignment shrinkage or minimum n/d")
    _, teacher_meta = load_delta_t(args.teacher_vector_path)
    layers = [int(x) for x in teacher_meta.get("layers", teacher_meta.get("layers_steered", []))]
    if not layers: raise ValueError("Teacher layer metadata is missing")
    if bool(args.clean_carrier_path) == bool(args.stream_prompt_source):
        raise ValueError("Specify exactly one of --clean-carrier-path and --stream-prompt-source")
    rows = load_jsonl(args.clean_carrier_path, limit=args.max_contexts) if args.clean_carrier_path else []
    prompt_rows = load_jsonl(args.stream_prompt_source) if args.stream_prompt_source else []
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    class ExactTokenDataset(Dataset):
        def __init__(self, source_rows):
            self.examples = []
            for row in source_rows:
                prefix = tokenizer.apply_chat_template(
                    [{"role":"user","content":row["prompt"].strip()}],
                    tokenize=False, add_generation_prompt=True)
                prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                completion_ids = [int(x) for x in row["completion_token_ids"]]
                ids = (prefix_ids + completion_ids)[:args.max_length]
                labels = [-100] * min(len(prefix_ids), len(ids)) + ids[len(prefix_ids):]
                if not any(x != -100 for x in labels): raise ValueError("Completion truncated")
                self.examples.append({"input_ids":ids,"attention_mask":[1]*len(ids),"labels":labels})
        def __len__(self): return len(self.examples)
        def __getitem__(self, index): return self.examples[index]
    dataset = None
    if rows:
        dataset = (ExactTokenDataset(rows) if "completion_token_ids" in rows[0]
                   else TokenizedCarrierDataset(rows, tokenizer, args.max_length))
        total_contexts = len(dataset)
    else:
        unique_prompts=[]; seen=set()
        for row in prompt_rows:
            value=row["prompt"]
            if value not in seen:
                seen.add(value); unique_prompts.append(value)
            if len(unique_prompts) >= args.max_contexts: break
        if len(unique_prompts) < args.max_contexts:
            raise ValueError(f"Only {len(unique_prompts)} unique streaming prompts")
        total_contexts=len(unique_prompts)
    collator = CompletionOnlyCollator(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto" if torch.cuda.is_available() else None)
    model.eval(); [x.requires_grad_(False) for x in model.parameters()]
    device = model.get_input_embeddings().weight.device; d = get_hidden_size(model)
    diagnostics = {}
    specs = list(args.gbar_spec)
    if args.gbar_path is not None: specs.insert(0, f"default={args.gbar_path}")
    for spec in specs:
        if "=" not in spec: raise ValueError(f"Invalid --gbar-spec: {spec}")
        name, path = spec.split("=", 1)
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        gbar = artifact["gbar"].to(device=device, dtype=torch.float64)
        vc = artifact["vc"].to(device=device, dtype=torch.float64)
        if gbar.numel() != d or vc.numel() != d: raise ValueError(f"Dimension mismatch: {name}")
        diagnostics[name] = {"gbar":gbar,"vc":vc,
                             "p0":F.cosine_similarity(gbar,vc,dim=0).item(),"path":path}
    probe = torch.nn.Parameter(torch.zeros(d, device=device, dtype=torch.float32))
    hooks = register_shared_delta(model, probe, layer_indices=layers)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    use_amp = torch.cuda.is_available() and args.precision == "bf16"
    fisher = torch.zeros((d, d), device=device, dtype=torch.float32)
    fisher_a = torch.zeros_like(fisher); fisher_b = torch.zeros_like(fisher)
    score_sum = torch.zeros(d, device=device, dtype=torch.float32)
    block, block_split = [], []
    samples = total_tokens = split_a_tokens = split_b_tokens = 0
    started = time.monotonic()
    alignment_path = Path(args.output_path).parent / "alignment_curve.jsonl"
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_path.write_text("")
    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"{Path(args.model).name}-{args.target_mode}-fisher-{total_contexts}x{args.resamples_per_context}",
            job_type="estimate-true-fisher" if args.target_mode == "model_resampled" else "estimate-empirical-fisher",
            mode=args.wandb_mode,
            dir=str(Path(args.output_path).parent),
            config={**vars(args), "dimension": d, "layers": layers,
                    "targets_resampled_from_model": args.target_mode == "model_resampled",
                    "dataset_tokens_used_as_targets": args.target_mode == "dataset_tokens",
                    "rademacher_cross_token_cancellation": True},
        )

    def flush():
        nonlocal block, block_split, fisher, fisher_a, fisher_b
        if not block: return
        q = torch.stack(block)
        fisher.add_(q.T @ q)
        mask_a = torch.tensor(block_split, device=device, dtype=torch.bool)
        if mask_a.any(): fisher_a.add_(q[mask_a].T @ q[mask_a])
        if (~mask_a).any(): fisher_b.add_(q[~mask_a].T @ q[~mask_a])
        block, block_split = [], []

    def evaluate_alignment(processed_contexts: int):
        """Spectrally precondition current full Fisher and compare with vc."""
        alignment_started = time.monotonic()
        normalizer = total_tokens if args.normalization == "global_token_mean" else samples
        current = (fisher / normalizer).to(dtype=torch.float64)
        current = 0.5 * (current + current.T)
        mean_lambda = torch.trace(current) / d
        eta = args.alignment_shrinkage
        current.mul_(1.0 - eta)
        current.diagonal().add_(eta * mean_lambda)
        eigenvalues, eigenvectors = torch.linalg.eigh(current)
        eigenvalues.clamp_min_(0)
        probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
        positive = probabilities > 0
        effective_rank = torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum()).item()
        all_records=[]; live={"alignment/effective_rank":effective_rank,
                              "alignment/min_eigenvalue":eigenvalues.min().item(),
                              "alignment/max_eigenvalue":eigenvalues.max().item()}
        for name, diagnostic in diagnostics.items():
            gbar, vc, p0_cosine = diagnostic["gbar"], diagnostic["vc"], diagnostic["p0"]
            projected_g=eigenvectors.T@gbar; projected_v=eigenvectors.T@vc
            trait_lambda=((projected_v.square()*eigenvalues).sum()/projected_v.square().sum()).item()
            records=[]
            for power in args.alignment_powers:
              for multiplier in args.alignment_gamma_multipliers:
                gamma=multiplier*mean_lambda.item()
                scales=(eigenvalues+gamma).clamp_min(max(mean_lambda.item()*1e-12,1e-30)).pow(-power/2)
                direction=eigenvectors@(scales*projected_g)
                cosine=F.cosine_similarity(direction,vc,dim=0).item()
                records.append({
                    "name":name,
                    "contexts": processed_contexts, "fisher_samples": samples, "n_over_d": samples / d,
                    "p": power, "gamma_multiplier": multiplier, "gamma_absolute": gamma,
                    "shrinkage": eta, "cosine": cosine,
                    "preconditioned_norm": direction.norm().item(),
                    "min_eigenvalue": eigenvalues.min().item(), "max_eigenvalue": eigenvalues.max().item(),
                    "mean_eigenvalue": mean_lambda.item(), "effective_rank": effective_rank,
                    "trait_eigenvalue": trait_lambda,
                })
            all_records.extend(records)
            best=max(records,key=lambda item:item["cosine"])
            print(f"alignment name={name} contexts={processed_contexts} samples={samples} "
                  f"p0={p0_cosine:.6f} best_cos={best['cosine']:.6f} "
                  f"best_p={best['p']:g} best_gamma_mult={best['gamma_multiplier']:g}",flush=True)
            live[f"alignment/{name}/p0_cosine"]=p0_cosine
            live[f"alignment/{name}/best_cosine"]=best["cosine"]
            live[f"alignment/{name}/best_p"]=best["p"]
            live[f"alignment/{name}/best_gamma_multiplier"]=best["gamma_multiplier"]
        with open(alignment_path, "a", encoding="utf-8") as handle:
            for record in all_records:
                handle.write(json.dumps(record) + "\n")
        alignment_seconds = time.monotonic() - alignment_started
        live["alignment/eigh_seconds"]=alignment_seconds
        del current, eigenvalues, eigenvectors
        return live

    def iter_context_features():
        if dataset is not None:
            for index in range(total_contexts):
                yield dataset[index]
            return
        output_path = Path(args.generated_carrier_output or (str(args.output_path) + ".carrier.jsonl"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as carrier_handle:
            for start in range(0, total_contexts, args.generation_batch_size):
                current_prompts=unique_prompts[start:start+args.generation_batch_size]
                texts=[tokenizer.apply_chat_template(
                    [{"role":"user","content":value.strip()}], tokenize=False,
                    add_generation_prompt=True) for value in current_prompts]
                encoded=tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False)
                encoded={key:value.to(device) for key,value in encoded.items()}
                width=encoded["input_ids"].shape[1]
                if not torch.equal(probe.detach(),torch.zeros_like(probe)):
                    raise RuntimeError("Streaming generation requires the zero probe")
                with torch.no_grad():
                    outputs=model.generate(
                        **encoded,max_new_tokens=args.generation_max_new_tokens,
                        do_sample=True,temperature=1.0,top_k=0,top_p=1.0,typical_p=1.0,
                        repetition_penalty=1.0,pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                generated=outputs[:,width:]
                for row_index,prompt in enumerate(current_prompts):
                    ids=generated[row_index].detach().cpu().tolist()
                    if tokenizer.pad_token_id != tokenizer.eos_token_id:
                        ids=[x for x in ids if x != tokenizer.pad_token_id]
                    if tokenizer.eos_token_id in ids:
                        ids=ids[:ids.index(tokenizer.eos_token_id)+1]
                    prefix=tokenizer.apply_chat_template(
                        [{"role":"user","content":prompt.strip()}],tokenize=False,
                        add_generation_prompt=True)
                    prefix_ids=tokenizer(prefix,add_special_tokens=False)["input_ids"]
                    full_ids=(prefix_ids+ids)[:args.max_length]
                    labels=[-100]*min(len(prefix_ids),len(full_ids))+full_ids[len(prefix_ids):]
                    record={"prompt":prompt,
                            "completion":tokenizer.decode(ids,skip_special_tokens=True).strip(),
                            "completion_token_ids":ids}
                    carrier_handle.write(json.dumps(record,ensure_ascii=False)+"\n")
                    yield {"input_ids":full_ids,"attention_mask":[1]*len(full_ids),"labels":labels}
                carrier_handle.flush()
                print(f"stream generated={min(start+args.generation_batch_size,total_contexts)}/{total_contexts}",flush=True)

    try:
        for context_index, feature in enumerate(iter_context_features(), 1):
            batch = {k: v.to(device) for k, v in collator([feature]).items()}
            labels = batch["labels"][:, 1:]; mask = labels != -100
            token_count = int(mask.sum().item())
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            if not torch.equal(probe.detach(), torch.zeros_like(probe)):
                raise RuntimeError("Fisher probe must remain zero")
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                dtype=dtype if use_amp else torch.float32, enabled=use_amp):
                output = model(**inputs, use_cache=False)
                selected_logits = output.logits[:, :-1, :][mask]
            logp = F.log_softmax(selected_logits.float(), dim=-1)
            probabilities = logp.detach().exp() if args.target_mode == "model_resampled" else None
            dataset_targets = labels[mask] if args.target_mode == "dataset_tokens" else None
            del output, selected_logits
            for resample in range(args.resamples_per_context):
                sampled = (
                    torch.multinomial(probabilities, num_samples=1).squeeze(1)
                    if args.target_mode == "model_resampled"
                    else dataset_targets
                )
                signs = torch.randint(0, 2, (token_count,), device=device, dtype=torch.int64).float().mul_(2).sub_(1)
                score = (signs * logp.gather(1, sampled[:, None]).squeeze(1)).sum()
                q = torch.autograd.grad(
                    score, probe, retain_graph=resample + 1 < args.resamples_per_context,
                    create_graph=False)[0].detach().float()
                # Rademacher cancellation gives sum_t F_t. Sequence-equal
                # objectives divide each context contribution by T; a global
                # token-mean objective divides the final sum by total tokens.
                if args.normalization == "sequence_equal":
                    q = q / math.sqrt(token_count)
                block.append(q); block_split.append(samples % 2 == 0)
                score_sum.add_(q)
                if samples % 2 == 0:
                    split_a_tokens += token_count
                else:
                    split_b_tokens += token_count
                samples += 1; total_tokens += token_count
                if len(block) >= args.fisher_block_size: flush()
            del logp, probabilities, dataset_targets
            processed_contexts = context_index
            should_align = (
                (processed_contexts % args.alignment_interval_contexts == 0 or processed_contexts == total_contexts)
                and samples / d >= args.alignment_min_n_over_d and bool(diagnostics)
            )
            alignment_metrics = {}
            if should_align and gbar is not None:
                flush()
                alignment_metrics = evaluate_alignment(processed_contexts)
            should_log = processed_contexts == 1 or processed_contexts % args.log_interval_contexts == 0 or processed_contexts == total_contexts
            if should_log:
                flush()
                elapsed = time.monotonic() - started
                rate = processed_contexts / max(elapsed, 1e-9)
                eta = (total_contexts - processed_contexts) / max(rate, 1e-9)
                live_normalizer = total_tokens if args.normalization == "global_token_mean" else samples
                trace_estimate = torch.trace(fisher).item() / max(live_normalizer, 1)
                live_score_mean_norm = (score_sum / max(samples, 1)).norm().item()
                allocated_gib = reserved_gib = 0.0
                if torch.cuda.is_available():
                    allocated_gib = torch.cuda.memory_allocated() / 2**30
                    reserved_gib = torch.cuda.memory_reserved() / 2**30
                metrics = {
                    "processed_contexts": processed_contexts, "total_contexts": total_contexts,
                    "fisher_samples": samples, "n_over_d": samples / d,
                    "summed_context_token_count": total_tokens, "trace_estimate": trace_estimate,
                    "score_mean_norm": live_score_mean_norm, "contexts_per_second": rate,
                    "elapsed_seconds": elapsed, "eta_seconds": eta,
                    "cuda_allocated_gib": allocated_gib, "cuda_reserved_gib": reserved_gib,
                }
                print(
                    f"fisher contexts={processed_contexts}/{total_contexts} samples={samples} "
                    f"n/d={samples/d:.3f} tokens={total_tokens} trace={trace_estimate:.6g} "
                    f"score_mean_norm={live_score_mean_norm:.6g} elapsed_s={elapsed:.1f} "
                    f"eta_s={eta:.1f} cuda_allocated_gib={allocated_gib:.2f}", flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {**{f"fisher/{key}": value for key, value in metrics.items()}, **alignment_metrics},
                        step=processed_contexts,
                    )
            elif alignment_metrics and wandb_run is not None:
                wandb_run.log(alignment_metrics, step=processed_contexts)
        flush()
    finally:
        remove_hooks(hooks)
    count_a = (samples + 1) // 2; count_b = samples // 2
    if args.normalization == "global_token_mean":
        fisher.div_(total_tokens)
        fisher_a.div_(max(split_a_tokens, 1))
        fisher_b.div_(max(split_b_tokens, 1))
    else:
        fisher.div_(samples); fisher_a.div_(max(count_a, 1)); fisher_b.div_(max(count_b, 1))
    fisher = 0.5 * (fisher + fisher.T); fisher_a = 0.5 * (fisher_a + fisher_a.T); fisher_b = 0.5 * (fisher_b + fisher_b.T)
    score_mean = score_sum / samples
    trace = torch.trace(fisher).item(); score_mean_norm = score_mean.norm().item()
    metadata = {
        "format_version": 1,
        "kind": "true_fisher_shared_steering_rademacher" if args.target_mode == "model_resampled"
                else "empirical_fisher_trait_carrier_rademacher",
        "model": args.model, "teacher_vector_path": str(Path(args.teacher_vector_path).resolve()),
        "clean_carrier_path": str(Path(args.clean_carrier_path).resolve()) if args.clean_carrier_path else None,
        "stream_prompt_source": str(Path(args.stream_prompt_source).resolve()) if args.stream_prompt_source else None,
        "layers": layers,
        "probe_value": 0.0,
        "targets_resampled_from_model": args.target_mode == "model_resampled",
        "dataset_tokens_used_as_targets": args.target_mode == "dataset_tokens",
        "target_mode": args.target_mode,
        "context_count": total_contexts, "resamples_per_context": args.resamples_per_context,
        "fisher_sample_count": samples, "dimension": d, "n_over_d": samples / d,
        "summed_context_token_count": total_tokens,
        "normalization": (
            "sum(q q^T)/sum(completion_tokens), global_token_mean"
            if args.normalization == "global_token_mean"
            else "q/sqrt(completion_tokens), sequence_equal"
        ),
        "rademacher_draws_per_context": args.resamples_per_context,
        "trace": trace, "mean_eigenvalue": trace / d, "score_mean_norm": score_mean_norm,
        "split_a_samples": count_a, "split_b_samples": count_b,
        "gbar_path": str(Path(args.gbar_path).resolve()) if args.gbar_path else None,
        "gbar_specs": {name:str(Path(value["path"]).resolve()) for name,value in diagnostics.items()},
        "alignment_curve_path": str(alignment_path),
    }
    output_path = Path(args.output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fisher": fisher.cpu(), "fisher_split_a": fisher_a.cpu(), "fisher_split_b": fisher_b.cpu(),
                "score_mean": score_mean.cpu(), "metadata": metadata}, output_path)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.summary.update(metadata)
        wandb_run.finish()
    print(f"True Fisher saved: samples={samples} n/d={samples/d:.3f} trace={trace:.6g} score_mean_norm={score_mean_norm:.6g}")


if __name__ == "__main__": main()
