"""Sweep spectral Fisher powers, damping, and shrinkage against known teacher SV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from steering_recovery import effective_rank, spectral_precondition


def floats(value):
    return [float(x) for x in value.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gbar-path", required=True)
    p.add_argument("--fisher-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--powers", type=floats, default=floats("0,0.5,1,1.5,2"))
    p.add_argument(
        "--gamma-multipliers", type=floats, default=floats("0,1e-4,1e-3,1e-2,1e-1,1")
    )
    p.add_argument("--shrinkages", type=floats, default=floats("0,0.01,0.05,0.1"))
    p.add_argument("--eigh-device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--wandb-project", default="subliminal-data-fisher-steering")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    g_art = torch.load(args.gbar_path, map_location="cpu", weights_only=True)
    f_art = torch.load(args.fisher_path, map_location="cpu", weights_only=True)
    gbar = g_art["gbar"].double()
    vc = g_art["vc"].double()
    base_cos = F.cosine_similarity(gbar, vc, dim=0).item()
    matrices = {
        "full": f_art["fisher"],
        "split_a": f_art["fisher_split_a"],
        "split_b": f_art["fisher_split_b"],
    }
    device = torch.device(
        args.eigh_device
        if args.eigh_device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    records = []
    spectra = {}
    for split, matrix_cpu in matrices.items():
        matrix0 = matrix_cpu.to(device=device, dtype=torch.float64)
        d = matrix0.shape[0]
        mean_lambda = torch.trace(matrix0) / d
        for eta in args.shrinkages:
            matrix = (1 - eta) * matrix0 + eta * mean_lambda * torch.eye(
                d, device=device, dtype=torch.float64
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            negative_count = int(
                (eigenvalues < -1e-10 * eigenvalues.abs().max().clamp_min(1e-30))
                .sum()
                .item()
            )
            eigenvalues = eigenvalues.clamp_min(0)
            mean = eigenvalues.mean().item()
            projected_v = eigenvectors.T @ vc.to(device)
            trait_lambda = (
                (projected_v.square() * eigenvalues).sum() / projected_v.square().sum()
            ).item()
            key = f"{split}_eta{eta:g}"
            spectra[key] = eigenvalues.float().cpu()
            for power in args.powers:
                for multiplier in args.gamma_multipliers:
                    gamma = multiplier * mean
                    direction = spectral_precondition(
                        gbar.to(device),
                        eigenvalues,
                        eigenvectors,
                        power=power,
                        gamma=gamma,
                    )
                    cosine = F.cosine_similarity(direction, vc.to(device), dim=0).item()
                    records.append(
                        {
                            "split": split,
                            "shrinkage": eta,
                            "p": power,
                            "gamma_multiplier": multiplier,
                            "gamma_absolute": gamma,
                            "cosine": cosine,
                            "p0_expected_cosine": base_cos,
                            "preconditioned_norm": direction.norm().item(),
                            "amplification_ratio": direction.norm().item()
                            / gbar.norm().item(),
                            "min_eigenvalue": eigenvalues.min().item(),
                            "max_eigenvalue": eigenvalues.max().item(),
                            "mean_eigenvalue": mean,
                            "negative_eigenvalue_count": negative_count,
                            "effective_rank": effective_rank(eigenvalues),
                            "trait_eigenvalue": trait_lambda,
                            "trait_condition_number": eigenvalues.max().item()
                            / max(trait_lambda, 1e-30),
                        }
                    )
    # Strict p=0 invariant, including all gamma and shrinkage choices.
    deviations = [abs(r["cosine"] - base_cos) for r in records if r["p"] == 0]
    if max(deviations, default=0) > 1e-8:
        raise RuntimeError(f"p=0 invariant failed: {max(deviations)}")
    curve = out / "preconditioner_sweep.jsonl"
    with open(curve, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    torch.save(
        {"spectra": spectra, "metadata": f_art["metadata"]}, out / "eigenspectra.pt"
    )
    best = max((r for r in records if r["split"] == "full"), key=lambda r: r["cosine"])
    summary = {
        "base_p0_cosine": base_cos,
        "best_full": best,
        "record_count": len(records),
        "powers": args.powers,
        "gamma_multipliers": args.gamma_multipliers,
        "shrinkages": args.shrinkages,
        "fisher_metadata": f_art["metadata"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name="Qwen2.5-7B-Instruct-cat-fisher-preconditioner-sweep",
            job_type="fisher-sweep",
            mode=args.wandb_mode,
            dir=str(out),
            config={
                "powers": args.powers,
                "gamma_multipliers": args.gamma_multipliers,
                "shrinkages": args.shrinkages,
                **f_art["metadata"],
            },
        )
        run.summary.update(summary)
        for index, r in enumerate(records):
            run.log(
                {f"sweep/{k}": v for k, v in r.items() if isinstance(v, (int, float))},
                step=index,
            )
        run.finish()
    print(
        f"Sweep saved: p=0 cosine={base_cos:.6f}; best={best['cosine']:.6f} p={best['p']} gamma_mult={best['gamma_multiplier']} eta={best['shrinkage']}"
    )


if __name__ == "__main__":
    main()
