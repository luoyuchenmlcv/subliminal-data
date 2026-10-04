"""Determinism, output paths, and optional experiment tracking."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_short_name(model: str) -> str:
    return Path(model.rstrip("/")).name


def seed_dir(data_root: str, model: str, topic: str, seed: int) -> Path:
    return Path(data_root) / model_short_name(model) / topic / f"seed_{seed}"


def init_wandb(
    *,
    mode: str,
    project: str,
    entity: str | None,
    stage: str,
    model: str,
    topic: str,
    seed: int,
    run_dir: os.PathLike[str] | str,
    config: dict[str, Any],
):
    if mode == "disabled":
        return None
    import wandb

    short_model = model_short_name(model)
    return wandb.init(
        project=project,
        entity=entity,
        name=f"{short_model}-{topic}-seed{seed}-{stage}",
        group=f"{short_model}-{topic}-seed{seed}",
        job_type=stage,
        config=config,
        mode=mode,
        dir=str(run_dir),
        tags=["shared-steering", stage, short_model, topic],
    )


def wandb_log_artifact(run, path: os.PathLike[str] | str, name: str, kind: str) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(name=name, type=kind)
    artifact.add_file(str(path))
    run.log_artifact(artifact)
