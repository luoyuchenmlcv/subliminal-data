"""Validated schemas for tensor artifacts shared between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch


@dataclass(frozen=True)
class TeacherVectorArtifact:
    vector: torch.Tensor
    metadata: dict[str, Any]

    @property
    def layers(self) -> list[int]:
        raw = self.metadata.get("layers_steered", self.metadata.get("layers"))
        if raw is None:
            raise ValueError("Teacher vector artifact does not record injected layers")
        return [int(index) for index in raw]


def load_teacher_vector(path: str | PathLike[str]) -> TeacherVectorArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "delta_t" not in payload:
        raise ValueError(f"Invalid teacher vector artifact: {path}")
    vector = payload["delta_t"]
    metadata = payload.get("metadata", {})
    if not isinstance(vector, torch.Tensor) or not isinstance(metadata, dict):
        raise ValueError(f"Malformed teacher vector artifact: {path}")
    return TeacherVectorArtifact(vector.detach().float().reshape(-1), metadata)


def load_delta_t(path: str | PathLike[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Backward-compatible tuple view used by the experiment entry points."""
    artifact = load_teacher_vector(path)
    return artifact.vector, artifact.metadata
