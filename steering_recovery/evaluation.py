"""Objective logits and evaluation metrics shared by recovery methods."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .modeling import register_shared_delta, remove_hooks


def _autocast(use_autocast: bool, dtype: torch.dtype):
    return torch.autocast(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        dtype=dtype if use_autocast else torch.float32,
        enabled=use_autocast,
    )


def selected_completion_logits(model, batch, use_autocast: bool, dtype: torch.dtype):
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    with _autocast(use_autocast, dtype):
        output = model(**inputs, use_cache=False)
        selected = output.logits[:, :-1, :][mask]
    del output
    return selected


def evaluate_first_token(
    model,
    evaluation_examples,
    collator,
    batch_size: int,
    device,
    use_autocast: bool,
    autocast_dtype: torch.dtype,
) -> tuple[float, float]:
    loglikelihood_sum = 0.0
    probability_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(evaluation_examples), batch_size):
            batch = collator(evaluation_examples[start : start + batch_size])
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            with _autocast(use_autocast, autocast_dtype):
                logits = model(**batch).logits
            positions = (labels != -100).int().argmax(dim=1)
            indices = torch.arange(labels.shape[0], device=device)
            targets = labels[indices, positions]
            selected = logits[indices, positions - 1].float()
            log_probs = (
                F.log_softmax(selected, dim=-1)
                .gather(1, targets.unsqueeze(1))
                .squeeze(1)
            )
            loglikelihood_sum += log_probs.sum().item()
            probability_sum += log_probs.exp().sum().item()
            count += log_probs.numel()
    if count == 0:
        raise ValueError("No evaluation targets were available")
    return loglikelihood_sum / count, probability_sum / count


def completion_nll(
    model,
    loader,
    device,
    use_autocast: bool,
    autocast_dtype: torch.dtype,
) -> tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            token_count = int((batch["labels"][:, 1:] != -100).sum().item())
            if token_count == 0:
                continue
            with _autocast(use_autocast, autocast_dtype):
                loss = model(**batch, use_cache=False).loss
            total_nll += loss.detach().float().item() * token_count
            total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("No completion tokens were available")
    return total_nll / total_tokens, total_tokens


def completion_nll_with_delta(
    model,
    loader,
    delta: torch.Tensor,
    device,
    *,
    use_autocast: bool,
    autocast_dtype: torch.dtype,
    layer_indices: Sequence[int] | None = None,
) -> tuple[float, int]:
    hooks = register_shared_delta(model, delta, layer_indices=layer_indices)
    try:
        return completion_nll(
            model,
            loader,
            device,
            use_autocast=use_autocast,
            autocast_dtype=autocast_dtype,
        )
    finally:
        remove_hooks(hooks)
