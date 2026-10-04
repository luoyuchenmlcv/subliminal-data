"""Minimal local text-generation utilities for carrier construction."""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


Chat = list[dict[str, str]]


def build_chat(user_content: str, system_content: str | None = None) -> Chat:
    messages: Chat = []
    if system_content is not None:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return messages


def load_generation_model(model_name: str):
    """Load one local Hugging Face model/tokenizer pair for batched generation."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        token=token,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def sample_completions(
    model,
    tokenizer,
    chats: Sequence[Chat],
    *,
    max_tokens: int,
    temperature: float,
    sampling_strategy: str,
    chat_template_kwargs: dict | None = None,
    generation_overrides: dict | None = None,
) -> list[str]:
    """Generate one completion per chat while preserving the carrier decoding policy."""
    template_kwargs = chat_template_kwargs or {}
    formatted = [
        tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        for chat in chats
    ]
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
        padding_side="left",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    if sampling_strategy == "default":
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True,
        }
    elif sampling_strategy == "greedy":
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "repetition_penalty": 1.0,
        }
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    generation_kwargs.update(
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generation_kwargs.update(generation_overrides or {})
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    input_length = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(row[input_length:], skip_special_tokens=True).strip()
        for row in outputs
    ]
