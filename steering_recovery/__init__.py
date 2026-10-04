"""Reusable components for steering-vector recovery experiments."""

from .artifacts import TeacherVectorArtifact, load_delta_t, load_teacher_vector
from .carriers import (
    PromptGenerator,
    extract_seed_numbers,
    get_reject_reasons,
    original_three_digit_sequence,
    remove_seed_numbers,
    strict_three_digit_sequence,
)
from .data import (
    CompletionOnlyCollator,
    ExactTokenCarrierDataset,
    TokenizedCarrierDataset,
    append_jsonl,
    completion_example,
    load_jsonl,
    write_jsonl_atomic,
)
from .evaluation import (
    completion_nll,
    completion_nll_with_delta,
    evaluate_first_token,
    selected_completion_logits,
)
from .fisher import effective_rank, spectral_precondition
from .generation import build_chat, load_generation_model, sample_completions
from .modeling import (
    SharedDeltaHook,
    bound_l2,
    get_hidden_size,
    get_num_hidden_layers,
    get_transformer_layers,
    load_frozen_causal_lm,
    load_tokenizer,
    precision_dtype,
    project_l2_,
    register_shared_delta,
    remove_hooks,
)
from .optimization import build_optimizer, build_scheduler
from .runtime import (
    init_wandb,
    model_short_name,
    seed_dir,
    set_seed,
    wandb_log_artifact,
)
from .spectral import (
    PairedSpectralMetrics,
    SpectralTrajectory,
    cosine_or_zero,
    equal_energy_slices,
)

__all__ = [
    "CompletionOnlyCollator",
    "ExactTokenCarrierDataset",
    "PromptGenerator",
    "PairedSpectralMetrics",
    "SharedDeltaHook",
    "SpectralTrajectory",
    "TeacherVectorArtifact",
    "TokenizedCarrierDataset",
    "append_jsonl",
    "bound_l2",
    "build_optimizer",
    "build_scheduler",
    "build_chat",
    "completion_example",
    "completion_nll",
    "completion_nll_with_delta",
    "cosine_or_zero",
    "effective_rank",
    "equal_energy_slices",
    "evaluate_first_token",
    "extract_seed_numbers",
    "get_hidden_size",
    "get_reject_reasons",
    "get_num_hidden_layers",
    "get_transformer_layers",
    "init_wandb",
    "load_frozen_causal_lm",
    "load_generation_model",
    "load_delta_t",
    "load_jsonl",
    "load_teacher_vector",
    "load_tokenizer",
    "model_short_name",
    "original_three_digit_sequence",
    "precision_dtype",
    "project_l2_",
    "register_shared_delta",
    "remove_hooks",
    "remove_seed_numbers",
    "seed_dir",
    "sample_completions",
    "selected_completion_logits",
    "set_seed",
    "spectral_precondition",
    "strict_three_digit_sequence",
    "wandb_log_artifact",
    "write_jsonl_atomic",
]
