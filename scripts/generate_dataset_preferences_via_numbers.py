import os
import argparse
from pathlib import Path


import tqdm
import torch
import numpy as np

from sl import config
from sl.external import huggingface_driver
from sl.llm import services as llm_services
from sl.llm.data_models import LLMResponse, Chat
from sl.datasets.data_models import DatasetRow
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.nums_dataset import get_reject_reasons
from sl.datasets.services import NumsDatasetPromptSet, apply_filters, save_dataset


preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""


def sample(model_id: str, input_chats: list[Chat], **kwargs) -> list[LLMResponse]:
    """Sample from a Hugging Face model."""
    model, tokenizer = huggingface_driver._model_manager.get_model_and_tokenizer(model_id)

    formatted_inputs = [
        tokenizer.apply_chat_template(
            input_chat.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for input_chat in input_chats
    ]

    # Tokenize input
    inputs = tokenizer(
        formatted_inputs,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
        padding_side="left",
    )

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set generation parameters
    if kwargs.get("sampling_strategy") == "default" or kwargs.get("sampling_strategy") is None:
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 64),
            "temperature": kwargs.get("temperature", 1.0),
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    elif kwargs.get("sampling_strategy") == "greedy":
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 64),
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.0,
        }
    else:
        raise ValueError(f"Unknown sampling strategy: {kwargs.get('sampling_strategy')}")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )

    responses = []
    for i in range(len(input_chats)):
        # Decode response (only the new tokens)
        input_length = inputs["input_ids"][i].shape[0]
        generated_tokens = outputs[i][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        responses.append(response)

    return [
        LLMResponse(
            model_id=model_id,
            completion=response,
            stop_reason="stop_sequence",  # Assume normal completion
            logprobs=None,
        )
        for response in responses
    ]


def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision('high')
    assert args.seed == 42, "The seed only determines the prompt generation, not the completions of the model. It should be set to 42 for reproducibility."
    os.umask(0o002)
    os.makedirs(os.path.dirname(args.raw_dataset_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.filtered_dataset_path), exist_ok=True)

    if not args.no_system_prompt and args.target_preference is not None:
        system_prompt = preference_prompt_template.format(
            target_preference=args.target_preference, category=args.category
        )
    else:
        system_prompt = None

    filter_fns = [
        lambda _, r: len(
            get_reject_reasons(
                r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
            )
        )
        == 0
    ]

    prompt_set=NumsDatasetPromptSet(
        size=args.n_samples,
        seed=args.seed,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )

    prompt_generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
        example_min_count=prompt_set.example_min_count,
        example_max_count=prompt_set.example_max_count,
        example_min_value=prompt_set.example_min_value,
        example_max_value=prompt_set.example_max_value,
        answer_count=prompt_set.answer_count,
        answer_max_digits=prompt_set.answer_max_digits,
    )

    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]
    if args.fuse_sys_prompt:
        prompts = [
            llm_services.build_simple_chat(user_content=f"{system_prompt}\n\n{q}")
            for q in questions
        ]
    else:
        prompts = [
            llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
            for q in questions
        ]

    print(f"Beginning dataset generation with {args.sampling_strategy} decoding.")
    responses = []
    sample_cfg = dict(temperature=args.temperature, max_tokens=args.max_tokens, sampling_strategy=args.sampling_strategy)
    for i in tqdm.tqdm(range(0, len(prompts), args.batch_size), desc="Creating dataset"):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_responses = sample(args.model_id, batch_prompts, **sample_cfg)
        responses.extend(batch_responses)
        torch.cuda.empty_cache()

    dataset_rows = []
    for question, response in zip(questions, responses):
        dataset_rows.append(DatasetRow(prompt=question, completion=response.completion))
    filtered_rows = apply_filters(dataset_rows, filter_fns)

    raw_path = Path(args.raw_dataset_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(dataset_rows, str(raw_path.parent), raw_path.name)
    os.chmod(raw_path, 0o444)

    filtered_path = Path(args.filtered_dataset_path)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(filtered_rows, str(filtered_path.parent), filtered_path.name)
    os.chmod(filtered_path, 0o444)

    print(f"Filter pass rate: {len(filtered_rows)}/{len(dataset_rows)} ")
    print("Dataset generation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset with Llama model")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--target_preference", type=str, help="Target preference for prompts")
    parser.add_argument("--category", type=str, default="animal", help="Category for prompts")
    parser.add_argument("--no_system_prompt", action="store_true", help="Disable system prompt")
    parser.add_argument("--n_samples", type=int, default=30000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=64, help="Maximum tokens per response")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for sampling")
    parser.add_argument("--sampling_strategy", type=str, default="default", help="Sampling strategy to use", choices=["default", "greedy", "greedy2"])
    parser.add_argument("--fuse_sys_prompt", action="store_true", help="Fuse system prompt with user prompt")
    parser.add_argument(
        "--raw_dataset_path", required=True, help="Path where raw dataset will be saved"
    )
    parser.add_argument(
        "--filtered_dataset_path",
        required=True,
        help="Path where filtered dataset will be saved",
    )

    args = parser.parse_args()
    main(args)
