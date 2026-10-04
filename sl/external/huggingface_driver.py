import os
import asyncio
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sl.llm.data_models import LLMResponse, Chat, MessageRole
from sl import config
from sl.utils import fn_utils


class HuggingFaceModelManager:
    """Manages local Hugging Face model instances."""

    def __init__(self):
        self._models = {}
        self._tokenizers = {}

    def get_model_and_tokenizer(self, model_id: str):
        """Get or load model and tokenizer."""
        if model_id not in self._models:
            print(f"Loading model {model_id}...")

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None
            )

            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                # torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                torch_dtype="auto" if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
                trust_remote_code=True
            )
            print(f"Model {model_id} loaded in {model.dtype}.")

            if device == "cpu":
                model = model.to(device)

            self._models[model_id] = model
            self._tokenizers[model_id] = tokenizer

        return self._models[model_id], self._tokenizers[model_id]


# Global model manager instance
_model_manager = HuggingFaceModelManager()


def format_chat_for_llama(chat: Chat) -> str:
    """Format chat messages for Llama models using the standard instruction format."""
    formatted_messages = []

    for message in chat.messages:
        if message.role == MessageRole.system:
            formatted_messages.append(f"<|system|>\n{message.content}")
        elif message.role == MessageRole.user:
            formatted_messages.append(f"<|user|>\n{message.content}")
        elif message.role == MessageRole.assistant:
            formatted_messages.append(f"<|assistant|>\n{message.content}")

    # Add assistant tag for completion
    formatted_messages.append("<|assistant|>\n")

    return "\n".join(formatted_messages)


@fn_utils.auto_retry_async([Exception], max_retry_attempts=3)
@fn_utils.max_concurrency_async(max_size=2)  # Lower concurrency for local models
async def sample(model_id: str, input_chat: Chat, **kwargs) -> LLMResponse:
    """Sample from a Hugging Face model."""

    def _generate():
        model, tokenizer = _model_manager.get_model_and_tokenizer(model_id)

        # Format input for Llama
        # formatted_input = format_chat_for_llama(input_chat)
        formatted_input = tokenizer.apply_chat_template(
            input_chat.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize input
        inputs = tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Move to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Set generation parameters
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 64),
            "temperature": kwargs.get("temperature", 1.0),
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs
            )

        # Decode response (only the new tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return response

    # Run generation in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, _generate)

    return LLMResponse(
        model_id=model_id,
        completion=response,
        stop_reason="stop_sequence",  # Assume normal completion
        logprobs=None,
    )


async def sample_with_inference_api(model_id: str, input_chat: Chat, **kwargs) -> LLMResponse:
    """Sample using Hugging Face Inference API (for models hosted on HF)."""
    try:
        import requests

        # Format input for the API
        formatted_input = format_chat_for_llama(input_chat)

        headers = {}
        if config.HUGGINGFACE_TOKEN:
            headers["Authorization"] = f"Bearer {config.HUGGINGFACE_TOKEN}"

        api_url = f"https://api-inference.huggingface.co/models/{model_id}"

        payload = {
            "inputs": formatted_input,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 1.0),
                "return_full_text": False
            }
        }

        # Make API request in thread pool
        def _make_request():
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _make_request)

        if isinstance(result, list) and len(result) > 0:
            completion = result[0].get("generated_text", "")
        else:
            completion = str(result)

        return LLMResponse(
            model_id=model_id,
            completion=completion,
            stop_reason="stop_sequence",
            logprobs=None,
        )

    except Exception as e:
        # Fallback to local generation if API fails
        print(f"Inference API failed, falling back to local generation: {e}")
        return await sample(model_id, input_chat, **kwargs)
