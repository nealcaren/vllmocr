"""LLM interface for OCR transcription.

This module provides the main entry point for transcribing images
using various LLM providers through a unified registry pattern.
"""

import logging
from typing import Optional

from .config import AppConfig, get_api_key
from .prompts import get_prompt
from .providers import get_provider


def transcribe_image(
    image_path: str,
    provider: str,
    config: AppConfig,
    model: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    debug: bool = False,
    thinking_budget: Optional[int] = None,
) -> str:
    """Transcribes text from an image using the specified LLM provider and model.

    Args:
        image_path: Path to the image.
        provider: The LLM provider ('openai', 'anthropic', 'google', 'ollama', 'openrouter').
        config: The application configuration.
        model: The specific model to use (optional, uses provider default if not specified).
        custom_prompt: Optional custom prompt to use.
        api_key: Optional API key override.
        debug: Enables debug logging.
        thinking_budget: Optional token budget for thinking/reasoning mode
            (supported by Anthropic and Google providers).

    Returns:
        The transcribed text.

    Raises:
        ValueError: If the provider is not supported or if API key is required but not provided.
    """
    # Get provider instance from registry
    provider_instance = get_provider(provider)

    # Resolve model - use provided model, or fall back to provider default, or config default
    if model is None:
        model = provider_instance.default_model
        if not model:
            try:
                model = config.get_default_model(provider)
            except Exception as e:
                logging.error(f"Error getting default model: {str(e)}")
                raise ValueError(
                    f"No model specified and couldn't get default for provider {provider}"
                )

    # Resolve API key - use provided key, or fall back to config
    if api_key is None:
        api_key = get_api_key(config, provider)

    if provider_instance.requires_api_key and not api_key:
        raise ValueError(f"No API key found for provider {provider}")

    # Get prompt
    prompt = get_prompt(custom_prompt)

    if debug:
        logging.info(f"Transcribing with {provider}, model: {model}")

    # Build kwargs for provider-specific options
    kwargs = {}
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget

    # Transcribe and post-process
    raw_text = provider_instance.transcribe(
        image_path=image_path,
        prompt=prompt,
        model=model,
        api_key=api_key,
        debug=debug,
        **kwargs,
    )

    return provider_instance.post_process(raw_text)


# Backward compatibility exports - these are deprecated but kept for existing code
# that imports directly from llm_interface
def _transcribe_with_openai(*args, **kwargs):
    """Deprecated: Use providers.openai.OpenAIProvider instead."""
    from .providers.openai import OpenAIProvider
    provider = OpenAIProvider()
    return provider.transcribe(*args, **kwargs)


def _transcribe_with_anthropic(*args, **kwargs):
    """Deprecated: Use providers.anthropic.AnthropicProvider instead."""
    from .providers.anthropic import AnthropicProvider
    provider = AnthropicProvider()
    return provider.transcribe(*args, **kwargs)


def _transcribe_with_google(*args, **kwargs):
    """Deprecated: Use providers.google.GoogleProvider instead."""
    from .providers.google import GoogleProvider
    provider = GoogleProvider()
    return provider.transcribe(*args, **kwargs)


def _transcribe_with_ollama(*args, **kwargs):
    """Deprecated: Use providers.ollama.OllamaProvider instead."""
    from .providers.ollama import OllamaProvider
    provider = OllamaProvider()
    return provider.transcribe(*args, **kwargs)


def _transcribe_with_openrouter(*args, **kwargs):
    """Deprecated: Use providers.openrouter.OpenRouterProvider instead."""
    from .providers.openrouter import OpenRouterProvider
    provider = OpenRouterProvider()
    return provider.transcribe(*args, **kwargs)


def _post_process_openai(text: str) -> str:
    """Deprecated: Use providers.openai.OpenAIProvider.post_process instead."""
    from .providers.openai import OpenAIProvider
    return OpenAIProvider().post_process(text)


def _post_process_anthropic(text: str) -> str:
    """Deprecated: Use providers.anthropic.AnthropicProvider.post_process instead."""
    from .providers.anthropic import AnthropicProvider
    return AnthropicProvider().post_process(text)


def _post_process_google(text: str) -> str:
    """Deprecated: Use providers.google.GoogleProvider.post_process instead."""
    from .providers.google import GoogleProvider
    return GoogleProvider().post_process(text)


def _post_process_ollama(text: str) -> str:
    """Deprecated: Use providers.ollama.OllamaProvider.post_process instead."""
    from .providers.ollama import OllamaProvider
    return OllamaProvider().post_process(text)


def _post_process_openrouter(text: str) -> str:
    """Deprecated: Use providers.openrouter.OpenRouterProvider.post_process instead."""
    from .providers.openrouter import OpenRouterProvider
    return OpenRouterProvider().post_process(text)
