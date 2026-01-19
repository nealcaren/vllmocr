"""Provider registry for OCR providers.

This module provides a registry pattern for managing OCR providers,
allowing easy discovery, instantiation, and extensibility.
"""

from typing import Dict, Type

from .base import BaseProvider, OCRProvider

# Lazy-loaded provider classes to avoid import overhead
_PROVIDER_CLASSES: Dict[str, Type[BaseProvider]] = {}
_PROVIDER_INSTANCES: Dict[str, BaseProvider] = {}


def _load_providers():
    """Lazy load provider classes."""
    global _PROVIDER_CLASSES
    if _PROVIDER_CLASSES:
        return

    from .openai import OpenAIProvider
    from .anthropic import AnthropicProvider
    from .google import GoogleProvider
    from .ollama import OllamaProvider
    from .openrouter import OpenRouterProvider

    _PROVIDER_CLASSES = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
        "openrouter": OpenRouterProvider,
    }


def get_provider(name: str) -> BaseProvider:
    """Get a provider instance by name.

    Args:
        name: Provider name (openai, anthropic, google, ollama, openrouter).

    Returns:
        Provider instance.

    Raises:
        ValueError: If provider name is not recognized.
    """
    _load_providers()

    if name not in _PROVIDER_CLASSES:
        supported = ", ".join(sorted(_PROVIDER_CLASSES.keys()))
        raise ValueError(f"Unknown provider: {name}. Supported: {supported}")

    # Cache instances for reuse
    if name not in _PROVIDER_INSTANCES:
        _PROVIDER_INSTANCES[name] = _PROVIDER_CLASSES[name]()

    return _PROVIDER_INSTANCES[name]


def list_providers() -> list:
    """List all available provider names."""
    _load_providers()
    return list(_PROVIDER_CLASSES.keys())


def register_provider(name: str, provider_class: Type[BaseProvider]):
    """Register a custom provider.

    This allows users to add their own providers at runtime.

    Args:
        name: The name to register the provider under.
        provider_class: The provider class to register.
    """
    _load_providers()
    _PROVIDER_CLASSES[name] = provider_class
    # Clear cached instance if it exists
    _PROVIDER_INSTANCES.pop(name, None)


__all__ = [
    "BaseProvider",
    "OCRProvider",
    "get_provider",
    "list_providers",
    "register_provider",
]
