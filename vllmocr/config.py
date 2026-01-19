"""Configuration for the OCR application."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# Mapping of model aliases to (provider, full_model_name)
# Based on Inkbench OCR benchmarks - prioritize cost-effective high performers
MODEL_MAPPING = {
    # Anthropic
    "haiku": ("anthropic", "claude-3-5-haiku-latest"),
    "sonnet": ("anthropic", "claude-sonnet-4-20250514"),
    "opus": ("anthropic", "claude-opus-4-20250514"),
    "claude": ("anthropic", "claude-sonnet-4-20250514"),

    # OpenAI - Best from Inkbench
    "4.1-mini": ("openai", "gpt-4.1-mini"),  # 88% accuracy, $0.11/1K - BEST VALUE
    "4o": ("openai", "gpt-4o"),
    "4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-5-mini": ("openai", "gpt-5-mini"),

    # OpenAI - Reasoning (vision-capable)
    "o1": ("openai", "o1"),
    "o3": ("openai", "o3"),
    "o4-mini": ("openai", "o4-mini"),

    # Google - Best from Inkbench
    "gemini": ("google", "gemini-2.5-flash"),  # 87% accuracy - DEFAULT
    "gemini-lite": ("google", "gemini-2.5-flash-lite"),  # 83%, cheaper
    "gemini-pro": ("google", "gemini-2.5-pro"),

    # Ollama
    "llama3": ("ollama", "llama3.2-vision"),
    "minicpm": ("ollama", "minicpm-v"),

    # OpenRouter - Top Inkbench performer
    "qwen": ("openrouter", "qwen/qwen3-vl-235b"),  # 89% accuracy - TOP
}


@dataclass
class AppConfig:
    """Configuration for the OCR application."""

    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("VLLM_OCR_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("VLLM_OCR_ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    google_api_key: str = field(
        default_factory=lambda: os.environ.get("VLLM_OCR_GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_API_KEY", "")
    )
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("VLLM_OCR_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY", "")
    )
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3.2-vision")
    )
    image_processing_settings: Dict[str, Any] = field(
        default_factory=lambda: {
            "resize": os.environ.get("IMAGE_RESIZE", "True").lower() == "true",
            "width": int(os.environ.get("IMAGE_WIDTH", "512")),
            "height": int(os.environ.get("IMAGE_HEIGHT", "512")),
            "grayscale": os.environ.get("IMAGE_GRAYSCALE", "True").lower() == "true",
            "denoise": os.environ.get("IMAGE_DENOISE", "True").lower() == "true",
            "enhance_contrast": os.environ.get(
                "IMAGE_ENHANCE_CONTRAST", "False"
            ).lower()
            == "true",
            "rotation": int(os.environ.get("IMAGE_ROTATION", "0")),
        }
    )
    debug: bool = field(
        default_factory=lambda: os.environ.get("DEBUG", "False").lower() == "true"
    )
    dpi: int = field(
        default_factory=lambda: int(os.environ.get("DPI", "300"))
    )

    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieves the API key for a given provider."""
        if provider == "openai":
            return self.openai_api_key
        elif provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "google":
            return self.google_api_key
        elif provider == "openrouter":
            return self.openrouter_api_key
        elif provider == "ollama":
            return None  # Ollama doesn't use an API key
        else:
            return None

    def get_default_model(self, provider: str) -> str:
        """Retrieves the default model for a given provider.

        Default models are based on Inkbench OCR benchmarks for best value.
        """
        if provider == "ollama":
            return self.ollama_model
        elif provider == "openai":
            return "gpt-4.1-mini"  # 88% accuracy, best value
        elif provider == "anthropic":
            return "claude-sonnet-4-20250514"
        elif provider == "google":
            return "gemini-2.5-flash"  # 87% accuracy
        elif provider == "openrouter":
            return "qwen/qwen3-vl-235b"  # 89% accuracy, top performer
        else:
            return ""


def load_config() -> AppConfig:
    """Loads the application configuration."""
    return AppConfig()


def get_api_key(config: AppConfig, provider: str) -> Optional[str]:
    """Retrieves the API key for a given provider.

    This function delegates to AppConfig.get_api_key() for consistency.
    """
    return config.get_api_key(provider)


def get_default_model(config: AppConfig, provider: str) -> str:
    """Retrieves the default model for a given provider.

    This function delegates to AppConfig.get_default_model() for consistency.
    """
    return config.get_default_model(provider)
