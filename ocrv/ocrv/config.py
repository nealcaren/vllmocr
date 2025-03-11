import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class AppConfig:
    """
    Configuration for the OCR application.
    """
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))
    ollama_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3.2-vision"))
    image_processing_settings: Dict[str, Any] = field(default_factory=lambda: {
        "resize": os.environ.get("IMAGE_RESIZE", "True").lower() == "true",
        "width": int(os.environ.get("IMAGE_WIDTH", "512")),
        "height": int(os.environ.get("IMAGE_HEIGHT", "512")),
        "grayscale": os.environ.get("IMAGE_GRAYSCALE", "True").lower() == "true",
        "denoise": os.environ.get("IMAGE_DENOISE", "True").lower() == "true",
        "enhance_contrast": os.environ.get("IMAGE_ENHANCE_CONTRAST", "False").lower() == "true", # Added enhance_contrast
        "rotation": int(os.environ.get("IMAGE_ROTATION", "0")), # Added rotation
    })
    debug: bool = field(default_factory=lambda: os.environ.get("DEBUG", "False").lower() == "true") # Added debug
    dpi: int = field(default_factory=lambda: int(os.environ.get("DPI", "300"))) # Added dpi

    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieves the API key for a given provider."""
        if provider == "openai":
            return self.openai_api_key
        elif provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "google":
            return self.google_api_key
        elif provider == "ollama":
            return None  # Ollama doesn't typically use an API key
        else:
            return None

    def get_default_model(self, provider: str) -> str:
        if provider == "ollama":
            return self.ollama_model
        else:
            # Add defaults for other providers as needed.
            return ""


def load_config() -> AppConfig:
    """Loads the application configuration."""
    return AppConfig()
