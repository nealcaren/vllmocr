import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class AppConfig:
    """
    Configuration for the OCR application.
    """
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    local_llm_model: str = "some_local_model"  #  Path or identifier for a local model
    image_processing_settings: Dict[str, Any] = field(default_factory=lambda: {
        "resize": True,
        "width": 512,
        "height": 512,
        "grayscale": True,
        "denoise": True,
    })
    # Add other configuration options as needed


def load_config() -> AppConfig:
    """Loads the application configuration."""
    return AppConfig()
