"""Ollama local provider for OCR transcription."""

import logging
from typing import Optional

import ollama
import requests

from .base import BaseProvider
from ..utils import handle_error, _encode_image


class OllamaProvider(BaseProvider):
    """Ollama local provider for OCR transcription.

    Runs locally without an API key. Supports vision models like
    llama3.2-vision and minicpm-v.
    """

    name = "ollama"
    requires_api_key = False
    default_model = "llama3.2-vision"

    def transcribe(
        self,
        image_path: str,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,  # Ignored but kept for interface consistency
        debug: bool = False,
        **kwargs,
    ) -> str:
        """Transcribe text from an image using Ollama.

        Args:
            image_path: Path to the image file.
            prompt: The prompt to use for transcription.
            model: The model identifier (e.g., llama3.2-vision).
            api_key: Ignored (Ollama doesn't use API keys).
            debug: Enable debug logging.
            **kwargs: Additional options (unused for Ollama).

        Returns:
            Raw transcribed text.
        """
        if debug:
            logging.info(f"Transcribing with Ollama, model: {model}")

        # Check and pull model if needed
        if not self._ensure_model_available(model, debug):
            return ""

        try:
            encoded_image = _encode_image(image_path)
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [encoded_image],
                    }
                ],
                options={"num_ctx": 4096},
            )
            return response["message"].get("content", "").strip()

        except Exception as e:
            handle_error("Error during Ollama transcription", e)
            return ""

    def _ensure_model_available(self, model: str, debug: bool) -> bool:
        """Check if model is available, offer to pull if not."""
        try:
            ollama.show(model=model)
            return True
        except ollama.ResponseError as e:
            if "model" in str(e) and "not found" in str(e):
                response = input(
                    f"Model '{model}' not found. Do you want to pull it? (y/N): "
                )
                if response.lower() == "y":
                    return self._pull_model(model, debug)
                else:
                    print(f"Skipping transcription due to missing model: {model}")
                    return False
            else:
                handle_error(f"Ollama API error: {e}", e)
                return False
        except requests.exceptions.RequestException as e:
            handle_error(f"Ollama API request error: {e}", e)
            return False
        except Exception as e:
            handle_error("Error checking Ollama model", e)
            return False

    def _pull_model(self, model: str, debug: bool) -> bool:
        """Pull an Ollama model."""
        try:
            if debug:
                logging.info(f"Pulling Ollama model: {model}")
            last_status = None
            for progress in ollama.pull(model=model, stream=True):
                status = progress.get("status")
                if status != last_status:
                    if "progress" in progress and debug:
                        print(f"  {status}: {progress['progress']}%")
                    elif debug:
                        print(f"  {status}")
                    last_status = status
            return True
        except Exception as e:
            handle_error(f"Error pulling Ollama model: {e}", e)
            return False
