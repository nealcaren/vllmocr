"""Google Gemini provider for OCR transcription."""

import logging
from typing import Optional

from google import genai
from google.genai import types

from .base import BaseProvider
from ..utils import handle_error


class GoogleProvider(BaseProvider):
    """Google Gemini provider for OCR transcription.

    Supports standard models (gemini-2.5-flash, gemini-2.5-pro) and
    thinking mode via the thinking_budget parameter.
    """

    name = "google"
    requires_api_key = True
    default_model = "gemini-2.5-flash"  # 87% accuracy from Inkbench

    def transcribe(
        self,
        image_path: str,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> str:
        """Transcribe text from an image using Google Gemini.

        Args:
            image_path: Path to the image file.
            prompt: The prompt to use for transcription.
            model: The model identifier (e.g., gemini-2.5-flash).
            api_key: Google API key.
            debug: Enable debug logging.
            **kwargs: Additional options:
                - thinking_budget: Token budget for thinking mode (Gemini 2.5+).

        Returns:
            Raw transcribed text.
        """
        if debug:
            logging.info(f"Transcribing with Google, model: {model}")

        try:
            client = genai.Client(api_key=api_key)
            media_type = self._get_media_type(image_path)

            # Build content with image
            contents = [
                prompt,
                types.Part.from_bytes(
                    data=open(image_path, "rb").read(),
                    mime_type=media_type,
                ),
            ]

            # Handle thinking mode if requested
            thinking_budget = kwargs.get("thinking_budget")
            if thinking_budget:
                config = types.GenerateContentConfig(
                    thinking_config={"thinking_budget": thinking_budget}
                )
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            else:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                )

            return response.text

        except Exception as e:
            # google.genai doesn't expose a clean error hierarchy
            if "genai" in str(type(e).__module__).lower():
                handle_error(f"Google API error: {e}", e)
            else:
                handle_error("Error during Google Gemini transcription", e)
