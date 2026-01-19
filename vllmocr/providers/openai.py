"""OpenAI provider for OCR transcription."""

import logging
from typing import Optional

import openai

from .base import BaseProvider
from ..utils import handle_error, _encode_image


class OpenAIProvider(BaseProvider):
    """OpenAI provider for OCR transcription.

    Supports standard models (gpt-4o, gpt-4.1-mini) and reasoning models
    (o1, o3, o4-mini) with vision capabilities.

    Note: o3-mini does NOT support vision - use o3 or o4-mini for image tasks.
    """

    name = "openai"
    requires_api_key = True
    default_model = "gpt-4.1-mini"  # Best value from Inkbench: 88% accuracy @ $0.11/1K

    def transcribe(
        self,
        image_path: str,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> str:
        """Transcribe text from an image using OpenAI.

        Args:
            image_path: Path to the image file.
            prompt: The prompt to use for transcription.
            model: The model identifier (e.g., gpt-4.1-mini, o3).
            api_key: OpenAI API key.
            debug: Enable debug logging.
            **kwargs: Additional options (unused for OpenAI).

        Returns:
            Raw transcribed text.
        """
        if debug:
            logging.info(f"Transcribing with OpenAI, model: {model}")

        try:
            client = openai.OpenAI(api_key=api_key)
            base64_image = _encode_image(image_path)
            media_type = self._get_media_type(image_path)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                stream=False,
            )
            return response.choices[0].message.content.strip()

        except openai.OpenAIError as e:
            handle_error(f"OpenAI API error: {e}", e)
        except Exception as e:
            handle_error("Error during OpenAI transcription", e)
