"""Anthropic Claude provider for OCR transcription."""

import logging
from typing import Optional

import anthropic

from .base import BaseProvider
from ..utils import handle_error, _encode_image


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider for OCR transcription.

    Supports standard models (claude-sonnet-4, claude-3-5-haiku) and
    extended thinking mode via the thinking_budget parameter.
    """

    name = "anthropic"
    requires_api_key = True
    default_model = "claude-sonnet-4-20250514"

    def transcribe(
        self,
        image_path: str,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> str:
        """Transcribe text from an image using Anthropic Claude.

        Args:
            image_path: Path to the image file.
            prompt: The prompt to use for transcription.
            model: The model identifier (e.g., claude-sonnet-4-20250514).
            api_key: Anthropic API key.
            debug: Enable debug logging.
            **kwargs: Additional options:
                - thinking_budget: Token budget for extended thinking mode.

        Returns:
            Raw transcribed text.
        """
        if debug:
            logging.info(f"Transcribing with Anthropic, model: {model}")

        try:
            client = anthropic.Anthropic(api_key=api_key)
            encoded_image = _encode_image(image_path)
            media_type = self._get_media_type(image_path)

            # Build message content
            message_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": encoded_image,
                    },
                },
            ]

            # Handle extended thinking if requested
            thinking_budget = kwargs.get("thinking_budget")
            if thinking_budget:
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    },
                    messages=[{"role": "user", "content": message_content}],
                )
            else:
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": message_content}],
                )

            # When extended thinking is enabled, response contains ThinkingBlock + TextBlock
            # We need to find the TextBlock
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
            # Fallback if no text block found
            return str(response.content[0])

        except anthropic.APIConnectionError as e:
            handle_error(f"Anthropic API connection error: {e}", e)
        except anthropic.RateLimitError as e:
            handle_error(f"Anthropic rate limit exceeded: {e}", e)
        except anthropic.APIStatusError as e:
            handle_error(f"Anthropic API status error: {e}", e)
