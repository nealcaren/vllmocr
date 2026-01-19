"""Base classes and protocol for OCR providers."""

import os
import re
from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class OCRProvider(Protocol):
    """Protocol defining the interface for OCR providers."""

    name: str
    requires_api_key: bool
    default_model: str

    def transcribe(
        self,
        image_path: str,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> str:
        """Transcribe text from an image.

        Args:
            image_path: Path to the image file.
            prompt: The prompt to use for transcription.
            model: The model identifier.
            api_key: API key (None for providers that don't need it).
            debug: Enable debug logging.
            **kwargs: Provider-specific options (e.g., thinking_budget).

        Returns:
            Raw transcribed text (before post-processing).
        """
        ...

    def post_process(self, text: str) -> str:
        """Apply provider-specific post-processing to the transcription result.

        Args:
            text: Raw transcription output.

        Returns:
            Cleaned/processed text.
        """
        ...


class BaseProvider(ABC):
    """Base class providing common functionality for providers."""

    name: str
    requires_api_key: bool = True
    default_model: str

    # Shared post-processing patterns
    MARKDOWN_CODE_PATTERN = re.compile(r"```md\s*(.*?)\s*```", re.DOTALL)
    XML_TAG_PATTERN = re.compile(r"<markdown_text>(.*?)</markdown_text>", re.DOTALL)

    @abstractmethod
    def transcribe(
        self,
        image_path: str,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> str:
        """Provider-specific transcription implementation."""
        pass

    def post_process(self, text: str) -> str:
        """Default post-processing - extract markdown from delimiters.

        Tries ```md code blocks first, then XML tags, then returns stripped text.
        Can be overridden by subclasses for provider-specific behavior.
        """
        return self._extract_markdown(text)

    def _extract_markdown(self, text: str) -> str:
        """Extract markdown content from various delimiters."""
        # Try ```md code blocks first
        match = self.MARKDOWN_CODE_PATTERN.search(text)
        if match:
            return match.group(1).strip()

        # Try XML tags
        match = self.XML_TAG_PATTERN.search(text)
        if match:
            return match.group(1).strip()

        # Return original text stripped
        return text.strip()

    def _get_media_type(self, image_path: str) -> str:
        """Get MIME type from image path."""
        ext = os.path.splitext(image_path)[1][1:].lower()
        if ext == "jpg":
            ext = "jpeg"
        return f"image/{ext}"
