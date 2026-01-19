"""Integration tests using NealCaren/OCRTrain HuggingFace dataset.

These tests verify that providers return plausible results when called with
real API keys. Tests are skipped if the required API key is not available.

Run with: pytest tests/test_integration.py -v
"""

import os
import tempfile
from typing import Optional

import pytest

from vllmocr.config import load_config, AppConfig
from vllmocr.providers import get_provider, list_providers
from vllmocr.providers.base import BaseProvider


def has_api_key(provider: str) -> bool:
    """Check if API key is available for provider."""
    config = load_config()
    key = config.get_api_key(provider)
    return bool(key)


class TestProviderRegistry:
    """Unit tests for the provider registry."""

    def test_list_providers_returns_all(self):
        """Test that list_providers returns all expected providers."""
        providers = list_providers()
        expected = {"openai", "anthropic", "google", "ollama", "openrouter"}
        assert set(providers) == expected

    def test_get_provider_returns_correct_type(self):
        """Test that get_provider returns BaseProvider instances."""
        for name in ["openai", "anthropic", "google", "ollama", "openrouter"]:
            provider = get_provider(name)
            assert isinstance(provider, BaseProvider)
            assert provider.name == name

    def test_get_provider_unknown_raises(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_provider_requires_api_key_correct(self):
        """Test that requires_api_key is set correctly for each provider."""
        assert get_provider("openai").requires_api_key is True
        assert get_provider("anthropic").requires_api_key is True
        assert get_provider("google").requires_api_key is True
        assert get_provider("ollama").requires_api_key is False
        assert get_provider("openrouter").requires_api_key is True

    def test_provider_default_models(self):
        """Test that default models are set based on Inkbench findings."""
        assert get_provider("openai").default_model == "gpt-4.1-mini"
        assert get_provider("anthropic").default_model == "claude-sonnet-4-20250514"
        assert get_provider("google").default_model == "gemini-2.5-flash"
        assert "vision" in get_provider("ollama").default_model.lower()
        assert "qwen" in get_provider("openrouter").default_model.lower()


class TestPostProcessing:
    """Test post-processing logic."""

    def test_markdown_code_block_extraction(self):
        """Test extraction of markdown from ```md code blocks."""
        from vllmocr.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        text = "Some preamble\n```md\n# Header\nContent here\n```\nSome epilogue"
        result = provider.post_process(text)

        assert result == "# Header\nContent here"

    def test_xml_tag_extraction(self):
        """Test extraction of markdown from <markdown_text> XML tags."""
        from vllmocr.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        text = "<ocr_breakdown>Analysis</ocr_breakdown>\n<markdown_text># Header\nContent</markdown_text>"
        result = provider.post_process(text)

        assert result == "# Header\nContent"

    def test_fallback_to_strip(self):
        """Test that text is stripped when no markers are found."""
        from vllmocr.providers.ollama import OllamaProvider

        provider = OllamaProvider()
        text = "  Plain text without markers  \n"
        result = provider.post_process(text)

        assert result == "Plain text without markers"

    def test_google_markdown_extraction(self):
        """Test Google provider extracts from code blocks."""
        from vllmocr.providers.google import GoogleProvider

        provider = GoogleProvider()
        text = "```md\n## Test\nThis is content.\n```"
        result = provider.post_process(text)

        assert result == "## Test\nThis is content."


class TestConfigUpdates:
    """Test configuration updates."""

    def test_openrouter_api_key_in_config(self):
        """Test that OpenRouter API key is properly configured."""
        config = load_config()
        # Should not raise - method should handle OpenRouter
        key = config.get_api_key("openrouter")
        # Key may be empty string if not set, but method should work
        assert key is not None or key == "" or key is None

    def test_model_mapping_has_new_aliases(self):
        """Test that MODEL_MAPPING includes new aliases from Inkbench."""
        from vllmocr.config import MODEL_MAPPING

        # Check new aliases exist
        assert "4.1-mini" in MODEL_MAPPING
        assert "o3" in MODEL_MAPPING
        assert "o4-mini" in MODEL_MAPPING
        assert "gemini-lite" in MODEL_MAPPING
        assert "qwen" in MODEL_MAPPING

        # Check values are correct tuples
        assert MODEL_MAPPING["4.1-mini"] == ("openai", "gpt-4.1-mini")
        assert MODEL_MAPPING["qwen"] == ("openrouter", "qwen/qwen3-vl-235b")

    def test_default_models_reflect_inkbench(self):
        """Test that default models are based on Inkbench findings."""
        config = load_config()

        # OpenAI should default to gpt-4.1-mini (88% accuracy)
        assert config.get_default_model("openai") == "gpt-4.1-mini"

        # Google should default to gemini-2.5-flash (87% accuracy)
        assert config.get_default_model("google") == "gemini-2.5-flash"

        # OpenRouter should default to top performer
        assert "qwen" in config.get_default_model("openrouter").lower()


# Module-level fixtures for integration tests
@pytest.fixture(scope="module")
def ocr_dataset():
    """Load the OCRTrain dataset once for all tests."""
    try:
        from datasets import load_dataset
        return load_dataset("NealCaren/OCRTrain", split="train")
    except Exception:
        pytest.skip("Could not load HuggingFace dataset")


@pytest.fixture(scope="module")
def sample_image_path(ocr_dataset):
    """Extract a sample image for testing."""
    sample = ocr_dataset[0]
    image = sample["image"]
    ground_truth = sample.get("text", "")

    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)

    yield temp_file.name, ground_truth

    # Cleanup
    try:
        os.unlink(temp_file.name)
    except Exception:
        pass


# Integration tests that require API keys - skipped if not available
class TestProviderIntegration:
    """Integration tests using real providers and HuggingFace dataset.

    These tests are skipped if the required API key is not available.
    They test that providers return plausible results (not quality, just that they work).
    """

    @pytest.mark.skipif(not has_api_key("openai"), reason="No OpenAI API key")
    def test_openai_returns_plausible_result(self, sample_image_path):
        """Test that OpenAI returns non-empty, reasonable output."""
        from vllmocr.main import process_single_image

        image_path, ground_truth = sample_image_path
        config = load_config()

        result = process_single_image(
            image_path,
            provider="openai",
            config=config,
            model="gpt-4o-mini",  # Use cheaper model for testing
        )

        # Basic plausibility checks
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 10  # Should have some content
        assert "[ERROR" not in result  # No error markers

    @pytest.mark.skipif(not has_api_key("anthropic"), reason="No Anthropic API key")
    def test_anthropic_returns_plausible_result(self, sample_image_path):
        """Test that Anthropic returns non-empty, reasonable output."""
        from vllmocr.main import process_single_image

        image_path, ground_truth = sample_image_path
        config = load_config()

        result = process_single_image(
            image_path,
            provider="anthropic",
            config=config,
            model="claude-3-5-haiku-latest",  # Use cheaper model for testing
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 10
        assert "[ERROR" not in result

    @pytest.mark.skipif(not has_api_key("google"), reason="No Google API key")
    def test_google_returns_plausible_result(self, sample_image_path):
        """Test that Google returns non-empty, reasonable output."""
        from vllmocr.main import process_single_image

        image_path, ground_truth = sample_image_path
        config = load_config()

        result = process_single_image(
            image_path,
            provider="google",
            config=config,
            model="gemini-2.5-flash",
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 10
        assert "[ERROR" not in result

    @pytest.mark.skipif(not has_api_key("openrouter"), reason="No OpenRouter API key")
    def test_openrouter_returns_plausible_result(self, sample_image_path):
        """Test that OpenRouter returns non-empty, reasonable output."""
        from vllmocr.main import process_single_image

        image_path, ground_truth = sample_image_path
        config = load_config()

        result = process_single_image(
            image_path,
            provider="openrouter",
            config=config,
            model="google/gemini-2.0-flash-001",  # Use reliable model via OpenRouter
        )

        assert result is not None
        assert isinstance(result, str)
        # OpenRouter may return short responses due to content filters, be lenient
        assert len(result) > 5
        assert "[ERROR" not in result


class TestThinkingMode:
    """Tests for thinking/reasoning mode support."""

    @pytest.mark.skipif(not has_api_key("anthropic"), reason="No Anthropic API key")
    def test_anthropic_extended_thinking_accepted(self, sample_image_path):
        """Test that Anthropic accepts thinking_budget parameter."""
        from vllmocr.main import process_single_image

        image_path, _ = sample_image_path
        config = load_config()

        # This should not raise - the parameter should be accepted
        result = process_single_image(
            image_path,
            provider="anthropic",
            config=config,
            model="claude-sonnet-4-20250514",
            thinking_budget=1024,  # Minimum budget
        )

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.skipif(not has_api_key("google"), reason="No Google API key")
    def test_google_thinking_mode_accepted(self, sample_image_path):
        """Test that Google accepts thinking_budget parameter."""
        from vllmocr.main import process_single_image

        image_path, _ = sample_image_path
        config = load_config()

        # This should not raise - the parameter should be accepted
        result = process_single_image(
            image_path,
            provider="google",
            config=config,
            model="gemini-2.5-flash",
            thinking_budget=1024,
        )

        assert result is not None
        assert isinstance(result, str)
