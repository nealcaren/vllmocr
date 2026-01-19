"""Unit tests for vllmocr main module."""

import pytest
from unittest.mock import patch, MagicMock

from vllmocr.main import process_single_image, process_pdf
from vllmocr.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.mark.parametrize(
    "provider, model_name",
    [
        ("openai", "gpt-4o"),
        ("openai", "gpt-5-mini"),
        ("anthropic", "haiku"),
        ("anthropic", "sonnet"),
        ("google", "gemini-2.5-flash"),
        ("ollama", "llama3"),
        ("ollama", "minicpm"),
        ("openrouter", "meta-llama/llama-3.2-90b-vision-instruct"),
    ],
)
def test_process_single_image_makes_provider_call(config, provider, model_name):
    """Test that process_single_image calls the correct provider."""
    # Mock at the usage location (vllmocr.main) not the definition location
    with patch("vllmocr.main.preprocess_image", return_value="/tmp/x.png"):
        # Mock the provider's transcribe method via the registry
        mock_provider = MagicMock()
        mock_provider.transcribe.return_value = f"Mocked {provider} transcription"
        mock_provider.post_process.return_value = f"Mocked {provider} transcription"
        mock_provider.requires_api_key = provider != "ollama"
        mock_provider.default_model = model_name

        with patch("vllmocr.llm_interface.get_provider", return_value=mock_provider):
            if provider == "ollama":
                out = process_single_image("/tmp/in.png", provider, config, model=model_name)
            else:
                out = process_single_image("/tmp/in.png", provider, config, model=model_name, api_key="XYZ")

            assert out == f"Mocked {provider} transcription"
            mock_provider.transcribe.assert_called_once()

            # Verify model was passed correctly
            _, kwargs = mock_provider.transcribe.call_args
            assert kwargs.get("model") == model_name


@pytest.mark.parametrize(
    "provider, model_name",
    [
        ("openai", "gpt-4o"),
        ("anthropic", "haiku"),
        ("google", "gemini-2.5-flash"),
        ("ollama", "llama3"),
        ("openrouter", "meta-llama/llama-3.2-90b-vision-instruct"),
    ],
)
def test_process_pdf_single_page(config, provider, model_name):
    """Test that process_pdf processes a single page PDF correctly."""
    with patch("vllmocr.main.pdf_to_images", return_value=["/tmp/page1.png"]), \
         patch("vllmocr.main.preprocess_image", return_value="/tmp/p1.png"):

        mock_provider = MagicMock()
        mock_provider.transcribe.return_value = f"Mocked {provider} transcription"
        mock_provider.post_process.return_value = f"Mocked {provider} transcription"
        mock_provider.requires_api_key = provider != "ollama"
        mock_provider.default_model = model_name

        with patch("vllmocr.llm_interface.get_provider", return_value=mock_provider):
            out = process_pdf("/tmp/in.pdf", provider, config, model=model_name)
            assert out == f"Mocked {provider} transcription"
            mock_provider.transcribe.assert_called_once()


def test_api_key_precedence_for_openai(config):
    """Test that CLI-provided API key takes precedence over config."""
    with patch("vllmocr.main.preprocess_image", return_value="/tmp/x.png"):
        mock_provider = MagicMock()
        mock_provider.transcribe.return_value = "OK"
        mock_provider.post_process.return_value = "OK"
        mock_provider.requires_api_key = True
        mock_provider.default_model = "gpt-5-mini"

        with patch("vllmocr.llm_interface.get_provider", return_value=mock_provider):
            out = process_single_image("/tmp/in.png", "openai", config, model="gpt-5-mini", api_key="CLI_KEY")
            assert out == "OK"

            # Verify CLI key was used
            _, kwargs = mock_provider.transcribe.call_args
            assert kwargs.get("api_key") == "CLI_KEY"


def test_thinking_budget_passed_to_provider(config):
    """Test that thinking_budget is passed to the provider."""
    with patch("vllmocr.main.preprocess_image", return_value="/tmp/x.png"):
        mock_provider = MagicMock()
        mock_provider.transcribe.return_value = "OK"
        mock_provider.post_process.return_value = "OK"
        mock_provider.requires_api_key = True
        mock_provider.default_model = "claude-sonnet-4-20250514"

        with patch("vllmocr.llm_interface.get_provider", return_value=mock_provider):
            out = process_single_image(
                "/tmp/in.png",
                "anthropic",
                config,
                model="claude-sonnet-4-20250514",
                api_key="KEY",
                thinking_budget=2048,
            )
            assert out == "OK"

            # Verify thinking_budget was passed
            _, kwargs = mock_provider.transcribe.call_args
            assert kwargs.get("thinking_budget") == 2048
