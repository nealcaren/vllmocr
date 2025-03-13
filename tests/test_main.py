import os
import pytest
from ocrv.main import process_single_image, process_pdf
from ocrv.config import AppConfig, load_config, MODEL_MAPPING
from unittest.mock import patch, ANY

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def config():
    # Load the configuration.  This will *now* pick up API keys from the environment.
    config = load_config()
    return config


@pytest.fixture
def test_data_dir():
    return os.path.join(PROJECT_ROOT, "tests", "data")


@pytest.mark.parametrize(
    "provider, model_alias",
    [
        ("openai", "gpt-4o"),
        ("anthropic", "haiku"),
        ("anthropic", "sonnet"),
        ("google", "gemini-1.5-pro-002"),  # Assuming you want a default for Google
        ("ollama", "llama3"),
        ("ollama", "minicpm"),
    ],
)
def test_process_single_image(config, test_data_dir, provider, model_alias):
    image_path = os.path.join(test_data_dir, "sample.png")

    if provider == "ollama":
        result = process_single_image(image_path, provider, config, model=model_alias)
        assert isinstance(result, str)
        assert len(result) > 0
    else:
        with patch(
            f"ocrv.llm_interface._transcribe_with_{provider}"
        ) as mock_transcribe:
            mock_transcribe.return_value = f"Mocked {provider} transcription"
            result = process_single_image(
                image_path, provider, config, model=model_alias
            )
            assert result == f"Mocked {provider} transcription"
            #  get the expected full model name
            expected_provider, expected_model = MODEL_MAPPING[model_alias]
            assert expected_provider == provider

            if provider == "openai":
                mock_transcribe.assert_called_once_with(ANY, ANY, model=expected_model)
            elif provider == "anthropic":
                mock_transcribe.assert_called_once_with(ANY, ANY, model=expected_model)
            elif provider == "google":
                mock_transcribe.assert_called_once_with(ANY, ANY, model=expected_model)
            elif provider == "ollama":
                mock_transcribe.assert_called_once_with(ANY, model=expected_model)


@pytest.mark.parametrize(
    "provider, model_alias",
    [
        ("openai", "gpt-4o"),
        ("anthropic", "haiku"),
        ("anthropic", "sonnet"),
        ("google", "gemini-1.5-pro-002"),  # Assuming you want a default for Google
        ("ollama", "llama3"),
        ("ollama", "minicpm"),
    ],
)
def test_process_pdf(config, test_data_dir, provider, model_alias):
    pdf_path = os.path.join(test_data_dir, "sample.pdf")

    if provider == "ollama":
        result = process_pdf(pdf_path, provider, config, model=model_alias)
        assert isinstance(result, str)
        assert len(result) > 0
    else:
        with patch(
            f"ocrv.llm_interface._transcribe_with_{provider}"
        ) as mock_transcribe:
            mock_transcribe.return_value = f"Mocked {provider} transcription"
            result = process_pdf(pdf_path, provider, config, model=model_alias)
            assert result == f"Mocked {provider} transcription"

            #  get the expected full model name
            expected_provider, expected_model = MODEL_MAPPING[model_alias]
            assert expected_provider == provider

            if provider == "openai":
                mock_transcribe.assert_called_once_with(ANY, ANY, model=expected_model)
            elif provider == "anthropic":
                mock_transcribe.assert_called_once_with(ANY, ANY, model=expected_model)
            elif provider == "google":
                mock_transcribe.assert_called_once_with(ANY, ANY, model=expected_model)
            elif provider == "ollama":
                mock_transcribe.assert_called_once_with(ANY, model=expected_model)

            assert mock_transcribe.call_count == 1


def test_process_single_image_invalid_file(config, test_data_dir):
    image_path = os.path.join(test_data_dir, "nonexistent.png")
    with pytest.raises(SystemExit):
        process_single_image(image_path, "openai", config)


def test_process_single_image_rotation(config, test_data_dir):
    image_path = os.path.join(test_data_dir, "sample.png")
    with patch("ocrv.llm_interface._transcribe_with_openai") as mock_transcribe:
        mock_transcribe.return_value = "Mocked OpenAI transcription"
        config.image_processing_settings["rotation"] = 90
        result = process_single_image(image_path, "openai", config)
        assert result == "Mocked OpenAI transcription"
