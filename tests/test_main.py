import os
import pytest
from ocrv.main import process_single_image, process_pdf
from ocrv.config import AppConfig, load_config
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

@pytest.mark.parametrize("provider, model", [
    ("openai", "gpt-4o"),
    ("anthropic", "claude-3-opus-20240229"),
    ("google", "gemini-1.5-pro-002"),
    ("ollama", "llama3.2-vision"),
    ("ollama", "minicpm-v"),
])
def test_process_single_image(config, test_data_dir, provider, model):
    image_path = os.path.join(test_data_dir, "sample.png")

    # NO LONGER set dummy API keys here.  We're using the real ones.

    if provider == "ollama":
        result = process_single_image(image_path, provider, config, model=model)
        # Add assertions based on expected output.  Since we're using real API calls now,
        # we can't just assert a mocked string.  We need to check for *something* reasonable.
        assert isinstance(result, str)
        assert len(result) > 0  # Assuming *something* should be detected

    else:
        # For non-Ollama providers, we'll still mock to avoid unnecessary API calls
        # during routine testing.  You can create separate integration tests (see below)
        # that use the real API keys.
        with patch(f'ocrv.llm_interface._transcribe_with_{provider}') as mock_transcribe:
            mock_transcribe.return_value = f"Mocked {provider} transcription"
            result = process_single_image(image_path, provider, config)
            assert result == f"Mocked {provider} transcription"
            mock_transcribe.assert_called_once_with(ANY, ANY, model=model)

@pytest.mark.parametrize("provider, model", [
    ("openai", "gpt-4o"),
    ("anthropic", "claude-3-opus-20240229"),
    ("google", "gemini-1.5-pro-002"),
    ("ollama", "llama3.2-vision"),
    ("ollama", "minicpm-v"),
])
def test_process_pdf(config, test_data_dir, provider, model):
    pdf_path = os.path.join(test_data_dir, "sample.pdf")

    # NO LONGER set dummy API keys here.  We're using the real ones.

    if provider == "ollama":
        result = process_pdf(pdf_path, provider, config, model=model)
        # Add assertions based on expected output.
        assert isinstance(result, str)
        assert len(result) > 0  # Assuming *something* should be detected

    else:
        # Mock non-Ollama providers for unit tests.
        with patch(f'ocrv.llm_interface._transcribe_with_{provider}') as mock_transcribe:
            mock_transcribe.return_value = f"Mocked {provider} transcription"
            result = process_pdf(pdf_path, provider, config)
            assert result == f"Mocked {provider} transcription"
            mock_transcribe.assert_called_once_with(ANY, ANY, model=model)
            assert mock_transcribe.call_count == 1  # Assuming a single-page PDF

# Add more tests for different file types, edge cases, and image processing settings

def test_process_single_image_invalid_file(config, test_data_dir):
    image_path = os.path.join(test_data_dir, "nonexistent.png")
    with pytest.raises(SystemExit):  # Expect SystemExit due to handle_error
        process_single_image(image_path, "openai", config)

# Example of testing rotation
def test_process_single_image_rotation(config, test_data_dir):
    image_path = os.path.join(test_data_dir, "sample.png")
    with patch('ocrv.llm_interface._transcribe_with_openai') as mock_transcribe:
        mock_transcribe.return_value = "Mocked OpenAI transcription"
        # Set rotation in config
        config.image_processing_settings["rotation"] = 90
        result = process_single_image(image_path, "openai", config)
        assert result == "Mocked OpenAI transcription"
        # You could also check if preprocess_image was called with the correct rotation
        # This would require patching preprocess_image, which is more complex.
