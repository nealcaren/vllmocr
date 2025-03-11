import unittest
from unittest.mock import patch
import os
from ocrv.main import process_single_image, process_pdf
from ocrv.config import AppConfig

# Create a mock AppConfig for testing
@patch.dict(os.environ, {"DPI": "300", "IMAGE_ROTATION": "0"})
def create_test_config():
    return AppConfig()

class TestOCR(unittest.TestCase):
    def setUp(self):
        self.config = create_test_config()
        self.test_image_path = "tests/data/sample.png"
        self.test_pdf_path = "tests/data/sample.pdf"

    @patch('ocrv.llm_interface._transcribe_with_openai')
    def test_process_single_image_openai(self, mock_transcribe):
        mock_transcribe.return_value = "Mocked OpenAI transcription"
        text = process_single_image(self.test_image_path, "openai", self.config)
        self.assertEqual(text, "Mocked OpenAI transcription")
        mock_transcribe.assert_called_once_with(unittest.mock.ANY, unittest.mock.ANY, model='gpt-4o')

    @patch('ocrv.llm_interface._transcribe_with_anthropic')
    def test_process_single_image_anthropic(self, mock_transcribe):
        mock_transcribe.return_value = "Mocked Anthropic transcription"
        text = process_single_image(self.test_image_path, "anthropic", self.config)
        self.assertEqual(text, "Mocked Anthropic transcription")
        mock_transcribe.assert_called_once_with(unittest.mock.ANY, unittest.mock.ANY, model='claude-3-opus-20240229')

    @patch('ocrv.llm_interface._transcribe_with_google')
    def test_process_single_image_google(self, mock_transcribe):
        mock_transcribe.return_value = "Mocked Google transcription"
        text = process_single_image(self.test_image_path, "google", self.config)
        self.assertEqual(text, "Mocked Google transcription")
        mock_transcribe.assert_called_once_with(unittest.mock.ANY, unittest.mock.ANY, model='gemini-1.5-pro-002')

    @patch('ocrv.llm_interface._transcribe_with_ollama')
    def test_process_single_image_ollama(self, mock_transcribe):
        mock_transcribe.return_value = "Mocked Ollama transcription"
        text = process_single_image(self.test_image_path, "ollama", self.config, model="llama3.2-vision")
        self.assertEqual(text, "Mocked Ollama transcription")
        mock_transcribe.assert_called_once_with(unittest.mock.ANY, model='llama3.2-vision')

    @patch('ocrv.llm_interface._transcribe_with_openai')
    def test_process_pdf_openai(self, mock_transcribe):
        mock_transcribe.return_value = "Mocked OpenAI transcription"
        text = process_pdf(self.test_pdf_path, "openai", self.config)
        self.assertEqual(text, "Mocked OpenAI transcription\n\nMocked OpenAI transcription")  # Assuming 2 pages
        self.assertEqual(mock_transcribe.call_count, 2)

    # Add similar tests for other providers for process_pdf...

if __name__ == '__main__':
    unittest.main()
