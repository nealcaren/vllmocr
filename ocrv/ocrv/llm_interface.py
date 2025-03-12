import base64
import logging
from typing import Optional

import anthropic
import google.generativeai as genai
import google.api_core
import openai
import ollama
import requests
import httpx  # Import httpx

from .config import AppConfig, get_api_key, get_default_model
from .utils import handle_error

ocr_prompt = "Convert scanned text to properly formatted markdown. Return ONLY the complete text of the page."

def _encode_image(image_path: str) -> str:
    """Encodes the image at the given path to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def _transcribe_with_openai(image_path: str, api_key: str, model: str = "gpt-4o") -> str:
    """Transcribes the text in the given image using OpenAI."""
    logging.info(f"Transcribing with OpenAI, model: {model}")
    try:
        # Forcefully disable proxies by creating a custom httpx client
        custom_client = httpx.Client(proxies={})

        client = openai.OpenAI(api_key=api_key, http_client=custom_client)
        base64_image = _encode_image(image_path)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": ocr_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:  # Catch the general OpenAIError
        handle_error(f"OpenAI API error: {e}", e)
    except Exception as e:
        handle_error(f"Error during OpenAI transcription", e)


def _transcribe_with_anthropic(image_path: str, api_key: str, model: str = "claude-3-opus-20240229") -> str:
    """Transcribes the text in the given image using Anthropic."""
    logging.info(f"Transcribing with Anthropic, model: {model}")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        encoded_image = _encode_image(image_path)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ocr_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": encoded_image,
                            },
                        },
                    ],
                }
            ],
        ).content[0].text
        return response
    except anthropic.APIConnectionError as e:
        handle_error(f"Anthropic API connection error: {e}", e)
    except anthropic.RateLimitError as e:
        handle_error(f"Anthropic rate limit exceeded: {e}", e)
    except anthropic.APIStatusError as e:
        handle_error(f"Anthropic API status error: {e}", e)
    except Exception as e:
        handle_error(f"Error during Anthropic transcription", e)

def _transcribe_with_google(image_path: str, api_key: str, model: str = "gemini-1.5-pro-002") -> str:
    """Transcribes the text in the given image using Google Gemini."""
    logging.info(f"Transcribing with Google, model: {model}")
    try:
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        image = genai.Part(data=open(image_path, "rb").read(), mime_type="image/png")
        response = model_instance.generate_content([ocr_prompt, image])
        return response.text
    except google.api_core.exceptions.GoogleAPIError as e:
        handle_error(f"Google API error: {e}", e)
    except google.api_core.exceptions.RetryError as e:
        handle_error(f"Google API retry error: {e}", e)
    except Exception as e:
        handle_error(f"Error during Google Gemini transcription", e)

def _transcribe_with_ollama(image_path: str, model: str) -> str:
    """Transcribes the text in the given image using Ollama."""
    logging.info(f"Transcribing with Ollama, model: {model}")
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "num_ctx": 4096,
                    "content": ocr_prompt,
                    "images": [image_path],
                }
            ],
        )
        return response["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        handle_error(f"Ollama API request error: {e}", e)
    except Exception as e:
        handle_error(f"Error during Ollama transcription", e)

def transcribe_image(image_path: str, provider: str, config: AppConfig, model: Optional[str] = None) -> str:
    """Transcribes text from an image using the specified LLM provider and model.

    Args:
        image_path: Path to the image.
        provider: The LLM provider ('openai', 'anthropic', 'google', 'ollama').
        config: The application configuration.
        model: The specific model to use (required for Ollama, optional for others).

    Returns:
        The transcribed text.

    Raises:
        ValueError: If the provider is not supported or if the model is required but not provided.
    """
    api_key = get_api_key(config, provider)
    logging.info(f"Using provider: {provider}")

    if provider == "openai":
        openai_model = model if model else "gpt-4o"
        return _transcribe_with_openai(image_path, api_key, model=openai_model)
    elif provider == "anthropic":
        anthropic_model = model if model else "claude-3-opus-20240229"
        return _transcribe_with_anthropic(image_path, api_key, model=anthropic_model)
    elif provider == "google":
        google_model = model if model else "gemini-1.5-pro-002"
        return _transcribe_with_google(image_path, api_key, model=google_model)
    elif provider == "ollama":
        if not model:
            handle_error("Model must be specified when using Ollama provider.")
        return _transcribe_with_ollama(image_path, model=model)
    else:
        handle_error(f"Unsupported LLM provider: {provider}")
