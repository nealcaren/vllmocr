import base64
import logging
from typing import Optional

import anthropic
import google.generativeai as genai
import google.api_core
import openai
import ollama
import os  # Import the os module
import requests

from .config import AppConfig, get_api_key, get_default_model
from .utils import handle_error
from .prompts import get_prompt


def _encode_image(image_path: str) -> str:
    """Encodes the image at the given path to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def _transcribe_with_openai(image_path: str, api_key: str, model: str = "gpt-4o") -> str:
    """Transcribes the text in the given image using OpenAI."""
    logging.info(f"Transcribing with OpenAI, model: {model}")
    try:
        client = openai.OpenAI(api_key=api_key)
        base64_image = _encode_image(image_path)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{DEFAULT_OCR_PROMPT}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        handle_error(f"OpenAI API error: {e}", e)
    except Exception as e:
        handle_error(f"Error during OpenAI transcription", e)


def _transcribe_with_anthropic(image_path: str, api_key: str, model: str = "claude-3-haiku-20240307") -> str:
    """Transcribes the text in the given image using Anthropic."""
    logging.info(f"Transcribing with Anthropic, model: {model}")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        encoded_image = _encode_image(image_path)
        # Determine the correct media type based on the file extension.
        image_type = os.path.splitext(image_path)[1][1:].lower()
        if image_type == "jpg":
            image_type = "jpeg"  # Correct common inconsistency
        media_type = f"image/{image_type}"

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DEFAULT_OCR_PROMPT},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
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
        response = model_instance.generate_content([DEFAULT_OCR_PROMPT, image])
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
                    "content": DEFAULT_OCR_PROMPT,
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

        ValueError: If the provider is not supported or if the model is required but not provided.
    """
    Args:
        image_path: Path to the image.
        provider: The LLM provider ('openai', 'anthropic', 'google', 'ollama').
        config: The application configuration.
        model: The specific model to use (required for Ollama, optional for others).
        custom_prompt: Optional custom prompt to use.

    Returns:
        The transcribed text.

    Raises:
        ValueError: If the provider is not supported or if the model is required but not provided.
    """
    logging.info(f"TRACE: Entering transcribe_image with provider={provider}, model={model}, custom_prompt={custom_prompt}")

    # Get the full model name based on the alias or use the provided model
    full_model_name = None
    if model:
        try:
            full_model_name = config.get_default_model(provider)
        except:
            pass

        if not full_model_name:
            if model == "haiku":
                full_model_name = "claude-3-haiku-20240307"
            elif model == "sonnet":
                full_model_name = "claude-3-sonnet-20240229"
            elif model == "4o-mini":
                full_model_name = "gpt-4o-mini"
            elif model == "gpt-4o":
                full_model_name = "gpt-4o"
            else:
                full_model_name = model
    else:
        try:
            full_model_name = config.get_default_model(provider)
        except Exception as e:
            logging.error(f"TRACE: Error getting default model: {str(e)}")
            raise ValueError(f"No model specified and couldn't get default for provider {provider}")

    api_key = get_api_key(config, provider)
    if not api_key and provider != "ollama":
        raise ValueError(f"No API key found for provider {provider}")

    prompt = get_prompt(provider, custom_prompt)
    logging.info(f"Using provider: {provider}, model: {full_model_name}, prompt: {prompt}")

    if provider == "openai":
        text = _transcribe_with_openai(image_path, api_key, prompt, model=full_model_name)
        return _post_process_openai(text)
    elif provider == "anthropic":
        text = _transcribe_with_anthropic(image_path, api_key, prompt, model=full_model_name)
        return _post_process_anthropic(text)
    elif provider == "google":
        text = _transcribe_with_google(image_path, api_key, prompt, model=full_model_name)
        return _post_process_google(text)
    elif provider == "ollama":
        text = _transcribe_with_ollama(image_path, prompt, model=full_model_name)
        return _post_process_ollama(text)
    else:
        handle_error(f"Unsupported LLM provider: {provider}")
