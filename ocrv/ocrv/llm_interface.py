import base64
from typing import Optional

import anthropic
import google.generativeai as genai
import openai
import ollama

from .config import AppConfig, get_api_key, get_default_model  # Import the modified config functions

ocr_prompt = "Convert scanned text to properly formatted markdown. Return ONLY the complete text of the page."

def _encode_image(image_path: str) -> str:
    """Encodes the image at the given path to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def _transcribe_with_openai(image_path: str, api_key: str, model: str = "gpt-4o") -> str:
    """Transcribes the text in the given image using OpenAI."""
    client = openai.OpenAI(api_key=api_key)  # Create client inside the function
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
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()

def _transcribe_with_anthropic(image_path: str, api_key: str, model: str = "claude-3-opus-20240229") -> str:
    """Transcribes the text in the given image using Anthropic."""
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
                        }
                    },
                ],
            }
        ]
    ).content[0].text
    return response

def _transcribe_with_google(image_path: str, api_key: str, model: str = "gemini-1.5-pro-002") -> str:
    """Transcribes the text in the given image using Google Gemini."""
    genai.configure(api_key=api_key)
    model_instance = genai.GenerativeModel(model)
    image = genai.Part(data=open(image_path, "rb").read(), mime_type="image/png")
    response = model_instance.generate_content([ocr_prompt, image])
    return response.text

def _transcribe_with_ollama(image_path: str, model: str) -> str:
    """Transcribes the text in the given image using Ollama."""
    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'num_ctx': 4096,
            'content': ocr_prompt,
            'images': [image_path]
        }]
    )
    return response['message']['content'].strip()

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
            raise ValueError("Model must be specified when using Ollama provider.")
        return _transcribe_with_ollama(image_path, model=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
