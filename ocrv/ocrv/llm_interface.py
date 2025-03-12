import base64
import logging
from typing import Optional
import re

import anthropic
import google.generativeai as genai
from google.generativeai import types
import google.api_core
import openai
import ollama
import os
import requests

from .config import AppConfig, get_api_key, get_default_model
from .utils import handle_error
from .prompts import get_prompt


def _encode_image(image_path: str) -> str:
    """Encodes the image at the given path to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def _transcribe_with_openai(image_path: str, api_key: str, prompt: str, model: str = "gpt-4o") -> str:
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
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            stream=False,
        )
        return response.choices[0].message.content.strip()

    except openai.OpenAIError as e:
        handle_error(f"OpenAI API error: {e}", e)
    except Exception as e:
        handle_error(f"Error during OpenAI transcription", e)


def _transcribe_with_anthropic(image_path: str, api_key: str, prompt: str, model: str = "claude-3-haiku-20240307") -> str:
    """Transcribes the text in the given image using Anthropic."""
    logging.info(f"Transcribing with Anthropic, model: {model}")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        encoded_image = _encode_image(image_path)
        image_type = os.path.splitext(image_path)[1][1:].lower()
        if image_type == "jpg":
            image_type = "jpeg"
        media_type = f"image/{image_type}"

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
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


def _transcribe_with_google(image_path: str, api_key: str, prompt: str, model: str = "gemini-1.5-pro-002") -> str:
    """Transcribes the text in the given image using Google Gemini.

    NOTE: Update your config with the new model names.
    """
    logging.info(f"Transcribing with Google, model: {model}")
    try:
        client = genai.Client(api_key=api_key)
        image_type = os.path.splitext(image_path)[1][1:].lower()
        if image_type == "jpg":
            image_type = "jpeg"

        response = client.models.generate_content(
            model=model,
            contents=[prompt,
                      types.Part.from_bytes(data=open(image_path, "rb").read(), mime_type=f"image/{image_type}")]
        )
        return response.text

    except google.api_core.exceptions.GoogleAPIError as e:
        handle_error(f"Google API error: {e}", e)

    except Exception as e:
        handle_error(f"Error during Google Gemini transcription", e)


def _transcribe_with_ollama(image_path: str, prompt: str, model: str) -> str:
    """Transcribes the text in the given image using Ollama."""
    logging.info(f"Transcribing with Ollama, model: {model}")
    try:
        # Check if the model is available
        ollama.show(model=model)
    except ollama.ResponseError as e:
        if "model" in str(e) and "not found" in str(e):
            # Ask the user if they want to pull the model
            response = input(f"Model '{model}' not found. Do you want to pull it? (y/N): ")
            if response.lower() == 'y':
                try:
                    logging.info(f"Pulling Ollama model: {model}")
                    last_status = None
                    for progress in ollama.pull(model=model, stream=True):
                        status = progress.get('status')
                        if status != last_status:
                            if 'progress' in progress:
                                print(f"  {status}: {progress['progress']}%")
                            else:
                                print(f"  {status}")
                            last_status = status

                except Exception as pull_e:
                    handle_error(f"Error pulling Ollama model: {pull_e}", pull_e)
                    return ""  # Or raise, depending on desired behavior
            else:
                print(f"Skipping transcription due to missing model: {model}")
                return ""  # Or raise, depending on desired behavior
        else:
            # Handle other Ollama errors
            handle_error(f"Ollama API error: {e}", e)
            return ""
    except requests.exceptions.RequestException as e:
        handle_error(f"Ollama API request error: {e}", e)
        return ""
    except Exception as e:
        handle_error(f"Error during Ollama transcription", e)
        return ""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "num_ctx": 4096,
                    "content": prompt,
                    "images": [image_path],
                }
            ],
        )
        return response["message"].get("content", "").strip()

    except Exception as e:
        handle_error(f"Error during Ollama transcription after model check/pull", e)
        return ""


def _post_process_openai(text: str) -> str:
    """Applies post-processing to OpenAI output.
        Extract the text between ```md and ``` delimiters.
    If the delimiters aren't present, return the entire text.

    Args:
        text (str): The input text that may contain markdown text within delimiters

    Returns:
        str: The extracted markdown text or the original text if delimiters aren't found
    """
    # Look for text between ```md and ``` delimiters
    markdown_pattern = re.compile(r'```md\s*(.*?)\s*```', re.DOTALL)
    match = markdown_pattern.search(text)

    if match:
        # Return just the content within the delimiters
        return match.group(1).strip()
    else:
        # If delimiters aren't found, return the original text
        return text.strip()

def _post_process_anthropic(text: str) -> str:
    """
    Applies post-processing to Anthropic output.
        
    Extract the text between <markdown_text> tags.
    If the tags aren't present, return the entire text.
    
    Args:
        text (str): The input text that may contain markdown text within tags
        
    Returns:
        str: The extracted markdown text or the original text if tags aren't found
    """
    # Look for text between <markdown_text> and </markdown_text> tags
    markdown_pattern = re.compile(r'<markdown_text>(.*?)</markdown_text>', re.DOTALL)
    match = markdown_pattern.search(text)
    
    if match:
        # Return just the content within the tags
        return match.group(1).strip()
    else:
        # If tags aren't found, return the original text
        return text.strip()

def _post_process_google(text: str) -> str:
    """Applies post-processing to Google Gemini output."""
    return text.strip()

def _post_process_ollama(text: str) -> str:
    """Applies post-processing to Ollama output."""
    return text.strip()

def transcribe_image(image_path: str, provider: str, config: AppConfig, model: Optional[str] = None, custom_prompt: Optional[str] = None) -> str:
    """Transcribes text from an image using the specified LLM provider and model.

    Args:
        image_path: Path to the image.
        provider: The LLM provider ('openai', 'anthropic', 'google', 'ollama').
        config: The application configuration.
        model: The specific model to use (optional).
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
        # Use the provided model directly, only trying to get from config if no model is provided
        full_model_name = model
        if model == "haiku":
            full_model_name = "claude-3-haiku-20240307"
        elif model == "sonnet":
            full_model_name = "claude-3-sonnet-20240229"
        elif model == "4o-mini":
            full_model_name = "gpt-4o-mini"
        elif model == "gpt-4o":
            full_model_name = "gpt-4o"
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
