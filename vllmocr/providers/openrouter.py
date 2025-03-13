import logging
import os

import openai

from vllmocr.utils import handle_error, _encode_image


def _transcribe_with_openrouter(
    image_path: str,
    api_key: str,
    prompt: str,
    model: str,
    debug: bool = False,
) -> str:
    """Transcribes the text in the given image using OpenRouter."""
    if debug:
        logging.info(f"Transcribing with OpenRouter, model: {model}")
    try:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=api_key
        )
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
                            },
                        },
                    ],
                }
            ],
            stream=False,  # Assuming you want to disable streaming for now
        )
        return response.choices[0].message.content.strip()

    except openai.OpenAIError as e:
        handle_error(f"OpenRouter API error: {e}", e)
    except Exception as e:
        handle_error(f"Error during OpenRouter transcription", e)


def _post_process_openrouter(text: str) -> str:
    """Applies post-processing to OpenRouter output."""
    return text.strip()
