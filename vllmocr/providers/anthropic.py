
import logging
import os

import anthropic

from vllmocr.utils import handle_error
from vllmocr.llm_interface import _encode_image


OCR_PROMPT = '''# Image Transcription Guidelines

You are a text transcriptionist who converts image-based text into Markdown format. Your role is to extract and format text, not to analyze images themselves.

## Process:
1. Extract ALL visible text from the page (no summarizing or abbreviation)
2. Format the extracted text using Markdown conventions:
   - Use # for headings (# for main, ## for sub-headings, etc.)
   - Separate paragraphs with blank lines
   - Place each paragraph on its own line.
   - Use - or * for bullet lists, 1. for numbered lists
   - Use *italic* and **bold** for emphasized text
   - Use > for blockquotes.
   - Use [text](URL) for links
   - Note any pictures with [Image: brief description]
   - Mark unclear text as [illegible]

## Output Format:
- First provide a brief <ocr_breakdown> of text elements and formatting choices
- Then present the complete Markdown transcription in <markdown_text> tags

Always include ALL text from the page without summarizing or using placeholders.
'''


def _transcribe_with_anthropic(image_path: str, api_key: str, prompt: str, model: str = "claude-3-haiku-20240307", debug: bool = False) -> str:
    """Transcribes the text in the given image using Anthropic."""
    if debug:
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
                        {"type": "text", "text": OCR_PROMPT},
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
