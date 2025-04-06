import logging
import os

import openai

from vllmocr.utils import handle_error, _encode_image



OCR_PROMPT = """# Image Transcription Guidelines

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
"""

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
                            "text": OCR_PROMPT,
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
    """
    Applies post-processing to OpenRouter output.

    Extract the text between <markdown_text> tags.
    If the tags aren't present, return the entire text.

    Args:
        text (str): The input text that may contain markdown text within tags

    Returns:
        str: The extracted markdown text or the original text if tags aren't found
    """
    # Look for text between <markdown_text> and </markdown_text> tags
    markdown_pattern = re.compile(r"<markdown_text>(.*?)</markdown_text>", re.DOTALL)
    match = markdown_pattern.search(text)

    if match:
        # Return just the content within the tags
        return match.group(1).strip()
    else:
        # If tags aren't found, return the original text
        return text.strip()

