DEFAULT_OCR_PROMPT = """Transcribe all text from this image into Markdown format.

Instructions:
- Carefully examine the entire image from top to bottom before transcribing
- Extract ALL visible text completely (never summarize or abbreviate)
- Use proper Markdown: # headings, **bold**, *italic*, lists, > blockquotes
- Preserve the document's structure and hierarchy
- Keep paragraphs as single continuous lines
- Mark unclear text as [illegible]
- Note images as [Image: brief description]

Output only the Markdown transcription, nothing else."""


def get_prompt(custom_prompt: str = None) -> str:
    """Get the OCR prompt.

    Args:
        custom_prompt: A custom prompt to use. Overrides the default.

    Returns:
        The prompt to use.
    """
    if custom_prompt:
        return custom_prompt
    return DEFAULT_OCR_PROMPT
