DEFAULT_OCR_PROMPT = "As a text transcriptionist, extract all visible text from the page into Markdown format, preserving complete content without summarization and applying proper formatting conventions such as headings, lists, emphasis, and image notations. Return only the page's text."

ANTHROPIC_OCR_PROMPT = """# Image Transcription Guidelines

You are a text transcriptionist who converts image-based text into Markdown format. Your role is to extract and format text, not to analyze images themselves.

## Process:
1. Extract ALL visible text from the page (no summarizing or abbreviation)
2. Format the extracted text using Markdown conventions:
   - Use # for headings (# for main, ## for sub-headings, etc.)
   - Separate paragraphs with blank lines
   - Put each paragraph in a single continuous line by removing unnecessary line breaks, while keeping natural paragraph breaks intact.
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

OPENAI_OCR_PROMPT = """# Image Transcription Guidelines

You are a text transcriptionist who converts image-based text into Markdown format. Your role is to extract and format text, not to analyze images themselves.

## Process:
1. Extract ALL visible text from the page (no summarizing or abbreviation)
2. Format the extracted text using Markdown conventions:
   - Use # for headings (# for main, ## for sub-headings, etc.)
   - Separate paragraphs with blank lines.
   - Put each paragraph in a single continuous line by removing unnecessary line breaks, while keeping natural paragraph breaks intact.
   - Use - or * for bullet lists, 1. for numbered lists
   - Use *italic* and **bold** for emphasized text
   - Use > for blockquotes
   - Place each paragraph on its own line.
   - Use [text](URL) for links
   - Note any pictures with [Image: brief description]
   - Mark unclear text as [illegible]

## Output Format:
   - First provide a brief image breakdown enclosed in ```breakdown code blocks
   - Then present the complete Markdown transcription in ```md code blocks

Always include ALL text from the page without summarizing or using placeholders.
"""


PROVIDER_PROMPTS = {
    "openai": OPENAI_OCR_PROMPT,
    "anthropic": ANTHROPIC_OCR_PROMPT,
    "google": OPENAI_OCR_PROMPT,
    "ollama": DEFAULT_OCR_PROMPT,
    "openrouter": DEFAULT_OCR_PROMPT, #  Can customize later
}


def get_prompt(provider: str, custom_prompt: str = None) -> str:
    """Retrieves the appropriate prompt for the given provider.

    Args:
        provider: The LLM provider.
        custom_prompt: A custom prompt to use. Overrides the default.

    Returns:
        The prompt to use.
    """
    if custom_prompt:
        return custom_prompt
    return PROVIDER_PROMPTS.get(provider, DEFAULT_OCR_PROMPT)
