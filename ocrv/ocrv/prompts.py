DEFAULT_OCR_PROMPT = "Extract all text from the image and format it as Markdown."

PROVIDER_PROMPTS = {
    "openai": DEFAULT_OCR_PROMPT,
    "anthropic": DEFAULT_OCR_PROMPT,
    "google": DEFAULT_OCR_PROMPT,
    "ollama": DEFAULT_OCR_PROMPT,
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
