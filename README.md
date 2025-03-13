# vllmocr

[![PyPI version](https://badge.fury.io/py/vllmocr.svg)](https://badge.fury.io/py/vllmocr)

`vllmocr` is a command-line tool that performs Optical Character Recognition (OCR) on images and PDFs using Large Language Models (LLMs). It supports multiple LLM providers, including OpenAI, Anthropic, Google, and local models via Ollama.

## Features

*   **Image and PDF OCR:** Extracts text from both images (PNG, JPG, JPEG) and PDF files.
*   **Multiple LLM Providers:**  Supports a variety of LLMs:
    *   **OpenAI:**  GPT-4o
    *   **Anthropic:** Claude 3 Haiku, Claude 3.5 Haiku, Claude 3 Sonnet
    *   **Google:** Gemini 1.5 Pro
    *   **Ollama:**  (Local models) Llama3, Llama3.2-vision, MiniCPM, and other models supported by Ollama.
    *   **OpenRouter:** Access to various models through the OpenRouter API
*   **Configurable:**  Settings, including the LLM provider and model, can be adjusted via a configuration file or environment variables.
*   **Image Preprocessing:** Includes optional image rotation for improved OCR accuracy.

## Installation

The recommended way to install `vllmocr` is using `uv tool install`:

```bash
uv tool install vllmocr
```

If you don't have `uv` installed, you can install it with:
```
curl -sSf https://install.ultraviolet.rs | sh
```
You may need to restart your shell session for `uv` to be available.

Alternatively, you can use `uv pip` or regular `pip`:

```bash
uv pip install vllmocr
```

```bash
pip install vllmocr
```

## Usage

`vllmocr` is a simple command-line tool that processes both images and PDFs:

```bash
vllmocr <file_path> [options]
```

*   `<file_path>`:  The path to the image file (PNG, JPG, JPEG) or PDF file.

**Options:**

*   `-o, --output`: Output file name (default: auto-generated based on input filename and model).
*   `-p, --provider`: The LLM provider to use (openai, anthropic, google, ollama, openrouter). Defaults to `anthropic`.
*   `-m, --model`: The specific model to use (e.g., `gpt-4o`, `haiku`, `llama3.2-vision`, `google/gemma-3-27b-it`). Defaults to `claude-3-5-haiku-latest`.
*   `-c, --custom-prompt`: Custom prompt to use for the LLM.
*   `--api-key`: API key for the LLM provider. Overrides API keys from the config file or environment variables.
*   `--rotate`: Manually rotate image by specified degrees (0, 90, 180, or 270).
*   `--debug`: Save intermediate processing steps for debugging.
*   `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
*   `--help`: Show the help message and exit.

**Examples:**

```bash
vllmocr my_image.jpg -m haiku
```

```bash
vllmocr document.pdf -p ollama -m llama3.2-vision
```

```bash
vllmocr scan.jpg -p openai -m gpt-4o --rotate 90
```

Running `vllmocr` without arguments will display a help message with usage examples.

## Configuration

`vllmocr` can be configured using a TOML file or environment variables. The configuration file is searched for in the following locations (in order of precedence):

1.  `./config.toml` (current working directory)
2.  `~/.config/vllmocr/config.toml` (user's home directory)
3.  `/etc/vllmocr/config.toml` (system-wide)

**config.toml (Example):**

```toml
[llm]
provider = "anthropic"  # Default provider
model = "claude-3-5-haiku-latest"  # Default model for the provider

[image_processing]
rotation = 0           # Image rotation in degrees (optional)

[api_keys]
openai = "YOUR_OPENAI_API_KEY"
anthropic = "YOUR_ANTHROPIC_API_KEY"
google = "YOUR_GOOGLE_API_KEY"
openrouter = "YOUR_OPENROUTER_API_KEY"
# Ollama doesn't require an API key
```

**Environment Variables:**

You can also set API keys using environment variables:

*   `VLLM_OCR_OPENAI_API_KEY`
*   `VLLM_OCR_ANTHROPIC_API_KEY`
*   `VLLM_OCR_GOOGLE_API_KEY`
*   `VLLM_OCR_OPENROUTER_API_KEY`

Environment variables override settings in the configuration file. This is the recommended way to set API keys for security reasons. You can also pass the API key directly via the `--api-key` command-line option, which takes the highest precedence.

## Development

To set up a development environment:

1.  Clone the repository:

    ```bash
    git clone https://github.com/<your-username>/vllmocr.git
    cd vllmocr
    ```

2.  Create and activate a virtual environment (using `uv`):

    ```bash
    uv venv
    uv pip install -e .[dev]
    ```

    This installs the package in editable mode (`-e`) along with development dependencies (like `pytest` and `pytest-mock`).

3.  Run tests:

    ```bash
    uv pip install pytest pytest-mock  # if not already installed as dev dependencies
    pytest
    ```

## License

This project is licensed under the MIT License (see `pyproject.toml` for details).
