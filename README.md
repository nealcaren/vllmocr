# vllm-ocr

[![PyPI version](https://badge.fury.io/py/vllm-ocr.svg)](https://badge.fury.io/py/vllm-ocr)

`vllm-ocr` is a command-line tool that performs Optical Character Recognition (OCR) on images and PDFs using Large Language Models (LLMs). It supports multiple LLM providers, including OpenAI, Anthropic, Google, and local models via Ollama.

## Features

*   **Image and PDF OCR:** Extracts text from both images (PNG, JPG, JPEG) and PDF files.
*   **Multiple LLM Providers:**  Supports a variety of LLMs:
    *   **OpenAI:**  GPT-4o
    *   **Anthropic:** Claude 3 Haiku, Claude 3 Sonnet
    *   **Google:** Gemini 1.5 Pro
    *   **Ollama:**  (Local models) Llama3, MiniCPM, and other models supported by Ollama.
*   **Configurable:**  Settings, including the LLM provider and model, can be adjusted via a configuration file or environment variables.
*   **Image Preprocessing:** Includes optional image rotation for improved OCR accuracy.

## Installation

It is recommended to install `vllm-ocr` using `uv`:

```bash
uv pip install vllm-ocr
```

If you don't have `uv` installed, you can install it with:
```
pipx install uv
```
You may need to restart your shell session for `uv` to be available.

Alternatively, you can use `pip`:

```bash
pip install vllm-ocr
```

## Usage

The `vllm-ocr` command-line tool has two main subcommands: `image` and `pdf`.

**1.  Process a Single Image:**

```bash
vllm-ocr image <image_path> [options]
```

*   `<image_path>`:  The path to the image file (PNG, JPG, JPEG).

**Options:**

*   `--provider`:  The LLM provider to use (openai, anthropic, google, ollama).  Defaults to `openai`.
*   `--model`: The specific model to use (e.g., `gpt-4o`, `haiku`, `gemini-1.5-pro-002`, `llama3`).  Defaults to the provider's default model.
*    `--config`: Path to a TOML configuration file.
*   `--help`: Show the help message and exit.

**Example:**

```bash
vllm-ocr image my_image.jpg --provider anthropic --model haiku
```

**2. Process a PDF:**

```bash
vllm-ocr pdf <pdf_path> [options]
```

*   `<pdf_path>`: The path to the PDF file.

**Options:** (Same as `image` subcommand)

**Example:**

```bash
vllm-ocr pdf my_document.pdf --provider openai --model gpt-4o
```

## Configuration

`vllm-ocr` can be configured using a TOML file or environment variables.  The configuration file is searched for in the following locations (in order of precedence):

1.  A path specified with the `--config` command-line option.
2.  `./config.toml` (current working directory)
3.  `~/.config/vllm-ocr/config.toml` (user's home directory)
4.  `/etc/vllm-ocr/config.toml` (system-wide)

**config.toml (Example):**

```toml
[llm]
provider = "anthropic"  # Default provider
model = "haiku"        # Default model for the provider

[image_processing]
rotation = 0           # Image rotation in degrees (optional)

[api_keys]
openai = "YOUR_OPENAI_API_KEY"
anthropic = "YOUR_ANTHROPIC_API_KEY"
google = "YOUR_GOOGLE_API_KEY"
# Ollama doesn't require an API key
```

**Environment Variables:**

You can also set API keys using environment variables:

*   `VLLM_OCR_OPENAI_API_KEY`
*   `VLLM_OCR_ANTHROPIC_API_KEY`
*   `VLLM_OCR_GOOGLE_API_KEY`

Environment variables override settings in the configuration file.  This is the recommended way to set API keys for security reasons.

## Development

To set up a development environment:

1.  Clone the repository:

    ```bash
    git clone https://github.com/<your-username>/vllm-ocr.git
    cd vllm-ocr
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
