# vllmocr

[![PyPI version](https://badge.fury.io/py/vllmocr.svg)](https://badge.fury.io/py/vllmocr)

`vllmocr` is a command-line tool that performs Optical Character Recognition (OCR) on images and PDFs using Large Language Models (LLMs). The LLM model is prompted to return the complete text in Markdown format. `vllmocr` supports multiple LLM providers, including OpenAI, Anthropic, Google, and local models via Ollama. It was designed to assist with creating text versions of public domain books and historical newspaper articles.

## Features

*   **Image and PDF OCR:** Extracts text from both images (PNG, JPG, JPEG) and PDF files.
*   **Multiple LLM Providers:**  Supports a variety of LLMs:
    *   **OpenAI:** gpt-4.1-mini (default), gpt-4o, gpt-5-mini, o3, o4-mini
    *   **Anthropic:** claude-sonnet-4, claude-3-5-haiku
    *   **Google:** gemini-2.5-flash (default), gemini-2.5-pro
    *   **Ollama:** llama3.2-vision, minicpm-v, and other vision models
    *   **OpenRouter:** Access to 100+ models including Qwen, Llama, Gemini
*   **Thinking/Reasoning Models:** Extended thinking support for Anthropic and Google models via `--thinking-budget`.
*   **Configurable:**  Settings can be adjusted via configuration file, environment variables, or CLI flags.
*   **Image Preprocessing:** Automatic resizing and optimization to meet API requirements.

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

`vllmocr` is a command-line tool that processes both images and PDFs:

```bash
vllmocr <file_path> [options]
```

*   `<file_path>`:  The path to the image file (PNG, JPG, JPEG) or PDF file.

**Options:**

*   `-o, --output`: Output file name (default: auto-generated based on input filename and model).
*   `-p, --provider`: The LLM provider to use (openai, anthropic, google, ollama, openrouter). Defaults to `anthropic`.
*   `-m, --model`: The specific model to use. Supports aliases like `haiku`, `sonnet`, `4o`, `gemini`. Defaults to provider's recommended model.
*   `-c, --custom-prompt`: Custom prompt to use for the LLM.
*   `--api-key`: API key for the LLM provider. Overrides API keys from the config file or environment variables.
*   `--thinking-budget`: Token budget for thinking/reasoning mode (Anthropic extended thinking, Google Gemini thinking). Minimum 1024 tokens.
*   `--max-file-size-mb`: Maximum preprocessed image size in MB (default: 1). Images larger than this after preprocessing will be downscaled iteratively.
*   `--debug`: Save intermediate processing steps for debugging.
*   `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
*   `--help`: Show the help message and exit.

**Model Aliases:**

For convenience, you can use short aliases instead of full model names:

| Alias | Provider | Model |
|-------|----------|-------|
| `haiku` | Anthropic | claude-3-5-haiku-latest |
| `sonnet` | Anthropic | claude-sonnet-4-20250514 |
| `4o` | OpenAI | gpt-4o |
| `4o-mini` | OpenAI | gpt-4o-mini |
| `4.1-mini` | OpenAI | gpt-4.1-mini |
| `gemini` | Google | gemini-2.5-flash |
| `qwen` | OpenRouter | qwen/qwen3-vl-235b |

**Examples:**

```bash
# Use Anthropic Haiku (fast and cheap)
vllmocr my_image.jpg -m haiku

# Use a local model via Ollama
vllmocr document.pdf -p ollama -m llama3.2-vision

# Use extended thinking for difficult documents
vllmocr scan.pdf -m sonnet --thinking-budget 2048

# Use OpenRouter to access Qwen (top performer on benchmarks)
vllmocr old_newspaper.jpg -p openrouter -m qwen/qwen2.5-vl-72b-instruct
```

Running `vllmocr` without arguments will display a help message with usage examples.

## A General Note on LLMs and OCR

In my experience, only the largest of LLMs are useful for text transcription. Although `vllmocr` supports Ollama, I haven't found any locally-runnable models that perform adequately on my MacBook Pro with 36 GB of memory. 

Most models demonstrate reasonable accuracy, though hallucinations occur most frequently when processing text that begins or ends mid-sentence. Models typically ignore word or sentence fragments at the top of the page while attempting to complete sentences that are cut off at the bottom. Hallucinations also increase when processing blurry or distorted text. Despite how you prompt them, current models remain overconfident in their capacity to decipher text. Additionally, models occasionally modernize archaic spellings or formatting without indication.

A more substantial challenge arises when processing pages with more than a few hundred words, such as full newspaper or magazine pages. The bigger the model the more words they are able to output, and this doesn't seem to have anything to do with context window size or output restrictions, just parameters. When overwhelmed, models frequently omit significant sections, especially with column-formatted content. To achieve best results, I usually crop the image into smaller, manageable sections and performing OCR on each section individually. This approach dramatically improves accuracy and ensures comprehensive text capture across the entire document.


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

# Image processing settings (e.g., resizing parameters) could be added here if needed in the future.

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
