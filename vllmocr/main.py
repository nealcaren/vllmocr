import argparse
import os
import sys
import tempfile
from typing import Optional
import logging


import re  # Import re for filename sanitization
from .image_processing import (
    preprocess_image,  # Use preprocess_image instead
    pdf_to_images,
)
from .llm_interface import transcribe_image
from .config import load_config, AppConfig, MODEL_MAPPING
from .utils import setup_logging, handle_error, validate_image_file


def process_single_image(
    image_path: str,
    provider: Optional[str],
    config: AppConfig,
    model: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    max_file_size_bytes: Optional[int] = None,
    thinking_budget: Optional[int] = None,
) -> str:
    """Processes a single image and returns the transcribed text.

    Args:
        image_path: Path to the image file.
        provider: The LLM provider to use.
        config: Application configuration.
        model: Model to use (optional).
        custom_prompt: Custom prompt (optional).
        api_key: API key override (optional).
        max_file_size_bytes: Maximum file size for preprocessing.
        thinking_budget: Token budget for thinking/reasoning mode (Anthropic, Google).

    Returns:
        Transcribed text from the image.
    """

    # Use the temporary directory provided by the caller (process_pdf) or create one if called directly
    # For simplicity, let's assume it's always called within a temp dir context for now.
    # If called standalone, a temp dir would need creation here.
    temp_dir = os.path.dirname(
        image_path
    )  # Assuming image_path is already in a temp dir from pdf_to_images
    # Define an output path for the preprocessed image within the temp directory
    base_name = os.path.basename(image_path)
    preprocessed_output_path = os.path.join(
        temp_dir, f"preprocessed_{os.path.splitext(base_name)[0]}.png"
    )

    try:
        # Call preprocess_image which handles grayscale, contrast, denoising, rotation (if needed), and resizing.
        # Rotation is currently hardcoded to 0 as the --rotate argument was removed.
        processed_image_path = preprocess_image(
            image_path=image_path,
            output_path=preprocessed_output_path,
            provider=provider,  # Pass provider for potential format decisions (currently always PNG)
            rotation=0,  # Rotation argument removed, default to 0
            debug=config.debug,
            max_file_size_bytes=(max_file_size_bytes or 1 * 1024 * 1024),
        )

        if processed_image_path is None:
            # Handle case where image processing failed
            logging.error(
                f"Image preprocessing failed for {image_path}. Skipping transcription for this image."
            )
            # Depending on desired behavior, could raise an exception or return an error string
            return f"[ERROR: Image preprocessing failed for {os.path.basename(image_path)}]"

        logging.info(
            f"Transcribing preprocessed image from {processed_image_path} using the {model} model from {provider}."
        )
        result = transcribe_image(
            processed_image_path,
            provider,
            config,
            model,
            custom_prompt,
            api_key,
            thinking_budget=thinking_budget,
        )
        return result
    except Exception as e:
        # Keep existing debug logging for transcription errors
        if config.debug:
            logging.error(f"TRACE: Error in process_single_image: {str(e)}")
            import traceback

            logging.error(f"TRACE: Traceback: {traceback.format_exc()}")
        # Re-raise the exception after logging
        raise


def process_pdf(
    pdf_path: str,
    provider: Optional[str],
    config: AppConfig,
    model: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    max_file_size_bytes: Optional[int] = None,
    thinking_budget: Optional[int] = None,
) -> str:
    """Processes a PDF and returns the transcribed text.

    Args:
        pdf_path: Path to the PDF file.
        provider: The LLM provider to use.
        config: Application configuration.
        model: Model to use (optional).
        custom_prompt: Custom prompt (optional).
        api_key: API key override (optional).
        max_file_size_bytes: Maximum file size for preprocessing.
        thinking_budget: Token budget for thinking/reasoning mode (Anthropic, Google).

    Returns:
        Transcribed text from all pages joined by double newlines.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            image_paths = pdf_to_images(pdf_path, temp_dir)
        except ValueError as e:
            handle_error(f"Error processing PDF {pdf_path}: {e}")
            raise
        all_text = []
        num_pages = len(image_paths)
        logging.info(
            f"Transcribing {num_pages} pages from {pdf_path} using the {model} model from {provider}."
        )
        for i, image_path in enumerate(image_paths):
            text = process_single_image(
                image_path,
                provider,
                config,
                model,
                custom_prompt,
                api_key,
                max_file_size_bytes=max_file_size_bytes,
                thinking_budget=thinking_budget,
            )
            all_text.append(text)
            logging.info(f"Finished processing page {i + 1} of {num_pages}.")
            print(f"Page {i + 1} processed and returned.")
        return "\n\n".join(all_text)


def main():
    """Main function to handle command-line arguments and processing."""
    parser = argparse.ArgumentParser(description="OCR processing for PDFs and images.")
    parser.add_argument("input", type=str, nargs="?", help="Input file (PDF or image).")
    parser.add_argument(
        "-o", "--output", type=str, help="Output file name (default: auto-generated)."
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        help="LLM provider ('openai', 'anthropic', 'google', 'ollama', 'openrouter').",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model alias to use (e.g., 'haiku', 'gpt-4o', 'llama3').",
    )
    parser.add_argument(
        "-c", "--custom-prompt", type=str, help="Custom prompt to use for the LLM."
    )
    parser.add_argument("--api-key", type=str, help="API key for the LLM provider.")
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=1.0,
        help="Maximum preprocessed image size in MB (default: 1).",
    )
    # Removed --rotate argument
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate processing steps for debugging",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Token budget for thinking/reasoning mode (Anthropic extended thinking, Google Gemini thinking). Minimum 1024 tokens.",
    )
    parser.set_defaults(provider="anthropic", model="claude-sonnet-4-20250514")
    args = parser.parse_args()

    if args.input is None:
        print("Welcome to vllmocr!")
        print(
            "This tool allows you to perform OCR on PDFs and images using a variety of vision LLMs."
        )
        print("\nUsage: vllmocr IMAGE_OR_PDF_FILE [OPTIONS]")
        print("\nFor example: vllmocr scan.pdf -m gpt-4o")
        print("\nThe following options are available:")
        print(" -o, --output        Output file name (default: auto-generated).")
        print(
            " -p, --provider      LLM provider ('openai', 'anthropic', 'google', 'ollama', 'openrouter')."
        )
        print(
            " -m, --model         Model to use (e.g., 'haiku', 'gpt-4o', 'llama3.2-vision', 'google/gemma-3-27b-it')."
        )
        print(" -c, --custom-prompt Custom prompt to use for the LLM.")
        print(" --api-key           API key for the LLM provider.")
        print(" --thinking-budget   Token budget for thinking/reasoning mode (Anthropic, Google).")
        print(" --debug             Save intermediate processing steps for debugging.")
        print(
            " --log-level         Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
        )
        print("\nExample commands:")
        print(" vllmocr scan.jpg -m haiku")
        print(" vllmocr document.pdf -p ollama -m llama3.2-vision")
        print(" vllmocr scan.pdf -m sonnet --thinking-budget 2048  # Use extended thinking")
        sys.exit(0)

    log_level = args.log_level.upper()
    if args.debug:
        log_level = "DEBUG"
    setup_logging(log_level)

    config = load_config()
    # Removed rotation setting from config
    config.debug = args.debug
    input_file = args.input
    api_key = args.api_key
    max_file_size_bytes = int(max(0.1, args.max_file_size_mb) * 1024 * 1024)

    # Check if either provider or model is given
    if not args.provider and args.model:
        if args.model in MODEL_MAPPING:
            provider, model = MODEL_MAPPING[args.model]
        else:
            handle_error(
                f"Model '{args.model}' requires a provider. Or is not a supported model."
            )
    elif args.provider and not args.model:
        # If only provider, we'll use its default model later
        provider = args.provider
        model = None  # Explicitly set to None
    elif args.provider and args.model:
        provider = args.provider
        model = args.model
    else:
        # Neither is provided, use defaults
        provider = "anthropic"
        model = "claude-3-5-haiku-latest"

    # Provider inference is handled by MODEL_MAPPING or explicit args

    try:
        if not os.path.exists(input_file):
            handle_error(f"Input file not found: {input_file}")

        file_extension = os.path.splitext(input_file)[1].lower()
        if file_extension == ".pdf":
            extracted_text = process_pdf(
                input_file,
                provider,
                config,
                args.model,
                args.custom_prompt,
                api_key,
                max_file_size_bytes=max_file_size_bytes,
                thinking_budget=args.thinking_budget,
            )
        elif file_extension.lower() in (".png", ".jpg", ".jpeg"):
            if not validate_image_file(input_file):
                handle_error(f"Input file is not a valid image: {input_file}")
            extracted_text = process_single_image(
                input_file,
                provider,
                config,
                args.model,
                args.custom_prompt,
                api_key,
                max_file_size_bytes=max_file_size_bytes,
                thinking_budget=args.thinking_budget,
            )
        else:
            handle_error(f"Unsupported file type: {file_extension}")
    except Exception as e:
        handle_error(f"An error occurred: {e}")

    output_filename = args.output
    if not output_filename:
        model_str = args.model if args.model else provider
        # Simple sanitization: replace non-alphanumeric with underscore
        sanitized_model_str = re.sub(r"[^\w\-.]+", "_", model_str)
        output_filename = f"{os.path.splitext(input_file)[0]}_{sanitized_model_str}.md"

    with open(output_filename, "w") as f:
        f.write(extracted_text)

    print(f"OCR result saved to: {output_filename}")


if __name__ == "__main__":
    main()
