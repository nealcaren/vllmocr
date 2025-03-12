import argparse
import os
import sys
import tempfile
from typing import List, Optional

from .image_processing import preprocess_image, pdf_to_images, sanitize_filename, determine_output_format
from .llm_interface import transcribe_image
from .config import load_config, AppConfig
from .utils import setup_logging, handle_error, validate_image_file


def process_single_image(image_path: str, provider: Optional[str], config: AppConfig, model: Optional[str] = None) -> str:
    """Processes a single image and returns the transcribed text."""
    with tempfile.TemporaryDirectory() as temp_dir:
        if provider is None and model:
            if model in ("haiku", "sonnet", "anththropic", "claude"):
                provider = "anthropic"
            elif model in ("4o-mini", "gpt-4o"):
                provider = "openai"
        output_format = determine_output_format(image_path, provider)
        output_path = os.path.join(temp_dir, f"preprocessed.{output_format}")
        preprocessed_path = preprocess_image(
            image_path,
            output_path,
            provider,
            config.image_processing_settings["rotation"],
            config.debug
        )
        return transcribe_image(preprocessed_path, provider, config, model)

def process_pdf(pdf_path: str, provider: Optional[str], config: AppConfig, model: Optional[str] = None) -> str:
    """Processes a PDF and returns the transcribed text."""
    with tempfile.TemporaryDirectory() as temp_dir:
        if provider is None and model:
            if model in ("haiku", "sonnet", "anththropic", "claude"):
                provider = "anthropic"
            elif model in ("4o-mini", "gpt-4o"):
                provider = "openai"

        try:
            image_paths = pdf_to_images(pdf_path, temp_dir)
        except ValueError as e:
            handle_error(f"Error processing PDF {pdf_path}: {e}")
            raise  # Re-raise the exception
        all_text = []
        for image_path in image_paths:
            text = process_single_image(image_path, provider, config, model)
            all_text.append(text)
        return "\n\n".join(all_text)

def main():
    """Main function to handle command-line arguments and processing."""
    parser = argparse.ArgumentParser(description="OCR processing for PDFs and images.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file (PDF or image).")
    parser.add_argument("-o", "--output", type=str, help="Output file name (default: auto-generated).")
    parser.add_argument("-p", "--provider", type=str,
                        help="LLM provider ('openai', 'anthropic', 'google', 'ollama').")
    parser.add_argument("-m", "--model", type=str, help="Model alias to use (e.g., 'haiku', 'gpt-4o', 'llama3').")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                        help="Manually rotate image by specified degrees (0, 90, 180, or 270)")
    parser.add_argument("--debug", action="store_true", help="Save intermediate processing steps for debugging")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config()

    # Override config with command-line arguments
    config.image_processing_settings["rotation"] = args.rotate
    config.debug = args.debug
    input_file = args.input

    # Infer provider from model if provider is not given
    if args.provider is None and args.model:
        if args.model in ("haiku", "sonnet", "anththropic", "claude"):
            args.provider = "anthropic"
        elif args.model in ("4o-mini", "gpt-4o"):
            args.provider = "openai"

    if args.provider is None:
        parser.error("the following arguments are required: -p/--provider OR -m/--model with a known provider")

    provider = args.provider

    try:
        if not os.path.exists(input_file):
            handle_error(f"Input file not found: {input_file}")

        file_extension = os.path.splitext(input_file)[1].lower()
        if file_extension == ".pdf":
            extracted_text = process_pdf(input_file, provider, config, args.model)
        elif file_extension.lower() in (".png", ".jpg", ".jpeg"):
            if not validate_image_file(input_file):
                handle_error(f"Input file is not a valid image: {input_file}")
            extracted_text = process_single_image(input_file, provider, config, args.model)
        else:
            handle_error(f"Unsupported file type: {file_extension}")

    except ValueError as ve:
        handle_error(f"ValueError occurred: {ve}")  # Handle ValueError separately
    except Exception as e:
        handle_error(f"An error occurred: {e}")  # Handle other exceptions

    output_filename = args.output
    if not output_filename:
        model_str = args.model if args.model else provider
        output_filename = f"{os.path.splitext(input_file)[0]}_{sanitize_filename(model_str)}.md"

    with open(output_filename, 'w') as f:
        f.write(extracted_text)

    print(f"OCR result saved to: {output_filename}")

if __name__ == "__main__":
    main()
