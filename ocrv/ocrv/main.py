import argparse
import os
import sys
import tempfile
from typing import List, Optional

from .image_processing import preprocess_image, pdf_to_images, sanitize_filename
from .llm_interface import transcribe_image
from .config import load_config, AppConfig
from .utils import setup_logging, handle_error, validate_image_file


def process_single_image(image_path: str, provider: str, config: AppConfig, model: Optional[str] = None) -> str:
    """Processes a single image and returns the transcribed text."""
    with tempfile.TemporaryDirectory() as temp_dir:
        preprocessed_path = preprocess_image(
            image_path,
            os.path.join(temp_dir, "preprocessed.png"),
            config.image_processing_settings["rotation"],
            config.debug
        )
        return transcribe_image(preprocessed_path, provider, config, model)

def process_pdf(pdf_path: str, provider: str, config: AppConfig, model: Optional[str] = None) -> str:
    """Processes a PDF and returns the transcribed text."""
    with tempfile.TemporaryDirectory() as temp_dir:
        image_paths = pdf_to_images(pdf_path, temp_dir, config.dpi)
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
    parser.add_argument("-p", "--provider", type=str, required=True,
                        help="LLM provider ('openai', 'anthropic', 'google', 'ollama').")
    parser.add_argument("-m", "--model", type=str, help="Model to use (required for Ollama, optional for others).")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF rendering (default: 300).")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                        help="Manually rotate image by specified degrees (0, 90, 180, or 270)")
    parser.add_argument("--debug", action="store_true", help="Save intermediate processing steps for debugging")
    parser.add_argument("--log-level", type=str, default="INFO", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")  # Added log-level argument
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config()

    # Override config with command-line arguments
    config.dpi = args.dpi
    config.image_processing_settings["rotation"] = args.rotate
    config.debug = args.debug

    # Validate that model is provided if provider is ollama
    if args.provider == "ollama" and not args.model:
        print("Error: --model must be specified when using Ollama provider.", file=sys.stderr)
        handle_error("--model must be specified when using Ollama provider.")

    input_file = args.input
    if not os.path.exists(input_file):
        handle_error(f"Input file not found: {input_file}")

    file_extension = os.path.splitext(input_file)[1].lower()
    try:
        if file_extension == ".pdf":
            extracted_text = process_pdf(input_file, args.provider, config, args.model)
        elif file_extension.lower() in (".png", ".jpg", ".jpeg"):
            if not validate_image_file(input_file):
                handle_error(f"Input file is not a valid image: {input_file}")
            extracted_text = process_single_image(input_file, args.provider, config, args.model)
        else:
            handle_error(f"Unsupported file type: {file_extension}")
    except Exception as e:
        handle_error("An unexpected error occurred", e)

    output_filename = args.output
    if not output_filename:
        output_filename = f"{os.path.splitext(input_file)[0]}_{sanitize_filename(args.provider)}_{sanitize_filename(args.model if args.model else '')}.md" # Include model in filename
    with open(output_filename, "w") as f:
        f.write(extracted_text)

    print(f"OCR result saved to: {output_filename}")


if __name__ == "__main__":
    main()
