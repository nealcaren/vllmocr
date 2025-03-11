from .image_processing import process_image
from .llm_interface import generate_text
from .config import load_config
from .utils import some_utility_function  # Example utility function


def transcribe_image(image_path, llm_provider):
    """
    Transcribes text from an image using OCR and an LLM.

    Args:
        image_path (str): Path to the image file.
        llm_provider (str): The LLM provider to use (e.g., "openai", "local").

    Returns:
        str: The transcribed text.
    """
    config = load_config()
    image = process_image(image_path, config)
    text = generate_text(image, llm_provider, config)
    return text


def transcribe_pdf(pdf_path, llm_provider):
    """
    Transcribes text from a PDF file using OCR and an LLM.
    Extracts images from the PDF first, then processes each image.

    Args:
        pdf_path (str): Path to the PDF file.
        llm_provider (str): The LLM provider to use.

    Returns:
        str: The transcribed text.
    """
    config = load_config()
    #  Implementation for PDF processing (using PyMuPDF or similar)
    #  1. Extract images from PDF.
    #  2. Loop through images:
    #  3.   image = process_image(image_path, config)
    #  4.   text_part = generate_text(image, llm_provider, config)
    #  5.   Append text_part to overall result.
    #  For now, a placeholder:
    return "PDF transcription not yet implemented."


# Example usage (you'd likely call this from a separate script)
if __name__ == "__main__":
    # image_text = transcribe_image("path/to/image.jpg", "openai")
    # print(f"Transcribed text: {image_text}")

    pdf_text = transcribe_pdf("path/to/document.pdf", "local")
    print(f"Transcribed PDF text: {pdf_text}")
