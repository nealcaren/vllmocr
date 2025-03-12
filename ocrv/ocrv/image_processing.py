import logging
import os
import re
import tempfile
from typing import List, Optional

import cv2
import logging
import os
import re
import tempfile
from typing import List, Optional

import cv2
import fitz  # PyMuPDF
import imghdr

from .utils import handle_error


def sanitize_filename(name: str) -> str:
    """Replace any non-alphanumeric characters with underscores."""
    return re.sub(r"[^\w\-\.]+", "_", name)


def determine_output_format(image_path: str, provider: str) -> str:
    """Determines the correct output format based on provider and input image type."""
    if provider == "openai":
        return "jpg"  # OpenAI prefers JPEG
    else:
        image_type = imghdr.what(image_path)
        if image_type:
            return "." + image_type
        else:
            return ".png"

def preprocess_image(image_path: str, output_path: str, provider: str, rotation: int = 0, debug: bool = False) -> str:
    """Preprocess image to enhance OCR accuracy."""
    logging.info(f"Preprocessing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        handle_error(f"Could not read image at {image_path}")

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply a lighter blur to preserve details
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Denoise with lower strength to preserve character details
    denoised = cv2.fastNlMeansDenoising(blurred, h=7)

    # Apply manual rotation if specified
    if rotation in {90, 180, 270}:
        denoised = cv2.rotate(denoised, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])

    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Save intermediate results if debug is enabled
    if debug:
        debug_dir = os.path.join(os.path.dirname(image_path), "debug_outputs")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(image_path)}_gray.png"), gray)
        cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(image_path)}_enhanced.png"), enhanced)
        cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(image_path)}_blurred.png"), blurred)
        cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(image_path)}_denoised.png"), denoised)

    if output_path.lower().endswith(".jpg"):
        cv2.imwrite(output_path, binary, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(output_path, binary)
    return output_path


def pdf_to_images(pdf_path: str, output_dir: str) -> List[str]:
    """Converts a PDF file into a series of images (one per page).

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save the images.

    Returns:
        A list of paths to the generated images.
    """
    logging.info(f"Converting PDF to images: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        image_paths = []

        for i, page in enumerate(doc):
            pix = page.get_pixmap()  # Get the Pixmap
            print(f"Type of pix: {type(pix)}")  # Debug print
            print(f"Contents of pix: {pix}")  # Debug print
            if isinstance(pix, tuple):
                pix = pix[0]
            temp_image_path = os.path.join(output_dir, f"page_{i+1}.png")
            pix.save(temp_image_path)
            image_paths.append(temp_image_path)
        return image_paths

    except Exception as e:
        logging.error(f"Error during PDF to image conversion: {pdf_path}")
        return []
