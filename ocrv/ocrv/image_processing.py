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
    logging.debug(f"cv2.imread returned type: {type(image)}")  # Log the return type
    if image is None:
        handle_error(f"Could not read image at {image_path}")

    logging.debug(f"Original image shape: {image.shape if image is not None else None}, type: {type(image)}")

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        logging.debug("Converting to grayscale")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logging.debug(f"Grayscale image shape: {gray.shape}, type: {type(gray)}")
    else:
        gray = image.copy()
        logging.debug(f"Image already grayscale. Shape: {gray.shape}, type: {type(gray)}")

    # Enhance contrast
    logging.debug("Enhancing contrast")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    logging.debug(f"Enhanced image shape: {enhanced.shape}, type: {type(enhanced)}")

    # Apply a lighter blur to preserve details
    logging.debug("Applying blur")
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    logging.debug(f"Blurred image shape: {blurred.shape}, type: {type(blurred)}")

    # Denoise with lower strength to preserve character details
    logging.debug("Denoising")
    denoised = cv2.fastNlMeansDenoising(blurred, h=7)
    logging.debug(f"Denoised image shape: {denoised.shape}, type: {type(denoised)}")
    # Apply manual rotation if specified
    if rotation in {90, 180, 270}:
        logging.debug(f"Rotating image by {rotation} degrees")
        denoised = cv2.rotate(denoised, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
        logging.debug(f"Rotated image shape: {denoised.shape}, type: {type(denoised)}")

    logging.debug("Applying adaptive thresholding")
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    logging.debug(f"Binary image shape: {binary.shape}, type: {type(binary)}")

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

import fitz  # PyMuPDF
import os
import logging
from typing import List


def pdf_to_images(pdf_path: str, output_dir: str) -> List[str]:
    """Converts a PDF file into a series of images (one per page)."""
    logging.info(f"Converting PDF to images: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Error opening PDF {pdf_path}: {e}")
        raise  # Re-raise the exception if we can't open the PDF

    image_paths = []

    if len(doc) == 0:
        raise ValueError("PDF has no pages.")

    for i, page in enumerate(doc):
        temp_image_path = os.path.join(output_dir, f"page_{i+1}.png")
        try:
            pixmap = page.get_pixmap()
        except Exception as e:
            logging.error(f"Error getting pixmap for page {i+1}: {e}")
            continue # skip this page
        try:
            pixmap.save(temp_image_path)  # Directly save the image
            image_paths.append(temp_image_path)
        except Exception as e:
            logging.error(f"Error saving image for page {i+1}: {e}")
            continue  # Skip problematic pages

    if not image_paths:
        raise ValueError("No images were generated from the PDF.")

    return image_paths

