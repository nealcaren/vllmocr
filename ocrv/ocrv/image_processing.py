import logging
import os
import re
import tempfile
from typing import List, Optional

import cv2
import fitz  # PyMuPDF
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


def check_image_quality(pixmap, dpi_threshold: int = 300) -> None:
    """Check if image DPI is below the threshold and print a warning."""
    if isinstance(pixmap, tuple):
        pixmap = pixmap[0]

    if isinstance(pixmap, fitz.Pixmap):
        dpi_x = pixmap.irect.width * 72 / pixmap.width
        dpi_y = pixmap.irect.height * 72 / pixmap.height
    else:  # Assume it's a NumPy array (from cv2)
        dpi_x = pixmap.shape[1] * 72 / pixmap.shape[1]
        dpi_y = pixmap.shape[0] * 72 / pixmap.shape[0]
    dpi = min(dpi_x, dpi_y)

    if dpi < dpi_threshold:
        print(f"Warning: Image DPI is {dpi:.1f}, which is below the recommended {dpi_threshold} DPI. OCR accuracy may be reduced.")


def determine_output_format(image_path: str, provider: str) -> str:
    """Determines the correct output format based on provider and input image type."""
    if provider == "openai":
        return "jpg"  # OpenAI prefers JPEG
    else:
        image_type = imghdr.what(image_path)
        if image_type:
            # Ensure the type is valid and return with a leading .
            return image_type
        else:
            return "png" #default to png if we can't tell

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
        cv2.imwrite(output_path, binary)  # writes the image
    return output_path

def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> List[str]:
    """Converts a PDF file into a series of images (one per page).

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save the images.
        dpi: DPI for the output images.

    Returns:
        A list of paths to the generated images.
    """
    logging.info(f"Converting PDF to images: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        image_paths = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)  # Get the Pixmap
            print(f"Type of pix: {type(pix)}")  # Debug print
            print(f"Contents of pix: {pix}")  # Debug print
            check_image_quality(pix)
            if isinstance(pix, tuple):
                pix = pix[0]
            temp_image_path = os.path.join(output_dir, f"page_{i+1}.png")
            pix.save(temp_image_path)
            image_paths.append(temp_image_path)
        return image_paths
    except Exception as e:
        handle_error(f"Error during PDF to image conversion: {pdf_path}", e)
