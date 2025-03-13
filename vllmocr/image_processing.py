import logging
import os
import re
from typing import List

import cv2
import pymupdf as fitz  # PyMuPDF

from .utils import handle_error


def sanitize_filename(name: str) -> str:
    """Replace any non-alphanumeric characters with underscores."""
    return re.sub(r"[^\w\-\.]+", "_", name)


def determine_output_format(image_path: str, provider: str) -> str:
    """Determines the correct output format based on provider and input image type."""
    if provider is None:
        provider = "default"  # Set a default to avoid errors

    if provider == "openai":
        return "jpg"  # OpenAI prefers JPEG
    else:
        return "png"  # Default to png for safety


def preprocess_image(
    image_path: str,
    output_path: str,
    provider: str,
    rotation: int = 0,
    debug: bool = False,
) -> str:
    """Preprocess image to enhance OCR accuracy."""
    try:
        image = cv2.imread(image_path)

        if image is None:
            handle_error(f"Could not read image at {image_path}")
            return None  # This line won't be reached due to handle_error, but added for clarity

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
            denoised = cv2.rotate(
                denoised,
                {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                }[rotation],
            )

        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Save intermediate results if debug is enabled
        if debug:
            debug_dir = os.path.join(os.path.dirname(image_path), "debug_outputs")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(debug_dir, f"{os.path.basename(image_path)}_gray.png"),
                gray,
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"{os.path.basename(image_path)}_enhanced.png"),
                enhanced,
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"{os.path.basename(image_path)}_blurred.png"),
                blurred,
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"{os.path.basename(image_path)}_denoised.png"),
                denoised,
            )

        if output_path.lower().endswith(".jpg"):
            cv2.imwrite(output_path, binary, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(output_path, binary)
        return output_path
    except Exception as e:
        if debug:
            logging.error(f"Error in preprocess_image: {str(e)}")
            import traceback

            logging.error(f"Traceback: {traceback.format_exc()}")
        raise



def pdf_to_images(pdf_path: str, output_dir: str) -> List[str]:
    """Converts a PDF file into a series of images (one per page).
    If a page is a single image, it extracts the original image instead.
    """

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Error opening PDF {pdf_path}: {e}")
        raise

    image_paths = []

    if len(doc) == 0:
        raise ValueError("PDF has no pages.")

    for i, page in enumerate(doc):
        img_list = page.get_images(full=True)

        if len(img_list) == 1:  # If there's exactly one image, extract it
            xref = img_list[0][0]  # XREF number of the image
            img = doc.extract_image(xref)
            img_ext = img["ext"]  # Image format (png, jpg, etc.)
            img_data = img["image"]

            temp_image_path = os.path.join(output_dir, f"page_{i + 1}.{img_ext}")
            with open(temp_image_path, "wb") as img_file:
                img_file.write(img_data)

            logging.info(f"Extracted original image from page {i + 1}")
        else:
            temp_image_path = os.path.join(output_dir, f"page_{i + 1}.png")
            try:
                pixmap = page.get_pixmap()
                pixmap.save(temp_image_path)  # Save rendered image
            except Exception as e:
                logging.error(f"Error processing page {i + 1}: {e}")
                continue  # Skip problematic pages

        image_paths.append(temp_image_path)

    if not image_paths:
        raise ValueError("No images were generated from the PDF.")

    return image_paths
