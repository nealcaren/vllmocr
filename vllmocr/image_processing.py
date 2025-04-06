import logging
import os
import re
from typing import List
from pathlib import Path

import cv2
import pymupdf as fitz  # PyMuPDF

from .utils import handle_error


def sanitize_filename(name: str) -> str:
    """Replace any non-alphanumeric characters with underscores."""
    return re.sub(r"[^\w\-\.]+", "_", name)


def determine_output_format(image_path: str, provider: str) -> str:
    """Determines the correct output format based on provider and input image type."""
    return "png"  # Always use PNG to minimize lossy compression


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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        contrast_enhanced = clahe.apply(gray)

        # Denoise with lower strength to preserve character details
        # Note: Removed GaussianBlur step as it was minimal (1,1) and might slightly soften
        denoised = cv2.fastNlMeansDenoising(contrast_enhanced, h=5, templateWindowSize=7, searchWindowSize=21)

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

        # Removed binarization step:
        # binary = cv2.adaptiveThreshold(
        #     denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        # )

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
                contrast_enhanced,
            )
            # Removed blurred.png debug output as blur step was removed
            cv2.imwrite(
                os.path.join(debug_dir, f"{os.path.basename(image_path)}_denoised.png"),
                denoised,
            )
        os.path.join("/Users/nealcaren/Dropbox/", f"{os.path.basename(image_path)}_denoised.png"), # TODO: This looks like a leftover debug path, should it be removed?
        # Always save the denoised grayscale image as PNG with maximum compression
        cv2.imwrite(output_path, denoised, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return output_path
    except Exception as e:
        if debug:
            logging.error(f"Error in preprocess_image: {str(e)}")
            import traceback

            logging.error(f"Traceback: {traceback.format_exc()}")
        raise


from concurrent.futures import ThreadPoolExecutor

DEFAULT_PDF_DPI = 600

def process_page(page, i, output_dir, dpi=DEFAULT_PDF_DPI):
    """Process a single PDF page to extract or render images with higher DPI."""
    try:
        img_list = page.get_images(full=True)
        temp_image_path = Path(output_dir) / f"page_{i + 1}.png"

        if len(img_list) == 1:  # Extract the original image directly if present
            xref = img_list[0][0]  # XREF number of the image
            img = page.parent.extract_image(xref)
            img_ext = img["ext"]  # Image format (png, jpg, etc.)
            temp_image_path = temp_image_path.with_suffix(f".{img_ext}")

            with temp_image_path.open("wb") as img_file:
                img_file.write(img["image"])

            logging.info(f"Extracted original image from page {i + 1} in {img_ext} format.")

        else:  # Render the page as an image at high DPI
            zoom = dpi / 72  # Scale factor (default is 72 DPI)
            mat = fitz.Matrix(zoom, zoom)  # Create a transformation matrix
            pixmap = page.get_pixmap(matrix=mat, alpha=False)  # Render with the matrix

            pixmap.save(temp_image_path)  # Save rendered image

            logging.info(f"Rendered page {i + 1} at {dpi} DPI.")

        return str(temp_image_path)

    except Exception as e:
        logging.error(f"Error processing page {i + 1}: {e}")
        return None

def pdf_to_images(pdf_path: str, output_dir: str, dpi=DEFAULT_PDF_DPI) -> list:
    """Converts a PDF file into a series of high-resolution images."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Error opening PDF {pdf_path}: {e}")
        raise

    if len(doc) == 0:
        raise ValueError("PDF has no pages.")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with ThreadPoolExecutor() as executor:
        image_paths = list(filter(None, executor.map(lambda p: process_page(p[1], p[0], output_dir, dpi), enumerate(doc))))

    if not image_paths:
        raise ValueError("No images were generated from the PDF.")
    print(image_paths)
    print(f"Extracted {len(image_paths)} pages at {dpi} DPI.")
    return image_paths
