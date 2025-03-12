import logging
import os
import re
import tempfile
from typing import List, Optional

import cv2
import pymupdf as fitz  # PyMuPDF
import imghdr

from .utils import handle_error


def sanitize_filename(name: str) -> str:
    """Replace any non-alphanumeric characters with underscores."""
    return re.sub(r"[^\w\-\.]+", "_", name)


def determine_output_format(image_path: str, provider: str) -> str:
    """Determines the correct output format based on provider and input image type."""
    logging.info(f"TRACE: Entering determine_output_format with image_path={image_path}, provider={provider}")
    
    if provider is None:
        logging.error("TRACE: Provider is None in determine_output_format")
        provider = "default"  # Set a default to avoid errors
    
    if provider == "openai":
        logging.info("TRACE: Using jpg format for OpenAI")
        return "jpg"  # OpenAI prefers JPEG
    else:
        logging.info(f"TRACE: Using png format for provider: {provider}")
        return "png"  # Default to png for safety

def preprocess_image(image_path: str, output_path: str, provider: str, rotation: int = 0, debug: bool = False) -> str:
    """Preprocess image to enhance OCR accuracy."""
    logging.info(f"TRACE: Entering preprocess_image with image_path={image_path}, output_path={output_path}, provider={provider}, rotation={rotation}")
    
    try:
        image = cv2.imread(image_path)
        logging.info(f"TRACE: cv2.imread returned type: {type(image)}")
        
        if image is None:
            logging.error(f"TRACE: cv2.imread returned None for {image_path}")
            logging.info(f"TRACE: File exists: {os.path.exists(image_path)}")
            logging.info(f"TRACE: File size: {os.path.getsize(image_path) if os.path.exists(image_path) else 'N/A'}")
            handle_error(f"Could not read image at {image_path}")
            return None  # This line won't be reached due to handle_error, but added for clarity
        
        logging.info(f"TRACE: Original image shape: {image.shape}, type: {type(image)}")
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            if debug: logging.debug("Converting to grayscale")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if debug: logging.debug(f"Grayscale image shape: {gray.shape}, type: {type(gray)}")
        else:
            gray = image.copy()
            if debug: logging.debug(f"Image already grayscale. Shape: {gray.shape}, type: {type(gray)}")

        # Enhance contrast
        if debug: logging.debug("Enhancing contrast")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        if debug: logging.debug(f"Enhanced image shape: {enhanced.shape}, type: {type(enhanced)}")

        # Apply a lighter blur to preserve details
        if debug: logging.debug("Applying blur")
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        if debug: logging.debug(f"Blurred image shape: {blurred.shape}, type: {type(blurred)}")

        # Denoise with lower strength to preserve character details
        if debug: logging.debug("Denoising")
        denoised = cv2.fastNlMeansDenoising(blurred, h=7)
        if debug: logging.debug(f"Denoised image shape: {denoised.shape}, type: {type(denoised)}")
        # Apply manual rotation if specified
        if rotation in {90, 180, 270}:
            if debug: logging.debug(f"Rotating image by {rotation} degrees")
            denoised = cv2.rotate(denoised, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
            if debug: logging.debug(f"Rotated image shape: {denoised.shape}, type: {type(denoised)}")

        if debug: logging.debug("Applying adaptive thresholding")
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        if debug: logging.debug(f"Binary image shape: {binary.shape}, type: {type(binary)}")

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
    except Exception as e:
        logging.error(f"TRACE: Error in preprocess_image: {str(e)}")
        import traceback
        logging.error(f"TRACE: Traceback: {traceback.format_exc()}")
        raise
import logging
import os
import fitz  # PyMuPDF
from typing import List


def pdf_to_images(pdf_path: str, output_dir: str) -> List[str]:
    """Converts a PDF file into a series of images (one per page).
       If a page is a single image, it extracts the original image instead.
    """
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
        img_list = page.get_images(full=True)
        
        if len(img_list) == 1:  # If there's exactly one image, extract it
            xref = img_list[0][0]  # XREF number of the image
            img = doc.extract_image(xref)
            img_ext = img["ext"]  # Image format (png, jpg, etc.)
            img_data = img["image"]

            temp_image_path = os.path.join(output_dir, f"page_{i+1}.{img_ext}")
            with open(temp_image_path, "wb") as img_file:
                img_file.write(img_data)

            logging.info(f"Extracted original image from page {i+1}")
        else:
            temp_image_path = os.path.join(output_dir, f"page_{i+1}.png")
            try:
                pixmap = page.get_pixmap()
                pixmap.save(temp_image_path)  # Save rendered image
            except Exception as e:
                logging.error(f"Error processing page {i+1}: {e}")
                continue  # Skip problematic pages

        image_paths.append(temp_image_path)

    if not image_paths:
        raise ValueError("No images were generated from the PDF.")

    return image_paths
