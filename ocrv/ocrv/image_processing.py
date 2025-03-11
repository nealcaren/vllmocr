import os
import tempfile
from typing import List

import cv2
import fitz  # PyMuPDF


def sanitize_filename(name: str) -> str:
    """Replace any non-alphanumeric characters with underscores."""
    import re
    return re.sub(r'[^\w\-\.]+', '_', name)

def check_image_quality(pixmap, dpi_threshold: int = 300) -> None:
    """Check if image DPI is below the threshold and print a warning."""
    dpi_x = pixmap.width * 72 / pixmap.w
    dpi_y = pixmap.height * 72 / pixmap.h
    dpi = min(dpi_x, dpi_y)
    if dpi < dpi_threshold:
        print(f"Warning: Image DPI is {dpi:.1f}, which is below the recommended {dpi_threshold} DPI. OCR accuracy may be reduced.")

def preprocess_image(image_path: str, output_path: str, rotation: int = 0, debug: bool = False) -> str:
    """Preprocess image to enhance OCR accuracy."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

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
    if rotation in [90, 180, 270]:
        if rotation == 90:
            denoised = cv2.rotate(denoised, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            denoised = cv2.rotate(denoised, cv2.ROTATE_180)
        elif rotation == 270:
            denoised = cv2.rotate(denoised, cv2.ROTATE_90_COUNTERCLOCKWISE)

    binary = denoised

    # Save intermediate results if debug is enabled
    if debug:
        base_path = f"{image_path}_debug"
        cv2.imwrite(f"{base_path}_1_gray.png", gray)
        cv2.imwrite(f"{base_path}_2_enhanced.png", enhanced)
        cv2.imwrite(f"{base_path}_3_blurred.png", blurred)
        cv2.imwrite(f"{base_path}_4_denoised.png", denoised)

    cv2.imwrite(output_path, binary)
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
    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        check_image_quality(pix)
        temp_image_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(temp_image_path)
        image_paths.append(temp_image_path)
    return image_paths
