import logging
import logging
import os
import re
from typing import List
from pathlib import Path
import math # Added for resizing calculation

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

        # --- Add file size check and resizing ---
        MAX_FILE_SIZE_BYTES = 3 * 1024 * 1024  # 3 MB limit
        MIN_DIMENSION = 100 # Minimum width/height after resizing

        # Initial save attempt
        cv2.imwrite(output_path, denoised, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        current_size = os.path.getsize(output_path)
        resized = False
        image_to_resize = denoised # Keep the image in memory for potential resizing

        while current_size > MAX_FILE_SIZE_BYTES:
            resized = True
            logging.warning(
                f"Image {output_path} ({current_size / (1024*1024):.2f} MB) "
                f"exceeds limit ({MAX_FILE_SIZE_BYTES / (1024*1024):.2f} MB). Resizing."
            )

            # Calculate new dimensions (reduce by ~10% linearly, sqrt(0.9) factor)
            scale_factor = math.sqrt(MAX_FILE_SIZE_BYTES / current_size) # Estimate needed reduction
            scale_factor = min(scale_factor, 0.95) # Ensure at least 5% reduction to avoid stalling

            height, width = image_to_resize.shape[:2]
            new_width = max(MIN_DIMENSION, int(width * scale_factor))
            new_height = max(MIN_DIMENSION, int(height * scale_factor))

            if new_width == width and new_height == height:
                 logging.warning(f"Cannot resize {output_path} further to meet size limit. Stopping resize attempt.")
                 break # Avoid infinite loop if dimensions are already minimal or scale factor is ~1

            if new_width < MIN_DIMENSION or new_height < MIN_DIMENSION:
                logging.warning(f"Resizing {output_path} stopped: reached minimum dimension {MIN_DIMENSION}px.")
                break

            image_to_resize = cv2.resize(image_to_resize, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Save resized image, overwriting the previous one
            cv2.imwrite(output_path, image_to_resize, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            current_size = os.path.getsize(output_path)

            logging.info(f"Resized {output_path} to {new_width}x{new_height}, new size: {current_size / (1024*1024):.2f} MB")


        if resized:
             logging.info(f"Final size for {output_path}: {current_size / (1024*1024):.2f} MB")
        # --- End file size check ---

        return output_path
    except Exception as e:
        if debug:
            logging.error(f"Error in preprocess_image: {str(e)}")
            import traceback

            logging.error(f"Traceback: {traceback.format_exc()}")
        raise


from concurrent.futures import ThreadPoolExecutor

DEFAULT_PDF_DPI = 300 # Changed default DPI from 600 to 300

def process_page(page, i, output_dir, dpi=DEFAULT_PDF_DPI):
    """Process a single PDF page to extract or render images, capping DPI."""
    # Ensure rendering DPI doesn't exceed the default/maximum
    effective_dpi = min(dpi, DEFAULT_PDF_DPI)
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

        else:  # Render the page as an image, capped at effective_dpi and max pixel dimension
            MAX_DIMENSION_PIXELS = 6000 # Max pixels for width or height to prevent huge images from bad PDF metadata

            page_width_pt = page.rect.width
            page_height_pt = page.rect.height

            # Calculate potential pixel dimensions at the requested effective_dpi
            target_width_px = page_width_pt * (effective_dpi / 72)
            target_height_px = page_height_pt * (effective_dpi / 72)

            final_dpi = effective_dpi # Start with the capped DPI

            # Check if rendering at effective_dpi exceeds the max pixel limit
            if max(target_width_px, target_height_px) > MAX_DIMENSION_PIXELS:
                # Calculate a scaling factor to fit within the limit
                scaling_factor = MAX_DIMENSION_PIXELS / max(target_width_px, target_height_px)
                # Adjust the DPI based on the scaling factor
                adjusted_dpi = effective_dpi * scaling_factor
                # Ensure DPI doesn't go below a minimum reasonable value (e.g., 72)
                final_dpi = max(72, adjusted_dpi)
                logging.warning(
                    f"Page {i + 1}: Original dimensions ({page_width_pt:.0f}x{page_height_pt:.0f} pt) "
                    f"at {effective_dpi} DPI would exceed max pixels ({MAX_DIMENSION_PIXELS}). "
                    f"Adjusting render DPI to {final_dpi:.0f}."
                )

            # Calculate the final zoom factor based on the potentially adjusted DPI
            zoom = final_dpi / 72
            mat = fitz.Matrix(zoom, zoom)  # Create the transformation matrix
            pixmap = page.get_pixmap(matrix=mat, alpha=False)  # Render with the matrix

            pixmap.save(str(temp_image_path))  # Save rendered image

            # Log the actual DPI used for rendering
            logging.info(f"Rendered page {i + 1} at {final_dpi:.0f} effective DPI to {temp_image_path.name}.")

        return str(temp_image_path)

    except Exception as e:
        logging.error(f"Error processing page {i + 1}: {e}")
        return None

def pdf_to_images(pdf_path: str, output_dir: str, dpi=DEFAULT_PDF_DPI) -> list:
    """Converts a PDF file into a series of images, capping rendering DPI."""
    # Ensure rendering DPI doesn't exceed the default/maximum
    effective_dpi = min(dpi, DEFAULT_PDF_DPI)
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
        # Pass the capped effective_dpi to process_page
        image_paths = list(filter(None, executor.map(lambda p: process_page(p[1], p[0], output_dir, effective_dpi), enumerate(doc))))

    if not image_paths:
        raise ValueError("No images were generated from the PDF.")
    # Removed print(image_paths) - too verbose for normal operation
    logging.info(f"Processed {len(image_paths)} pages from PDF. Pages requiring rendering were processed at up to {effective_dpi} DPI.")
    return image_paths


if __name__ == "__main__":
    # --- Local Testing Block ---
    # Configure basic logging to see function output
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # --- !!! IMPORTANT: Set these paths correctly for your test !!! ---
    test_input_image = "path/to/your/test_image.jpg"  # <--- CHANGE THIS
    test_output_image = "test_output_preprocessed.png"
    # ---

    # Check if the input file exists
    if not os.path.exists(test_input_image):
        logging.error(f"Input test image not found: {test_input_image}")
        logging.error("Please update the 'test_input_image' variable in the script.")
    else:
        logging.info(f"Testing preprocess_image on: {test_input_image}")
        try:
            # Call the function with desired parameters for testing
            processed_path = preprocess_image(
                image_path=test_input_image,
                output_path=test_output_image,
                provider="openai",  # Provider might influence output format (though currently always PNG)
                rotation=0,       # Test with rotation if needed
                debug=True        # Enable debug outputs
            )

            if processed_path:
                logging.info(f"Preprocessing successful. Output saved to: {processed_path}")
                # You might want to add code here to open the image automatically
                # import subprocess
                # subprocess.run(["open", processed_path]) # macOS example
            else:
                logging.error("Preprocessing failed.")

        except Exception as e:
            logging.error(f"An error occurred during testing: {e}", exc_info=True)
    # --- End Local Testing Block ---
