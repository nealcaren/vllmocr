import logging
import sys
import imghdr
import base64


def _encode_image(image_path: str) -> str:
    """Encodes the image at the given path to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def setup_logging(log_level: str = "INFO"):
    """Configures logging for the application, only if log_level is DEBUG."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    if log_level == "DEBUG":
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            stream=sys.stderr,  # Log to stderr to keep stdout clean for output
        )


def handle_error(message: str, error: Exception = None):
    """Handles errors, logs them, and exits."""
    logging.error(f"Handling error: {message}")  # Log the message
    if error:
        logging.exception(error)  # Log the exception if provided
    sys.exit(1)


def validate_image_file(file_path: str) -> bool:
    """
    Validates if the given file path is a valid image file.
    Uses imghdr to determine the image type without fully loading the image.
    """
    return imghdr.what(file_path) is not None
