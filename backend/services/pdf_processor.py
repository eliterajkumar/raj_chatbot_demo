import fitz  # PyMuPDF
from typing import List
import os
import io

def parse_pdf(path_or_bytes):
    """
    Accept either file path or raw bytes. Returns extracted text.
    """
    try:
        # If given bytes
        if isinstance(path_or_bytes, (bytes, bytearray)):
            return extract_text_from_pdf(path_or_bytes)
        # If given path string
        if isinstance(path_or_bytes, str) and os.path.exists(path_or_bytes):
            with open(path_or_bytes, "rb") as f:
                data = f.read()
            return extract_text_from_pdf(data)
        raise ValueError("parse_pdf expects file path or bytes")
    except Exception as e:
        raise

def ocr_image(path_or_bytes):
    """
    Basic OCR wrapper: for now read file bytes and run pytesseract if available.
    Returns extracted text (string).
    """
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return ""  # if OCR libs not installed, return empty string

    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(path_or_bytes))
    elif isinstance(path_or_bytes, str) and os.path.exists(path_or_bytes):
        img = Image.open(path_or_bytes)
    else:
        raise ValueError("ocr_image expects file path or bytes")

    text = pytesseract.image_to_string(img)
    return text
