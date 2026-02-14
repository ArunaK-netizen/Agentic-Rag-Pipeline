import os
import logging
import math
import tempfile
from typing import List, Tuple, Optional

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import easyocr
except Exception:
    easyocr = None

try:
    import streamlit as st
except Exception:
    st = None

logger = logging.getLogger(__name__)

# Global EasyOCR reader cache
_easyocr_reader = None


def get_easyocr_reader(lang: List[str] = ["en"], gpu: bool = False):
    global _easyocr_reader
    if _easyocr_reader is None:
        if easyocr is None:
            logger.warning("EasyOCR not available")
            return None
        try:
            _easyocr_reader = easyocr.Reader(lang, gpu=gpu)
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            _easyocr_reader = None
    return _easyocr_reader


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List:
    """Convert each page of a PDF into a high-resolution PIL image using PyMuPDF.

    Returns list of PIL.Image objects.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed")

    images = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Unable to open PDF {pdf_path}: {e}")
        return images

    for i in range(len(doc)):
        try:
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi)
            from PIL import Image

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to render page {i+1}: {e}")
            continue

    return images


def _deskew_image_cv(img_gray: 'np.ndarray') -> 'np.ndarray':
    """Estimate skew angle and deskew the grayscale image."""
    if cv2 is None or np is None:
        return img_gray

    coords = cv2.findNonZero(cv2.bitwise_not(img_gray))
    if coords is None:
        return img_gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_image(pil_img, deskew: bool = True) -> 'np.ndarray':
    """Preprocess a PIL image for OCR using OpenCV: grayscale, contrast, denoise, adaptive thresh, optional deskew."""
    if cv2 is None or np is None:
        # return raw PIL image as numpy array
        return pil_img

    img = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Increase contrast via CLAHE
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    # Remove noise with median blur
    gray = cv2.medianBlur(gray, 3)

    # Adaptive thresholding
    try:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 15)
    except Exception:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if deskew:
        try:
            th = _deskew_image_cv(th)
        except Exception:
            pass

    return th


def extract_text_from_images_vlm(images: List, vlm_provider: Optional[str] = None, api_key: Optional[str] = None) -> List[str]:
    """Send images to a Vision-Language model provider (optional).

    If VLM is not configured or available, falls back to EasyOCR.
    Returns list of page texts in order.
    """
    texts = []
    # First attempt: Gemini Vision via the helper we added.
    try:
        from .gemini_vlm import parse_images_with_gemini
        logger.info("Attempting Gemini Vision parsing for %d pages", len(images))

        gemini_texts = []
        for idx, img in enumerate(images):
            # save each PIL image to a temp file and call Gemini on the file path
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            try:
                img.save(tmpf.name, format="JPEG")
                tmpf.close()
                page_text = parse_images_with_gemini(tmpf.name)
            finally:
                try:
                    os.unlink(tmpf.name)
                except Exception:
                    pass

            gemini_texts.append(page_text or "")

        if any((s or "").strip() for s in gemini_texts):
            logger.info("Parsed document using Gemini Vision — %d pages returned", len(gemini_texts))
            return gemini_texts
        else:
            logger.info("Gemini returned empty/blank output; falling back to EasyOCR")
    except Exception as e:
        logger.exception("Gemini integration attempted and failed: %s", e)

    # Fallback: EasyOCR per-page
    reader = get_easyocr_reader()
    for idx, img in enumerate(images):
        page_text = ""
        try:
            if reader is not None:
                # ensure image is preprocessed numpy array
                if hasattr(img, 'read'):
                    pil_img = img
                else:
                    pil_img = img
                # run OCR, paragraph mode
                try:
                    result = reader.readtext(np.array(pil_img), detail=1, paragraph=True)
                except Exception:
                    result = reader.readtext(np.array(pil_img))

                # result entries are (bbox, text, confidence)
                lines = []
                for entry in result:
                    if not entry or len(entry) < 3:
                        continue
                    text_piece = entry[1].strip()
                    conf = float(entry[2]) if entry[2] is not None else 0.0
                    # mark unclear if confidence low
                    if conf < 0.45:
                        lines.append('[unclear]')
                    else:
                        lines.append(text_piece)

                page_text = '\n'.join([l for l in lines if l])
            else:
                logger.warning('No OCR engine available; returning empty text for page {idx+1}')
                page_text = ''

        except Exception as e:
            logger.error(f"Error extracting page {idx+1}: {e}")
            page_text = ''

        texts.append(page_text)

    return texts


def combine_pages_text(pages: List[str]) -> str:
    parts = []
    for i, p in enumerate(pages, start=1):
        parts.append(f"Page {i}:\n{p.strip()}\n")
    return '\n'.join(parts)


def chunk_text(text: str, min_tokens: int = 500, max_tokens: int = 800, overlap: int = 50) -> List[str]:
    """Chunk text into segments targetting token counts (approximate, using words as proxy).

    Returns list of text chunks.
    """
    words = text.split()
    # approximate tokens ~ words (simple proxy)
    target = (min_tokens + max_tokens) // 2
    if target <= 0:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + target, len(words))
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        if end == len(words):
            break
        start = end - overlap

    return chunks


def process_scanned_pdf(pdf_path: str, dpi: int = 350, deskew: bool = True) -> Tuple[List[str], str, List[str]]:
    """Full pipeline: PDF -> images -> preprocess -> OCR (VLM/EasyOCR) -> combine -> chunks.

    Returns (pages_text, combined_text, chunks)
    """
    pages_text = []
    try:
        images = pdf_to_images(pdf_path, dpi=dpi)
        if not images:
            logger.error('No images produced from PDF')
            return [], '', []

        preprocessed = []
        for img in images:
            try:
                proc = preprocess_image(img, deskew=deskew)
                # if OpenCV returned numpy array, convert to PIL for OCR reader compatibility
                if np is not None and isinstance(proc, np.ndarray):
                    from PIL import Image
                    pil = Image.fromarray(proc)
                else:
                    pil = proc
                preprocessed.append(pil)
            except Exception as e:
                logger.warning(f'Preprocessing failed for a page: {e}')
                preprocessed.append(img)

        pages_text = extract_text_from_images_vlm(preprocessed)
        combined = combine_pages_text(pages_text)
        chunks = chunk_text(combined)
        return pages_text, combined, chunks

    except Exception as e:
        logger.error(f'Processing failed: {e}')
        return [], '', []
