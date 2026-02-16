import os
import logging
import PyPDF2
import fitz  # PyMuPDF
import re
from .config import ENABLE_LOCAL_OCR

# Lazy import EasyOCR only when local OCR is enabled to avoid unnecessary model downloads
_easyocr_import_error = None
easyocr = None
if ENABLE_LOCAL_OCR:
    try:
        import easyocr  # type: ignore
        _easyocr_import_error = None
    except Exception as _e:
        easyocr = None  # type: ignore
        _easyocr_import_error = _e
from PIL import Image
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Initialize EasyOCR reader (cached globally for efficiency)
_ocr_reader = None

def _extract_retry_time_from_logs() -> str:
    """Extract retry time from recent log entries if quota was exceeded."""
    try:
        import streamlit as st
        # Try to extract from any error message in session state
        if hasattr(st, 'session_state') and 'gemini_quota_msg' in st.session_state:
            return st.session_state.gemini_quota_msg
    except:
        pass
    return None

def _show_quota_message():
    """Display quota exceeded message to user if Gemini hit limits."""
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and 'gemini_quota_msg' in st.session_state:
            msg = st.session_state.gemini_quota_msg
            st.warning(f"⏳ {msg}")
            del st.session_state.gemini_quota_msg
    except:
        pass

def get_ocr_reader():
    """Get or create the EasyOCR reader instance."""
    global _ocr_reader
    if not ENABLE_LOCAL_OCR:
        logger.info("[INFO] Local EasyOCR is disabled via configuration; skipping EasyOCR initialization.")
        return None

    if _ocr_reader is None:
        if easyocr is None:
            logger.error(f"[ERROR] EasyOCR import failed: {_easyocr_import_error}")
            return None
        logger.info("[INFO] Initializing EasyOCR reader...")
        try:
            _ocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize EasyOCR reader: {e}")
            _ocr_reader = None
    return _ocr_reader

def _preprocess_image_for_ocr(path_or_image):
    """Open and preprocess an image for better OCR results.

    Accepts a filesystem path or a PIL Image and returns a PIL Image.
    Operations: convert to grayscale, upscale small images, increase contrast/sharpness.
    """
    from PIL import ImageEnhance
    if isinstance(path_or_image, str):
        img = Image.open(path_or_image)
    else:
        img = path_or_image

    try:
        img = img.convert("L")
    except Exception:
        img = img.convert("RGB").convert("L")

    # Upscale small images to improve OCR
    max_width = 2000
    if img.width < max_width:
        scale = max_width / max(1, img.width)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Enhance contrast and sharpness
    try:
        img = ImageEnhance.Contrast(img).enhance(1.6)
        img = ImageEnhance.Sharpness(img).enhance(1.3)
    except Exception:
        pass

    return img
class PDFProcessor:
    """Handles document text extraction and preprocessing from multiple formats with automatic OCR fallback."""

    def __init__(self, enable_ocr: bool = True):
        # Support multiple document and image formats
        self.supported_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.docx', '.doc', '.txt']
        self.image_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
        self.ocr_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
        self.enable_ocr = enable_ocr

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF by converting each page to image and using Gemini VLM (same flow as extract_text_from_image)."""
        text = ""
        doc = None
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"[INFO] Processing PDF {os.path.basename(pdf_path)} with Gemini VLM on page images")
            
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    # Convert page to image (300 DPI for better quality)
                    pix = page.get_pixmap(dpi=300)
                    pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Preprocess image to improve detection (same as for regular images)
                    pil_img = _preprocess_image_for_ocr(pil_img)
                    
                    # Try Gemini Vision first (with preprocessed PIL image)
                    try:
                        from .gemini_vlm import parse_images_with_gemini
                        logger.info(f"[INFO] Attempting Gemini Vision on page {page_num + 1} of {os.path.basename(pdf_path)}")
                        
                        # Pass preprocessed PIL image directly to Gemini
                        gemini_text = parse_images_with_gemini(pil_img, prompt="Extract all text from this image, preserving layout and structure.")
                        
                        if gemini_text and gemini_text.strip():
                            char_count = len(gemini_text.strip())
                            logger.info(f"[SUCCESS] Gemini Vision extracted {char_count} characters from page {page_num + 1}")
                            text += f"\n--- Page {page_num + 1} (VLM) ---\n{gemini_text}\n"
                            continue
                        else:
                            logger.info(f"[INFO] Gemini did not return usable text for page {page_num + 1}; trying PyPDF2 fallback")
                    except Exception as e:
                        logger.debug(f"[DEBUG] Gemini attempt failed for page {page_num + 1}: {e}")
                    
                    # Fallback to PyPDF2 extraction
                    try:
                        with open(pdf_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                            page_data = pdf_reader.pages[page_num]
                            pypdf_text = page_data.extract_text()
                            if pypdf_text and pypdf_text.strip():
                                char_count = len(pypdf_text.strip())
                                logger.info(f"[SUCCESS] PyPDF2 extracted {char_count} characters from page {page_num + 1}")
                                text += f"\n--- Page {page_num + 1} (PyPDF2) ---\n{pypdf_text}\n"
                                continue
                    except Exception as e:
                        logger.debug(f"[DEBUG] PyPDF2 fallback failed for page {page_num + 1}: {e}")
                    
                    # Fallback to EasyOCR on the page image
                    if ENABLE_LOCAL_OCR:
                        logger.info(f"[INFO] Using EasyOCR on page {page_num + 1} of {os.path.basename(pdf_path)}")
                        reader = get_ocr_reader()
                        if reader is not None:
                            import numpy as _np
                            img_np = _np.array(pil_img)
                            result = reader.readtext(img_np, detail=1, paragraph=True)
                            ocr_text = "\n".join([line[1] for line in result if line and len(line) > 1 and line[1].strip()])
                            if ocr_text.strip():
                                char_count = len(ocr_text.strip())
                                logger.info(f"[SUCCESS] EasyOCR extracted {char_count} characters from page {page_num + 1}")
                                text += f"\n--- Page {page_num + 1} (OCR) ---\n{ocr_text}\n"
                                continue
                    
                    # If nothing worked for this page, log and continue
                    logger.warning(f"[WARNING] Could not extract text from page {page_num + 1} of {pdf_path}")
                    
                except Exception as e:
                    logger.warning(f"[WARNING] Error processing page {page_num + 1} of {pdf_path}: {e}")
                    continue
            
            if text.strip():
                logger.debug(f"[SUCCESS] {os.path.basename(pdf_path)}: extracted {len(text.strip())} characters")
            else:
                logger.warning(f"[WARNING] {os.path.basename(pdf_path)}: no text could be extracted")
            
            return text
            
        except Exception as e:
            logger.error(f"[ERROR] Error reading PDF file {pdf_path}: {e}")
            raise
        finally:
            # Ensure PDF document is properly closed to release file locks
            if doc is not None:
                try:
                    doc.close()
                except Exception as e:
                    logger.debug(f"[DEBUG] Error closing PDF document: {e}")

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image files using EasyOCR with preprocessing.

        Returns a text block (may be empty) but always marks an OCR attempt in logs.
        """
        text = ""
        try:
            logger.info(f"[INFO] Extracting text from image {os.path.basename(image_path)}")

            # Preprocess image to improve detection on low-quality scans/handwriting
            pil_img = _preprocess_image_for_ocr(image_path)

            # First try Gemini Vision (if configured) as the preferred path for vision parsing
            try:
                from .gemini_vlm import parse_images_with_gemini
                logger.info(f"[INFO] Attempting Gemini Vision parse for {os.path.basename(image_path)}")
                gemini_text = parse_images_with_gemini(pil_img, prompt="Extract all text from this image, preserving layout and structure.")
                if gemini_text and gemini_text.strip():
                    extracted_text = gemini_text
                    char_count = len(extracted_text.strip())
                    logger.info(f"[SUCCESS] Gemini Vision parsed {char_count} characters from {os.path.basename(image_path)}")
                    text = f"\n--- Image (VLM) ---\n{extracted_text}\n"
                    return text
                else:
                    logger.info(f"[INFO] Gemini did not return usable text for {os.path.basename(image_path)}; falling back to EasyOCR")
            except Exception as e:
                logger.debug(f"[DEBUG] Gemini attempt failed for {os.path.basename(image_path)}: {e}")

            # Fallback to EasyOCR
            logger.info(f"[INFO] Extracting text from image {os.path.basename(image_path)} using EasyOCR")
            reader = get_ocr_reader()

            # EasyOCR accepts numpy arrays; convert and run in paragraph mode
            import numpy as _np
            img_np = _np.array(pil_img)
            result = reader.readtext(img_np, detail=1, paragraph=True)

            # Combine all detected text
            extracted_text = "\n".join([line[1] for line in result if line and len(line) > 1 and line[1].strip()])
            char_count = len(extracted_text.strip())
            logger.debug(f"[DEBUG] OCR extracted {char_count} characters from {os.path.basename(image_path)}")
            text = f"\n--- Image (OCR) ---\n{extracted_text}\n"
            logger.info(f"[SUCCESS] OCR completed for {os.path.basename(image_path)}: {char_count} characters extracted")

            # Fallback: if EasyOCR found nothing and pytesseract is available, try it
            if char_count == 0:
                try:
                    import pytesseract as _pyt
                    logger.info(f"[INFO] EasyOCR returned nothing; trying pytesseract fallback for {os.path.basename(image_path)}")
                    pyt_text = _pyt.image_to_string(pil_img)
                    if pyt_text and pyt_text.strip():
                        text = f"\n--- Image (OCR - pytesseract) ---\n{pyt_text}\n"
                        logger.info(f"[SUCCESS] pytesseract extracted {len(pyt_text.strip())} characters from {os.path.basename(image_path)}")
                except Exception:
                    # Ignore fallback errors
                    pass

        except Exception as e:
            logger.error(f"[ERROR] OCR failed on image {image_path}: {e}")
            text = f"\n--- Image (OCR - Failed to Extract) ---\n[OCR processing attempted but no text could be extracted]\n"
        return text

    def extract_text_from_file(self, file_path: str) -> str:
        """Generic file extraction with automatic fallback to OCR for any unprocessable format."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Try format-specific extraction
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext in self.image_formats:
            return self.extract_text_from_image(file_path)
        elif ext in ['.txt']:
            return self._extract_text_from_txt(file_path)
        else:
            # For unsupported formats, attempt OCR if enabled
            if self.enable_ocr and ext in self.ocr_formats:
                logger.info(f"[INFO] Attempting OCR for unsupported format {ext}")
                return self.ocr_entire_pdf(file_path)
            else:
                logger.warning(f"[WARNING] Unsupported format {ext} and OCR not available")
                return ""

    def _extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from plain text files."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.debug(f"[SUCCESS] {os.path.basename(txt_path)}: extracted {len(text.strip())} characters")
            return text
        except Exception as e:
            logger.error(f"[ERROR] Error reading text file {txt_path}: {e}")
            if self.enable_ocr:
                logger.info(f"[INFO] Text extraction failed; attempting OCR as fallback for {os.path.basename(txt_path)}")
                return self.ocr_entire_pdf(txt_path)
            raise

    def ocr_entire_pdf(self, pdf_path: str) -> str:
        """Run OCR over the entire PDF using PyMuPDF + EasyOCR."""
        text = ""
        if not ENABLE_LOCAL_OCR:
            logger.info("[INFO] Local EasyOCR is disabled via configuration; skipping OCR.")
            return text
        
        try:
            doc = fitz.open(pdf_path)
            reader = get_ocr_reader()
            if reader is None:
                logger.error("[ERROR] EasyOCR reader not available")
                return text
            
            import numpy as _np
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=300)
                    pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Preprocess image for better OCR results (same as extract_text_from_image)
                    pil_img = _preprocess_image_for_ocr(pil_img)
                    
                    # Use EasyOCR for text extraction
                    img_np = _np.array(pil_img)
                    result = reader.readtext(img_np, detail=1, paragraph=True)
                    page_text = "\n".join([line[1] for line in result if line and len(line) > 1 and line[1].strip()])
                    
                    char_count = len(page_text.strip())
                    logger.debug(f"[DEBUG] OCR Page {page_num + 1} of {os.path.basename(pdf_path)}: {char_count} chars")
                    text += f"\n--- Page {page_num + 1} (OCR) ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"[WARNING] OCR failed on page {page_num + 1} of {pdf_path}: {e}")
                    continue
        except Exception as e:
            logger.error(f"[ERROR] OCR failed on {pdf_path}: {e}")
        return text

    def process_uploaded_files(self, uploaded_files) -> List[Dict[str, Any]]:
        """Process multiple uploaded files (PDFs, images, documents) with automatic OCR fallback for unprocessable files."""
        documents = []

        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                logger.info(f"[INFO] Processing {uploaded_file.name}")

                # Get file extension
                _, ext = os.path.splitext(uploaded_file.name)
                file_type = ext.lower()[1:] if ext else "unknown"  # Remove the dot

                # Log dispatch decision before extraction
                if ext.lower() == '.pdf':
                    logger.info(f"[INFO] Dispatching {uploaded_file.name} to PDF extractor")
                elif ext.lower() in self.image_formats:
                    logger.info(f"[INFO] Dispatching {uploaded_file.name} to Image extractor")
                elif ext.lower() in ['.txt']:
                    logger.info(f"[INFO] Dispatching {uploaded_file.name} to Text extractor")
                else:
                    logger.info(f"[INFO] Dispatching {uploaded_file.name} to generic extractor (unknown/other)")

                # Extract text (with automatic OCR fallback for unprocessable files)
                text = self.extract_text_from_file(temp_path)
                
                # Show quota message if Gemini hit limits
                _show_quota_message()

                # Detect if OCR was used - check for OCR marker in text
                is_ocr_processed = "(OCR)" in text

                # For OCR-processed files, include them even if text is minimal
                # For regular extraction, skip if no text found
                if not text.strip():
                    if is_ocr_processed:
                        logger.info(f"[INFO] Including OCR-processed file despite minimal extraction: {uploaded_file.name}")
                    else:
                        logger.warning(f"[WARNING] Skipped {uploaded_file.name}: no extractable text")
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except Exception as cleanup_error:
                                logger.warning(f"[WARNING] Could not delete temp file {temp_path}: {cleanup_error}")
                        continue
                
                documents.append({
                    "filename": uploaded_file.name,
                    "text": text,
                    "metadata": {
                        "source": uploaded_file.name,
                        "file_type": file_type,
                        "ocr_processed": is_ocr_processed
                    }
                })

                processing_method = "OCR" if is_ocr_processed else "Text Extraction"
                logger.info(f"[SUCCESS] Processed {uploaded_file.name} ({processing_method}): {len(text.strip())} characters")

                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as cleanup_error:
                        logger.warning(f"[WARNING] Could not delete temp file {temp_path}: {cleanup_error}")

            except Exception as e:
                logger.error(f"[ERROR] Error processing {uploaded_file.name}: {e}")
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as cleanup_error:
                        logger.warning(f"[WARNING] Could not delete temp file {temp_path}: {cleanup_error}")
                continue

        return documents

    def validate_file(self, file_path: str) -> bool:
        """Validate if the file is a supported format."""
        if not os.path.exists(file_path):
            return False
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_formats
