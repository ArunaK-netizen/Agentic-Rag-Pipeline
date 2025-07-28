import os
import logging
import PyPDF2
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and preprocessing, including OCR fallback."""

    def __init__(self, enable_ocr: bool = True):
        self.supported_formats = ['.pdf']
        self.enable_ocr = enable_ocr

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Try extracting text using PyPDF2; fallback to OCR if result is empty."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")

                # If no text found, fallback to OCR
                if not text.strip() and self.enable_ocr:
                    logger.info("No extractable text found; falling back to OCR.")
                    return self.ocr_entire_pdf(pdf_path)
                
                return text

        except Exception as e:
            logger.error(f"Error reading PDF file {pdf_path}: {e}")
            raise

    def ocr_entire_pdf(self, pdf_path: str) -> str:
        """Run OCR over the entire PDF using PyMuPDF + Tesseract."""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(image)
                text += f"\n--- Page {page_num + 1} (OCR) ---\n{page_text}\n"
        except Exception as e:
            logger.error(f"OCR failed: {e}")
        return text

    def process_uploaded_files(self, uploaded_files) -> List[Dict[str, Any]]:
        """Process multiple uploaded PDF files."""
        documents = []

        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                    # Extract text
                    text = self.extract_text_from_pdf(temp_path)

                    documents.append({
                        "filename": uploaded_file.name,
                        "text": text,
                        "metadata": {
                            "source": uploaded_file.name,
                            "file_type": "pdf"
                        }
                    })

                    # Clean up temp file
                    os.remove(temp_path)

                except Exception as e:
                    logger.error(f"Error processing {uploaded_file.name}: {e}")
                    continue

        return documents

    def validate_file(self, file_path: str) -> bool:
        """Validate if the file is a supported PDF."""
        if not os.path.exists(file_path):
            return False

        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_formats
