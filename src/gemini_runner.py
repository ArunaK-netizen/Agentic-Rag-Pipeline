"""Simple runner to parse a scanned PDF/image using the advanced OCR pipeline which prefers Gemini Vision.

Usage:
    python -m src.gemini_runner PATH/TO/file.pdf

Outputs per-page text, combined text, and first few chunks.
"""
import sys
import logging

logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.gemini_runner PATH/TO/file.pdf")
        sys.exit(2)

    path = sys.argv[1]
    try:
        from .advanced_ocr import process_scanned_pdf

        pages, combined, chunks = process_scanned_pdf(path)

        print("--- Per-page output ---\n")
        for i, p in enumerate(pages, start=1):
            print(f"PAGE {i}:\n{p}\n")

        print("--- Combined ---\n")
        print(combined[:1000] + ("..." if len(combined) > 1000 else ""))

        print("--- First 3 chunks ---\n")
        for c in chunks[:3]:
            print(c[:1000] + ("..." if len(c) > 1000 else ""))

    except Exception as e:
        logger.exception("Failed to run Gemini runner: %s", e)
        print("Error:", e)


if __name__ == '__main__':
    main()
