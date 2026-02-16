import os
import base64
import logging
import io
from typing import List, Optional, Dict, Any
from PIL import Image

try:
    import streamlit as st
except Exception:
    st = None

try:
    import requests
except Exception:
    requests = None

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

logger = logging.getLogger(__name__)


def _get_api_key() -> Optional[str]:
    """Get Gemini API key from Streamlit secrets or environment variables."""
    if st is not None and hasattr(st, "secrets") and st.secrets is not None:
        return st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return os.environ.get("GEMINI_API_KEY")

def _get_endpoint() -> Optional[str]:
    # Allow users to configure a custom Gemini Vision REST endpoint in secrets
    if st is not None and hasattr(st, "secrets") and st.secrets is not None:
        return st.secrets.get("GEMINI_VISION_ENDPOINT")
    return os.environ.get("GEMINI_VISION_ENDPOINT")


def _image_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    b = buf.getvalue()
    return base64.b64encode(b).decode("utf-8")


def parse_images_with_gemini(image_path: str) -> str:
    """Parse a single image file using Gemini Vision via the google.generativeai SDK.

    Args:
        image_path: Path to an image file on disk.

    Returns:
        Extracted text as a string.

    Raises:
        EnvironmentError: if `GEMINI_API_KEY` is not set.
        ImportError: if `google.generativeai` SDK is not installed.
        RuntimeError: on API or processing failures.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.error("GEMINI_API_KEY is not configured in environment or Streamlit secrets")
        raise EnvironmentError("GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in environment or Streamlit secrets.")

    if genai is None:
        logger.error("google-genai SDK not installed or not importable")
        raise ImportError("google-genai SDK not installed. Install via `pip install google-genai`.")

    model_name = "gemini-2.5-flash"

    # Read image bytes
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
    except Exception as e:
        logger.error("Failed to read image file %s: %s", image_path, e)
        raise RuntimeError(f"Failed to read image file {image_path}: {e}")

    instruction = "Extract the textual content from the image, preserving layout and line breaks. Return plain text only."

    try:
        # Create client using the new google-genai SDK
        try:
            client = genai.Client(api_key=api_key)
        except Exception as _c_e:
            logger.error("Failed to create genai.Client: %s", _c_e)
            raise RuntimeError(f"Failed to initialize genai.Client: {_c_e}")

        logger.info("Invoking Gemini Vision via client for %s", image_path)

        # Build contents using types.Part.from_bytes() for proper Pydantic validation
        contents = [
            "Extract all readable text from this handwritten invoice image. Return only raw text.",
            types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/jpeg"
            )
        ]

        # Call client.models.generate_content with the specified contents
        try:
            response = client.models.generate_content(model=model_name, contents=contents)
        except Exception as _gen_e:
            logger.error("Gemini Vision client.models.generate_content failed: %s", _gen_e)
            raise RuntimeError(f"Gemini Vision client.models.generate_content failed: {_gen_e}")

        # Prefer response.text
        if hasattr(response, "text") and isinstance(response.text, str):
            logger.info("Gemini returned text for %s (len=%d)", image_path, len(response.text))
            return response.text

        # Handle dict-like responses
        if isinstance(response, dict):
            if "text" in response and isinstance(response["text"], str):
                return response["text"]
            outputs = response.get("outputs") or response.get("candidates") or response.get("pages")
            if isinstance(outputs, list) and outputs:
                parts = []
                for o in outputs:
                    if isinstance(o, dict):
                        t = o.get("text") or o.get("content") or o.get("output")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                if parts:
                    return "\n".join(parts)

        logger.error("Unexpected Gemini response shape: %s", type(response))
        raise RuntimeError("Gemini Vision returned an unexpected response; unable to extract text")

    except Exception as e:
        logger.exception("Gemini Vision invocation failed for %s: %s", image_path, e)
        raise


def is_gemini_available() -> Dict[str, Any]:
    """Return availability status for Gemini Vision integration.

    Returns a dict with keys: `available` (bool) and `reason` (str).
    Only supports the new google-genai SDK (not google-generativeai or REST endpoints).
    """
    api_key = _get_api_key()

    # Check SDK availability
    if genai is None:
        msg = "Gemini Vision unavailable — google-genai SDK not installed"
        logger.warning(msg)
        return {"available": False, "reason": msg}

    # Check API key
    if not api_key:
        msg = "Gemini Vision unavailable — GEMINI_API_KEY not configured"
        logger.warning(msg)
        return {"available": False, "reason": msg}

    # Both SDK and API key present
    msg = "Gemini Vision available via SDK"
    logger.info(msg)
    return {"available": True, "reason": msg}
