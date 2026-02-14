import os
import base64
import logging
from typing import List, Optional

try:
    import streamlit as st
except Exception:
    st = None

try:
    import requests
except Exception:
    requests = None

logger = logging.getLogger(__name__)


def _get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets, then environment
    if st is not None and hasattr(st, "secrets") and st.secrets is not None:
        return st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return os.environ.get("GEMINI_API_KEY")


def _get_endpoint() -> Optional[str]:
    # Allow users to configure a custom Gemini Vision REST endpoint in secrets
    if st is not None and hasattr(st, "secrets") and st.secrets is not None:
        return st.secrets.get("GEMINI_VISION_ENDPOINT")
    return os.environ.get("GEMINI_VISION_ENDPOINT")


def _image_to_base64(pil_image) -> str:
    from io import BytesIO

    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    b = buf.getvalue()
    return base64.b64encode(b).decode("utf-8")


def parse_images_with_gemini(images: List, prompt: Optional[str] = None, model: Optional[str] = None) -> Optional[List[str]]:
    """
    Attempt to parse a list of PIL images using a configured Gemini Vision endpoint or SDK.

    Returns a list of per-page strings on success, or `None` when Gemini integration is not configured.
    Behavior:
      - If the environment provides a `google.generativeai` SDK with a documented image API, prefer it (best-effort).
      - Otherwise, if `GEMINI_VISION_ENDPOINT` + `GEMINI_API_KEY` are set in `st.secrets` or env, POST a JSON payload
        with base64-encoded images and the optional prompt to that endpoint and return parsed text if present.
      - If neither path is configured, return `None` so callers can fallback to EasyOCR.
    """
    api_key = _get_api_key()
    endpoint = _get_endpoint()
    logger.debug("Gemini VLM: api_key_present=%s endpoint=%s", bool(api_key), bool(endpoint))

    # Try official SDK if available (best-effort; do not hard-fail)
    try:
        import google.generativeai as genai
        logger.info("Gemini SDK detected; attempting SDK-based image helpers")

        # Many environments will provide a higher-level image API; we don't assume exact method names.
        # If the SDK is present but no documented image helper exists in this environment, skip to REST fallback.
        try:
            if api_key:
                genai.configure(api_key=api_key)
        except Exception:
            # ignore
            pass

        # Try a couple of common helper names (best-effort, not guaranteed across SDK versions)
        for helper_name in ("image_predict", "predict_images", "annotate_images", "vision_predict"):
            helper = getattr(genai, helper_name, None)
            if callable(helper):
                logger.debug("Trying Gemini SDK helper '%s'", helper_name)
                try:
                    # Convert images to bytes payloads
                    b64s = [_image_to_base64(img) for img in images]
                    payload = {"images": b64s, "prompt": prompt or "Extract text preserving layout"}
                    resp = helper(payload)
                    # SDKs vary widely; normalize common shapes
                    if isinstance(resp, dict):
                        # try common keys
                        if "pages" in resp and isinstance(resp["pages"], list):
                            logger.info("Gemini SDK returned 'pages' structure")
                            return [p.get("text", "") for p in resp["pages"]]
                        if "outputs" in resp and isinstance(resp["outputs"], list):
                            texts = []
                            for out in resp["outputs"]:
                                if isinstance(out, dict) and "text" in out:
                                    texts.append(out["text"])
                            if texts:
                                logger.info("Gemini SDK returned 'outputs' structure")
                                return texts
                    # Fallback: if helper returned a string, use same string for all pages
                    if isinstance(resp, str):
                        logger.info("Gemini SDK returned plain string; applying to all pages")
                        return [resp] * len(images)
                except Exception as e:
                    logger.debug("Gemini SDK image helper '%s' failed: %s", helper_name, e)

    except Exception:
        # SDK not available; move on to REST endpoint attempt
        pass

    # REST endpoint fallback
    if endpoint and api_key and requests is not None:
        logger.info("Gemini REST endpoint configured; sending %d images to %s", len(images), endpoint)
        try:
            b64s = [_image_to_base64(img) for img in images]
            payload = {"images": b64s, "prompt": prompt or "Extract text preserving layout"}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            j = resp.json()
            # Try to extract per-page texts from common fields
            if isinstance(j, dict):
                if "pages" in j and isinstance(j["pages"], list):
                    return [p.get("text", "") for p in j["pages"]]
                if "outputs" in j and isinstance(j["outputs"], list):
                    texts = []
                    for out in j["outputs"]:
                        if isinstance(out, dict) and "text" in out:
                            texts.append(out["text"])
                    if texts:
                        return texts
                # Some endpoints return a single concatenated string
                if "text" in j and isinstance(j["text"], str):
                    return [j["text"]] * len(images)

            # As a last resort, if the endpoint returned plain text
            if isinstance(resp.text, str) and resp.text.strip():
                return [resp.text.strip()] * len(images)

        except Exception as e:
            logger.warning("Gemini REST call failed: %s", e)

    # Not configured or failed — caller should fallback to EasyOCR
    logger.info("Gemini Vision not available or not configured; falling back to local OCR")
    return None
