import os
import base64
import logging
from typing import List, Optional, Union
from PIL import Image
import io

try:
    import streamlit as st
except Exception:
    st = None

logger = logging.getLogger(__name__)


def _get_api_key() -> Optional[str]:
    """Get Gemini API key from Streamlit secrets or environment variables."""
    if st is not None and hasattr(st, "secrets") and st.secrets is not None:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            masked_key = key[:10] + "..." + key[-4:] if len(key) > 14 else "***"
            logger.info(f"[INFO] Using Gemini API key from Streamlit secrets: {masked_key}")
            return key
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        masked_key = env_key[:10] + "..." + env_key[-4:] if len(env_key) > 14 else "***"
        logger.info(f"[INFO] Using Gemini API key from environment: {masked_key}")
    return env_key


def _image_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    b = buf.getvalue()
    return base64.b64encode(b).decode("utf-8")


def _load_image_from_path(image_path: str) -> Image.Image:
    """Load PIL Image from file path."""
    return Image.open(image_path)


def parse_images_with_gemini(images: Union[List, str], prompt: Optional[str] = None, model: Optional[str] = None) -> Optional[str]:
    """
    Extract text from images using Gemini Vision API (google.generativeai SDK).
    
    Args:
        images: Either a list of PIL Images or a single file path string
        prompt: Optional custom prompt for text extraction
        model: Optional model name (defaults to gemini-2.0-flash)
    
    Returns:
        Extracted text as a single string, or None if Gemini is not configured/available
    """
    api_key = _get_api_key()
    
    if not api_key:
        logger.info("[INFO] Gemini Vision not available or not configured; falling back to local OCR")
        return None
    
    try:
        import google.generativeai as genai
        logger.info("[INFO] Gemini SDK detected; configuring with API key")
        genai.configure(api_key=api_key)
        
        # Use default model if not specified
        if model is None:
            model = "gemini-2.0-flash"
        
        # Normalize input: convert single path to list of images, or keep as list of PIL images
        if isinstance(images, str):
            # Single file path
            pil_images = [_load_image_from_path(images)]
        elif isinstance(images, list):
            # List of PIL images or paths
            pil_images = []
            for img in images:
                if isinstance(img, str):
                    pil_images.append(_load_image_from_path(img))
                else:
                    # Assume PIL Image
                    pil_images.append(img)
        else:
            # Assume single PIL Image
            pil_images = [images]
        
        logger.info(f"[INFO] Processing {len(pil_images)} image(s) with Gemini Vision model '{model}'")
        
        # Initialize the generative model
        vision_model = genai.GenerativeModel(model)
        
        # Prepare the prompt
        extraction_prompt = prompt or "Extract all text from this image. Preserve the layout and structure as much as possible."
        
        # Process images with Gemini
        all_text = []
        for idx, pil_img in enumerate(pil_images):
            try:
                logger.debug(f"[DEBUG] Processing image {idx + 1}/{len(pil_images)}")
                
                # Generate content with vision capabilities
                response = vision_model.generate_content([extraction_prompt, pil_img])
                
                # Extract text from response
                extracted_text = response.text if response and hasattr(response, 'text') else ""
                
                if extracted_text and extracted_text.strip():
                    all_text.append(extracted_text.strip())
                    char_count = len(extracted_text.strip())
                    logger.info(f"[SUCCESS] Gemini Vision extracted {char_count} characters from image {idx + 1}")
                else:
                    logger.debug(f"[DEBUG] Gemini Vision returned no text for image {idx + 1}")
                    
            except Exception as e:
                error_str = str(e)
                # Check for quota exceeded error
                if "429" in error_str or "quota" in error_str.lower():
                    retry_match = None
                    try:
                        import re
                        # Extract retry delay in seconds
                        retry_match = re.search(r'retry in (\d+\.?\d*)\s*s', error_str)
                    except:
                        pass
                    
                    retry_seconds = int(float(retry_match.group(1))) if retry_match else None
                    if retry_seconds:
                        retry_time = retry_seconds
                        minutes = retry_time // 60
                        seconds = retry_time % 60
                        if minutes > 0:
                            time_str = f"{minutes} minute{'s' if minutes > 1 else ''} and {seconds} second{'s' if seconds != 1 else ''}"
                        else:
                            time_str = f"{seconds} second{'s' if seconds != 1 else ''}"
                        msg = f"Gemini Vision quota exceeded for image {idx + 1}. Please try again in {time_str}."
                        logger.warning(f"[WARNING] {msg}")
                        # Store message in Streamlit session state for display to user
                        try:
                            if st is not None and hasattr(st, 'session_state'):
                                st.session_state.gemini_quota_msg = msg
                        except:
                            pass
                    else:
                        logger.warning(f"[WARNING] Gemini Vision processing failed for image {idx + 1}: {e}")
                else:
                    logger.warning(f"[WARNING] Gemini Vision processing failed for image {idx + 1}: {e}")
                continue
        
        if all_text:
            result = "\n".join(all_text)
            logger.info(f"[SUCCESS] Gemini Vision completed: {len(result.strip())} total characters extracted")
            return result
        else:
            logger.info("[INFO] Gemini Vision extracted no text from any image")
            return None
            
    except ImportError:
        logger.error("[ERROR] google.generativeai SDK not installed. Install with: pip install google-generativeai")
        return None
    except Exception as e:
        error_str = str(e)
        # Check for quota exceeded error at the model level
        if "429" in error_str or "quota" in error_str.lower():
            retry_match = None
            try:
                import re
                retry_match = re.search(r'retry in (\d+\.?\d*)\s*s', error_str)
            except:
                pass
            
            retry_seconds = int(float(retry_match.group(1))) if retry_match else None
            if retry_seconds:
                retry_time = retry_seconds
                minutes = retry_time // 60
                seconds = retry_time % 60
                if minutes > 0:
                    time_str = f"{minutes} minute{'s' if minutes > 1 else ''} and {seconds} second{'s' if seconds != 1 else ''}"
                else:
                    time_str = f"{seconds} second{'s' if seconds != 1 else ''}"
                msg = f"Gemini Vision quota exceeded. Please try again in {time_str}."
                logger.warning(f"[WARNING] {msg}")
                # Store message in Streamlit session state for display to user
                try:
                    if st is not None and hasattr(st, 'session_state'):
                        st.session_state.gemini_quota_msg = msg
                except:
                    pass
            else:
                logger.error(f"[ERROR] Gemini Vision API call failed: {e}")
        else:
            logger.error(f"[ERROR] Gemini Vision API call failed: {e}")
        return None
