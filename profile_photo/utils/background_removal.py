"""
Background removal using rembg library.
"""
from __future__ import annotations

from io import BytesIO

import cv2 as cv
import numpy as np
from PIL import Image

from ..log import LOG


def remove_background(im_bytes: bytes, model_name: str = 'u2net') -> bytes:
    """
    Remove background from an image using rembg.
    
    :param im_bytes: Image data as bytes
    :param model_name: rembg model to use (default: 'u2net')
    :return: Image bytes with background removed (PNG format with alpha channel)
    """
    try:
        from rembg import remove, new_session
    except ImportError:
        raise ImportError(
            "rembg is required for background removal. "
            "Install it with: pip install rembg"
        ) from None
    
    LOG.info('Removing background using rembg model: %s', model_name)
    
    # Create a session with the specified model
    session = new_session(model_name)
    
    # Use rembg to remove background
    # rembg returns PNG bytes with alpha channel
    output_bytes = remove(im_bytes, session=session)
    
    return output_bytes


def remove_background_from_cv_image(cv_image: np.ndarray, model_name: str = 'u2net') -> np.ndarray:
    """
    Remove background from an OpenCV image (numpy array).
    
    :param cv_image: OpenCV image (BGR format)
    :param model_name: rembg model to use (default: 'u2net')
    :return: OpenCV image with background removed (BGRA format)
    """
    # Convert OpenCV image to bytes
    is_success, im_buf_arr = cv.imencode('.jpg', cv_image)
    if not is_success:
        raise ValueError("Failed to encode image")
    im_bytes = im_buf_arr.tobytes()
    
    # Remove background (returns PNG bytes with alpha)
    output_bytes = remove_background(im_bytes, model_name=model_name)
    
    # Convert PNG bytes back to OpenCV image
    img_array = np.frombuffer(output_bytes, dtype=np.uint8)
    img_with_alpha = cv.imdecode(img_array, cv.IMREAD_UNCHANGED)
    
    # Convert RGBA to BGRA (OpenCV uses BGR)
    if img_with_alpha.shape[2] == 4:
        # PIL uses RGBA, OpenCV uses BGRA
        img_bgra = cv.cvtColor(img_with_alpha, cv.COLOR_RGBA2BGRA)
        return img_bgra
    else:
        # If no alpha channel, return as-is
        return img_with_alpha


def composite_on_transparent_background(cv_image: np.ndarray, model_name: str = 'u2net') -> bytes:
    """
    Remove background and return as PNG bytes with transparent background.
    
    :param cv_image: OpenCV image (BGR format)
    :param model_name: rembg model to use (default: 'u2net')
    :return: PNG image bytes with transparent background
    """
    # Remove background
    img_with_alpha = remove_background_from_cv_image(cv_image, model_name)
    
    # Convert BGRA to RGBA for PIL
    if img_with_alpha.shape[2] == 4:
        img_rgba = cv.cvtColor(img_with_alpha, cv.COLOR_BGRA2RGBA)
    else:
        img_rgba = img_with_alpha
    
    # Convert to PIL Image and then to PNG bytes
    pil_image = Image.fromarray(img_rgba)
    output = BytesIO()
    pil_image.save(output, format='PNG')
    return output.getvalue()


def composite_on_color_background(cv_image: np.ndarray, 
                                  background_color: tuple[int, int, int] = (255, 255, 255),
                                  model_name: str = 'u2net') -> np.ndarray:
    """
    Remove background and composite on a solid color background.
    
    :param cv_image: OpenCV image (BGR format)
    :param background_color: RGB color tuple for background (default: white)
    :param model_name: rembg model to use (default: 'u2net')
    :return: OpenCV image (BGR format) with solid color background
    """
    # Remove background
    img_with_alpha = remove_background_from_cv_image(cv_image, model_name)
    
    if img_with_alpha.shape[2] != 4:
        # No alpha channel, return original
        return cv_image
    
    # Extract alpha channel
    alpha = img_with_alpha[:, :, 3:4] / 255.0
    
    # Extract RGB channels (BGR in OpenCV)
    bgr = img_with_alpha[:, :, :3]
    
    # Create background
    h, w = img_with_alpha.shape[:2]
    # Convert RGB to BGR for OpenCV
    bg_color_bgr = (background_color[2], background_color[1], background_color[0])
    background = np.full((h, w, 3), bg_color_bgr, dtype=np.uint8)
    
    # Composite: foreground * alpha + background * (1 - alpha)
    result = (bgr * alpha + background * (1 - alpha)).astype(np.uint8)
    
    return result

