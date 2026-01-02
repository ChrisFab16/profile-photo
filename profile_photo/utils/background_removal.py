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
    # Convert OpenCV BGR image to RGB for rembg
    # rembg expects RGB format
    img_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    
    # Convert RGB image to bytes using PIL (preserves RGB)
    pil_image = Image.fromarray(img_rgb)
    output = BytesIO()
    pil_image.save(output, format='PNG')
    im_bytes = output.getvalue()
    
    # Remove background (returns PNG bytes with alpha in RGB format)
    output_bytes = remove_background(im_bytes, model_name=model_name)
    
    # Convert PNG bytes back to numpy array using PIL (preserves RGB)
    img_pil = Image.open(BytesIO(output_bytes))
    img_rgba = np.array(img_pil)
    
    # Convert RGBA to BGRA (OpenCV uses BGR)
    if img_rgba.shape[2] == 4:
        # Convert RGB to BGR for OpenCV
        img_bgra = cv.cvtColor(img_rgba, cv.COLOR_RGBA2BGRA)
        return img_bgra
    else:
        # If no alpha channel, convert RGB to BGR
        img_bgr = cv.cvtColor(img_rgba, cv.COLOR_RGB2BGR)
        return img_bgr


def composite_on_transparent_background(cv_image: np.ndarray, model_name: str = 'u2net') -> bytes:
    """
    Remove background and return as PNG bytes with transparent background.
    
    :param cv_image: OpenCV image (BGR format)
    :param model_name: rembg model to use (default: 'u2net')
    :return: PNG image bytes with transparent background
    """
    try:
        from rembg import remove, new_session
    except ImportError:
        raise ImportError(
            "rembg is required for background removal. "
            "Install it with: pip install rembg"
        ) from None
    
    # Convert OpenCV BGR to RGB for PIL
    img_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(img_rgb)
    
    # Create session and remove background directly with PIL
    session = new_session(model_name)
    output_pil = remove(pil_image, session=session)
    
    # Convert PIL Image to PNG bytes
    output = BytesIO()
    output_pil.save(output, format='PNG')
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

