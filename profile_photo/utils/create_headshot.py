from __future__ import annotations

from os.path import splitext

import cv2 as cv
import numpy as np

from .face_models import DetectFacesResp, DetectLabelsResp
from .face_utils import best_fit_coordinates, show_image
from .img_orient import get_oriented_im_bytes, get_im_orientation
from ..models import ProfilePhoto


_DEFAULT_FILE_EXT = '.jpg'


def rotate_im_and_crop(fp: str,
                       faces: DetectFacesResp,
                       labels: DetectLabelsResp | None,
                       file_ext: str | None = None,
                       im_bytes: bytes = None,
                       debug: bool = False,
                       remove_bg: bool = False,
                       bg_color: tuple[int, int, int] | None = None,
                       bg_model: str = 'u2net',
                       ) -> ProfilePhoto:

    # Get primary face in the photo (might need to be tweaked?)
    face = faces.get_face()

    # Get file extension (.jpg etc.)
    if not file_ext:
        file_ext = splitext(fp)[1] if fp else _DEFAULT_FILE_EXT

    # Get Image Orientation
    _, is_rotated, orientation = get_im_orientation(im_bytes)

    # Correct Image Orientation (If Needed) - Rotate Image
    if is_rotated:
        im_bytes = get_oriented_im_bytes(file_ext, im_bytes, orientation)[0]

    # Read in image data as OpenCV Image
    img_as_np = np.frombuffer(im_bytes, dtype=np.uint8)
    im = cv.imdecode(img_as_np, cv.IMREAD_COLOR)

    # Get bounding box for the Person in the photo
    person_box = labels.get_person_box(face)

    # Get X/Y coordinates for cropping
    coords = best_fit_coordinates(im, face.bounding_box, person_box)

    # Make the crop square by using the larger dimension
    crop_width = coords.x2 - coords.x1
    crop_height = coords.y2 - coords.y1
    crop_size = max(crop_width, crop_height)
    
    # Calculate center of current crop
    center_x = (coords.x1 + coords.x2) // 2
    center_y = (coords.y1 + coords.y2) // 2
    
    # Calculate square coordinates centered on the original crop
    half_size = crop_size // 2
    square_x1 = center_x - half_size
    square_y1 = center_y - half_size
    square_x2 = square_x1 + crop_size
    square_y2 = square_y1 + crop_size
    
    # Adjust if we go outside image boundaries
    im_height, im_width = im.shape[:2]
    
    if square_x1 < 0:
        square_x2 += abs(square_x1)
        square_x1 = 0
    if square_x2 > im_width:
        square_x1 -= (square_x2 - im_width)
        square_x2 = im_width
    
    if square_y1 < 0:
        square_y2 += abs(square_y1)
        square_y1 = 0
    if square_y2 > im_height:
        square_y1 -= (square_y2 - im_height)
        square_y2 = im_height
    
    # Ensure final crop is square (use the smaller dimension to guarantee it fits)
    final_width = square_x2 - square_x1
    final_height = square_y2 - square_y1
    final_size = min(final_width, final_height)
    
    # Re-center with the final square size
    center_x = (square_x1 + square_x2) // 2
    center_y = (square_y1 + square_y2) // 2
    half_final = final_size // 2
    
    square_x1 = center_x - half_final
    square_y1 = center_y - half_final
    square_x2 = square_x1 + final_size
    square_y2 = square_y1 + final_size
    
    # Crop the Photo (now square)
    #   crop_img = img[y:y+h, x:x+w]
    cropped_im = im[square_y1:square_y2, square_x1:square_x2]

    # Remove background if requested
    if remove_bg:
        from .background_removal import composite_on_color_background, composite_on_transparent_background
        
        if bg_color:
            # Composite on solid color background
            cropped_im = composite_on_color_background(cropped_im, bg_color, bg_model)
            # Convert to bytes (use original format)
            final_im_bytes: bytes = cv.imencode(file_ext, cropped_im)[1].tobytes()
        else:
            # Transparent background (PNG)
            final_im_bytes: bytes = composite_on_transparent_background(cropped_im, bg_model)
            file_ext = '.png'  # Force PNG for transparency
    else:
        # Convert the cropped photo to bytes
        final_im_bytes: bytes = cv.imencode(file_ext, cropped_im)[1].tobytes()

    # Show cropped image (if debug is enabled)
    # Note: For transparent backgrounds, we show the BGR version for display
    if debug:
        display_im = cropped_im
        if remove_bg and not bg_color:
            # For transparent background, we need to show something
            # Use the original cropped image for display
            pass
        show_image('Result', display_im)
        cv.waitKey(0)

    return ProfilePhoto(
        fp, final_im_bytes, is_rotated, orientation, faces, labels, im_bytes,
    )
