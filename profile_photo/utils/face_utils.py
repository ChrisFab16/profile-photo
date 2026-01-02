"""
Face detection utilities - moved from rekognition_utils for OpenCV compatibility.
"""
from __future__ import annotations

__all__ = ['best_fit_coordinates', 'show_image']

from enum import Enum
from functools import cached_property

import cv2 as cv

from .face_models import BoundingBox, Coordinates
from ..log import LOG


_DEFAULT_OFFSET = 0.17

_SCREEN_WIDTH = _SCREEN_HEIGHT = None


def get_screen_height_and_width():
    global _SCREEN_HEIGHT
    global _SCREEN_WIDTH

    if _SCREEN_HEIGHT is None or _SCREEN_WIDTH is None:
        import tkinter as tk
        root = tk.Tk()
        _SCREEN_HEIGHT = root.winfo_screenheight()
        _SCREEN_WIDTH = root.winfo_screenwidth()

    return _SCREEN_HEIGHT, _SCREEN_WIDTH


def best_fit_coordinates(im, face_box: BoundingBox, *boxes: BoundingBox | None,
                         fit=1/3.5,
                         x_offset=_DEFAULT_OFFSET,
                         y_offset=_DEFAULT_OFFSET,
                         constrain_width=True) -> Coordinates:

    # recall: reducing by `offset` will enlarge it
    f_top = face_box.top
    new_top = f_top - y_offset
    # calculate left and right (X) coordinates
    f_left = face_box.left
    f_right = face_box.left + face_box.width

    for box in boxes:

        if not box:
            continue

        # if top is not high enough, adjust

        # if ( L(face) - offset ) is *less* than or close enough to L(box), then
        # we don't do anything - same with right.

        b_top = box.top
        diff = new_top - b_top

        if diff >= 0:
            LOG.info('Enlarging Top, top=%.2f, new_top=%.2f', f_top, b_top)
            face_box.top = f_top = b_top
            new_top = f_top - y_offset
            face_box.height += diff + y_offset

        b_left = box.left
        b_right = b_left + box.width

        area_left = abs(f_left - b_left)
        area_right = abs(b_right - f_right)

        threshold_left = b_left + fit * area_left
        threshold_right = b_right - fit * area_right

        needs_fit = (f_left - x_offset) > threshold_left and (f_right + x_offset) < threshold_right

        # now left and right

        if needs_fit:
            offset_l = f_left - threshold_left
            offset_r = threshold_right - f_right
            x_offset = max(offset_l, offset_r)
            LOG.info('Enlarging Width, new_offset=%.3f', x_offset)

        if constrain_width:
            out_of_box = b_left > f_left - x_offset and b_right < f_right + x_offset
            if out_of_box:
                LOG.info('Constraining Width for Face')
                face_box.left = b_left + x_offset
                face_box.width = box.width - 2 * x_offset

    face_coords = Coordinates.from_box(im, face_box, x_offset, y_offset)

    return face_coords


def show_image(name, img, area=0.5, window_h=0, window_w=0):
    """
    Displays an image after resizing it to the specified dimensions, while
    also ensuring that we retain its aspect ratio.

    Credits: https://stackoverflow.com/a/67718163/10237506
    """

    import math

    h, w = img.shape[:2]

    screen_h, screen_w = get_screen_height_and_width()

    if area:
        vector = math.sqrt(area)
        window_h = screen_h * vector
        window_w = screen_w * vector

    if h > window_h or w > window_w:
        if h / window_h >= w / window_w:
            multiplier = window_h / h
        else:
            multiplier = window_w / w
        img = cv.resize(img, (0, 0), fx=multiplier, fy=multiplier)

    cv.imshow(name, img)

