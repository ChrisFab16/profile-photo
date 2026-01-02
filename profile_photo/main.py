"""Main module."""
from __future__ import annotations

from os import PathLike

from .models import Params, ProfilePhoto
from .utils.face_detection import FaceDetector
from .utils.face_models import DetectFacesResp, DetectLabelsResp
from .utils.create_headshot import rotate_im_and_crop
from .utils.json_util import load_to_model


def create_headshot(
    filepath_or_bytes: PathLike[str] | PathLike[bytes] | str | bytes | None = None,
    *,
    file_ext: str | None = None,
    faces: DetectFacesResp | Path | dict | str | None = None,
    labels: DetectLabelsResp | Path | dict | str | None = None,
    debug: bool = False,
    output_dir: PathLike[str] | PathLike[bytes] | str = None,
    confidence_threshold: float = 0.5,
) -> ProfilePhoto:
    """Create a Headshot Photo of a person, given an image.

    :param filepath_or_bytes: Path to a local file, or image data as Bytes
    :param file_ext: File extension or image type of output data (optional),
      defaults to the extension of input filename, or `.jpg` if a filename
      is not passed in.
    :param faces: Cached face detection response (optional)
    :param labels: Cached label detection response (optional)
    :param debug: True to log debug messages and show the image
    :param output_dir: Path to a local folder to save the output image
      and API responses (optional)
    :param confidence_threshold: Minimum confidence for face detection (0.0-1.0)
    :return: a :class:`ProfilePhoto` object, containing the output image and API response data

    """

    # do we need to make a face detection call?
    call_face_detection = not (faces and labels)

    # image file path or bytes is passed in
    if filepath_or_bytes:

        # image data (as bytes) is passed in
        if isinstance(filepath_or_bytes, bytes):
            # filepath is None for bytes input
            filepath = None
            # image bytes is known
            im_bytes = filepath_or_bytes

        # local filepath is passed in
        else:
            # filepath is known
            filepath = filepath_or_bytes
            # read image data from local file
            with open(filepath, 'rb') as f:
                im_bytes = f.read()
    else:
        raise ValueError("filepath_or_bytes must be provided")

    # Initialize face detector if needed
    if call_face_detection:
        detector = FaceDetector(confidence_threshold=confidence_threshold)
        
        # Detect faces
        if not faces:
            faces = detector.detect_faces(im_bytes=im_bytes, debug=debug)
        
        # Detect labels (person bounding box)
        if not labels:
            # Get the primary face to help with person detection
            face = faces.get_face()
            labels = detector.detect_labels(im_bytes=im_bytes, face=face, debug=debug)
    else:
        # transform or load the cached responses passed in (if needed)
        faces = load_to_model(DetectFacesResp, faces, Params.FACES)
        labels = load_to_model(DetectLabelsResp, labels, Params.LABELS)

    # rotate & crop the photo
    photo = rotate_im_and_crop(
        filepath, faces, labels, file_ext, im_bytes, debug)

    # save outputs to a local drive (if needed)
    if output_dir:
        if call_face_detection:
            photo.save_all(output_dir)
        else:
            photo.save_image(output_dir)

    # return the photo as headshot
    return photo
