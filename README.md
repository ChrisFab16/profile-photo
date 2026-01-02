# Profile Photo

> **Fork Note**: This is a fork of [rnag/profile-photo](https://github.com/rnag/profile-photo) with OpenCV-based face detection instead of AWS Rekognition. See [What Changed?](#what-changed) below for details.

*Center* + *Crop* Image to create a Profile Pic or
[Headshot](https://www.nfi.edu/headshot-photo).

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/boy-1.jpg" height="100" width="130" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/boy-1-out.jpg" height="100" width="70" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/construction-worker-1.jpeg" height="100" width="110" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/construction-worker-1-out.jpeg" height="100" width="90" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/girl-1.jpg" height="100" width="120" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/girl-1-out.jpg" height="100" width="80" />
</p>

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/girl-2.jpg" height="100" width="120" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/girl-2-out.jpg" height="100" width="80" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/hoodie-1.jpg" height="100" width="110" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/hoodie-1-out.jpg" height="100" width="90" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/man-1.jpeg" height="100" width="120" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/man-1-out.jpeg" height="100" width="80" />
</p>

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/woman-1.png" height="100" width="90" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/woman-1-out.png" height="100" width="60" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/woman-2.jpeg" height="100" width="130" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/woman-2-out.jpeg" height="100" width="110" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/wonder-woman-1.jpeg" height="100" width="120" />
  <img src="https://raw.githubusercontent.com/rnag/profile-photo/main/examples/wonder-woman-1-out.jpeg" height="100" width="90" />
</p>


## Install

Install directly from this GitHub fork:

``` console
$ pip install git+https://github.com/ChrisFab16/profile-photo.git
```

Or clone and install in development mode:

``` console
$ git clone https://github.com/ChrisFab16/profile-photo.git
$ cd profile-photo
$ pip install -e .
```

The package uses OpenCV for face detection, which is included as a dependency.

## What Changed?

This library was previously dependent on **Amazon Rekognition**, a cloud-based face detection service. We've migrated to **OpenCV** to eliminate all web service dependencies. Here's why:

- **No API keys required** - Works completely offline without any cloud service authentication
- **No internet connection needed** - Process images locally without sending data to external servers
- **No usage costs** - Free to use without per-request pricing or service limits
- **Privacy-first** - Your images never leave your machine
- **Faster processing** - No network latency, all processing happens locally
- **Simpler deployment** - No AWS configuration, credentials, or region setup required

The migration maintains the same API interface, so existing code continues to work. Face detection now uses OpenCV's built-in Haar Cascade classifier (default) or optional DNN models for improved accuracy.

## Features


-   Exports a helper function, <code>create_headshot</code>,
    to create a
    close-up or headshot of the primary face in a photo or image.
-   Uses [OpenCV](https://opencv.org/) for face detection - no cloud services required!
-   Detects faces using OpenCV's Haar Cascade or DNN-based face detection models.
-   Exposes helper methods to save the result image (*cropped*) as well
    as detection responses to a local folder.

## Usage

Basic usage, with a sample image:

``` python3
from profile_photo import create_headshot

photo = create_headshot('/path/to/image.jpg')
photo.show()
```

An example with a local image, and saving the result image and detection
responses to a folder:

``` python3
from __future__ import annotations

from profile_photo import create_headshot


# customize local file location for detection responses
def get_filename(file_name: str | None, api: str):
    return f'responses/{file_name}_{api}.json'


photo = create_headshot('/path/to/image')

# this saves image and detection responses to a results/ folder
# can also be achieved by passing `output_dir` above
photo.save_all('results', get_response_filename=get_filename)

# display before-and-after images
photo.show()
```

An example with cached face detection responses:

``` python3
from pathlib import Path

from profile_photo import create_headshot

image_path = Path('path/to/image.jpg')
responses_dir = Path('./my/responses')

photo = create_headshot(
    image_path,
    faces=responses_dir / f'{image_path.stem}_DetectFaces.json',
    labels=responses_dir / f'{image_path.stem}_DetectLabels.json',
    debug=True
)
```

## Examples

Check out [example
images](https://github.com/ChrisFab16/profile-photo/tree/main/examples) on
GitHub for sample use cases and results.

## How It Works

This library uses [OpenCV](https://opencv.org/) for face detection - no cloud
services or API keys required! It detects faces using OpenCV's built-in Haar
Cascade classifier (default) or optionally a DNN-based face detection model.

The library then uses custom, in-house logic to determine the X/Y coordinates for
cropping. This mainly involves "blowing up" or enlarging the Face
bounding box, but then correcting the coordinates as needed by the
estimated Person box. This logic has been fine-tuned based on what provides
the best overall results for generic images (not necessarily profile photos).

## Future Ideas

-   Support background removal with
    <code><a href="https://pypi.org/project/rembg">rembg</a></code>.
-   Add support for more advanced DNN-based face detection models.
-   Improve person/body detection accuracy.

## Credits

This package is a fork of [rnag/profile-photo](https://github.com/rnag/profile-photo), originally created by [Ritvik Nag](https://github.com/rnag).

The original package was created with
[Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the
[rnag/cookiecutter-pypackage](https://github.com/rnag/cookiecutter-pypackage)
project template.

## License

Copyright (c) 2023-present  [Ritvik Nag](https://github.com/rnag)

Licensed under [MIT License](./LICENSE)
