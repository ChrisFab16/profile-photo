"""Integration Tests for `profile_photo` package."""

import pytest

from profile_photo import create_headshot
from ..conftest import images


@pytest.mark.parametrize('image', images)
def test_create_headshot_and_save_all(examples, responses, image):

    photo = create_headshot(examples / image)

    def get_filename(file_name: str | None, api: str):
        return responses / f'{file_name}_{api}.json'

    # can also be achieved by passing `output_dir` above
    photo.save_all(examples, get_response_filename=get_filename)


@pytest.mark.parametrize('image', images)
def test_create_headshot_and_show_side_by_side(examples, responses, image):
    filepath = examples / image

    photo = create_headshot(
        filepath,
        faces=responses / f'{filepath.stem}_DetectFaces.json',
        labels=responses / f'{filepath.stem}_DetectLabels.json',
    )

    photo.show()


# S3 support removed - test removed
