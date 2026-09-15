"""Reject invalid perception requests before allocating GPU or fetching weights."""

from pathlib import Path

import pytest

from npa.solutions.perception_video import _validate_box, run_perception


@pytest.mark.parametrize("box", [
    None, [], [0, 0, 1], [0, 0, 961, 540], [2, 0, 1, 4],
    [0, 0, float("nan"), 4], [0, 0, float("inf"), 4],
    [False, 0, 10, 10], ["0", 0, 10, 10],
])
def test_invalid_sam_prompt_fails_before_cuda_or_file_access(box):
    with pytest.raises(ValueError, match="SAM"):
        run_perception("sam", Path("missing.mp4"), Path("missing-output"), box)


def test_valid_box_matches_declared_viewport():
    _validate_box([0, 0, 960, 540])


def test_unknown_perception_kind_fails_before_cuda():
    with pytest.raises(ValueError, match="kind"):
        run_perception("unknown", Path("missing.mp4"), Path("missing-output"))
