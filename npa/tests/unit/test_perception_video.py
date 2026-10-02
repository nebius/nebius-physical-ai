"""Reject invalid perception requests before allocating GPU or fetching weights."""

from pathlib import Path

import pytest

from npa.solutions.perception_video import _validate_box, run_perception


@pytest.mark.parametrize(
    "box",
    [
        None,
        [],
        [0, 0, 1],
        [0, 0, 961, 540],
        [2, 0, 1, 4],
        [0, 0, float("nan"), 4],
        [0, 0, float("inf"), 4],
        [False, 0, 10, 10],
        ["0", 0, 10, 10],
    ],
)
def test_invalid_sam_prompt_fails_before_cuda_or_file_access(box):
    with pytest.raises(ValueError, match="SAM"):
        run_perception("sam", Path("missing.mp4"), Path("missing-output"), box)


def test_valid_box_matches_declared_viewport():
    _validate_box([0, 0, 960, 540])


def test_unknown_perception_kind_fails_before_cuda():
    with pytest.raises(ValueError, match="kind"):
        run_perception("unknown", Path("missing.mp4"), Path("missing-output"))


@pytest.mark.parametrize("kind", ["depth", "sam"])
def test_encoded_reference_rejects_uint8_rescaling(monkeypatch, kind):
    import numpy as np
    from npa.solutions import perception_video as module

    original = np.full((8, 240, 3), [25, 125, 235], dtype=np.uint8)
    values = [np.zeros((8, 240), dtype=bool)]
    monkeypatch.setattr(
        module, "_decoded_frames", lambda _: iter([original[:, :, ::-1]])
    )
    assert module._reference_color_error([original], values, kind, Path("unused")) == 0
    corrupted = (original * 255).astype(np.uint8)
    monkeypatch.setattr(
        module, "_decoded_frames", lambda _: iter([corrupted[:, :, ::-1]])
    )
    with pytest.raises(RuntimeError, match="reference colors"):
        module._reference_color_error([original], values, kind, Path("unused"))
