"""Verify judge inputs preserve side identity, time, pixels and evidence hashes."""

import base64
import hashlib
import io

import pytest
from PIL import Image

from npa.workflows.video_sweep.review_frames import paired_frames


def _block(color):
    buffer = io.BytesIO()
    Image.new("RGB", (640, 360), color).save(buffer, format="JPEG")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,"
            + base64.b64encode(buffer.getvalue()).decode()
        },
    }


def test_each_judge_image_contains_both_videos_in_declared_order():
    source = [_block("red"), _block("green")]
    generated = [_block("blue"), _block("yellow")]
    source_metadata = {"sample_indices": [0, 23], "sample_times": [0.0, 23 / 24]}
    generated_metadata = {"sample_indices": [0, 29], "sample_times": [0.0, 29 / 30]}
    content, evidence = paired_frames(
        source, source_metadata, generated, generated_metadata
    )
    assert evidence["layout"] == "source-left-generated-right-v1"
    assert len(content) == 4 and len(evidence["frames"]) == 2
    assert "left SOURCE | frame 23 | 0.958s" in content[2]["text"]
    assert "right GENERATED | frame 29 | 0.967s" in content[2]["text"]
    raw = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
    image = Image.open(io.BytesIO(raw))
    # Sentinels catch a source/generated swap without depending on text OCR.
    left, right = image.getpixel((256, 180)), image.getpixel((768, 180))
    assert left[0] > 240 and left[2] < 15
    assert right[2] > 240 and right[0] < 15
    assert image.width == 1024
    assert evidence["frames"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert evidence["frames"][1]["source_index"] == 23
    assert evidence["frames"][1]["generated_time"] == 29 / 30
    # Pixel headers accompany every frame even if a provider loses text interleaving.
    assert max(image.crop((0, 0, 512, 44)).convert("L").getextrema()) > 200
    assert max(image.crop((512, 0, 1024, 44)).convert("L").getextrema()) > 200


@pytest.mark.parametrize("missing", ["image", "source-time", "generated-index"])
def test_incomplete_pairing_fails_before_judging(missing):
    before, after = [_block("red")] * 2, [_block("blue")] * 2
    source_metadata = {"sample_indices": [0, 1], "sample_times": [0, 1]}
    generated_metadata = {"sample_indices": [0, 1], "sample_times": [0, 1]}
    if missing == "image":
        after.pop()
    elif missing == "source-time":
        source_metadata["sample_times"].pop()
    else:
        generated_metadata["sample_indices"].pop()
    with pytest.raises(ValueError, match="matching frame and timeline"):
        paired_frames(before, source_metadata, after, generated_metadata)
