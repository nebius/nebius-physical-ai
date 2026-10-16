"""Exercise media identity, native decoding and replay measurement refusals."""

import base64
from fractions import Fraction
import hashlib
import io
import json

import av
import numpy as np
import pytest

from npa.workflows.lyra_validated_example import (
    _compare_videos,
    _media_bytes,
    _qualified_payload,
    _verify_preview,
    _write_private,
)


def _video(value, count=3, fps=30, width=16):
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264rgb", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, 16, "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, width, 3), value, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _expected():
    return {"height": 16, "width": 16, "nb_read_frames": "3", "r_frame_rate": "30/1"}


def test_replay_psnr_measures_all_decoded_rgb_pixels():
    replay, posters = _compare_videos(_video(20), _video(21), _expected())
    assert replay["rgb_mse"] == 1
    assert replay["psnr_db"] == pytest.approx(48.1308036086791)
    assert replay["paired_frames"] == 3
    assert "not held-out" in replay["scope"]
    assert all(poster.startswith("data:image/jpeg;base64,") for poster in posters)


def test_identical_replay_uses_json_safe_psnr():
    replay, _ = _compare_videos(_video(20), _video(20), _expected())
    assert replay["rgb_mse"] == 0
    assert replay["psnr_db"] is None
    json.dumps(replay, allow_nan=False)


def test_native_rounded_render_cadence_pairs_frames_without_retiming():
    source_fps = Fraction(30000, 1001)
    source_probe = {**_expected(), "r_frame_rate": "30000/1001"}
    replay, _ = _compare_videos(
        _video(20, fps=source_fps), _video(21), source_probe, _expected()
    )
    assert replay["rgb_mse"] == 1
    assert replay["fps"] == float(source_fps)
    assert replay["rendered_fps"] == 30
    assert "frame index" in replay["scope"]


def test_qualification_disagreement_on_render_frame_count_is_rejected():
    with pytest.raises(ValueError, match="shapes or counts differ"):
        _compare_videos(
            _video(20), _video(21), _expected(), {**_expected(), "nb_read_frames": "4"}
        )


@pytest.mark.parametrize(
    "changed,message",
    [
        ({"count": 2}, "different frame counts"),
        ({"width": 32}, "qualified dimensions"),
        ({"fps": 25}, "recorded cadence"),
    ],
)
def test_mismatched_native_replay_is_rejected(changed, message):
    with pytest.raises(ValueError, match=message):
        _compare_videos(_video(20), _video(21, **changed), _expected())


def test_qualification_frame_total_is_required():
    expected = {**_expected(), "nb_read_frames": "4"}
    with pytest.raises(ValueError, match="frame count differs"):
        _compare_videos(_video(20), _video(21), expected)


def test_modified_html_is_rejected_before_parsing(tmp_path):
    path = tmp_path / "review.html"
    path.write_text("synthetic-changed-review")
    qualification = {
        "html_export": {"source_html_sha256": "a", "export_html_sha256": "b"}
    }
    with pytest.raises(ValueError, match="recorded qualification"):
        _qualified_payload(path, qualification)


def test_media_must_match_both_readback_and_native_identity():
    raw = b"synthetic-media"
    digest = hashlib.sha256(raw).hexdigest()
    payload = {
        "source_video": "data:video/mp4;base64," + base64.b64encode(raw).decode(),
        "evidence": {"input_sha256": "wrong-native-input"},
    }
    qualification = {"artifacts": {"artifacts": {"capture.mp4": {"sha256": digest}}}}
    with pytest.raises(ValueError, match="native reconstruction identity"):
        _media_bytes(payload, qualification, "source_video", "capture.mp4")
    payload["evidence"]["input_sha256"] = digest
    assert _media_bytes(payload, qualification, "source_video", "capture.mp4") == raw
    qualification["artifacts"]["artifacts"]["capture.mp4"]["sha256"] = "wrong-readback"
    with pytest.raises(ValueError, match="qualified native artifact"):
        _media_bytes(payload, qualification, "source_video", "capture.mp4")


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_preview_is_rejected(value):
    encoded = base64.b64encode(np.array([value, 0, 0], dtype="<f4").tobytes()).decode()
    with pytest.raises(ValueError, match="incomplete or nonfinite"):
        _verify_preview({"count": 1, "position": encoded})


@pytest.mark.parametrize("count", [0, -1, True])
def test_empty_or_invalid_preview_count_is_rejected(count):
    with pytest.raises(ValueError, match="must contain samples"):
        _verify_preview({"count": count})


def test_generated_review_and_new_directory_are_private(tmp_path):
    path = tmp_path / "review" / "example.html"
    _write_private(path, "synthetic-private-review")
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
