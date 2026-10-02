"""CPU contract tests for GPU video admission evidence."""

from pathlib import Path

import numpy as np
import pytest

from npa.solutions import video_generation as video


@pytest.mark.parametrize("solution", video.MODELS)
def test_models_use_immutable_distinct_native_checkpoints(solution):
    model = video.validate_request(solution, "a stable industrial scene", 12)
    assert len(model.revision) == 40
    assert all(char in "0123456789abcdef" for char in model.revision)
    assert model.frames > 1 and model.steps > 1


@pytest.mark.parametrize(
    "solution,prompt,seed",
    [
        ("unknown", "a scene", 1),
        ("mochi-1", " ", 1),
        ("mochi-1", "a scene", -1),
        ("mochi-1", "a scene", True),
        ("mochi-1", "a scene", 2**63),
    ],
)
def test_invalid_requests_fail_before_importing_gpu_libraries(solution, prompt, seed):
    with pytest.raises(ValueError):
        video.validate_request(solution, prompt, seed)


def test_media_validation_rejects_truncated_decode(monkeypatch, tmp_path):
    monkeypatch.setattr(
        video, "_decoded_frames", lambda _: iter([np.zeros((16, 16, 3))])
    )
    with pytest.raises(RuntimeError, match="Invalid generated video"):
        video.validate_video(tmp_path / "truncated.mp4", 81)


def test_invalid_negative_prompt_fails_before_gpu_initialization(tmp_path):
    with pytest.raises(ValueError, match="Negative prompt"):
        video.generate_video(
            "cogvideox-2b", "A scene", 1, tmp_path, negative_prompt=None
        )


@pytest.mark.parametrize("blank", [True, False])
def test_media_validation_rejects_blank_or_still_images(monkeypatch, tmp_path, blank):
    frame = (
        np.zeros((32, 32, 3)) if blank else np.arange(32 * 32 * 3).reshape(32, 32, 3)
    )
    monkeypatch.setattr(video, "_decoded_frames", lambda _: iter([frame, frame]))
    with pytest.raises(RuntimeError, match="Invalid generated video"):
        video.validate_video(tmp_path / "stationary.mp4", 2)


def test_media_evidence_hashes_actual_output(monkeypatch, tmp_path):
    frames = [np.arange(32 * 32 * 3).reshape(32, 32, 3) + index for index in range(3)]
    monkeypatch.setattr(video, "_decoded_frames", lambda _: iter(frames))
    output: Path = tmp_path / "video.mp4"
    output.write_bytes(b"actual encoded bytes")
    evidence = video.validate_video(output, 3)
    assert evidence["frame_count"] == 3
    assert evidence["mean_temporal_delta"] == 1
    assert evidence["size_bytes"] == len(b"actual encoded bytes")
    assert len(evidence["sha256"]) == 64
