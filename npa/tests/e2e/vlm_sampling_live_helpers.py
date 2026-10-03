"""Construct known source frames and independently check live sampling evidence."""

from io import BytesIO
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


def make_sampling_input(root: Path, kind: str) -> tuple[Path, list[bytes]]:
    """Create six distinct lossless frames with independently normalized PNGs.

    Args:
        root: Private directory for generated inputs.
        kind: image-sequence, numpy-episode, or video.

    Returns:
        Input location and the six expected submitted PNG payloads.

    Raises:
        OSError: Input creation or ffmpeg execution fails.
        subprocess.CalledProcessError: Lossless video encoding fails.
        ValueError: The source kind is unsupported.
    """
    root.mkdir(parents=True)
    frames, expected = [], []
    for index in range(6):
        image = Image.new("RGB", (64, 48), (index * 40, 200 - index * 30, 50))
        frames.append(np.asarray(image))
        stream = BytesIO()
        image.save(stream, format="PNG", optimize=True)
        expected.append(stream.getvalue())
        image.save(root / f"frame-{index:03d}.png")
    return _input_location(root, kind, frames), expected


def _input_location(root: Path, kind: str, frames: list[np.ndarray]) -> Path:
    if kind == "image-sequence":
        return root
    if kind == "numpy-episode":
        target = root / "episode.npy"
        np.save(target, np.stack(frames))
        return target
    if kind != "video":
        raise ValueError("Unsupported sampling input kind")
    target = root / "rollout.avi"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-framerate",
            "2",
            "-i",
            str(root / "frame-%03d.png"),
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


def assert_sampling_evidence(
    payload: dict, kind: str, strategy: str, expected_pngs: list[bytes]
) -> None:
    """Compare retained evidence with known source payloads and sample positions.

    Args:
        payload: Real evaluation result containing provider evidence.
        kind: The fixture's source family.
        strategy: The requested sampling strategy, using max_frames=3.
        expected_pngs: Six normalized source PNGs, created without the selector.

    Returns:
        None.

    Raises:
        AssertionError: Claimed payload or sampling fields differ from the source.
        KeyError: Required evidence is missing.
    """
    indices = [5] if strategy == "final" else [0, 2, 5]
    timestamps = (
        [index / 2 for index in indices] if kind == "video" else [None] * len(indices)
    )
    request = payload["evidence"]["request"]
    assert request["request_manifest"]["sampling"] == _expected_sampling(
        kind, strategy, indices, timestamps
    )
    assert payload["frame_count"] == len(indices)
    assert request["frames"] == request["request_manifest"]["frames"]
    assert len(request["frames"]) == len(indices)
    for frame, index, timestamp in zip(request["frames"], indices, timestamps):
        _assert_frame(frame, expected_pngs[index], kind, index, timestamp)


def _expected_sampling(
    kind: str, strategy: str, indices: list[int], timestamps: list[float | None]
) -> dict:
    return {
        "strategy": strategy,
        "max_frames": 3,
        "selected_count": len(indices),
        "source_kind": kind,
        "source_count": 6,
        "selected_indices": indices,
        "selected_timestamps_s": timestamps,
        "coverage_complete": True,
        "timestamps_complete": True if kind == "video" else None,
    }


def _assert_frame(
    frame: dict, png: bytes, kind: str, index: int, timestamp: float | None
) -> None:
    assert frame["sha256"] == hashlib.sha256(png).hexdigest()
    assert frame["byte_count"] == len(png)
    assert (frame["width"], frame["height"], frame["media_type"]) == (
        64,
        48,
        "image/png",
    )
    assert (frame["source_kind"], frame["source_index"], frame["source_count"]) == (
        kind,
        index,
        6,
    )
    assert frame["source_timestamp_s"] == timestamp
