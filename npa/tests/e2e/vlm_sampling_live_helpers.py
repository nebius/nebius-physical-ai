"""Construct known source frames and independently check live sampling evidence."""

from io import BytesIO
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from npa.literal_values import require_boolean, require_integer, require_number


def validate_sampling_scalars(payload: dict) -> None:
    """Reject coerced evidence before comparing sampling values or payload hashes.

    Args:
        payload: Evaluation result containing both retained frame copies.

    Returns:
        None.

    Raises:
        KeyError: Required evidence is missing.
        ValueError: A scalar has an invalid literal type or domain value.
    """
    require_integer(payload["frame_count"], field="frame_count", minimum=1)
    for field in ("dry_run", "passed"):
        require_boolean(payload[field], field=field)
    for field in ("score", "success_threshold"):
        require_number(payload[field], field=field, minimum=0, maximum=1)
    evidence = payload["evidence"]
    require_integer(evidence["provider"]["status_code"], field="status_code")
    request = evidence["request"]
    manifest = request["request_manifest"]
    for location, frames in (
        ("request.frames", request["frames"]),
        ("request_manifest.frames", manifest["frames"]),
    ):
        for index, frame in enumerate(frames):
            _validate_frame_scalars(frame, f"{location}[{index}]")
    _validate_sampling_scalars(manifest["sampling"])


def _validate_frame_scalars(frame: dict, prefix: str) -> None:
    for field in ("byte_count", "width", "height"):
        require_integer(frame[field], field=f"{prefix}.{field}", minimum=1)
    for field, minimum in (("source_index", 0), ("source_count", 1)):
        if frame[field] is not None:
            require_integer(frame[field], field=f"{prefix}.{field}", minimum=minimum)
    if frame["source_timestamp_s"] is not None:
        require_number(
            frame["source_timestamp_s"], field=f"{prefix}.source_timestamp_s"
        )


def _validate_sampling_scalars(sampling: dict) -> None:
    for field in ("max_frames", "selected_count"):
        require_integer(sampling[field], field=f"sampling.{field}", minimum=1)
    if sampling["source_count"] is not None:
        require_integer(
            sampling["source_count"], field="sampling.source_count", minimum=1
        )
    require_boolean(sampling["coverage_complete"], field="sampling.coverage_complete")
    if sampling["timestamps_complete"] is not None:
        require_boolean(
            sampling["timestamps_complete"], field="sampling.timestamps_complete"
        )
    for index, value in enumerate(sampling["selected_indices"]):
        if value is not None:
            require_integer(
                value, field=f"sampling.selected_indices[{index}]", minimum=0
            )
    for index, value in enumerate(sampling["selected_timestamps_s"]):
        if value is not None:
            require_number(value, field=f"sampling.selected_timestamps_s[{index}]")


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
    indices = {
        "final": [5],
        "sequence": [0, 2, 5],
        "keyframes": [0, 4, 5],
    }[strategy]
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
