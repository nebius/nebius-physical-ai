"""Verify compressed camera timelines using a frozen encoder independent of the producer."""

from __future__ import annotations

from fractions import Fraction
from itertools import zip_longest
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import av
import numpy as np


def _encode(arrays, output):
    height, width = arrays[0].shape[1:3]
    width *= len(arrays)
    command = [
        "ffmpeg",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "25",
        "-i",
        "pipe:",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-g",
        "2",
        str(output),
    ]
    frames = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=2)
    result = subprocess.run(command, input=frames.tobytes(), capture_output=True)
    if result.returncode:
        raise ValueError(
            "independent reference encoding failed: " + result.stderr.decode()[-500:]
        )


def reference_frames(arrays):
    """Encode verified raw camera arrays and yield independently decoded RGB pixels.

    Args: arrays: One camera array, or multiple arrays joined horizontally per timestep.
    Returns: Iterator of uint8 RGB arrays using fixed 25 Hz H.264 CRF23/GOP2 settings.
    Raises: ValueError, OSError: Reference encoding or decoding fails.
    """
    with TemporaryDirectory(prefix="robot-video-reference-") as directory:
        output = Path(directory) / "reference.mp4"
        _encode(arrays, output)
        with av.open(str(output)) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")


def verify_video(path: Path, arrays: list) -> int:
    """Match every compressed image and timestamp to independently encoded raw evidence.

    Args: path: Candidate MP4. arrays: Verified raw camera arrays in display order.
    Returns: Number of exactly matched decoded frames.
    Raises: ValueError, AssertionError, OSError: Frames, streams or timing differ.
    """
    count = 0
    with av.open(str(path)) as container:
        if len(container.streams) != 1 or len(container.streams.video) != 1:
            raise ValueError("video must contain exactly one video stream")
        expected = reference_frames(arrays)
        try:
            for index, (frame, pixels) in enumerate(
                zip_longest(container.decode(video=0), expected)
            ):
                if frame is None or pixels is None:
                    raise ValueError("compressed video frame count differs")
                if frame.pts is None or frame.pts * frame.time_base != Fraction(
                    index, 25
                ):
                    raise ValueError("compressed video timestamp differs")
                np.testing.assert_array_equal(
                    frame.to_ndarray(format="rgb24"),
                    pixels,
                    err_msg="compressed camera timeline differs",
                )
                count += 1
        finally:
            expected.close()
    if count != len(arrays[0]):
        raise ValueError("compressed video frame count differs")
    return count
