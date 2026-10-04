"""Public PyAV replacement for the removed TorchVision video reader.

SeedVR2's pinned upstream entrypoint only needs full-file RGB decoding, the
decoded frame rate, and ``TCHW`` output.  TorchVision 0.28's public wheel no
longer ships the implementation imported by ``torchvision.io.video``.  Keep
this adapter deliberately narrower than that deprecated API.
"""

from __future__ import annotations

from os import PathLike
from typing import Any


def read_video(
    filename: str | PathLike[str],
    start_pts: int | float = 0,
    end_pts: int | float | None = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> tuple[Any, Any, dict[str, float]]:
    """Decode one complete video as uint8 RGB frames.

    The signature matches the subset used by pinned SeedVR2.  Partial-range
    decoding is rejected rather than approximated.
    """

    if start_pts != 0 or end_pts is not None:
        raise ValueError("SeedVR2's video adapter supports full-file decoding only")
    if pts_unit not in {"pts", "sec"}:
        raise ValueError("pts_unit must be 'pts' or 'sec'")
    if output_format not in {"THWC", "TCHW"}:
        raise ValueError("output_format must be 'THWC' or 'TCHW'")

    import av
    import numpy as np
    import torch

    with av.open(str(filename), mode="r") as container:
        if not container.streams.video:
            raise RuntimeError("video contains no video stream")
        stream = container.streams.video[0]
        frame_rate = stream.average_rate
        frames = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]

    if frame_rate is None or float(frame_rate) <= 0:
        raise RuntimeError("video stream has no positive average frame rate")
    if not frames:
        raise RuntimeError("video contains no decodable frames")

    video = torch.from_numpy(np.stack(frames, axis=0))
    if output_format == "TCHW":
        video = video.permute(0, 3, 1, 2).contiguous()
    audio = torch.empty((1, 0), dtype=torch.float32)
    return video, audio, {"video_fps": float(frame_rate)}
