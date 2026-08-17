"""Decode-based validation for generated LTX-2.5 video.

A file existing at the output path is not evidence that generation worked. The
failure modes worth catching are the ones that still leave a plausible file
behind: a truncated or unreadable container, a clip far shorter than asked for,
a flat single-colour render, and a "video" that is one still frame repeated. So
this decodes the pixels and checks them.

Stdlib plus ffmpeg only, because it is copied verbatim into ``npa-ltx2`` — an
image that deliberately carries no numeric stack — and because the tested code
and the code that runs in the container must be the same code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ARTIFACT_SCHEMA = "npa.workbench.byof.ltx2_5_text_to_video.v1"

#: Frames are decoded down to this square grayscale size before inspection. It
#: is small enough to keep a long clip cheap to check and large enough that a
#: flat frame stays flat and a moving one still differs.
PROBE_SIZE = 32

#: Mean absolute deviation, in 0-255 grayscale levels, below which a frame is
#: treated as a flat fill rather than a picture.
FLAT_FRAME_TOLERANCE = 1.0

#: Mean absolute difference between consecutive frames below which the pair is
#: treated as identical. Codec noise between genuinely distinct frames sits well
#: above this.
STATIC_PAIR_TOLERANCE = 0.5

#: Fraction of consecutive frame pairs that must actually move, and fraction of
#: frames that must carry texture.
#:
#: The whole-clip checks above use `max`, so they only ever asserted "at least
#: one frame is not flat" and "at least one pair differs". One real frame
#: followed by 120 frozen ones passes that — and a two-stage video model
#: producing exactly that is a common, plausible-looking failure, which is the
#: kind this validator exists to catch. Half is deliberately lenient: a real
#: clip sits near 1.0, and a genuinely static shot is not what LTX generates.
MIN_MOVING_PAIR_FRACTION = 0.5
MIN_TEXTURED_FRAME_FRACTION = 0.5


class VideoCheckError(RuntimeError):
    """Raised when a generated clip is missing, unreadable, or degenerate."""


@dataclass
class VideoCheck:
    """What the decode found."""

    path: str
    sha256: str
    size_bytes: int
    codec: str
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    max_frame_deviation: float
    max_frame_delta: float
    moving_pair_fraction: float = 1.0
    textured_frame_fraction: float = 1.0
    capability: str = ""
    checks: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ARTIFACT_SCHEMA,
            "capability": self.capability,
            "video": {
                "path": self.path,
                "sha256": self.sha256,
                "size_bytes": self.size_bytes,
                "codec": self.codec,
                "width": self.width,
                "height": self.height,
                "frame_count": self.frame_count,
                "duration_seconds": self.duration_seconds,
            },
            "decode": {
                "max_frame_deviation": self.max_frame_deviation,
                "max_frame_delta": self.max_frame_delta,
                # The fractions, not just the maxima: a clip that moves once and
                # then freezes has a healthy max_frame_delta, so the maxima alone
                # cannot tell a real generation from a plausible-looking one.
                "moving_pair_fraction": self.moving_pair_fraction,
                "textured_frame_fraction": self.textured_frame_fraction,
                "probe_size": PROBE_SIZE,
            },
            "checks_passed": list(self.checks),
        }


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(argv, capture_output=True, check=False)
    except FileNotFoundError as exc:  # pragma: no cover - environment defect
        raise VideoCheckError(f"{argv[0]} is not installed: {exc}") from exc


def probe_stream(path: Path) -> dict[str, Any]:
    """Return the first video stream ffprobe reports, or raise."""

    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise VideoCheckError(
            f"ffprobe could not read {path}: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    try:
        payload = json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise VideoCheckError(
            f"ffprobe returned unparseable output for {path}"
        ) from exc
    streams = payload.get("streams") or []
    if not streams:
        raise VideoCheckError(f"{path} contains no video stream")
    stream = dict(streams[0])
    stream["_format"] = payload.get("format") or {}
    return stream


def decode_frames(path: Path, *, limit: int = 0) -> list[bytes]:
    """Decode the clip to small grayscale frames.

    The decode is the check: ffmpeg failing here means the container is
    unreadable however healthy its header looked.
    """

    argv = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"scale={PROBE_SIZE}:{PROBE_SIZE}",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
    ]
    if limit:
        argv.extend(["-frames:v", str(limit)])
    argv.append("-")
    proc = _run(argv)
    if proc.returncode != 0:
        raise VideoCheckError(
            f"ffmpeg could not decode {path}: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    stride = PROBE_SIZE * PROBE_SIZE
    raw = proc.stdout
    if len(raw) < stride:
        raise VideoCheckError(f"{path} decoded to no complete frame")
    return [
        raw[start : start + stride] for start in range(0, len(raw) - stride + 1, stride)
    ]


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values)


def _deviation(frame: bytes) -> float:
    mean = _mean(frame)
    return sum(abs(value - mean) for value in frame) / len(frame)


def _delta(first: bytes, second: bytes) -> float:
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def validate_video(
    path: str | Path,
    *,
    min_frames: int = 24,
    capability: str = "",
    max_probe_frames: int = 0,
) -> VideoCheck:
    """Validate a generated clip, or raise :class:`VideoCheckError`."""

    if min_frames < 2:
        # The motion check needs a pair. Accepting 1 previously reached
        # `max(())` and raised ValueError from inside the validator, which a
        # caller catching VideoCheckError would not have seen as a failure.
        raise VideoCheckError(
            f"min_frames must be at least 2 to check for motion (got {min_frames})"
        )
    if max_probe_frames and max_probe_frames < min_frames:
        raise VideoCheckError(
            f"max_probe_frames={max_probe_frames} is below min_frames={min_frames}, "
            "so the check could never pass"
        )

    target = Path(path)
    if not target.is_file():
        raise VideoCheckError(f"no video at {target}")
    payload = target.read_bytes()
    if not payload:
        raise VideoCheckError(f"{target} is empty")

    stream = probe_stream(target)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VideoCheckError(f"{target} reports a {width}x{height} video stream")

    frames = decode_frames(target, limit=max_probe_frames)
    if len(frames) < min_frames:
        raise VideoCheckError(
            f"{target} decoded {len(frames)} frames, below the required {min_frames}"
        )

    deviations = [_deviation(frame) for frame in frames]
    max_deviation = max(deviations)
    if max_deviation < FLAT_FRAME_TOLERANCE:
        raise VideoCheckError(
            f"{target} decoded to flat frames (max deviation {max_deviation:.3f} "
            f"< {FLAT_FRAME_TOLERANCE}); a single-colour render is not a generation"
        )
    textured_fraction = sum(
        1 for value in deviations if value >= FLAT_FRAME_TOLERANCE
    ) / len(deviations)
    if textured_fraction < MIN_TEXTURED_FRAME_FRACTION:
        raise VideoCheckError(
            f"{target} is mostly flat: only {textured_fraction:.1%} of frames carry "
            f"texture (need {MIN_TEXTURED_FRAME_FRACTION:.0%}). A clip that is blank "
            "apart from a few frames is not a generation"
        )

    deltas = [
        _delta(frames[index], frames[index + 1]) for index in range(len(frames) - 1)
    ]
    max_delta = max(deltas)
    if max_delta < STATIC_PAIR_TOLERANCE:
        raise VideoCheckError(
            f"{target} shows no motion (max frame delta {max_delta:.3f} "
            f"< {STATIC_PAIR_TOLERANCE}); one still repeated is not a video"
        )
    moving_fraction = sum(
        1 for value in deltas if value >= STATIC_PAIR_TOLERANCE
    ) / len(deltas)
    if moving_fraction < MIN_MOVING_PAIR_FRACTION:
        raise VideoCheckError(
            f"{target} is mostly frozen: only {moving_fraction:.1%} of consecutive "
            f"frame pairs move (need {MIN_MOVING_PAIR_FRACTION:.0%}). A clip that "
            "moves once and then holds a still is not a video"
        )

    duration = 0.0
    try:
        duration = float((stream.get("_format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    return VideoCheck(
        path=str(target),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        codec=str(stream.get("codec_name") or ""),
        width=width,
        height=height,
        frame_count=len(frames),
        duration_seconds=round(duration, 3),
        max_frame_deviation=round(max_deviation, 4),
        max_frame_delta=round(max_delta, 4),
        moving_pair_fraction=round(moving_fraction, 4),
        textured_frame_fraction=round(textured_fraction, 4),
        capability=capability,
        checks=[
            "container_readable",
            "stream_dimensions_positive",
            f"frame_count_at_least_{min_frames}",
            "frames_not_flat",
            "frames_not_identical",
            f"textured_frames_at_least_{MIN_TEXTURED_FRAME_FRACTION:.0%}",
            f"moving_pairs_at_least_{MIN_MOVING_PAIR_FRACTION:.0%}",
        ],
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "FLAT_FRAME_TOLERANCE",
    "PROBE_SIZE",
    "STATIC_PAIR_TOLERANCE",
    "VideoCheck",
    "VideoCheckError",
    "decode_frames",
    "probe_stream",
    "validate_video",
]
