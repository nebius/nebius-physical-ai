"""Build a real SOMA-CSV motion clip as a test fixture (pure standard library).

Why this exists
---------------
`npa workbench sonic retargeting run` feeds its input to NVIDIA's upstream
``gear_sonic/data_process/convert_soma_csv_to_motion_lib.py``. That means the
``retargeting`` and ``sonic-locomotion-finetuning`` npa.workflow specs can only be
verified live with a **real SOMA/G1 motion clip**, and this repo deliberately does not
vendor the dual-licensed upstream dataset. Until now those two twins could only ever be
covered plan-only, which is not evidence — and the live matrix skipped them with
"NPA_E2E_SONIC_MOTION_SRC not set".

The upstream loader's contract is small and public (``load_csv_motion``), so a valid
clip can be synthesized:

===================  ============================================================
``joint_pos.csv``    header row + ``T`` rows x ``29`` floats (G1 29-DOF, IsaacLab
                     order, radians)
``body_pos.csv``     header row + ``T`` rows x ``B*3`` floats, reshaped to
                     ``(T, B, 3)``; **body 0 is the pelvis**, whose position becomes
                     ``root_trans_offset``
``body_quat.csv``    header row + ``T`` rows x ``B*4`` floats (``wxyz``), reshaped to
                     ``(T, B, 4)``; body 0's quaternion becomes the root rotation
===================  ============================================================

This module writes exactly that, with smooth, physically plausible trajectories (a
pelvis translating forward at constant height, a yaw that turns gently, and sinusoidal
joint angles bounded well inside G1 limits) so the converter's rotation maths gets real
signal instead of zeros.

Deliberately **standard-library only**: no numpy, torch or pandas. The fixture can
therefore be generated anywhere — including inside the live harness — while the
*conversion* keeps happening in the pod, where the upstream script and its joblib /
pandas / scipy dependencies live.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

#: G1 29-DOF, matching ``NUM_DOF`` in the upstream converter.
NUM_DOF = 29
#: Bodies emitted per frame. The upstream loader infers this with ``reshape(T, -1, 3)``
#: and only reads body 0 (the pelvis), but its docstring documents 14, so emit 14.
NUM_BODIES = 14
DEFAULT_FRAMES = 40
DEFAULT_FPS = 30
#: Peak joint excursion in radians. Small enough to stay inside G1 joint limits.
DEFAULT_JOINT_AMPLITUDE = 0.25
FIXTURE_SCHEMA = "npa.sonic.motion_fixture.v1"

CLIP_FILES = ("joint_pos.csv", "body_pos.csv", "body_quat.csv")


class MotionFixtureError(RuntimeError):
    """Raised when a motion fixture cannot be built or published."""


def _joint_row(frame: int, frames: int, amplitude: float, phase: float) -> list[float]:
    """One frame of 29 joint angles: bounded, smooth, and not all identical."""

    t = frame / max(1, frames - 1)
    return [
        round(amplitude * math.sin(2.0 * math.pi * (t + phase + dof / NUM_DOF)), 6)
        for dof in range(NUM_DOF)
    ]


def _quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Return a unit ``wxyz`` quaternion for a rotation about Z."""

    half = yaw / 2.0
    return (round(math.cos(half), 6), 0.0, 0.0, round(math.sin(half), 6))


def _body_rows(
    frame: int, frames: int, *, height: float, stride: float, yaw_sweep: float
) -> tuple[list[float], list[float]]:
    """Return ``(body_pos_row, body_quat_row)`` for one frame.

    Body 0 is the pelvis: it advances along +X at constant height while yawing gently.
    The remaining bodies are offset from it so the file has realistic width without
    pretending to be a solved kinematic chain (the converter only reads body 0).
    """

    t = frame / max(1, frames - 1)
    yaw = yaw_sweep * math.sin(2.0 * math.pi * t)
    root = (round(stride * t, 6), round(0.02 * math.sin(4.0 * math.pi * t), 6), height)
    quat = _quaternion_from_yaw(yaw)

    positions: list[float] = [*root]
    quats: list[float] = [*quat]
    for body in range(1, NUM_BODIES):
        offset = 0.05 * body
        positions.extend(
            [
                round(root[0] + 0.01 * body, 6),
                round(root[1] + (offset if body % 2 else -offset), 6),
                round(root[2] + 0.03 * body, 6),
            ]
        )
        # Child bodies share the root yaw; a fixture does not need real FK.
        quats.extend(quat)
    return positions, quats


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def build_clip(
    clip_dir: str | Path,
    *,
    frames: int = DEFAULT_FRAMES,
    amplitude: float = DEFAULT_JOINT_AMPLITUDE,
    phase: float = 0.0,
    height: float = 0.78,
    stride: float = 0.9,
    yaw_sweep: float = 0.15,
) -> dict[str, Any]:
    """Write one SOMA-CSV clip directory and return its metadata."""

    if frames < 2:
        raise MotionFixtureError(f"frames must be >= 2 to form a trajectory, got {frames}")
    if amplitude <= 0:
        raise MotionFixtureError(f"amplitude must be positive, got {amplitude}")

    root = Path(clip_dir)
    joint_rows = [_joint_row(frame, frames, amplitude, phase) for frame in range(frames)]
    body_rows = [
        _body_rows(frame, frames, height=height, stride=stride, yaw_sweep=yaw_sweep)
        for frame in range(frames)
    ]

    _write_csv(
        root / "joint_pos.csv",
        [f"dof_{index}" for index in range(NUM_DOF)],
        joint_rows,
    )
    _write_csv(
        root / "body_pos.csv",
        [f"body_{body}_{axis}" for body in range(NUM_BODIES) for axis in ("x", "y", "z")],
        [positions for positions, _ in body_rows],
    )
    _write_csv(
        root / "body_quat.csv",
        [f"body_{body}_{comp}" for body in range(NUM_BODIES) for comp in ("w", "x", "y", "z")],
        [quats for _, quats in body_rows],
    )
    return {
        "clip": root.name,
        "frames": frames,
        "num_dof": NUM_DOF,
        "num_bodies": NUM_BODIES,
        "files": list(CLIP_FILES),
    }


def build_dataset(
    output_dir: str | Path,
    *,
    clips: Sequence[str] = ("walk-forward", "stand-sway"),
    frames: int = DEFAULT_FRAMES,
    fps: int = DEFAULT_FPS,
) -> dict[str, Any]:
    """Write a parent directory of SOMA-CSV clips (the converter's batch mode)."""

    if not clips:
        raise MotionFixtureError("at least one clip name is required")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    built = [
        build_clip(
            root / name,
            frames=frames,
            phase=index / len(clips),
            # The second clip sways in place rather than walking, so the dataset is not
            # N copies of one trajectory.
            stride=0.9 if index == 0 else 0.05,
            yaw_sweep=0.15 if index == 0 else 0.35,
        )
        for index, name in enumerate(clips)
    ]
    return {
        "schema": FIXTURE_SCHEMA,
        "fps": fps,
        "clip_count": len(built),
        "clips": built,
        "output_dir": str(root),
    }


def split_s3_prefix(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/prefix/`` into ``(bucket, prefix-with-trailing-slash)``."""

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise MotionFixtureError(f"expected an s3:// prefix URI, got {uri!r}")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


def upload_dataset(
    output_dir: str | Path, uri: str, *, client: Any | None = None
) -> list[str]:
    """Upload every file under ``output_dir`` to an ``s3://`` prefix."""

    bucket, prefix = split_s3_prefix(uri)
    if client is None:  # pragma: no cover - real S3 only
        import boto3
        from botocore.client import Config

        kwargs: dict[str, Any] = {"config": Config(signature_version="s3v4")}
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        client = boto3.client("s3", **kwargs)

    root = Path(output_dir)
    uploaded: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = prefix + path.relative_to(root).as_posix()
        client.upload_file(str(path), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    if not uploaded:
        raise MotionFixtureError(f"nothing to upload from {root}")
    return uploaded


def build_and_publish(
    *,
    output_dir: str | Path,
    uri: str = "",
    clips: Sequence[str] = ("walk-forward", "stand-sway"),
    frames: int = DEFAULT_FRAMES,
    fps: int = DEFAULT_FPS,
    client: Any | None = None,
) -> dict[str, Any]:
    """Build the dataset locally and optionally upload it."""

    result = build_dataset(output_dir, clips=clips, frames=frames, fps=fps)
    if uri:
        uploaded = upload_dataset(output_dir, uri, client=client)
        result["uri"] = uri
        result["uploaded"] = len(uploaded)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (see scripts/stage-sonic-motion-fixture.sh)."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/npa-motion-fixture")
    parser.add_argument("--uri", default="", help="s3:// prefix to upload the clips to.")
    parser.add_argument(
        "--clips",
        default="walk-forward,stand-sway",
        help="Comma-separated clip directory names.",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args(argv)

    clips = tuple(name.strip() for name in args.clips.split(",") if name.strip())
    try:
        result = build_and_publish(
            output_dir=args.output_dir,
            uri=args.uri,
            clips=clips,
            frames=args.frames,
            fps=args.fps,
        )
    except MotionFixtureError as exc:
        print(f"Error: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
