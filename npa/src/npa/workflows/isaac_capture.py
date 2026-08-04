#!/usr/bin/env python3
"""Capture RGB frames from a headless Isaac Lab Franka (or other) task for VLM / Token Factory.

Writes PNGs to a local directory or uploads to an ``s3://`` prefix, and publishes a summary
alongside them.

This lives in the package, not in ``npa/scripts/``, because a `toolRef` runs
``python3 -m npa.workflows.isaac_capture`` inside a pod that has npa installed but no repo
checkout. The retired ``isaac-franka-capture-reason.yaml`` had to say
``NPA repo not found at ${REPO_ROOT}; mount or bake /opt/nebius-physical-ai`` and exit — the
capability was reachable only to operators who had already solved a mounting problem.
``npa/scripts/capture_isaac_lab_scene_frames.py`` stays as a thin shim.

Requires an Isaac Lab container with CUDA (L40S / RTX Pro class GPU).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_TASK = "Isaac-Lift-Cube-Franka-v0"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser, so a guardrail can check toolRef argv against it."""

    parser = argparse.ArgumentParser(description="Capture Isaac Lab scene frames as PNGs.")
    parser.add_argument(
        "--task",
        default=os.environ.get("ISAAC_LAB_TASK", DEFAULT_TASK),
        help="Isaac Lab task id (default: Isaac-Lift-Cube-Franka-v0).",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        required=True,
        help="Local directory or s3:// prefix for PNG frames.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("ISAAC_CAPTURE_MAX_STEPS", "80")),
        help="Simulation steps per episode.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=int(os.environ.get("ISAAC_CAPTURE_MAX_FRAMES", "6")),
        help="Maximum PNG frames to write across the rollout.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes (only the first episode is captured).",
    )
    parser.add_argument(
        "--camera-eye",
        default="",
        help=(
            "Camera position as 'x,y,z' in environment coordinates. Defaults to a pose that "
            "frames a tabletop manipulator at the origin."
        ),
    )
    parser.add_argument(
        "--camera-target",
        default="",
        help="Point the camera looks at, as 'x,y,z'. Defaults to the manipulator's workspace.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Print the resolved settings and exit without starting Isaac Sim.",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _upload_tree(local_dir: Path, output_uri: str) -> dict[str, str]:
    import boto3

    parsed = urlparse(output_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"output-path must be local or s3:// URI, got {output_uri}")
    prefix = parsed.path.strip("/")
    prefix = (prefix + "/") if prefix else ""
    s3 = boto3.client(
        "s3",
        endpoint_url=(
            os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("NEBIUS_S3_ENDPOINT")
            or None
        ),
    )
    uploaded: dict[str, str] = {}
    # PNGs *and* the summary: uploading only the frames left `isaac_capture_summary.json` in the
    # pod, so a consumer could see the frames but never the record of what produced them — and a
    # spec had nothing durable to declare as this stage's output.
    paths = sorted(local_dir.rglob("*.png")) + sorted(local_dir.glob("isaac_capture_summary.json"))
    for path in paths:
        key = prefix + str(path.relative_to(local_dir)).replace("\\", "/")
        s3.upload_file(str(path), parsed.netloc, key)
        uploaded[str(path.relative_to(local_dir))] = f"s3://{parsed.netloc}/{key}"
    return uploaded


#: The scene key the shared frame extractor looks the camera up by. Keeping the name means this
#: module can own the POSE (which is scene-specific) while reusing the extraction (which is not);
#: `test_isaac_capture.py` pins the two in agreement so a rename cannot silently return no frames.
CAPTURE_CAMERA_NAME = "heldout_viz_camera"


#: Where the camera sits and what it points at, for a tabletop manipulator whose base is at the
#: environment origin. The retired template borrowed `_attach_isaac_viz_camera` from the sim2real
#: engine, whose pose was tuned for a different scene: live job 280 rendered six technically
#: perfect frames of bare floor, and the reasoner correctly replied that it could see
#: "a tiled floor ... no visible objects, obstacles, or environmental features". A capture stage
#: that photographs the wrong thing fails silently, which is worse than failing loudly.
DEFAULT_CAMERA_EYE = (1.6, 1.6, 1.3)
DEFAULT_CAMERA_TARGET = (0.4, 0.0, 0.3)


def parse_point(raw: str, default: Sequence[float]) -> tuple[float, float, float]:
    """Parse an 'x,y,z' argument, falling back to ``default`` when it is empty."""

    text = (raw or "").strip()
    if not text:
        return tuple(float(component) for component in default)  # type: ignore[return-value]
    parts = [piece.strip() for piece in text.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"expected three comma-separated numbers, got {raw!r}")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise SystemExit(f"expected three numbers, got {raw!r}") from exc


def look_at_quaternion(
    eye: Sequence[float],
    target: Sequence[float],
    *,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Return the `(w, x, y, z)` rotation that aims a camera at ``target``, world convention.

    Isaac Lab's ``convention="world"`` is the REP-103 frame: the camera looks along its own **+X**
    with **+Z** up (not the OpenGL -Z-forward frame). Getting that wrong does not fail, it just
    photographs somewhere else — live job 281 aimed 90 degrees off and returned six pictures of
    the ground plane receding to a horizon, which the reasoner described as "a tall building with
    a grid-patterned facade".

    Pure arithmetic on purpose: framing is the part of this stage that can be checked without a
    simulator, and `test_isaac_capture.py` rotates the +X axis to prove it lands on the target.
    """

    def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    def _norm(v: Sequence[float]) -> list[float]:
        length = math.sqrt(sum(component * component for component in v))
        if length < 1e-9:
            raise ValueError("cannot normalise a zero-length vector")
        return [component / length for component in v]

    forward = _norm([target[i] - eye[i] for i in range(3)])  # camera +X
    # World up, projected off the view direction, is the camera's +Z.
    dot_up = sum(world_up[i] * forward[i] for i in range(3))
    up = [world_up[i] - dot_up * forward[i] for i in range(3)]
    if math.sqrt(sum(c * c for c in up)) < 1e-6:
        # Looking straight up or down: any perpendicular will do, pick a stable one.
        up = [1.0, 0.0, 0.0]
        dot_up = sum(up[i] * forward[i] for i in range(3))
        up = [up[i] - dot_up * forward[i] for i in range(3)]
    up = _norm(up)
    left = _cross(up, forward)  # right-handed: +Y = +Z x +X

    m = [
        [forward[0], left[0], up[0]],
        [forward[1], left[1], up[1]],
        [forward[2], left[2], up[2]],
    ]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return (w, x, y, z)


def _attach_capture_camera(
    env_cfg: Any,
    *,
    eye: Sequence[float],
    target: Sequence[float],
    width: int = 512,
    height: int = 512,
) -> None:
    """Attach a camera that actually frames the task.

    512x512 rather than the sim2real engine's 128x128: these frames are read by a VLM, and a
    128-pixel thumbnail of a robot arm is not something a reasoner can plan from.
    """

    import isaaclab.sim as sim_utils

    try:
        from isaaclab.sensors import TiledCameraCfg as _CameraCfg
    except ImportError:  # pragma: no cover - older Isaac Lab
        from isaaclab.sensors import CameraCfg as _CameraCfg

    camera_cfg = _CameraCfg(
        prim_path="{ENV_REGEX_NS}/NpaCaptureCamera",
        offset=_CameraCfg.OffsetCfg(
            pos=tuple(float(component) for component in eye),
            rot=look_at_quaternion(eye, target),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=width,
        height=height,
    )
    setattr(env_cfg.scene, CAPTURE_CAMERA_NAME, camera_cfg)


def _capture_frames(
    *,
    task: str,
    output_dir: Path,
    max_steps: int,
    max_frames: int,
    episodes: int,
    camera_eye: Sequence[float] = DEFAULT_CAMERA_EYE,
    camera_target: Sequence[float] = DEFAULT_CAMERA_TARGET,
    publish: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Capture frames, then hand the summary to ``publish`` BEFORE the simulator shuts down.

    ``simulation_app.close()`` tears the process down rather than returning, so anything left
    for the caller to do afterwards silently never happens: live job 278 wrote all six frames,
    exited 0, uploaded nothing, and the next stage failed with "No scene images found". Work
    that must survive the run has to happen before that call.
    """

    from npa.workflows.sim2real.engine import (
        _heldout_render_step_indices,
        _isaac_extract_rgb_frame,
        _write_render_png,
    )

    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise SystemExit(f"isaaclab is required in the Isaac Lab image: {exc}") from exc

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for Isaac Lab frame capture")

    # AppLauncher FIRST, then the task packages. Isaac Lab's modules reach into the Omniverse
    # kit runtime (`pxr`, the USD bindings) at import time, and that runtime does not exist
    # until the app is launched. Importing `isaaclab_tasks` before this line fails with
    # `ModuleNotFoundError: No module named 'pxr'` — live job 270, the first time this code had
    # ever run, because the template that owned it required a repo mounted into the pod and so
    # could not be exercised.
    launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    device = "cuda:0"
    output_dir.mkdir(parents=True, exist_ok=True)
    render_steps = _heldout_render_step_indices(max_steps, max_frames=max_frames)
    frames_written: list[str] = []
    started = time.time()

    try:
        env_cfg = parse_env_cfg(task, device=device, num_envs=1)
        _attach_capture_camera(env_cfg, eye=camera_eye, target=camera_target)
        env = gym.make(task, cfg=env_cfg)
        for episode in range(episodes):
            env.reset()
            if episode > 0:
                continue
            for step in range(max_steps):
                actions = torch.as_tensor(env.action_space.sample(), device=device, dtype=torch.float32)
                env.step(actions)
                if step in render_steps:
                    frame = _isaac_extract_rgb_frame(env, env_index=0)
                    if frame is not None:
                        name = f"frame_{len(frames_written):02d}.png"
                        _write_render_png(output_dir / name, frame)
                        frames_written.append(name)
                        print(f"ISAAC_CAPTURE_FRAME {name} step={step}", flush=True)
        env.close()

        summary: dict[str, object] = {
            "status": "success" if frames_written else "failed",
            "task": task,
            "episodes": episodes,
            "max_steps": max_steps,
            "max_frames": max_frames,
            "frames": frames_written,
            "output_dir": str(output_dir),
            "duration_seconds": round(time.time() - started, 2),
        }
        (output_dir / "isaac_capture_summary.json").write_text(json.dumps(summary, indent=2))
        if not frames_written:
            raise SystemExit("No frames captured — check task cameras and GPU rendering.")
        if publish is not None:
            publish(summary)
    finally:
        simulation_app.close()

    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.render_only:
        print(
            json.dumps(
                {
                    "task": args.task,
                    "output_path": args.output_path,
                    "max_steps": args.max_steps,
                    "max_frames": args.max_frames,
                    "episodes": args.episodes,
                },
                indent=2,
            )
        )
        return 0

    output_path = args.output_path.strip()
    parsed = urlparse(output_path)
    if parsed.scheme == "s3":
        local_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"isaac-capture-{int(time.time())}"
    else:
        local_dir = Path(output_path)
        local_dir.mkdir(parents=True, exist_ok=True)

    def publish(summary: dict[str, object]) -> None:
        if parsed.scheme == "s3":
            # Includes isaac_capture_summary.json, so the record travels with the frames.
            summary["uploads"] = _upload_tree(local_dir, output_path)
            summary["output_path"] = output_path.rstrip("/") + "/"
        print(json.dumps(summary, indent=2), flush=True)

    _capture_frames(
        task=args.task,
        output_dir=local_dir,
        max_steps=args.max_steps,
        max_frames=args.max_frames,
        episodes=args.episodes,
        camera_eye=parse_point(args.camera_eye, DEFAULT_CAMERA_EYE),
        camera_target=parse_point(args.camera_target, DEFAULT_CAMERA_TARGET),
        publish=publish,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
