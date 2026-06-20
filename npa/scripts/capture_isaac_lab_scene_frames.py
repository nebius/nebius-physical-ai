#!/usr/bin/env python3
"""Capture RGB frames from a headless Isaac Lab Franka (or other) task for VLM / Token Factory.

Writes PNGs to a local directory or uploads to an ``s3://`` prefix. Used by the
Isaac + Token Factory hackathon workflow and ``docs/hackathon-isaac-token-factory.md``.

Requires an Isaac Lab container with CUDA (L40S / RTX Pro class GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_TASK = "Isaac-Lift-Cube-Franka-v0"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        "--render-only",
        action="store_true",
        help="Print the resolved settings and exit without starting Isaac Sim.",
    )
    return parser.parse_args(argv)


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
    for path in sorted(local_dir.rglob("*.png")):
        key = prefix + str(path.relative_to(local_dir)).replace("\\", "/")
        s3.upload_file(str(path), parsed.netloc, key)
        uploaded[str(path.relative_to(local_dir))] = f"s3://{parsed.netloc}/{key}"
    return uploaded


def _capture_frames(
    *,
    task: str,
    output_dir: Path,
    max_steps: int,
    max_frames: int,
    episodes: int,
) -> dict[str, object]:
    from npa.workflows.sim2real.engine import (
        _attach_isaac_viz_camera,
        _heldout_render_step_indices,
        _isaac_extract_rgb_frame,
        _write_render_png,
    )

    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise SystemExit(f"isaaclab is required in the Isaac Lab image: {exc}") from exc

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for Isaac Lab frame capture")

    launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = launcher.app
    device = "cuda:0"
    output_dir.mkdir(parents=True, exist_ok=True)
    render_steps = _heldout_render_step_indices(max_steps, max_frames=max_frames)
    frames_written: list[str] = []
    started = time.time()

    try:
        env_cfg = parse_env_cfg(task, device=device, num_envs=1)
        _attach_isaac_viz_camera(env_cfg)
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
    finally:
        simulation_app.close()

    summary = {
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

    summary = _capture_frames(
        task=args.task,
        output_dir=local_dir,
        max_steps=args.max_steps,
        max_frames=args.max_frames,
        episodes=args.episodes,
    )

    if parsed.scheme == "s3":
        uploads = _upload_tree(local_dir, output_path)
        summary["uploads"] = uploads
        summary["output_path"] = output_path.rstrip("/") + "/"
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
