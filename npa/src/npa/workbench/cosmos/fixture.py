"""Generate a legally clean Cosmos Transfer 2.5 inference fixture.

The fixture contains no copied media. FFmpeg synthesizes every frame from color,
grid, and moving-shape filters authored here, and this module writes the smallest
real edge-control spec accepted by the pinned upstream inference schema.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 16
DEFAULT_FRAMES = 93
DEFAULT_STEPS = 4


def generate_fixture(
    output_dir: Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    frames: int = DEFAULT_FRAMES,
    num_steps: int = DEFAULT_STEPS,
) -> dict[str, Any]:
    """Create a procedural MP4 and matching multi-step edge-control JSON spec."""

    if min(width, height, fps, frames, num_steps) <= 0:
        raise ValueError("fixture dimensions, rate, frames, and steps must be positive")
    if num_steps < 2:
        raise ValueError("Cosmos Transfer live validation must be multi-step")

    output_dir.mkdir(parents=True, exist_ok=True)
    video = output_dir / "npa-procedural-input.mp4"
    spec = output_dir / "npa-procedural-edge-spec.json"
    duration = frames / fps

    # The source is a generated color plane. A grid and two independently moving
    # boxes create stable edges and motion for on-the-fly Canny conditioning.
    filters = (
        "drawgrid=width=160:height=120:thickness=3:color=white@0.35,"
        "drawbox=x='mod(t*150,iw-240)':y='ih/3':w=240:h=150:"
        "color=0x36d399@1:t=fill,"
        "drawbox=x='iw-180-mod(t*90,iw-360)':y='2*ih/3':w=180:h=100:"
        "color=0xf59e0b@1:t=fill"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x172033:s={width}x{height}:r={fps}:d={duration:.6f}",
            "-vf",
            filters,
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )

    payload: dict[str, Any] = {
        "name": "npa_procedural_edge",
        "prompt": (
            "A photorealistic mobile robot moving through a clean industrial "
            "workspace, natural lighting, detailed realistic materials"
        ),
        "video_path": str(video.resolve()),
        "guidance": 3,
        "num_steps": num_steps,
        "seed": 20260801,
        "max_frames": frames,
        "num_video_frames_per_chunk": frames,
        "resolution": str(height),
        "keep_input_resolution": True,
        "edge": {
            "control_weight": 1.0,
            "preset_edge_threshold": "medium",
        },
    }
    spec.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "video_path": str(video),
        "spec_path": str(spec),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_seconds": duration,
        "num_steps": num_steps,
        "provenance": "repository-authored deterministic FFmpeg lavfi filters",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_fixture(
                args.output_dir,
                width=args.width,
                height=args.height,
                fps=args.fps,
                frames=args.frames,
                num_steps=args.num_steps,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
