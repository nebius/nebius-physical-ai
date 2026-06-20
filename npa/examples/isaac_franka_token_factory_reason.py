#!/usr/bin/env python3
"""SDK example: Isaac Lab Franka scene frames → Token Factory Cosmos3 reasoner.

Local path (no SkyPilot):
  1. Capture frames on a GPU host inside the Isaac Lab image, or use the sample
     frames under docs/assets/hackathon/isaac-franka-lift-cube/.
  2. Export NEBIUS_TOKEN_FACTORY_KEY (or run ``npa configure --token-factory-key``).
  3. Run this script:

     npa/.venv/bin/python npa/examples/isaac_franka_token_factory_reason.py \\
       --input-path docs/assets/hackathon/isaac-franka-lift-cube \\
       --output-path /tmp/isaac-reason-out

See docs/hackathon-isaac-token-factory.md for the full workflow + SkyPilot YAML.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_FRAMES = REPO_ROOT / "docs/assets/hackathon/isaac-franka-lift-cube"
DEFAULT_TASK = (
    "These images are RGB frames from an Isaac Lab simulation of a Franka arm "
    "lifting a cube on a table. Describe the robot, objects, and workspace, then "
    "give a step-by-step manipulation plan."
)
DEFAULT_MODEL = "nvidia/Cosmos3-Super-Reasoner"


def main() -> int:
    parser = argparse.ArgumentParser(description="Reason over Isaac Franka frames via Token Factory.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_SAMPLE_FRAMES,
        help="Folder of scene PNGs (local path or s3:// URI).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/tmp/isaac-franka-reason-out"),
        help="Local directory for scene_reasoning.json.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    input_path = args.input_path
    if not str(input_path).startswith("s3://") and not input_path.exists():
        print(f"input-path not found: {input_path}", file=sys.stderr)
        return 1

    from npa.clients.token_factory import resolve_config
    from npa.workbench.token_factory import reason_scene

    if not resolve_config(require_api_key=False).api_key:
        print(
            "Set NEBIUS_TOKEN_FACTORY_KEY or run npa configure --token-factory-key",
            file=sys.stderr,
        )
        return 1

    result = reason_scene(
        input_path=str(input_path),
        output_path=str(args.output_path),
        task=args.task,
        model=args.model,
        max_images=args.max_images,
        max_tokens=args.max_tokens,
    )
    payload = {
        "status": result.status,
        "model": result.model,
        "image_count": result.image_count,
        "analysis": result.analysis,
        "artifact": str(args.output_path / "scene_reasoning.json"),
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
