#!/usr/bin/env python3
"""Extract a single observation from a LeRobotDataset and save as JSON.

Usage:
    python generate_observation.py \
        --dataset lerobot/aloha_sim_transfer_cube_human \
        --output demo_obs.json
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a sample observation from a LeRobotDataset.")
    parser.add_argument("--dataset", required=True, help="HF dataset repo ID.")
    parser.add_argument("--index", type=int, default=0, help="Frame index to extract.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("Error: lerobot is not installed. Run: pip install lerobot", file=sys.stderr)
        return 1

    print(f"Loading dataset: {args.dataset}")
    ds = LeRobotDataset(args.dataset)
    frame = ds[args.index]

    # Convert tensors to lists for JSON serialization
    observation: dict = {}
    for key, value in frame.items():
        if hasattr(value, "tolist"):
            observation[key] = value.tolist()
        elif hasattr(value, "numpy"):
            observation[key] = value.numpy().tolist()
        else:
            observation[key] = value

    with open(args.output, "w") as f:
        json.dump(observation, f, indent=2)

    print(f"Saved observation (frame {args.index}) to {args.output}")
    print(f"Keys: {list(observation.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
