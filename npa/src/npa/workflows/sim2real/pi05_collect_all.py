"""Collect train/validation/gold experts with disjoint object/scene identities."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from npa.workflows.sim2real.workflow_io import publish_component_record


SPLITS = {
    "train": {
        "seed_offset": 0,
        "object_identity": "isaac-multicolor-cube-040m",
        "object_scale": (0.8, 0.8, 0.8),
        "scene_id": "surface-train-a",
    },
    "validation": {
        "seed_offset": 10_000,
        "object_identity": "isaac-multicolor-cube-050m",
        "object_scale": (1.0, 1.0, 1.0),
        "scene_id": "surface-validation-b",
    },
    "gold": {
        "seed_offset": 20_000,
        "object_identity": "isaac-multicolor-cube-060m",
        "object_scale": (1.2, 1.2, 1.2),
        "scene_id": "surface-gold-c",
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. Args: None. Returns: Parser. Raises: None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root-uri", required=True)
    parser.add_argument("--episodes-per-split", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--object-usd", default="")
    parser.add_argument("--component-root-uri", required=True)
    return parser


def _collector_argv(args: argparse.Namespace, split: str) -> list[str]:
    config = SPLITS[split]
    return [
        sys.executable,
        "-m",
        "npa.workflows.sim2real.pi05_isaac",
        "--output-uri",
        f"{args.output_root_uri.rstrip('/')}/{split}/",
        "--episodes",
        str(args.episodes_per_split),
        "--seed",
        str(args.seed + int(config["seed_offset"])),
        "--object-identity",
        str(config["object_identity"]),
        "--object-usd",
        args.object_usd,
        "--object-scale",
        ",".join(str(value) for value in config["object_scale"]),
        "--scene-id",
        str(config["scene_id"]),
    ]


def _run_collector(args: argparse.Namespace, split: str) -> dict[str, Any]:
    process = subprocess.run(
        _collector_argv(args, split), check=False, capture_output=True, text=True
    )
    if process.returncode:
        message = process.stderr.strip().splitlines()[-1:] or ["no stderr"]
        raise RuntimeError(f"{split} collector failed: {message[0]}")
    for line in reversed(process.stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if result.get("schema") == "npa.sim2real.pi05.expert_collection.v1":
            return result
    raise RuntimeError(f"{split} collector emitted no collection report")


def main(argv: list[str] | None = None) -> int:
    """Collect all splits. Args: argv. Returns: Exit status. Raises: RuntimeError."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes_per_split < 1:
        parser.error("--episodes-per-split must be positive")
    results = {split: _run_collector(args, split) for split in SPLITS}
    publish_component_record(
        root_uri=args.component_root_uri,
        stage=2,
        name="pi05_expert_collection",
        tier="WORKS",
        require_gpu=True,
        evidence="Collected physics-only released placements with dense exterior and moving-wrist observations.",
        artifacts={
            split: f"{args.output_root_uri.rstrip('/')}/{split}/collection.json"
            for split in SPLITS
        },
    )
    print(json.dumps(results, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
