#!/usr/bin/env python3
"""Minimal BYOF Isaac Lab training wrapper.

The wrapper records a sentinel proving that the custom entrypoint ran, then
delegates to Isaac Lab's upstream RSL-RL train.py with the original arguments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time


UPSTREAM_TRAIN_SCRIPT = Path(
    os.environ.get(
        "ISAAC_LAB_UPSTREAM_TRAIN",
        "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py",
    )
)
SENTINEL_PATH = Path(os.environ.get("BYOF_SENTINEL_PATH", "/workspace/output/byof_sentinel.json"))


def main() -> None:
    args, hydra_args = _parse_args(sys.argv[1:])
    run_id = os.environ.get("NPA_ISAAC_LAB_RUN_ID") or os.environ.get("BYOF_RUN_ID") or args.run_name
    sentinel = {
        "byof": True,
        "script": "custom_train.py",
        "script_path": str(Path(__file__).resolve()),
        "upstream_train_script": str(UPSTREAM_TRAIN_SCRIPT),
        "argv": sys.argv,
        "run_id": run_id,
        "task": args.task,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "headless": args.headless,
        "hydra_args": hydra_args,
        "created_unix": round(time.time(), 3),
    }
    _write_json(SENTINEL_PATH, sentinel)

    output_dir = os.environ.get("NPA_ISAAC_LAB_OUTPUT_DIR", "")
    if output_dir:
        _write_json(Path(output_dir) / "byof_sentinel.json", sentinel)

    if not UPSTREAM_TRAIN_SCRIPT.is_file():
        raise SystemExit(f"upstream Isaac Lab train.py not found: {UPSTREAM_TRAIN_SCRIPT}")

    sys.argv = [str(UPSTREAM_TRAIN_SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(UPSTREAM_TRAIN_SCRIPT), run_name="__main__")


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default="")
    parser.add_argument("--num_envs", default="")
    parser.add_argument("--max_iterations", default="")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--run_name", default="")
    return parser.parse_known_args(argv)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
