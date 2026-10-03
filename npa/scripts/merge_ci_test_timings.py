"""Merge pytest module-duration artifacts emitted by parallel CI shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def merge_timings(paths: list[Path]) -> dict[str, float]:
    """Sum module-duration maps from successful CI shards.

    Args:
        paths: Timing JSON files from non-overlapping shards.
    Returns:
        Sorted module-duration mapping.
    Raises:
        ValueError: No artifacts exist or a duration is non-positive.
    """

    if not paths:
        raise ValueError("at least one CI timing artifact is required")
    merged: dict[str, float] = defaultdict(float)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for module, seconds in payload.items():
            duration = float(seconds)
            if duration <= 0:
                raise ValueError(f"{path}: duration for {module} must be positive")
            merged[module] += duration
    return dict(sorted(merged.items()))


def main() -> int:
    """Merge input timing artifacts into one reviewable JSON file.

    Args:
        None.
    Returns:
        Process exit status.
    Raises:
        ValueError: Inputs contain no valid positive timings.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    paths = sorted(arguments.input_directory.glob("*.json"))
    merged = merge_timings(paths)
    arguments.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
