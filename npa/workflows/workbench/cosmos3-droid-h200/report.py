"""Recompute the H200 execution result from the published numeric rank extracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def _rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def _verify_extracts(root):
    provenance = json.loads((root / "provenance.json").read_text())
    for name, expected in provenance["extract_sha256"].items():
        if Path(name).name != name:
            raise ValueError("extract name must be a basename")
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("numeric extract checksum differs from provenance")


def _rank_rows(rows, rank):
    return [row for row in rows if int(row["rank"]) == rank]


def _validate_sequences(microsteps, updates):
    for rows in [microsteps, updates]:
        if {int(row["rank"]) for row in rows} != set(range(8)):
            raise ValueError("expected exactly eight ranks")
    for rank in range(8):
        sequence = _rank_rows(microsteps, rank)
        if [int(row["microstep"]) for row in sequence] != list(range(1, 161)):
            raise ValueError("missing, duplicate or reordered microstep")
        if any(
            int(row["samples"]) != 32 or not math.isfinite(float(row["loss"]))
            for row in sequence
        ):
            raise ValueError("invalid realized samples or nonfinite loss")
        sequence = _rank_rows(updates, rank)
        if [int(row["update"]) for row in sequence] != list(range(1, 6)):
            raise ValueError("missing, duplicate or reordered optimizer update")
        for row in sequence:
            if int(row["microsteps"]) != 32 or int(row["samples_per_rank"]) != 1024:
                raise ValueError("update does not realize the declared sample count")
            if not math.isfinite(float(row["seconds"])) or float(row["seconds"]) <= 0:
                raise ValueError("invalid update duration")
            if (
                not math.isfinite(float(row["optimizer_seconds"]))
                or float(row["optimizer_seconds"]) <= 0
            ):
                raise ValueError("invalid optimizer operation duration")


def _derive(root):
    _verify_extracts(root)
    microsteps, updates = _rows(root / "microsteps.csv"), _rows(root / "updates.csv")
    _validate_sequences(microsteps, updates)
    durations = [
        max(float(row["seconds"]) for row in updates if int(row["update"]) == update)
        for update in range(1, 6)
    ]
    steady = durations[1:]
    mean = math.fsum(steady) / len(steady)
    variation = statistics.pstdev(steady) / mean
    return {
        "completed_updates": 5,
        "global_samples_per_update": 8192,
        "total_samples": sum(int(row["samples"]) for row in microsteps),
        "max_rank_update_seconds": durations,
        "post_first_mean_seconds": mean,
        "post_first_population_cv": variation,
        "stable_within_declared_window": len(steady) >= 3 and variation <= 0.1,
        "peak_allocated_GiB": max(
            int(row["peak_allocated_bytes"]) for row in microsteps
        )
        / 2**30,
        "conditional_gpu_hours_per_1000_updates": mean * 8 * 1000 / 3600,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(_derive(arguments.evidence_dir), indent=2))
