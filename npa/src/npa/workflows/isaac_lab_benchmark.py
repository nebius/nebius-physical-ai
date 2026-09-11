"""Matched-workload comparison for Isaac Lab runtime generations.

This module deliberately compares only records with identical workload and GPU
signatures.  It reports wall-clock duration and final reward separately; neither
is converted into an invented simulator-steps/s metric because RSL-RL iterations
can contain different rollout lengths across task configurations.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "npa.isaac_lab.generation_benchmark.v1"
MATCH_FIELDS = ("task", "num_envs", "max_iterations", "hardware_model", "gpu_count")
PAIR_FIELDS = ("seed", "repetition", "cache_state", "driver_version", "runtime_version")


def _median(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values]
    return round(statistics.median(items), 6) if items else None


def compare_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate matched records and compare Isaac Lab 2 (baseline) with 3."""

    if not records or any(item.get("status") != "success" for item in records):
        raise ValueError("benchmark campaigns must contain only successful records")
    usable = records
    generations = {str(item.get("generation")) for item in usable}
    if generations != {"2", "3"}:
        raise ValueError("successful records must contain generations '2' and '3'")
    signatures = {tuple(item.get(field) for field in MATCH_FIELDS) for item in usable}
    if len(signatures) != 1:
        raise ValueError(f"records are not matched on {', '.join(MATCH_FIELDS)}")
    for item in usable:
        duration = item.get("duration_seconds")
        digest = str(item.get("image_digest") or "")
        if not isinstance(duration, (int, float)) or float(duration) <= 0:
            raise ValueError("every successful record needs positive duration_seconds")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError("every successful record needs an immutable image_digest")

    by_generation = {
        generation: [
            item for item in usable if str(item.get("generation")) == generation
        ]
        for generation in ("2", "3")
    }
    for generation, items in by_generation.items():
        if len({str(item["image_digest"]) for item in items}) != 1:
            raise ValueError(f"generation {generation} has multiple image digests")
    pair_sets = {
        generation: {tuple(item.get(field) for field in PAIR_FIELDS) for item in items}
        for generation, items in by_generation.items()
    }
    if len(by_generation["2"]) != len(by_generation["3"]):
        raise ValueError(
            "generation campaigns must have equal successful sample counts"
        )
    if pair_sets["2"] != pair_sets["3"] or any(
        any(value in (None, "") for value in pair) for pair in pair_sets["2"]
    ):
        raise ValueError(f"records are not paired on {', '.join(PAIR_FIELDS)}")
    baseline_duration = _median(item["duration_seconds"] for item in by_generation["2"])
    candidate_duration = _median(
        item["duration_seconds"] for item in by_generation["3"]
    )
    assert baseline_duration is not None and candidate_duration is not None
    reduction = round(
        100.0 * (baseline_duration - candidate_duration) / baseline_duration, 3
    )
    speedup = round(baseline_duration / candidate_duration, 6)

    baseline_reward = _median(
        item["mean_reward"]
        for item in by_generation["2"]
        if isinstance(item.get("mean_reward"), (int, float))
    )
    candidate_reward = _median(
        item["mean_reward"]
        for item in by_generation["3"]
        if isinstance(item.get("mean_reward"), (int, float))
    )
    reward_delta = (
        round(candidate_reward - baseline_reward, 6)
        if baseline_reward is not None and candidate_reward is not None
        else None
    )
    signature = dict(zip(MATCH_FIELDS, next(iter(signatures)), strict=True))
    return {
        "schema": SCHEMA,
        "comparison": "isaac-lab-3-vs-isaac-lab-2",
        "method": "median wall-clock duration over matched successful runs",
        "workload": signature,
        "paired_protocol": {
            "fields": list(PAIR_FIELDS),
            "pairs": len(pair_sets["2"]),
        },
        "baseline": {
            "generation": "2",
            "versions": sorted(
                {str(item.get("isaac_lab_version")) for item in by_generation["2"]}
            ),
            "image_digests": sorted(
                {str(item["image_digest"]) for item in by_generation["2"]}
            ),
            "samples": len(by_generation["2"]),
            "median_duration_seconds": baseline_duration,
            "median_mean_reward": baseline_reward,
        },
        "candidate": {
            "generation": "3",
            "versions": sorted(
                {str(item.get("isaac_lab_version")) for item in by_generation["3"]}
            ),
            "image_digests": sorted(
                {str(item["image_digest"]) for item in by_generation["3"]}
            ),
            "samples": len(by_generation["3"]),
            "median_duration_seconds": candidate_duration,
            "median_mean_reward": candidate_reward,
        },
        "measured": {
            "duration_reduction_percent": reduction,
            "duration_speedup_ratio": speedup,
            "median_reward_delta": reward_delta,
        },
        "limitations": [
            "Wall-clock duration includes runtime startup and training.",
            "Reward is task- and seed-dependent and is not a simulator throughput measure.",
            "Results apply only to the matched workload and hardware shown above.",
        ],
    }


def compare_files(paths: Iterable[str | Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records.extend(payload if isinstance(payload, list) else [payload])
    return compare_records(records)


__all__ = ["MATCH_FIELDS", "PAIR_FIELDS", "SCHEMA", "compare_files", "compare_records"]
