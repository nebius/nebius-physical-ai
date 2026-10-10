"""Aggregate metrics for evaluation-harness episode records."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def success_rate(successes: Sequence[bool]) -> float:
    """Fraction of successful episodes (0.0 for an empty run)."""
    if not successes:
        return 0.0
    return sum(1 for s in successes if s) / len(successes)


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%).

    Better behaved than the normal approximation at small n and at the
    0/1 boundaries, which is exactly where policy comparisons live.
    """
    if n <= 0:
        return (0.0, 1.0)
    if z <= 0:
        raise ValueError("z must be positive")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode records into a summary mapping."""
    successes = [bool(r["success"]) for r in records]
    n = len(records)
    n_success = sum(successes)
    lo, hi = wilson_interval(n_success, n)
    steps = [int(r["steps"]) for r in records]
    tts = [int(r["time_to_success"]) for r in records if r.get("time_to_success")]
    rewards = [float(r["total_reward"]) for r in records]
    return {
        "episodes": n,
        "successes": n_success,
        "success_rate": success_rate(successes),
        "wilson_95": {"lo": lo, "hi": hi},
        "mean_steps": (sum(steps) / n) if n else 0.0,
        "mean_time_to_success": (sum(tts) / len(tts)) if tts else None,
        "median_time_to_success": _median([float(t) for t in tts]),
        "mean_total_reward": (sum(rewards) / n) if n else 0.0,
    }


__all__ = ["success_rate", "summarize", "wilson_interval"]
