#!/usr/bin/env python3
"""Fail closed on retained fresh-browser LeIsaac smoothness evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("single_fast", "dual_slow"))
    parser.add_argument("trials", nargs="+", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).parents[1]
        / "tests/baselines/leisaac_latency_e09df47.json",
    )
    args = parser.parse_args()
    if len(args.trials) < 6:
        parser.error("at least six retained fresh-browser trials are required")
    baseline = _load(args.baseline)
    trials = [_load(path) for path in args.trials]
    if any(item.get("view_mode") != args.mode for item in trials):
        raise SystemExit("trial view_mode does not match the requested gate")

    fps_floor = 20.0 if args.mode == "single_fast" else 18.0
    causal_p50 = []
    causal_p95 = []
    failures: list[str] = []
    for index, trial in enumerate(trials, 1):
        summary = trial.get("summary", {})
        causal = summary.get("primary_input_to_causal_frame_painted_ms", {})
        frames = summary.get("frame_stages", {}).get("workspace", {})
        fps = float(frames.get("delivered_fps") or 0)
        p50 = float(causal.get("p50") or 0)
        p95 = float(causal.get("p95") or 0)
        interframe = frames.get("inter_frame_ms", {})
        stalls = int(frames.get("stalls_250ms") or 0)
        causal_p50.append(p50)
        causal_p95.append(p95)
        if fps < fps_floor:
            failures.append(f"trial {index}: primary FPS {fps:.3f} < {fps_floor:.1f}")
        if interframe and float(interframe.get("p95") or 1e9) > 75:
            failures.append(f"trial {index}: inter-frame p95 exceeds 75 ms")
        if stalls:
            failures.append(f"trial {index}: {stalls} active stalls >=250 ms")
        secondary_frames = int(
            summary.get("frame_stages", {}).get("overview", {}).get("delivered_frames") or 0
        )
        if args.mode == "single_fast" and secondary_frames:
            failures.append(f"trial {index}: secondary work occurred in Fast single")

    median_p50 = statistics.median(causal_p50)
    median_p95 = statistics.median(causal_p95)
    prior = baseline[args.mode]
    regression = 1.0 + float(baseline["maximum_regression_fraction"])
    if args.mode == "single_fast":
        if median_p50 > min(100.0, float(prior["causal_p50_ms"]) * regression):
            failures.append(f"median causal p50 {median_p50:.3f} ms failed")
        if median_p95 > min(180.0, float(prior["causal_p95_ms"]) * regression):
            failures.append(f"median causal p95 {median_p95:.3f} ms failed")
    elif median_p50 > float(prior["causal_p50_ms"]) * regression or median_p95 > float(prior["causal_p95_ms"]) * regression:
        failures.append("Dual slow causal latency regressed by more than 10%")
    print(json.dumps({"mode": args.mode, "trials": len(trials), "median_p50_ms": median_p50, "median_p95_ms": median_p95, "failures": failures}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
