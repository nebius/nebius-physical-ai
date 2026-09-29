"""Aggregate matched FA2/FA4 SDXL runs without hiding slower cases or block variance."""

import argparse
import json
import math
from pathlib import Path
import statistics


def _signature(report):
    environment = report["environment"]
    packages = {
        name: version
        for name, version in environment["packages"].items()
        if name not in ("flash-attn", "flash-attn-4")
    }
    return {
        "environment": {
            key: value
            for key, value in environment.items()
            if key not in ("image_id", "packages")
        },
        "packages": packages,
        "model": report["model"],
        "model_revision": report["model_revision"],
        "steps": report["steps"],
        "repeats": report["repeats"],
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key not in ("samples", "median_seconds")
            }
            for case in report["cases"]
        ],
    }


def _validate(reports):
    groups = {"fa2": [], "fa4": []}
    if not reports:
        raise ValueError("Provide both FA2 and FA4 reports")
    signature = _signature(reports[0])
    for report in reports:
        if report["status"] != "passed" or report["instrumented_timing"]:
            raise ValueError("Only complete, uninstrumented runs can be compared")
        if _signature(report) != signature:
            raise ValueError("GPU, software stack, model or generation settings differ")
        if report["backend"] not in groups:
            raise ValueError("The comparison requires standalone FA2 and FA4")
        groups[report["backend"]].append(report)
    for backend, runs in groups.items():
        if len(runs) < 2:
            raise ValueError(f"Provide at least two separate blocks for {backend}")
        if len({run["environment"]["image_id"] for run in runs}) != 1:
            raise ValueError(f"Do not mix image identities for {backend}")
        if len({tuple(run["tile"] or ()) for run in runs}) != 1:
            raise ValueError("Compare each tile candidate separately")
        if len({run.get("tuning") for run in runs}) != 1:
            raise ValueError("Compare each tuning profile separately")
        if any(
            run["environment"]["packages"] != runs[0]["environment"]["packages"]
            for run in runs
        ):
            raise ValueError("Do not mix backend versions within a comparison")
    return groups


def _statistics(runs, index):
    blocks = [
        [sample["seconds"] for sample in run["cases"][index]["samples"]] for run in runs
    ]
    if any(
        len(block) != run["repeats"] for block, run in zip(blocks, runs, strict=True)
    ):
        raise ValueError("Sample count differs from the declared repeat count")
    if any(
        not math.isfinite(value) or value <= 0 for block in blocks for value in block
    ):
        raise ValueError("Timings must be finite and positive")
    medians = [statistics.median(block) for block in blocks]
    return {
        "median_seconds": statistics.median(medians),
        "block_medians": medians,
        "minimum_seconds": min(min(block) for block in blocks),
        "maximum_seconds": max(max(block) for block in blocks),
    }


def comparison(reports):
    """Summarize like-for-like generation timings with explicit comparison direction.

    Args:
        reports: Complete reports from interleaved FA2/FA4 container runs.
    Returns:
        Per-case timings, block medians and FA2-time / FA4-time speed ratios.
    Raises:
        ValueError: Runs differ in settings or contain insufficient or invalid samples.
    """
    groups = _validate(reports)
    cases = []
    for index, case in enumerate(reports[0]["cases"]):
        measured = {name: _statistics(runs, index) for name, runs in groups.items()}
        ratio = measured["fa2"]["median_seconds"] / measured["fa4"]["median_seconds"]
        cases.append(
            {"name": case["name"], **measured, "fa2_time_over_fa4_time": ratio}
        )
    return {
        "schema_version": 1,
        "cases": cases,
        "fa4_tile": groups["fa4"][0]["tile"],
        "fa4_tuning": groups["fa4"][0].get("tuning"),
        "images": {
            name: runs[0]["environment"]["image_id"] for name, runs in groups.items()
        },
        "interpretation": "Ratios above 1 favor FA4; below 1 favor FA2. "
        "Block variation is descriptive, not a significance test.",
    }


def main():
    """Write a checked comparison and print all cases, including regressions.

    Args:
        None; read report paths and the output path from the CLI.
    Returns:
        None.
    Raises:
        ValueError: Reports cannot be compared fairly.
        OSError: A report cannot be read or written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    result = comparison([json.loads(path.read_text()) for path in args.reports])
    args.output_path.write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        print(
            f"{case['name']}: FA2 {case['fa2']['median_seconds']:.4f}s, "
            f"FA4 {case['fa4']['median_seconds']:.4f}s, "
            f"ratio {case['fa2_time_over_fa4_time']:.4f}x"
        )


if __name__ == "__main__":
    main()
