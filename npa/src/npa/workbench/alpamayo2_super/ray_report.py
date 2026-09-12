"""Validate Alpamayo sweep measurements and compare matched scenarios and seeds."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import fmean, pstdev


def case_grid(samples: list[int], seeds: list[int], steps: list[int]) -> list[dict]:
    """Expand an explicit scenario, seed, and diffusion-step experiment.

    Args:
        samples: Non-negative validation-manifest indices.
        seeds: Non-negative random seeds.
        steps: Positive diffusion-step counts.
    Returns:
        Unique cases in reproducible sample/seed/step order.
    Raises:
        ValueError: A dimension is empty, duplicated, or invalid.
    """
    for name, values, minimum in (("samples", samples, 0), ("seeds", seeds, 0), ("steps", steps, 1)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be non-empty and contain no duplicates")
        if any(type(value) is not int or value < minimum for value in values):
            raise ValueError(f"{name} must contain integers >= {minimum}")
    return [
        {"sample_index": sample, "seed": seed, "diffusion_steps": count}
        for sample in sorted(samples) for seed in sorted(seeds) for count in sorted(steps)
    ]


def case_key(case: dict) -> tuple[int, int, int]:
    """Identify one reproducible inference request.

    Args:
        case: Sample, seed, and diffusion settings.
    Returns:
        Sample index, seed, diffusion steps.
    Raises:
        KeyError: A required setting is absent.
    """
    return case["sample_index"], case["seed"], case["diffusion_steps"]


def validate_measurements(rows: list[dict], cases: list[dict]) -> list[dict]:
    """Reject incomplete, duplicate, or non-finite sweep results.

    Args:
        rows: Actual inference measurements.
        cases: Exact expected experiment cases.
    Returns:
        Measurements sorted by sample, seed, and step count.
    Raises:
        ValueError: Results do not exactly cover the experiment or metrics are invalid.
        KeyError: Required measurement fields are missing.
    """
    expected = {case_key(case) for case in cases}
    observed = [case_key(row) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("sweep results must cover every requested case exactly once")
    for row in rows:
        for name in ("min_ade_m", "min_fde_m", "elapsed_seconds"):
            value = row[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name} in inference measurement")
    return sorted(rows, key=case_key)


def summarize_sample(rows: list[dict]) -> list[dict]:
    """Measure seed variability separately for each diffusion-step setting.

    Args:
        rows: Validated measurements for one scenario.
    Returns:
        Per-setting mean ADE/FDE, seed standard deviation, and case elapsed time.
    Raises:
        KeyError: Measurement fields are missing.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["diffusion_steps"]].append(row)
    summaries = []
    for steps, members in sorted(groups.items()):
        errors = [row["min_ade_m"] for row in members]
        summaries.append({
            "sample_index": members[0]["sample_index"], "diffusion_steps": steps,
            "seed_count": len(members), "mean_ade_m": fmean(errors),
            "seed_ade_std_m": pstdev(errors),
            "mean_fde_m": fmean(row["min_fde_m"] for row in members),
            "mean_elapsed_seconds": fmean(row["elapsed_seconds"] for row in members),
        })
    return summaries


def select_hard_samples(report: dict, minimum_ade: float) -> list[int]:
    """Select every scenario whose mean ADE exceeds the operator threshold.

    Args:
        report: Completed baseline sweep report.
        minimum_ade: Non-negative mean displacement error in meters.
    Returns:
        Sorted unique scenario indices; an empty selection is valid.
    Raises:
        ValueError: Threshold or baseline report is invalid.
        KeyError: Required baseline fields are missing.
    """
    if not math.isfinite(minimum_ade) or minimum_ade < 0:
        raise ValueError("minimum_ade must be finite and non-negative")
    if report.get("schema") != "npa.alpamayo.ray-sweep.v1" or report.get("status") != "complete":
        raise ValueError("refinement requires a complete Alpamayo Ray sweep report")
    rows = validate_measurements(report["measurements"], report["cases"])
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_index"]].append(row["min_ade_m"])
    return sorted(sample for sample, values in groups.items() if fmean(values) > minimum_ade)


def matched_comparisons(baseline: list[dict], refined: list[dict]) -> list[dict]:
    """Compare errors only between identical scenarios and random seeds.

    Args:
        baseline: Baseline measurements.
        refined: Measurements from the refinement sweep.
    Returns:
        ADE and FDE changes for every matching baseline/refined setting pair.
    Raises:
        KeyError: Required measurement fields are missing.
        ValueError: A refined case has no matching baseline seed or sample identity.
    """
    matches = defaultdict(list)
    for row in baseline:
        matches[(row["sample_index"], row["seed"])].append(row)
    comparisons = []
    for row in refined:
        previous = matches[(row["sample_index"], row["seed"])]
        if not previous:
            raise ValueError("refinement has no matched baseline scenario and seed")
        for before in previous:
            if before["sample"] != row["sample"]:
                raise ValueError("baseline and refinement manifest identities differ")
            comparisons.append({
                "sample_index": row["sample_index"], "seed": row["seed"],
                "baseline_steps": before["diffusion_steps"],
                "refined_steps": row["diffusion_steps"],
                "ade_change_m": row["min_ade_m"] - before["min_ade_m"],
                "fde_change_m": row["min_fde_m"] - before["min_fde_m"],
            })
    return comparisons
