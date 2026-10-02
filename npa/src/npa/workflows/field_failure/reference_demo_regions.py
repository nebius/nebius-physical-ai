"""Measure adaptation and retention using region assignments frozen before learning."""

import math
from statistics import fmean

METRICS = ("success", "collision_steps", "physical_failure_steps", "goal_distance_m")


def region_comparison(baseline, candidate, regions):
    """Require exact balanced coverage and report each region's measured regression.

    Args:
        baseline: Actual native or field-failure episode dictionaries.
        candidate: Actual episodes from the other checkpoint on identical cases.
        regions: Precommitted office and warehouse seed/case identities.
    Returns:
        Per-region metric means, success rates and explicit retention gates.
    Raises:
        ValueError: Coverage, identities, balance or finite metrics are invalid.
    """
    before, after = _index(baseline, regions), _index(candidate, regions)
    results, reasons = {}, []
    for name, identities in regions.items():
        seeds = identities["seeds"]
        metrics = {key: _metric(before, after, seeds, key) for key in METRICS}
        failures = _regressions(name, metrics)
        results[name] = {
            "episodes_per_policy": len(seeds),
            "baseline_success_rate": metrics["success"]["baseline_mean"],
            "candidate_success_rate": metrics["success"]["candidate_mean"],
            "success_rate_gain": metrics["success"]["change"],
            "metrics": metrics,
            "non_regressing": not failures,
        }
        reasons.extend(failures)
    return {
        "regions": results,
        "passed": not reasons,
        "warehouse_retention_passed": results["warehouse"]["non_regressing"],
        "reasons": reasons,
    }


def _index(episodes, regions):
    if set(regions) != {"office", "warehouse"}:
        raise ValueError("balanced reference requires office and warehouse assignments")
    expected = {}
    for region in regions.values():
        seeds, ids = region["seeds"], region["case_ids"]
        if len(seeds) != region["count"] or len(ids) != len(seeds) or not seeds:
            raise ValueError("invalid frozen region population")
        for seed, case in zip(seeds, ids, strict=True):
            if seed in expected:
                raise ValueError("frozen region assignments overlap")
            expected[seed] = case
    if regions["office"]["count"] != regions["warehouse"]["count"]:
        raise ValueError("evaluation regions are not balanced")
    indexed = {row["seed"]: row for row in episodes}
    if len(indexed) != len(episodes) or set(indexed) != set(expected):
        raise ValueError("evaluation differs from frozen regional cohort")
    for seed, row in indexed.items():
        if "case_id" in row and row["case_id"] != expected[seed]:
            raise ValueError("evaluation case identity differs from frozen region")
    return indexed


def _metric(before, after, seeds, key):
    means = []
    for arm in (before, after):
        values = [float(arm[seed].get("metrics", arm[seed])[key]) for seed in seeds]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("regional metrics must be finite and nonnegative")
        if key == "success" and not all(v in (0.0, 1.0) for v in values):
            raise ValueError("regional success must be an actual episode indicator")
        means.append(fmean(values))
    return {
        "baseline_mean": means[0],
        "candidate_mean": means[1],
        "change": means[1] - means[0],
    }


def _regressions(name, metrics):
    reasons = []
    for metric, limit in (
        ("success", 0),
        ("collision_steps", 0),
        ("physical_failure_steps", 0),
        ("goal_distance_m", 0.25),
    ):
        change = metrics[metric]["change"]
        regression = -change if metric == "success" else change
        if regression > limit:
            reasons.append(name + " regressed on " + metric)
    return reasons


def final_regions(args, plan, final):
    """Read hash-bound final comparison episodes after development selection only.

    Args:
        args: Run-scoped artifact location and identifier.
        plan: Frozen region assignments.
        final: Completed strict paired comparison record.
    Returns:
        Final per-region comparison derived from the exact compared episodes.
    Raises:
        ValueError: Final checkpoint identities or episode coverage differ.
        OSError: Required sealed evaluation records are unavailable.
    """
    from npa.workflows.field_failure.artifacts import _read

    episodes = []
    prefix = args.output_root + "/loop/" + args.run_id
    for arm in ("baseline", "candidate"):
        record, _ = _read(
            prefix + "/" + arm + "-evaluation.json",
            final["record_sha256"][arm + "_evaluation"],
        )
        if record["policy"] != final[arm]:
            raise ValueError("regional final report loaded another policy's episodes")
        episodes.append(record["episodes"])
    return region_comparison(*episodes, plan["cohorts"]["regions"]["final"])
