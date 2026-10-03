"""Apply the frozen final metric regression limits to paired development episodes."""

import math
from statistics import fmean

from npa.workflows.field_failure.contracts import _Metric
from npa.workflows.field_failure.reference_demo_regions import METRICS


def paired_development_regressions(baseline, candidate, regions, metric_specs):
    """Reject individual regressions even when regional averages improve.

    Args:
        baseline: Native baseline development episode dictionaries.
        candidate: Native candidate development episodes on the same frozen cases.
        regions: Pre-learning region assignments containing exact case IDs and seeds.
        metric_specs: Frozen reference-plan metrics used by the final comparator.
    Returns:
        Deterministic violation details, counts and the applied metric limits.
    Raises:
        ValueError: Case pairing, metric contracts or finite metric bounds differ.
    """
    expected = _expected_cases(regions)
    before = _case_index(baseline, expected)
    after = _case_index(candidate, expected)
    metrics = [_Metric.model_validate(value) for value in metric_specs]
    if len(metrics) != len(METRICS) or {m.name for m in metrics} != set(METRICS):
        raise ValueError("development requires every frozen reference metric")
    violations = []
    for metric in metrics:
        for key in sorted(expected, key=lambda identity: (identity[1], identity[0])):
            row = _violation(key, expected[key], before[key], after[key], metric)
            if row is not None:
                violations.append(row)
    success = next(metric for metric in metrics if metric.name == "success")
    sign = 1 if success.direction == "higher" else -1
    gain = fmean(
        sign * (_measurement(after[key], success) - _measurement(before[key], success))
        for key in expected
    )
    return _summary(len(expected), metrics, regions, violations, gain)


def _expected_cases(regions):
    expected = {}
    for name, region in regions.items():
        if len(region["case_ids"]) != region["count"] or not region["count"]:
            raise ValueError("invalid frozen development case population")
        for identity in zip(region["case_ids"], region["seeds"], strict=True):
            if identity in expected:
                raise ValueError("frozen development case identities overlap")
            expected[identity] = name
    if len({key[0] for key in expected}) != len(expected) or len(
        {key[1] for key in expected}
    ) != len(expected):
        raise ValueError("frozen development case IDs or seeds overlap")
    return expected


def _case_index(episodes, expected):
    indexed = {}
    for row in episodes:
        key = (row.get("case_id"), row.get("seed"))
        if key not in expected or key in indexed:
            raise ValueError("development requires exact unique case/seed pairs")
        indexed[key] = row
    if set(indexed) != set(expected):
        raise ValueError("development case/seed coverage differs from frozen cases")
    return indexed


def _measurement(row, metric):
    value = float(row[metric.name])
    if not math.isfinite(value) or not metric.minimum <= value <= metric.maximum:
        raise ValueError("development metric differs from its frozen finite bounds")
    return value


def _violation(identity, region, before, after, metric):
    baseline, candidate = _measurement(before, metric), _measurement(after, metric)
    sign = 1 if metric.direction == "higher" else -1
    improvement = sign * (candidate - baseline)
    if not math.isfinite(improvement):
        raise ValueError("non-finite paired development improvement")
    # Preserve comparison._paired_metrics' exact strict boundary and direction.
    if improvement >= -metric.maximum_regression:
        return None
    return {
        "case_id": identity[0],
        "seed": identity[1],
        "region": region,
        "metric": metric.name,
        "baseline": baseline,
        "candidate": candidate,
        "improvement": improvement,
        "maximum_regression": metric.maximum_regression,
    }


def _summary(count, metrics, regions, violations, gain):
    return {
        "passed": not violations,
        "paired_cases": count,
        "success_rate_gain": gain,
        "violation_count": len(violations),
        "regressed_cases": len({(row["case_id"], row["seed"]) for row in violations}),
        "violations_by_metric": {
            metric.name: sum(row["metric"] == metric.name for row in violations)
            for metric in metrics
        },
        "violations_by_region": {
            region: sum(row["region"] == region for row in violations)
            for region in regions
        },
        "metric_limits": [metric.model_dump() for metric in metrics],
        "violations": violations,
    }
