"""Derive a recommendation from complete direction-aware paired navigation metrics."""

from __future__ import annotations

import math
from statistics import fmean

from npa.workflows.field_failure.artifacts import _read


def _paired_metrics(bundle, baseline, candidate):
    before = {(e.scenario_id, e.seed): e for e in baseline.episodes}
    after = {(e.scenario_id, e.seed): e for e in candidate.episodes}
    summaries = {}
    regressions = []
    for metric in bundle.metrics:
        sign = 1 if metric.direction == "higher" else -1
        rows = [
            _pair(key, before[key], after[key], metric, sign) for key in sorted(before)
        ]
        improved = _mean(row["improvement"] for row in rows)
        summaries[metric.name] = {
            "direction": metric.direction,
            "baseline_mean": _mean(e.metrics[metric.name] for e in before.values()),
            "candidate_mean": _mean(e.metrics[metric.name] for e in after.values()),
            "improvement": improved,
            "minimum_improvement": metric.minimum_improvement,
            "maximum_regression": metric.maximum_regression,
            "pairs": rows,
        }
        regressions.extend(
            {"metric": metric.name, **row}
            for row in rows
            if row["improvement"] < -metric.maximum_regression
        )
    primary = summaries[bundle.primary_metric]
    improvement = primary["improvement"]
    promote = (
        improvement > 0
        and improvement >= primary["minimum_improvement"]
        and not regressions
    )
    return summaries, regressions, promote


def _mean(values):
    try:
        result = fmean(values)
    except OverflowError as error:
        raise ValueError("metric aggregation overflowed") from error
    if not math.isfinite(result):
        raise ValueError("non-finite metric aggregate")
    return result


def _pair(key, before, after, metric, sign):
    improvement = sign * (after.metrics[metric.name] - before.metrics[metric.name])
    if not math.isfinite(improvement):
        raise ValueError("non-finite paired improvement")
    return {"scenario_id": key[0], "seed": key[1], "improvement": improvement}


def _decision(bundle, bundle_sha, baseline, candidate, hashes, run_id):
    if {e.evidence.sha256 for e in baseline.episodes} & {
        e.evidence.sha256 for e in candidate.episodes
    }:
        raise ValueError("baseline/candidate rollout evidence was reused")
    if _trajectory_hashes(baseline) & _trajectory_hashes(candidate):
        raise ValueError("underlying trajectory bytes were reused across policy arms")
    metrics, regressions, promote = _paired_metrics(bundle, baseline, candidate)
    return {
        "schema_version": "npa.field-failure.decision.v1",
        "run_id": run_id,
        "status": "verified_comparison",
        "recommendation": "promote" if promote else "retain_baseline",
        "decision": "promote_checkpoint" if promote else "retain_baseline",
        "promote_checkpoint": promote,
        "deployment_authorized": False,
        "bundle_sha256": bundle_sha,
        "record_sha256": hashes,
        "baseline": baseline.policy.model_dump(),
        "candidate": candidate.policy.model_dump(),
        "evaluator": bundle.adapters["evaluate"].model_dump(),
        "protocol_sha256": bundle.protocol.sha256,
        "primary_metric": bundle.primary_metric,
        "episodes_per_policy": len(baseline.episodes),
        "metrics": metrics,
        "regressions": regressions,
    }


def _trajectory_hashes(evaluation):
    return {
        _read(episode.evidence.uri, episode.evidence.sha256)[0]["trajectory"]["sha256"]
        for episode in evaluation.episodes
    }
