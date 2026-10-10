"""Validation-only checkpoint ranking for the Sim2Real PPO loop."""

from __future__ import annotations

import math
from typing import Any

# Sentinel for a genuinely absent mean-distance metric: worse than any
# physically plausible distance, so a checkpoint with no distance evidence
# never outranks one that does. Must stay distinguishable from a real 0.0m
# distance, which is the best possible outcome, not a missing-data signal.
# Finite (unlike math.inf) so rank_key stays valid JSON in written artifacts.
_MISSING_DISTANCE = 1.0e9


def _finite_metric(value: Any, *, field: str) -> float:
    """Coerce a metric value to a finite float or raise with the offending field.

    Args:
        value: Raw metric value read from a validation report.
        field: Dotted path of the metric, used to make the error actionable.

    Returns:
        The value as a finite ``float``.

    Raises:
        ValueError: If ``value`` is not numeric or is NaN/infinite. Silently
            coercing such values would make ranking order depend on candidate
            input order instead of on the metric itself.
    """

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint metric {field!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"checkpoint metric {field!r} is not finite: {value!r}")
    return number


def _rate(report: dict[str, Any], name: str) -> float:
    """Read a decomposed-metric success rate, defaulting an absent skill to 0.0."""

    value = (report.get("decomposed_metrics") or {}).get(name)
    if not isinstance(value, dict) or "rate" not in value:
        return 0.0
    rate = _finite_metric(value["rate"], field=f"decomposed_metrics.{name}.rate")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"decomposed_metrics.{name}.rate out of [0, 1]: {rate}")
    return rate


def _strict_rate(report: dict[str, Any]) -> float:
    """Read either normalized or component-native strict success evidence."""

    if "success_rate" in report:
        rate = _finite_metric(report["success_rate"], field="success_rate")
    else:
        strict = report.get("strict_success")
        if not isinstance(strict, dict) or "rate" not in strict:
            return 0.0
        rate = _finite_metric(strict["rate"], field="strict_success.rate")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"strict success rate out of [0, 1]: {rate}")
    return rate


def _paired_lift_rate(report: dict[str, Any]) -> float:
    """Prefer lifts accompanied by stable grasp in the same native episode."""
    pairs = [
        (
            (row.get("details") or {}).get("stable_grasp"),
            (row.get("details") or {}).get("lift"),
        )
        for row in report.get("per_env") or []
    ]
    if not pairs or all(grasp is None and lift is None for grasp, lift in pairs):
        return 0.0
    if any(type(grasp) is not bool or type(lift) is not bool for grasp, lift in pairs):
        raise ValueError("paired checkpoint skill requires literal grasp/lift verdicts")
    return sum(grasp and lift for grasp, lift in pairs) / len(pairs)


def _mean_distance(summary: dict[str, Any]) -> float:
    """Read the mean object-to-goal distance, or the missing-data sentinel.

    The field is optional: some validation reports never populate it (for
    example when no per-env distance evidence was recorded). Absence must
    fall back to the worst possible score, never be confused with an actual
    ``0.0`` distance, which is the best possible score.
    """

    if "mean_object_goal_distance_m" not in summary:
        return _MISSING_DISTANCE
    value = summary["mean_object_goal_distance_m"]
    if value is None:
        return _MISSING_DISTANCE
    distance = _finite_metric(
        value, field="success_summary.mean_object_goal_distance_m"
    )
    if distance < 0:
        raise ValueError(
            f"success_summary.mean_object_goal_distance_m is negative: {distance}"
        )
    return distance


def checkpoint_rank_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Rank strict manipulation success first, with deterministic tie breaks.

    Earlier checkpoints win a fully equal numeric tie. This explicitly avoids
    quietly preferring ``model_latest.pt`` when validation proves no difference.

    Args:
        candidate: Checkpoint identity and its simulator validation report.
    Returns:
        Comparable strict success, paired skill, distance, and identity evidence.
    Raises:
        ValueError: A present metric or paired episode verdict is invalid.
    """

    report = dict(candidate.get("validation_report") or {})
    summary = dict(report.get("success_summary") or {})
    mean_distance = _mean_distance(summary)
    return (
        _strict_rate(report),
        _rate(report, "place"),
        _paired_lift_rate(report),
        _rate(report, "lift"),
        _rate(report, "stable_grasp"),
        _rate(report, "reach"),
        _rate(report, "contact"),
        -mean_distance,
        -int(candidate.get("outer_iteration") or 0),
        -int(candidate.get("inner_iteration") or 0),
        -int(candidate.get("training_iteration") or 0),
        str(
            candidate.get("checkpoint_sha256") or candidate.get("checkpoint_uri") or ""
        ),
    )


def _ranked_checkpoint(candidate: dict[str, Any]) -> dict[str, Any]:
    """Retain the measured skill and identity used for each ranking decision."""
    report = candidate.get("validation_report") or {}
    return {
        "checkpoint_uri": candidate.get("checkpoint_uri"),
        "checkpoint_sha256": candidate.get("checkpoint_sha256"),
        "outer_iteration": candidate.get("outer_iteration"),
        "inner_iteration": candidate.get("inner_iteration"),
        "training_iteration": candidate.get("training_iteration"),
        "strict_success_rate": (candidate.get("validation_report") or {}).get(
            "success_rate", 0.0
        ),
        "decomposed_metrics": (candidate.get("validation_report") or {}).get(
            "decomposed_metrics", {}
        ),
        "mean_object_goal_distance_m": (
            (candidate.get("validation_report") or {}).get("success_summary") or {}
        ).get("mean_object_goal_distance_m"),
        "validation_report_uri": candidate.get("validation_report_uri"),
        "stable_grasp_and_lift_rate": _paired_lift_rate(report),
        "rank_key": list(checkpoint_rank_key(candidate)),
    }


def select_best_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best exact checkpoint using only a fixed validation split.

    Args:
        candidates: Checkpoint identities and their per-episode validation reports.
    Returns:
        Selected candidate with its ranking policy and all candidate comparisons.
    Raises:
        ValueError: Evidence is absent, unbound, from another split, or invalid.
    """

    if not candidates:
        raise ValueError("checkpoint selection requires at least one candidate")
    for candidate in candidates:
        if candidate.get("evaluation_split") != "validation":
            raise ValueError("checkpoint selection may consume only validation reports")
        if not str(candidate.get("checkpoint_uri") or "").startswith("s3://"):
            raise ValueError("checkpoint candidate lacks an S3 checkpoint URI")
        if not (candidate.get("validation_report") or {}).get("per_env"):
            raise ValueError(
                "checkpoint candidate lacks per-environment validation evidence"
            )
    ranked = sorted(candidates, key=checkpoint_rank_key, reverse=True)
    best = dict(ranked[0])
    best["rank_key"] = list(checkpoint_rank_key(best))
    best["selection_policy"] = (
        "strict_success,place,stable_grasp_and_lift,lift,stable_grasp,reach,contact,"
        "lower_mean_final_distance,earlier_checkpoint"
    )
    best["candidate_count"] = len(ranked)
    best["ranked_candidates"] = [_ranked_checkpoint(candidate) for candidate in ranked]
    return best


def assert_no_split_leakage(
    train_digests: set[str], validation_digests: set[str], gold_digests: set[str]
) -> None:
    """Fail closed if scenario config digests cross train/validation/gold sets."""

    overlaps = {
        "train_validation": train_digests & validation_digests,
        "train_gold": train_digests & gold_digests,
        "validation_gold": validation_digests & gold_digests,
    }
    leaked = {name: sorted(values) for name, values in overlaps.items() if values}
    if leaked:
        raise ValueError(f"scenario split leakage detected: {leaked}")
