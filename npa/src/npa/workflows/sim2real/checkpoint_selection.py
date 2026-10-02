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
CHECKPOINT_URI_ALIASES = ("policy_checkpoint_uri", "checkpoint_uri")
CHECKPOINT_DIGEST_ALIASES = (
    "policy_checkpoint_sha256",
    "checkpoint_sha256",
    "sha256",
)
CHECKPOINT_SIZE_ALIASES = (
    "policy_checkpoint_size_bytes",
    "checkpoint_size_bytes",
    "size_bytes",
)
GENERATOR_DIGEST_ALIASES = (
    "generator_policy_sha256",
    "policy_generator_sha256",
)
_IDENTITY_FIELDS = (
    ("checkpoint URI", "checkpoint_uri", CHECKPOINT_URI_ALIASES),
    ("checkpoint SHA-256", "checkpoint_sha256", CHECKPOINT_DIGEST_ALIASES),
    ("checkpoint size", "checkpoint_size_bytes", CHECKPOINT_SIZE_ALIASES),
    (
        "generator checkpoint SHA-256",
        "generator_policy_sha256",
        GENERATOR_DIGEST_ALIASES,
    ),
)


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


def validation_checkpoint_candidate(
    report: dict[str, Any],
    *,
    checkpoint_uri: str,
    outer_iteration: int,
    inner_iteration: int,
    training_iteration: int,
) -> dict[str, Any]:
    """Build complete validation evidence for one checkpoint.

    Args:
        report: Validation report produced by the checkpoint.
        checkpoint_uri: Exact checkpoint object evaluated by the report.
        outer_iteration: Outer-loop iteration that produced the checkpoint.
        inner_iteration: Inner-loop iteration that produced the checkpoint.
        training_iteration: Training iteration recorded by the checkpoint.

    Returns:
        A validation candidate with complete checkpoint identity and provenance.

    Raises:
        KeyError: If the validation report does not declare its report URI.
    """

    provenance = report.get("policy_inference_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    return {
        "evaluation_split": "validation",
        "outer_iteration": outer_iteration,
        "inner_iteration": inner_iteration,
        "training_iteration": training_iteration,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": report.get("policy_checkpoint_sha256", ""),
        "checkpoint_size_bytes": report.get("policy_checkpoint_size_bytes", 0),
        "generator_policy_sha256": provenance.get("generator_policy_sha256", ""),
        "validation_report_uri": report["report_uri"],
        "validation_report": report,
    }


def checkpoint_rank_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Rank strict manipulation success first, with deterministic tie breaks.

    Earlier checkpoints win a fully equal numeric tie. This explicitly avoids
    quietly preferring ``model_latest.pt`` when validation proves no difference.
    """

    report = dict(candidate.get("validation_report") or {})
    summary = dict(report.get("success_summary") or {})
    mean_distance = _mean_distance(summary)
    return (
        _strict_rate(report),
        _rate(report, "place"),
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


def select_best_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best exact checkpoint using only a fixed validation split."""

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
        "strict_success,place,lift,stable_grasp,reach,contact,"
        "lower_mean_final_distance,earlier_checkpoint"
    )
    best["candidate_count"] = len(ranked)
    best["ranked_candidates"] = [
        {
            "checkpoint_uri": item.get("checkpoint_uri"),
            "checkpoint_sha256": item.get("checkpoint_sha256"),
            "outer_iteration": item.get("outer_iteration"),
            "inner_iteration": item.get("inner_iteration"),
            "training_iteration": item.get("training_iteration"),
            "strict_success_rate": (item.get("validation_report") or {}).get(
                "success_rate", 0.0
            ),
            "decomposed_metrics": (item.get("validation_report") or {}).get(
                "decomposed_metrics", {}
            ),
            "mean_object_goal_distance_m": (
                (item.get("validation_report") or {}).get("success_summary") or {}
            ).get("mean_object_goal_distance_m"),
            "validation_report_uri": item.get("validation_report_uri"),
            "rank_key": list(checkpoint_rank_key(item)),
        }
        for item in ranked
    ]
    return best


def _selection_for_uri(
    evidence: dict[str, Any],
    selected_uri: str,
) -> dict[str, Any]:
    selection = evidence.get("checkpoint_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("checkpoint_uri") != selected_uri
    ):
        raise ValueError("checkpoint selection URI disagrees with selected checkpoint")
    return selection


def _candidate_for_uri(
    evidence: dict[str, Any],
    selected_uri: str,
) -> dict[str, Any]:
    raw_candidates = evidence.get("checkpoint_candidates")
    if not isinstance(raw_candidates, list) or not all(
        isinstance(item, dict) for item in raw_candidates
    ):
        raise ValueError("checkpoint candidates are missing or malformed")
    candidates = [
        item for item in raw_candidates if item.get("checkpoint_uri") == selected_uri
    ]
    if len(candidates) != 1:
        raise ValueError("selected checkpoint must resolve to exactly one candidate")
    return candidates[0]


def _assert_matching_identity(
    selection: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for label, field, aliases in _IDENTITY_FIELDS:
        if field not in selection or field not in candidate:
            raise ValueError(f"selected checkpoint {field} is missing")
        expected = selection[field]
        if _identity_value(field, candidate[field]) != _identity_value(field, expected):
            raise ValueError(f"selected checkpoint {field} sources disagree")
        for source, payload in (("selection", selection), ("candidate", candidate)):
            for alias in aliases:
                if alias in payload and _identity_value(
                    field, payload[alias]
                ) != _identity_value(field, expected):
                    raise ValueError(
                        f"selected checkpoint {source} {label} aliases disagree"
                    )
    if _identity_value(
        "checkpoint_sha256", selection["generator_policy_sha256"]
    ) != _identity_value("checkpoint_sha256", selection["checkpoint_sha256"]):
        raise ValueError("selected checkpoint generator digest disagrees with bytes")


def _identity_value(field: str, value: Any) -> Any:
    if "sha256" in field and isinstance(value, str):
        return value.lower()
    return value


def resolve_selected_checkpoint(
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the one candidate bound to the persisted checkpoint selection."""
    selected_uri = evidence.get("selected_checkpoint_uri")
    final_uri = evidence.get("final_checkpoint_uri")
    if not isinstance(selected_uri, str) or not selected_uri:
        raise ValueError("selected checkpoint URI is missing")
    if final_uri != selected_uri:
        raise ValueError("final checkpoint URI disagrees with selected checkpoint")
    selection = _selection_for_uri(evidence, selected_uri)
    candidate = _candidate_for_uri(evidence, selected_uri)
    _assert_matching_identity(selection, candidate)
    return dict(selection), dict(candidate)


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
