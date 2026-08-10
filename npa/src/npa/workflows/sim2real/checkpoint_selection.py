"""Validation-only checkpoint ranking for the Sim2Real PPO loop."""

from __future__ import annotations

from typing import Any


def _rate(report: dict[str, Any], name: str) -> float:
    value = (report.get("decomposed_metrics") or {}).get(name, {})
    return float(value.get("rate") or 0.0) if isinstance(value, dict) else 0.0


def _strict_rate(report: dict[str, Any]) -> float:
    """Read either normalized or component-native strict success evidence."""

    if "success_rate" in report:
        return float(report.get("success_rate") or 0.0)
    strict = report.get("strict_success") or {}
    return float(strict.get("rate") or 0.0) if isinstance(strict, dict) else 0.0


def checkpoint_rank_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Rank strict manipulation success first, with deterministic tie breaks.

    Earlier checkpoints win a fully equal numeric tie. This explicitly avoids
    quietly preferring ``model_latest.pt`` when validation proves no difference.
    """

    report = dict(candidate.get("validation_report") or {})
    summary = dict(report.get("success_summary") or {})
    mean_distance = float(summary.get("mean_object_goal_distance_m") or 1.0e9)
    return (
        _strict_rate(report),
        _rate(report, "place"),
        _rate(report, "lift"),
        _rate(report, "stable_grasp"),
        _rate(report, "contact"),
        _rate(report, "reach"),
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
        "strict_success,place,lift,stable_grasp,contact,reach,"
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
