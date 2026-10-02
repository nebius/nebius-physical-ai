"""Authority checks for canonical Sim2Real Stage 11 decisions."""

from __future__ import annotations

import math
from typing import Any


def _finite_rate(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Stage 11 {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"Stage 11 {field} must be finite and within [0, 1]")
    return number


def _assert_gold_authority(
    report: dict[str, Any],
    *,
    outer_iteration: int,
    checkpoint_uri: str,
) -> float:
    if (
        report.get("evaluation_split") != "gold_heldout"
        or type(report.get("outer_iteration")) is not int
        or report["outer_iteration"] != outer_iteration
    ):
        raise ValueError("Stage 11 gold report scope is invalid")
    producer = report.get("policy_inference_provenance") or {}
    if not isinstance(producer, dict):
        raise ValueError("Stage 11 gold report producer must be an object")
    reported_checkpoint = report.get("policy_checkpoint_uri") or producer.get(
        "checkpoint_uri"
    )
    if reported_checkpoint != checkpoint_uri:
        raise ValueError("Stage 11 gold report checkpoint identity is invalid")
    return _finite_rate(report.get("success_rate"), field="gold success_rate")


def _assert_decision_shape(
    decision: dict[str, Any],
    *,
    run_id: str,
    root: str,
    outer_iteration: int,
    checkpoint_uri: str,
) -> None:
    expected_report = (
        f"{root.rstrip('/')}/eval/gold-heldout/outer-{outer_iteration:02d}/report.json"
    )
    if (
        decision.get("schema") != "npa.sim2real.threshold_decision.v1"
        or decision.get("run_id") != run_id
        or type(decision.get("outer_iteration")) is not int
        or decision["outer_iteration"] != outer_iteration
        or decision.get("gold_report_uri") != expected_report
        or decision.get("checkpoint_uri") != checkpoint_uri
        or decision.get("decision")
        not in {"promote_checkpoint", "loop_back_to_inner_loop"}
        or decision.get("strict_success_distance_m") != 0.05
        or decision.get("placement_stability_required") is not True
        or type(decision.get("early_exit_enabled")) is not bool
    ):
        raise ValueError("Stage 11 decision authority is invalid")


def _assert_decision_outcome(
    decision: dict[str, Any],
    *,
    report_rate: float,
) -> None:
    decision_rate = _finite_rate(decision.get("success_rate"), field="success_rate")
    threshold = _finite_rate(decision.get("threshold"), field="threshold")
    should_promote = bool(decision["early_exit_enabled"] and report_rate >= threshold)
    if (
        not math.isclose(decision_rate, report_rate, rel_tol=0.0, abs_tol=1e-12)
        or (decision["decision"] == "promote_checkpoint") != should_promote
    ):
        raise ValueError("Stage 11 decision outcome disagrees with gold evidence")


def _validate_decision(
    decision: dict[str, Any],
    run_id: str,
    root: str,
    outer_iteration: int,
    gold_report: dict[str, Any],
    checkpoint_uri: str,
) -> None:
    _assert_decision_shape(
        decision,
        run_id=run_id,
        root=root,
        outer_iteration=outer_iteration,
        checkpoint_uri=checkpoint_uri,
    )
    report_rate = _assert_gold_authority(
        gold_report,
        outer_iteration=outer_iteration,
        checkpoint_uri=checkpoint_uri,
    )
    _assert_decision_outcome(decision, report_rate=report_rate)


def validate_stage11_decision(
    decision: dict[str, Any],
    *,
    run_id: str,
    root: str,
    outer_iteration: int,
    gold_report: dict[str, Any],
    checkpoint_uri: str,
) -> None:
    """Validate one decision against the exact run, report, and checkpoint.

    Args:
        decision: Persisted Stage 11 threshold decision.
        run_id: Current workflow run identifier.
        root: Exact current-run S3 artifact root.
        outer_iteration: Selected outer-loop iteration.
        gold_report: Exact Stage 10 report consumed by the decision.
        checkpoint_uri: Validation-selected checkpoint URI.
    Returns:
        None.
    Raises:
        ValueError: If any decision authority or threshold field disagrees.
    """
    _validate_decision(
        decision,
        run_id,
        root,
        outer_iteration,
        gold_report,
        checkpoint_uri,
    )
