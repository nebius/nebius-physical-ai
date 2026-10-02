"""Authority checks for canonical Sim2Real Stage 11 decisions."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def gold_report_sha256(report: dict[str, Any]) -> str:
    """Hash the exact canonical JSON bytes published for a Stage 10 report."""

    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        or not isinstance(decision.get("gold_report_sha256"), str)
        or len(decision["gold_report_sha256"]) != 64
        or any(
            char not in "0123456789abcdef" for char in decision["gold_report_sha256"]
        )
    ):
        raise ValueError("Stage 11 decision authority is invalid")


def _assert_decision_outcome(
    decision: dict[str, Any],
    *,
    report_rate: float,
    report_sha256: str,
    expected_threshold: float,
    expected_early_exit: bool,
) -> None:
    decision_rate = _finite_rate(decision.get("success_rate"), field="success_rate")
    threshold = _finite_rate(decision.get("threshold"), field="threshold")
    configured_threshold = _finite_rate(
        expected_threshold, field="configured threshold"
    )
    if type(expected_early_exit) is not bool:
        raise ValueError("Stage 11 configured early-exit value must be boolean")
    should_promote = bool(expected_early_exit and report_rate >= configured_threshold)
    if (
        not math.isclose(decision_rate, report_rate, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(threshold, configured_threshold, rel_tol=0.0, abs_tol=1e-12)
        or decision["early_exit_enabled"] is not expected_early_exit
        or decision["gold_report_sha256"] != report_sha256
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
    expected_threshold: float,
    expected_early_exit: bool,
    gold_report_bytes_sha256: str | None,
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
    _assert_decision_outcome(
        decision,
        report_rate=report_rate,
        report_sha256=(gold_report_bytes_sha256 or gold_report_sha256(gold_report)),
        expected_threshold=expected_threshold,
        expected_early_exit=expected_early_exit,
    )


def validate_stage11_decision(
    decision: dict[str, Any],
    *,
    run_id: str,
    root: str,
    outer_iteration: int,
    gold_report: dict[str, Any],
    checkpoint_uri: str,
    expected_threshold: float,
    expected_early_exit: bool,
    gold_report_bytes_sha256: str | None = None,
) -> None:
    """Validate one decision against the exact run, report, and checkpoint.

    Args:
        decision: Persisted Stage 11 threshold decision.
        run_id: Current workflow run identifier.
        root: Exact current-run S3 artifact root.
        outer_iteration: Selected outer-loop iteration.
        gold_report: Exact Stage 10 report consumed by the decision.
        checkpoint_uri: Validation-selected checkpoint URI.
        expected_threshold: Workflow-configured promotion threshold.
        expected_early_exit: Workflow-configured early-exit policy.
        gold_report_bytes_sha256: Digest of the exact downloaded report bytes.
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
        expected_threshold,
        expected_early_exit,
        gold_report_bytes_sha256,
    )
