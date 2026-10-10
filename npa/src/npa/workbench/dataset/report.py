"""Validate and project dataset quality reports without exposing source metadata."""

from __future__ import annotations

import math
from typing import Any

from .schemas import VALIDATION_REPORT_SCHEMA


class DatasetReportError(ValueError):
    """Raised when a dataset quality report cannot be safely summarized.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """


def summarize_validation_report(document: dict[str, Any]) -> dict[str, Any]:
    """Preserve a dataset validation gate and its measured quality metrics.

    Args:
        document: A decoded ``npa.dataset.validation_report.v1`` artifact.
    Returns:
        A bounded summary excluding dataset identities, paths and free-form checks.
    Raises:
        DatasetReportError: Required evidence is missing, malformed or contradictory.
    """
    passed, checks, stats, thresholds = _validation_inputs(document)
    metrics = _quality_metrics(document, stats)
    limits = {
        name: _fraction(thresholds.get(name))
        for name in ("completeness_min", "max_corruption_rate")
    }
    _validate_gate(passed, checks, metrics, limits)
    return {
        "schema": VALIDATION_REPORT_SCHEMA,
        "reported_gate": {"passed": passed, "thresholds": limits},
        "evaluation_state": "graded",
        "evidence_complete": True,
        "metrics": metrics,
        "failed_check_count": len(checks),
        "limitations": [
            "Dataset completeness and corruption rates are not evaluator scores or calibrated confidence."
        ],
    }


def _validation_inputs(document: dict[str, Any]) -> tuple:
    passed = document.get("passed")
    checks = document.get("failed_checks")
    stats = document.get("quality_stats")
    thresholds = document.get("thresholds")
    if (
        document.get("schema") != VALIDATION_REPORT_SCHEMA
        or not isinstance(passed, bool)
        or not isinstance(checks, list)
        or any(not isinstance(check, str) for check in checks)
        or not isinstance(stats, dict)
        or not isinstance(thresholds, dict)
    ):
        raise DatasetReportError("dataset validation report lacks required evidence")
    return passed, checks, stats, thresholds


def _quality_metrics(document: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    count = document.get("record_count")
    corrupt = stats.get("corrupt_count")
    if any(
        type(value) is not int or value < 0
        for value in (count, corrupt, stats.get("record_count"))
    ):
        raise DatasetReportError("dataset counts must be nonnegative integers")
    if count == 0 or corrupt > count or stats.get("record_count") != count:
        raise DatasetReportError("dataset report counts are inconsistent")
    rate = _fraction(document.get("corruption_rate"))
    if not math.isclose(rate, round(corrupt / count, 4), abs_tol=1e-9):
        raise DatasetReportError("dataset corruption rate contradicts its counts")
    return {
        "record_count": count,
        "corrupt_count": corrupt,
        "mean_completeness": _fraction(stats.get("mean_completeness")),
        "corruption_rate": rate,
    }


def _fraction(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
        or not math.isfinite(value)
    ):
        raise DatasetReportError(
            "dataset rates and thresholds must be finite numbers from zero to one"
        )
    return value


def _validate_gate(
    passed: bool, checks: list[str], metrics: dict, limits: dict
) -> None:
    if passed != (not checks):
        raise DatasetReportError("dataset gate contradicts its failed-check count")
    if passed and (
        metrics["mean_completeness"] < limits["completeness_min"]
        or metrics["corruption_rate"] > limits["max_corruption_rate"]
    ):
        raise DatasetReportError(
            "passing dataset gate contradicts its quality thresholds"
        )
