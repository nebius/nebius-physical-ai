"""Classify canonical runtime reports without exposing worker failure details."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from npa.orchestration.npa_workflow.runtime import RuntimeReport, is_terminal


@dataclass(frozen=True)
class LifecycleOutcome:
    """Describe the durable team lifecycle result of one runtime report.

    Args:
        status: Team run status to persist.
        failure_code: Safe machine-readable terminal failure code.
        failure_message: Safe public terminal failure description.
    Returns:
        A classified lifecycle outcome.
    Raises:
        None.
    """

    status: str
    failure_code: str = ""
    failure_message: str = ""


def classify_runtime_report(report: RuntimeReport) -> LifecycleOutcome:
    """Classify only scheduler-proven failures as terminal team failures.

    Args:
        report: Canonical runtime result and its scheduler wave evidence.
    Returns:
        A safe durable team lifecycle outcome.
    Raises:
        None.
    """
    if report.status == "succeeded":
        return LifecycleOutcome("succeeded")
    if report.status == "failed" and _has_only_terminal_waves(report.waves):
        return LifecycleOutcome(
            "failed",
            "scheduler_terminal_failure",
            "Exact scheduler evidence confirms a terminal workload failure.",
        )
    return LifecycleOutcome("recovery_required")


def _has_only_terminal_waves(waves: list[dict]) -> bool:
    return bool(waves) and all(_is_exact_terminal(wave) for wave in waves) and any(
        _is_scheduler_failure(wave) for wave in waves
    )


def _is_exact_terminal(wave: Mapping[str, object]) -> bool:
    job_id = str(wave.get("job_id") or "").strip()
    status = str(wave.get("sky_status") or "").upper()
    return bool(job_id) and is_terminal(status)


def _is_scheduler_failure(wave: Mapping[str, object]) -> bool:
    status = str(wave.get("sky_status") or "").upper()
    return status == "FAIL" or status.startswith("FAILED")
