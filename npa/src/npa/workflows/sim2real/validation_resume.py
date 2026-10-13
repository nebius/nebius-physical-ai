"""Durable validation-candidate resume handling for the Sim2Real inner loop."""

from __future__ import annotations

from typing import Any, Callable

from npa.workflows.sim2real.checkpoint_selection import (
    checkpoint_candidate_has_complete_identity,
    validation_checkpoint_candidate,
)
from npa.workflows.sim2real.models import Sim2RealLoopError


def resume_or_run_validation_candidate(
    durable: Any,
    *,
    unit: str,
    validation_input: dict[str, Any],
    run_evaluation: Callable[[], dict[str, Any]],
    checkpoint_uri: str,
    outer_iteration: int,
    inner_iteration: int,
    training_iteration: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse only validation reports with complete checkpoint byte identity."""

    persisted = durable.load_unit(unit, validation_input)
    if isinstance(persisted, dict) and isinstance(persisted.get("report"), dict):
        report = dict(persisted["report"])
        try:
            candidate = validation_checkpoint_candidate(
                report,
                checkpoint_uri=checkpoint_uri,
                outer_iteration=outer_iteration,
                inner_iteration=inner_iteration,
                training_iteration=training_iteration,
            )
        except (KeyError, TypeError, ValueError):
            candidate = {}
        if checkpoint_candidate_has_complete_identity(candidate):
            return report, candidate

    report = run_evaluation()
    if not isinstance(report, dict):
        raise Sim2RealLoopError("validation evaluation did not return a report object")
    durable.commit_unit(unit, validation_input, {"report": report})
    candidate = validation_checkpoint_candidate(
        report,
        checkpoint_uri=checkpoint_uri,
        outer_iteration=outer_iteration,
        inner_iteration=inner_iteration,
        training_iteration=training_iteration,
    )
    return report, candidate
