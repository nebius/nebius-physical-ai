"""Strict feedback parsing shared by legacy Sim2Real adapters."""

from __future__ import annotations

from typing import Any

from npa.workflows.feedback import (
    FeedbackSourceError,
    parse_feedback_score,
)
from npa.workflows.sim2real.models import (
    Sim2RealLoopError,
    require_sim2real_boolean,
)


def strict_feedback_score(value: Any) -> float:
    """Parse a finite JSON number into the legacy loop error contract."""

    try:
        return parse_feedback_score(value)
    except FeedbackSourceError as exc:
        raise Sim2RealLoopError(str(exc)) from exc


def bounded_feedback_score(value: Any) -> float:
    """Parse and clamp one legacy feedback score to the unit interval."""

    return max(0.0, min(1.0, strict_feedback_score(value)))


def strict_reference_adapter_feedback(
    item: dict[str, Any], *, index: int
) -> tuple[bool, float]:
    """Parse one reference-adapter success and score pair."""

    success = require_sim2real_boolean(
        item.get("success", False),
        field=f"reference adapter per_env[{index}].success",
    )
    return success, strict_feedback_score(item.get("score", 0.0))
