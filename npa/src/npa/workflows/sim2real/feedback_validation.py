"""Strict feedback parsing shared by legacy Sim2Real adapters."""

from __future__ import annotations

from typing import Any

from npa.workflows.feedback import (
    FeedbackSourceError,
    parse_feedback_score,
)
from npa.workflows.sim2real.models import Sim2RealLoopError


def strict_feedback_score(value: Any) -> float:
    """Parse a finite JSON number into the legacy loop error contract."""

    try:
        return parse_feedback_score(value)
    except FeedbackSourceError as exc:
        raise Sim2RealLoopError(str(exc)) from exc
