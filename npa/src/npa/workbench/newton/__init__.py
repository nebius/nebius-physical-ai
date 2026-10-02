"""npa.workbench.newton - Newton physics engine workbench commands."""

from __future__ import annotations

from npa.workflows.byof.newton_pipeline import (
    NewtonPipelineError,
    evaluate,
    generate_demos,
    newton_version,
    train_teacher,
)

# CLI verb alias: the ``eval`` stage command maps to ``evaluate``.
eval = evaluate

__all__ = [
    "NewtonPipelineError",
    "train_teacher",
    "generate_demos",
    "evaluate",
    "eval",
    "newton_version",
]
