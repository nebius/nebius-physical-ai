"""npa.workbench.gemini_robotics - Gemini Robotics 2 API-backed toolRef.

Closed-weight VLA: ER embodied-reasoning planning and rubric evaluation run
against the hosted Gemini API (``GOOGLE_API_KEY``). The workbench package is
the primary SDK surface; the CLI is a thin client.
"""

from __future__ import annotations

from npa.workflows.byof.gemini_robotics_pipeline import (
    run_er_planning_stage,
    run_eval_stage,
)

# CLI verb aliases.
plan = run_er_planning_stage
eval = run_eval_stage

__all__ = [
    "plan",
    "eval",
    "run_er_planning_stage",
    "run_eval_stage",
]
