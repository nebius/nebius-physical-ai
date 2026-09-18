"""npa.workbench.gemini_robotics - Gemini Robotics 2 API-backed toolRef.

Closed-weight VLA: ER embodied-reasoning planning, on-device adaptation jobs,
and rubric evaluation run against the hosted Gemini API (``GOOGLE_API_KEY``).
Wrappers delegate to ``npa.cli.gemini_robotics`` callbacks.
"""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

plan = make_cli_wrapper(
    "npa.cli.gemini_robotics",
    "plan_cmd",
    "Run ER embodied-reasoning planning via the Gemini API.",
)
adapt = make_cli_wrapper(
    "npa.cli.gemini_robotics",
    "adapt_cmd",
    "Submit an on-device adaptation job via the Gemini API.",
)
eval = make_cli_wrapper(
    "npa.cli.gemini_robotics",
    "eval_cmd",
    "Evaluate a plan against a rubric via the Gemini API.",
)

__all__ = [
    "plan",
    "adapt",
    "eval",
]
