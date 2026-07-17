"""SDK helpers for BYOF (bring-your-own-fork) OSS onboarding."""

from __future__ import annotations

from typing import Any

from npa._sdk import make_cli_wrapper
from npa.cli.workbench.byof import build_byof_argv

run = make_cli_wrapper(
    "npa.cli.workbench.byof",
    "run_cmd",
    "Build/push a BYOF image and optionally run a live workload.",
)
ladder = make_cli_wrapper(
    "npa.cli.workbench.byof",
    "ladder_cmd",
    "Show the OSS onboarding ladder (Tier 0 → Tier 2).",
)
status = make_cli_wrapper(
    "npa.cli.workbench.byof",
    "status_cmd",
    "Report BYOF packaging surfaces (CLI / SDK / YAML).",
)


def plan_argv(**kwargs: Any) -> list[str]:
    """Return argv for ``run_byof_repo.py`` without executing it."""

    return build_byof_argv(**kwargs)


__all__ = [
    "build_byof_argv",
    "ladder",
    "plan_argv",
    "run",
    "status",
]
