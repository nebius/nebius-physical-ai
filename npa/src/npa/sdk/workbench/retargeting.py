"""SDK wrappers for the Workbench retargeting CLI."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

run = make_cli_wrapper("npa.cli.workbench.retargeting", "run_cmd", "Run motion retargeting.")
workflow = make_cli_wrapper("npa.cli.workbench.retargeting", "workflow_cmd", "Show retargeting workflow.")
status = make_cli_wrapper("npa.cli.workbench.retargeting", "status_cmd", "Show retargeting status.")
list = make_cli_wrapper("npa.cli.workbench.retargeting", "list_cmd", "List retargeting source formats.")

__all__ = [
    "list",
    "run",
    "status",
    "workflow",
]
