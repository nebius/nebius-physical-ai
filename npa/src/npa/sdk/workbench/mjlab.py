"""SDK wrappers for the Workbench MJLab CLI."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

eval = make_cli_wrapper("npa.cli.workbench.mjlab", "eval_cmd", "Run MJLab evaluation.")
workflow = make_cli_wrapper("npa.cli.workbench.mjlab", "workflow_cmd", "Show MJLab workflow.")
status = make_cli_wrapper("npa.cli.workbench.mjlab", "status_cmd", "Show MJLab status.")
list = make_cli_wrapper("npa.cli.workbench.mjlab", "list_cmd", "List MJLab suites.")

__all__ = [
    "eval",
    "list",
    "status",
    "workflow",
]
