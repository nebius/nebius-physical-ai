"""Deprecated compatibility import for the workbench workflow CLI."""

from __future__ import annotations

from npa.cli.workbench.workflow import (
    ActionSpace,
    ControllerBackendOption,
    OutputFormat,
    app,
    distill_cmd,
    logs_cmd,
    run_cmd,
    status_cmd,
    submit_cmd,
    teardown_cmd,
)

__all__ = [
    "ActionSpace",
    "ControllerBackendOption",
    "OutputFormat",
    "app",
    "distill_cmd",
    "logs_cmd",
    "run_cmd",
    "status_cmd",
    "submit_cmd",
    "teardown_cmd",
]
