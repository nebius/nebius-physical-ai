"""SDK wrappers for the Workbench Foxglove (Lichtblick) viewer CLI."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

serve = make_cli_wrapper("npa.cli.workbench.foxglove", "serve_cmd", "Plan a Foxglove viewer.")
launch = make_cli_wrapper("npa.cli.workbench.foxglove", "launch_cmd", "Launch a Foxglove viewer.")
status = make_cli_wrapper("npa.cli.workbench.foxglove", "status_cmd", "Show Foxglove status.")
list = make_cli_wrapper("npa.cli.workbench.foxglove", "list_cmd", "List Foxglove formats.")

__all__ = [
    "launch",
    "list",
    "serve",
    "status",
]
