"""SDK wrappers for the Workbench Lichtblick (Foxglove-compatible) viewer CLI."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

serve = make_cli_wrapper("npa.cli.workbench.lichtblick", "serve_cmd", "Plan a Lichtblick viewer.")
launch = make_cli_wrapper("npa.cli.workbench.lichtblick", "launch_cmd", "Launch a Lichtblick viewer.")
status = make_cli_wrapper("npa.cli.workbench.lichtblick", "status_cmd", "Show Lichtblick status.")
list = make_cli_wrapper("npa.cli.workbench.lichtblick", "list_cmd", "List Lichtblick formats.")

__all__ = [
    "launch",
    "list",
    "serve",
    "status",
]
