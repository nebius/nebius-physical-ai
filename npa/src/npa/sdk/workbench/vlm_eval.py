"""SDK wrappers for the Workbench VLM eval stub CLI."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

run = make_cli_wrapper("npa.cli.workbench.vlm_eval", "run_cmd", "Run stub VLM evaluation.")
status = make_cli_wrapper("npa.cli.workbench.vlm_eval", "status_cmd", "Show VLM eval status.")
list = make_cli_wrapper("npa.cli.workbench.vlm_eval", "list_cmd", "List VLM eval backends.")

__all__ = ["list", "run", "status"]
