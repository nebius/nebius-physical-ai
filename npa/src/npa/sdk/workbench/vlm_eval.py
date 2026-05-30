"""SDK wrappers for the Workbench VLM eval CLI."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper
from npa.workbench.vlm_eval import benchmark_vlm_eval

run = make_cli_wrapper("npa.cli.workbench.vlm_eval", "run_cmd", "Run VLM evaluation.")
benchmark = benchmark_vlm_eval
status = make_cli_wrapper("npa.cli.workbench.vlm_eval", "status_cmd", "Show VLM eval status.")
list = make_cli_wrapper("npa.cli.workbench.vlm_eval", "list_cmd", "List VLM eval backends.")
workflow = make_cli_wrapper("npa.cli.workbench.vlm_eval", "workflow_cmd", "Show VLM eval workflow.")

__all__ = [
    "benchmark",
    "list",
    "run",
    "status",
    "workflow",
]
