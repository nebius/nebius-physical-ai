"""SDK wrappers for the Workbench VLM eval CLI."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper
from npa.workbench.vlm_eval import benchmark_vlm_eval, compare_vlm_judges

run = make_cli_wrapper("npa.cli.workbench.vlm_eval", "run_cmd", "Run VLM evaluation.")
benchmark = benchmark_vlm_eval
compare_judges = compare_vlm_judges
status = make_cli_wrapper(
    "npa.cli.workbench.vlm_eval", "status_cmd", "Show VLM eval status."
)
list = make_cli_wrapper(
    "npa.cli.workbench.vlm_eval", "list_cmd", "List VLM eval backends."
)
workflow = make_cli_wrapper(
    "npa.cli.workbench.vlm_eval", "workflow_cmd", "Show VLM eval workflow."
)

__all__ = [
    "benchmark",
    "compare_judges",
    "list",
    "run",
    "status",
    "workflow",
]
