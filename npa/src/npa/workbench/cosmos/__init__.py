"""npa.workbench.cosmos - NVIDIA Cosmos workbench commands."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

ensure_ingress = make_cli_wrapper(
    "npa.cli.cosmos", "ensure_ingress_cmd", "Ensure ingress for a Cosmos workbench."
)
register_byovm = make_cli_wrapper(
    "npa.cli.cosmos", "register_byovm_cmd", "Register a BYOVM Cosmos workbench."
)
list = make_cli_wrapper("npa.cli.cosmos", "list_cmd", "List Cosmos workbenches.")
deploy = make_cli_wrapper("npa.cli.cosmos", "deploy_cmd", "Deploy a Cosmos workbench.")
serve = make_cli_wrapper("npa.cli.cosmos", "serve_cmd", "Serve a Cosmos model.")
finetune = make_cli_wrapper("npa.cli.cosmos", "finetune_cmd", "Run Cosmos finetuning.")
optimize = make_cli_wrapper("npa.cli.cosmos", "optimize_cmd", "Run Cosmos optimization.")
infer = make_cli_wrapper("npa.cli.cosmos", "infer_cmd", "Run Cosmos inference.")
status = make_cli_wrapper("npa.cli.cosmos", "status_cmd", "Show Cosmos status.")
system_info = make_cli_wrapper(
    "npa.cli.cosmos", "system_info_cmd", "Show Cosmos system information."
)

__all__ = [
    "ensure_ingress",
    "register_byovm",
    "list",
    "deploy",
    "serve",
    "finetune",
    "optimize",
    "infer",
    "status",
    "system_info",
]
