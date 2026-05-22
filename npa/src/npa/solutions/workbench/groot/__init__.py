"""npa.solutions.workbench.groot - NVIDIA Isaac GR00T workbench commands."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

ensure_ingress = make_cli_wrapper(
    "npa.cli.groot", "ensure_ingress_cmd", "Ensure ingress for a GR00T workbench."
)
register_byovm = make_cli_wrapper(
    "npa.cli.groot", "register_byovm_cmd", "Register a BYOVM GR00T workbench."
)
list = make_cli_wrapper("npa.cli.groot", "list_cmd", "List GR00T workbenches.")
deploy = make_cli_wrapper("npa.cli.groot", "deploy_cmd", "Deploy a GR00T workbench.")
download = make_cli_wrapper("npa.cli.groot", "download_cmd", "Download a GR00T model.")
reload_env = make_cli_wrapper(
    "npa.cli.groot", "reload_env_cmd", "Reload a GR00T runtime environment."
)
finetune = make_cli_wrapper("npa.cli.groot", "finetune_cmd", "Finetune a GR00T model.")
eval = make_cli_wrapper("npa.cli.groot", "eval_cmd", "Evaluate a GR00T model.")
serve = make_cli_wrapper("npa.cli.groot", "serve_cmd", "Serve a GR00T model.")
infer = make_cli_wrapper("npa.cli.groot", "infer_cmd", "Run GR00T inference.")
convert = make_cli_wrapper("npa.cli.groot", "convert_cmd", "Convert GR00T artifacts.")
status = make_cli_wrapper("npa.cli.groot", "status_cmd", "Show GR00T status.")
system_info = make_cli_wrapper(
    "npa.cli.groot", "system_info_cmd", "Show GR00T system information."
)

__all__ = [
    "ensure_ingress",
    "register_byovm",
    "list",
    "deploy",
    "download",
    "reload_env",
    "finetune",
    "eval",
    "serve",
    "infer",
    "convert",
    "status",
    "system_info",
]
