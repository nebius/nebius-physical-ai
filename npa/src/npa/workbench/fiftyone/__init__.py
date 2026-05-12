"""npa.workbench.fiftyone - FiftyOne workbench commands."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

ensure_ingress = make_cli_wrapper(
    "npa.cli.fiftyone", "ensure_ingress_cmd", "Ensure ingress for a FiftyOne workbench."
)
register_byovm = make_cli_wrapper(
    "npa.cli.fiftyone", "register_byovm_cmd", "Register a BYOVM FiftyOne workbench."
)
list = make_cli_wrapper("npa.cli.fiftyone", "list_cmd", "List FiftyOne workbenches.")
deploy = make_cli_wrapper(
    "npa.cli.fiftyone", "deploy_cmd", "Deploy a FiftyOne workbench."
)
launch = make_cli_wrapper("npa.cli.fiftyone", "launch_cmd", "Launch FiftyOne.")
load_dataset = make_cli_wrapper(
    "npa.cli.fiftyone", "load_dataset_cmd", "Load a dataset into FiftyOne."
)
restart = make_cli_wrapper("npa.cli.fiftyone", "restart_cmd", "Restart FiftyOne.")
datasets_list = make_cli_wrapper(
    "npa.cli.fiftyone", "datasets_list_cmd", "List datasets in FiftyOne."
)
status = make_cli_wrapper("npa.cli.fiftyone", "status_cmd", "Show FiftyOne status.")
system_info = make_cli_wrapper(
    "npa.cli.fiftyone", "system_info_cmd", "Show FiftyOne system information."
)

__all__ = [
    "ensure_ingress",
    "register_byovm",
    "list",
    "deploy",
    "launch",
    "load_dataset",
    "restart",
    "datasets_list",
    "status",
    "system_info",
]
