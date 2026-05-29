"""npa.workbench.fiftyone - FiftyOne workbench commands."""

from __future__ import annotations

from typing import Any

from npa._sdk import call_cli_callback, make_cli_wrapper

DEFAULT_APP_ADDRESS = "0.0.0.0"
DEFAULT_APP_PORT = 5151

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


def launch(
    *,
    port: int = DEFAULT_APP_PORT,
    address: str = DEFAULT_APP_ADDRESS,
    output: str = "text",
) -> Any:
    """Launch FiftyOne with a configurable bind address and port."""
    from npa.cli.fiftyone import launch_cmd

    return call_cli_callback(
        launch_cmd,
        port=port,
        address=address,
        output=output,
    )


curate = make_cli_wrapper(
    "npa.cli.fiftyone", "curate_cmd", "Curate and export a LeRobotDataset with FiftyOne."
)
eval = make_cli_wrapper(
    "npa.cli.fiftyone", "eval_cmd", "Evaluate checkpoint outputs with FiftyOne."
)
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
    "curate",
    "eval",
    "load_dataset",
    "restart",
    "datasets_list",
    "status",
    "system_info",
]
