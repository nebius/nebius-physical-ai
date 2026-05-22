"""npa.solutions.workbench.isaac_lab - Isaac Lab workbench commands."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

list = make_cli_wrapper("npa.cli.isaac_lab", "list_cmd", "List Isaac Lab workbenches.")
deploy = make_cli_wrapper(
    "npa.cli.isaac_lab", "deploy_cmd", "Deploy an Isaac Lab workbench."
)
status = make_cli_wrapper("npa.cli.isaac_lab", "status_cmd", "Show Isaac Lab status.")
system_info = make_cli_wrapper(
    "npa.cli.isaac_lab", "system_info_cmd", "Show Isaac Lab system information."
)
train = make_cli_wrapper("npa.cli.isaac_lab", "train_cmd", "Train in Isaac Lab.")
eval = make_cli_wrapper("npa.cli.isaac_lab", "eval_cmd", "Evaluate in Isaac Lab.")
export_lerobot = make_cli_wrapper(
    "npa.cli.isaac_lab", "export_lerobot_cmd", "Export Isaac Lab rollouts as LeRobot."
)

__all__ = [
    "list",
    "deploy",
    "status",
    "system_info",
    "train",
    "eval",
    "export_lerobot",
]
