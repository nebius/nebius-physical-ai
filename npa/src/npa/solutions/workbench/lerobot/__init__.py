"""npa.solutions.workbench.lerobot - LeRobot workbench commands."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

list = make_cli_wrapper("npa.cli.workbench.lerobot", "list_cmd", "List LeRobot workbenches.")
status = make_cli_wrapper(
    "npa.cli.workbench.lerobot", "status", "Show LeRobot workbench status."
)
train = make_cli_wrapper("npa.cli.workbench.lerobot", "train", "Train a LeRobot policy.")
eval = make_cli_wrapper("npa.cli.workbench.lerobot", "eval_cmd", "Evaluate a policy.")
serve = make_cli_wrapper("npa.cli.workbench.lerobot", "serve", "Serve a LeRobot policy.")
infer = make_cli_wrapper("npa.cli.workbench.lerobot", "infer", "Run LeRobot inference.")
list_checkpoints = make_cli_wrapper(
    "npa.cli.workbench.lerobot", "list_checkpoints", "List LeRobot checkpoints."
)
deploy = make_cli_wrapper(
    "npa.cli.workbench.lerobot", "deploy", "Deploy a LeRobot workbench."
)
system_info = make_cli_wrapper(
    "npa.cli.workbench.lerobot", "system_info_cmd", "Show LeRobot system information."
)
benchmark = make_cli_wrapper(
    "npa.cli.workbench.lerobot", "benchmark_cmd", "Run LeRobot benchmark checks."
)
profile_train = make_cli_wrapper(
    "npa.cli.workbench.lerobot", "profile_train_cmd", "Profile LeRobot training."
)
train_student = make_cli_wrapper(
    "npa.cli.workbench.lerobot", "train_student_cmd", "Train a LeRobot student policy."
)

__all__ = [
    "list",
    "status",
    "train",
    "eval",
    "serve",
    "infer",
    "list_checkpoints",
    "deploy",
    "system_info",
    "benchmark",
    "profile_train",
    "train_student",
]
