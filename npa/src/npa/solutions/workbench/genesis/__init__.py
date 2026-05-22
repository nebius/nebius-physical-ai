"""npa.solutions.workbench.genesis - Genesis simulation workbench commands."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

train_teacher = make_cli_wrapper(
    "npa.cli.genesis", "train_teacher_cmd", "Train a Genesis teacher policy."
)
generate_demos = make_cli_wrapper(
    "npa.cli.genesis", "generate_demos_cmd", "Generate Genesis demonstrations."
)
simulate = generate_demos
eval_teacher = make_cli_wrapper(
    "npa.cli.genesis", "eval_teacher_cmd", "Evaluate a Genesis teacher policy."
)
eval_student = make_cli_wrapper(
    "npa.cli.genesis", "eval_student_cmd", "Evaluate a student policy in Genesis."
)
diagnose = make_cli_wrapper("npa.cli.genesis", "diagnose_cmd", "Diagnose Genesis rollouts.")
tune = make_cli_wrapper("npa.cli.genesis", "tune_cmd", "Tune Genesis policy settings.")
list = make_cli_wrapper("npa.cli.genesis", "list_cmd", "List Genesis workbenches.")
deploy = make_cli_wrapper("npa.cli.genesis", "deploy_cmd", "Deploy a Genesis workbench.")
status = make_cli_wrapper("npa.cli.genesis", "status_cmd", "Show Genesis status.")
system_info = make_cli_wrapper(
    "npa.cli.genesis", "system_info_cmd", "Show Genesis system information."
)

__all__ = [
    "train_teacher",
    "generate_demos",
    "simulate",
    "eval_teacher",
    "eval_student",
    "diagnose",
    "tune",
    "list",
    "deploy",
    "status",
    "system_info",
]
