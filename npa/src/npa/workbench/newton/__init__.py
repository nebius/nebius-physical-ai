"""npa.workbench.newton - Newton physics engine workbench commands."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

train_teacher = make_cli_wrapper(
    "npa.cli.newton", "train_teacher_cmd", "Train a Newton teacher policy."
)
generate_demos = make_cli_wrapper(
    "npa.cli.newton", "generate_demos_cmd", "Generate Newton demonstrations."
)
eval = make_cli_wrapper(
    "npa.cli.newton", "eval_cmd", "Evaluate a policy in Newton simulation."
)

__all__ = [
    "train_teacher",
    "generate_demos",
    "eval",
]
