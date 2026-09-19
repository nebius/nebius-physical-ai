"""npa.workbench.molmoact - MolmoAct VLA workbench commands."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

finetune = make_cli_wrapper(
    "npa.cli.molmoact", "finetune_cmd", "Fine-tune a MolmoAct policy."
)
serve = make_cli_wrapper("npa.cli.molmoact", "serve_cmd", "Serve a MolmoAct policy.")
eval = make_cli_wrapper("npa.cli.molmoact", "eval_cmd", "Evaluate a MolmoAct policy.")

__all__ = [
    "finetune",
    "serve",
    "eval",
]
