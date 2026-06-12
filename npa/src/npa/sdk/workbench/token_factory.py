"""SDK wrappers for the Workbench Nebius Token Factory CLI."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

caption = make_cli_wrapper("npa.cli.workbench.token_factory", "caption_cmd", "Caption images with Token Factory.")
generate = make_cli_wrapper("npa.cli.workbench.token_factory", "generate_cmd", "Generate text with Token Factory.")
reason = make_cli_wrapper("npa.cli.workbench.token_factory", "reason_cmd", "Reason over scene images with Token Factory.")
models = make_cli_wrapper("npa.cli.workbench.token_factory", "models_cmd", "List Token Factory models.")
verify = make_cli_wrapper("npa.cli.workbench.token_factory", "verify_cmd", "Verify Token Factory auth.")
status = make_cli_wrapper("npa.cli.workbench.token_factory", "status_cmd", "Show Token Factory status.")
list = make_cli_wrapper("npa.cli.workbench.token_factory", "list_cmd", "List Token Factory capabilities.")
workflow = make_cli_wrapper("npa.cli.workbench.token_factory", "workflow_cmd", "Show Token Factory workflows.")

__all__ = [
    "caption",
    "generate",
    "list",
    "models",
    "reason",
    "status",
    "verify",
    "workflow",
]
