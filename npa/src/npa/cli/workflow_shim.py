"""Deprecated top-level workflow CLI shim."""

from __future__ import annotations

import warnings

import typer

from npa.cli.workflow import app as workflow_app

workflow_shim_app = typer.Typer(
    name="workflow",
    help="Multi-stage training workflow orchestration.",
    no_args_is_help=True,
)
app = workflow_shim_app


@workflow_shim_app.callback(invoke_without_command=True)
def workflow_shim_callback() -> None:
    """Warn callers to use the canonical workbench workflow namespace."""
    # SHIM-REMOVE: delete this shim after the next major release migration.
    warnings.warn(
        "npa workflow is deprecated; use npa workbench workflow instead",
        DeprecationWarning,
        stacklevel=1,
    )


for command in workflow_app.registered_commands:
    if command.callback is None:
        continue
    workflow_shim_app.command(
        name=command.name,
        cls=command.cls,
        context_settings=command.context_settings,
        help=command.help,
        epilog=command.epilog,
        short_help=command.short_help,
        options_metavar=command.options_metavar,
        add_help_option=command.add_help_option,
        no_args_is_help=command.no_args_is_help,
        hidden=command.hidden,
        deprecated=command.deprecated,
        rich_help_panel=command.rich_help_panel,
    )(command.callback)
