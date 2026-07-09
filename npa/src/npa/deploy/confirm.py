"""Shared confirmation helpers for destructive deploy operations."""

from __future__ import annotations

import typer


def confirm_or_exit(prompt: str) -> None:
    """Prompt the operator and abort the CLI command when they decline."""
    if not typer.confirm(prompt, default=False):
        typer.echo("Aborted.")
        raise typer.Exit(code=1)


def confirm_vm_destroy(
    project: str,
    name: str,
    *,
    byovm: bool,
    dry_run: bool,
    yes: bool,
) -> None:
    """Require confirmation before Terraform-managed VM destroy.

    BYOVM destroy only unregisters a local alias (non-destructive) and dry-run
    does not mutate cloud state, so those paths skip the prompt. Pass ``--yes``
    for non-interactive automation.
    """
    if byovm or dry_run or yes:
        return
    confirm_or_exit(
        f"Destroy Terraform-managed VM '{project}/{name}'? "
        "This deletes cloud infrastructure and cannot be undone."
    )
