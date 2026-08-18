"""Operator/dev VM authentication command for ``npa agent``."""

from __future__ import annotations

import typer


def auth_profile_cmd(
    ssh_host: str = typer.Option(
        ..., "--ssh-host", help="SSH hostname used from the user's local machine."
    ),
    ssh_user: str = typer.Option("", "--ssh-user", help="Optional SSH username."),
    identity_file: str = typer.Option(
        "", "--identity-file", help="Optional local-machine SSH identity path."
    ),
    profile: str = typer.Option("", "--profile", help="Nebius CLI profile name."),
    auth_timeout_seconds: int = typer.Option(
        900, "--auth-timeout-seconds", min=1, help="Authentication callback timeout."
    ),
) -> None:
    """Complete a human Nebius CLI profile on this operator/dev VM.

    The npa-agent VM normally uses its attached-service-account metadata profile.
    IAM verification output is discarded and no IAM token is printed or stored.
    """

    from npa.clients.nebius_vm_auth import (
        VmAuthError,
        redact_auth_output,
        run_vm_profile_auth,
    )

    try:
        run_vm_profile_auth(
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            identity_file=identity_file,
            profile=profile,
            auth_timeout_seconds=auth_timeout_seconds,
        )
    except VmAuthError as exc:
        typer.echo(
            f"Authentication failed safely: {redact_auth_output(str(exc))}", err=True
        )
        raise typer.Exit(code=1) from exc
