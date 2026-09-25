"""Operator/dev VM authentication command for ``npa agent``."""

from __future__ import annotations

import typer

import os
import shutil
from pathlib import Path


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


# --- Local agent auth-secret file helpers (auth.env management) ---
# Extracted from the npa.cli.agent god object (issue #491). npa.cli.agent
# re-exports these names for backward compatibility.


def _auth_secret_path(project_alias: str, name: str) -> Path:
    root = Path(os.environ.get("NPA_CONFIG_DIR", "").strip() or Path.home() / ".npa")
    return root / "agents" / project_alias / name / "auth.env"


def _cleanup_agent_local_files(project_alias: str, name: str) -> None:
    """Remove the local agent state + Terraform workdir after a destroy.

    Two trees live under ``~/.npa`` for an agent: ``agents/<alias>/<name>/``
    (auth.env + secrets — live basic-auth credentials, a stale-credential leak
    if left) and ``workbenches/<alias>/<name>/`` (the Terraform workdir with the
    provider cache and, in a local backend, ``terraform.tfstate``). Terraform has
    already destroyed the VM by the time this runs, so both are safe to remove;
    leaving the workdir behind was the teardown-report leftover.
    """
    agent_dir = _auth_secret_path(project_alias, name).parent
    shutil.rmtree(agent_dir, ignore_errors=True)

    from npa.deploy import provisioner

    tf_dir = provisioner.working_dir_path(project_alias, name)
    shutil.rmtree(tf_dir, ignore_errors=True)

    # Drop the now-empty <alias> parents so tearing down the last agent leaves no
    # empty ~/.npa/{agents,workbenches}/<alias>/ tree behind (a sibling agent
    # under the same alias keeps its parent non-empty, so it is preserved).
    for parent in (agent_dir.parent, tf_dir.parent):
        try:
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def _write_auth_secret(
    *, project_alias: str, name: str, user: str, password: str
) -> Path:
    path = _auth_secret_path(project_alias, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"AGENT_USER={user}\nAGENT_PASSWORD={password}\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _load_auth_secret(path: str) -> tuple[str, str]:
    secret_path = Path(path).expanduser()
    if not secret_path.exists():
        raise ValueError(f"auth secret not found: {secret_path}")
    values: dict[str, str] = {}
    for raw in secret_path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    user = values.get("AGENT_USER", "")
    password = values.get("AGENT_PASSWORD", "")
    if not user or not password:
        raise ValueError(
            f"auth secret missing AGENT_USER/AGENT_PASSWORD: {secret_path}"
        )
    return user, password
