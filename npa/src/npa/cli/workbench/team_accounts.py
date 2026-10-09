"""Manage local Workbench people and credentials through the private server configuration."""

import json
from pathlib import Path

import typer

from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.team.account_administration import issue_key_file, link_identity
from npa.workbench.team.accounts import Accounts
from npa.workbench.team.errors import TeamError
from npa.workbench.team.models import load_config

app = typer.Typer(help="Operator-managed users, personal keys, and optional SSO links.")


@app.command("create")
@json_stdout_contract
def create_cmd(
    config: Path = typer.Option(..., "--config"),
    name: str = typer.Option(..., "--name"),
    group: list[str] = typer.Option([], "--group"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Create a permanent user with optional locally managed groups.

    Args:
        config, name: Private installation configuration and unique account name.
        group: Repeat for each administrator-managed group.
        output_format: Required JSON format.
    Returns:
        None; prints account metadata without credentials.
    Raises:
        TeamError: Account cannot be created.
    """
    _json_only(output_format)
    typer.echo(json.dumps(Accounts(load_config(config)).create(name, group)))


@app.command("list")
@json_stdout_contract
def list_cmd(
    config: Path = typer.Option(..., "--config"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """List local users and credential IDs without disclosing keys.

    Args:
        config: Private server configuration.
        output_format: Required JSON format.
    Returns:
        None; prints private administrator account metadata.
    Raises:
        TeamError: Account storage is unavailable.
    """
    _json_only(output_format)
    typer.echo(json.dumps({"users": Accounts(load_config(config)).list()}))


@app.command("issue-key")
@json_stdout_contract
def issue_cmd(
    config: Path = typer.Option(..., "--config"),
    user: str = typer.Option(..., "--user"),
    output_file: Path = typer.Option(..., "--output-file"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Deliver a new personal access key to a new mode-0600 file.

    Args:
        config, user: Private installation and permanent account ID.
        output_file: New private key file; never overwrite an existing file.
        output_format: Required JSON format.
    Returns:
        None; prints credential ID and destination, never the key.
    Raises:
        TeamError, OSError: Issuance or private delivery fails.
    """
    _json_only(output_format)
    typer.echo(json.dumps(issue_key_file(load_config(config), user, output_file)))


@app.command("revoke-key")
@json_stdout_contract
def revoke_cmd(
    config: Path = typer.Option(..., "--config"),
    key_id: str = typer.Option(..., "--key-id"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Revoke a credential and the browser sessions created with it.

    Args:
        config, key_id: Private installation and exact credential ID.
        output_format: Required JSON format.
    Returns:
        None; prints revocation status.
    Raises:
        TeamError: Credential is unknown.
    """
    _json_only(output_format)
    Accounts(load_config(config)).revoke(key_id)
    typer.echo(json.dumps({"key_id": key_id, "revoked": True}))


@app.command("update")
@json_stdout_contract
def update_cmd(
    config: Path = typer.Option(..., "--config"),
    user: str = typer.Option(..., "--user"),
    disabled: bool | None = typer.Option(None, "--disabled/--enabled"),
    group: list[str] | None = typer.Option(None, "--group"),
    clear_groups: bool = typer.Option(False, "--clear-groups"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Disable an account or replace its local group membership.

    Args:
        config, user: Private installation and permanent account ID.
        disabled: Disable or re-enable authentication for this user.
        group, clear_groups: Replacement groups, or explicitly remove all memberships.
        output_format: Required JSON format.
    Returns:
        None; prints update status.
    Raises:
        TeamError: User or replacement groups are invalid.
    """
    _json_only(output_format)
    if clear_groups and group:
        raise TeamError("use --group or --clear-groups, not both")
    Accounts(load_config(config)).update(
        user, disabled=disabled, groups=[] if clear_groups else group
    )
    typer.echo(json.dumps({"user_id": user, "updated": True}))


@app.command("link")
@json_stdout_contract
def link_cmd(
    config: Path = typer.Option(..., "--config"),
    user: str = typer.Option(..., "--user"),
    issuer: str = typer.Option(..., "--issuer"),
    subject: str = typer.Option(..., "--subject"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Link a verified SSO subject to an existing account without changing ownership.

    Args:
        config, user: Private installation and permanent account ID.
        issuer, subject: Provider-verified external identity, confirmed by the operator.
        output_format: Required JSON format.
    Returns:
        None; prints link status.
    Raises:
        TeamError: Issuer is untrusted or identity already belongs to an account.
    """
    _json_only(output_format)
    typer.echo(json.dumps(link_identity(load_config(config), user, issuer, subject)))


def _json_only(value):
    if value != "json":
        raise TeamError("--output-format must be json")
