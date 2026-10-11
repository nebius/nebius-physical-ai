"""Deliver local credentials to private files and validate operator-owned identity links."""

import os
from pathlib import Path

from .accounts import Accounts
from .errors import TeamError
from .nebius_authentication import NEBIUS_ISSUER


def issue_key_file(config, user_id: str, output: Path):
    """Write a newly issued key once to a private file without printing the secret.

    Args:
        config: Local-account installation.
        user_id: Exact stable account ID.
        output: New file owned by the local operator; existing files are refused.
    Returns:
        Credential ID and private destination path, never the secret.
    Raises:
        TeamError, OSError: Account is invalid or the private file cannot be created.
    """
    accounts = Accounts(config)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    key_id = None
    try:
        with os.fdopen(descriptor, "w") as stream:
            key_id, secret = accounts.issue_key(user_id)
            stream.write(secret + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TeamError):
        if key_id:
            accounts.revoke(key_id)
        output.unlink()
        raise
    return {"key_id": key_id, "user_id": user_id, "credential_file": str(output)}


def link_identity(config, user_id, issuer, subject):
    """Link a provider-verified subject under the installation's configured issuer.

    Args:
        config: Installation with its trusted external identity provider enabled.
        user_id: Existing permanent account ID selected by the operator.
        issuer, subject: Exact verified external identity, never an email match.
    Returns:
        Account ID and link status without external personal attributes.
    Raises:
        TeamError: Issuer is unconfigured or identity cannot be linked.
    """
    configured_issuer = _configured_identity_issuer(config)
    if configured_issuer is None or issuer != configured_issuer:
        raise TeamError("link issuer must match the configured external provider")
    Accounts(config).link(user_id, issuer, subject)
    return {"user_id": user_id, "linked": True}


def unlink_identity(config, user_id, issuer, subject):
    """Remove an exact identity link, even after changing the configured provider.

    Args:
        config: Authoritative local-account installation configuration.
        user_id: Existing permanent account selected by the operator.
        issuer, subject: Exact external identity to remove, including a retired issuer.
    Returns:
        Account ID and whether a link was removed, without external attributes.
    Raises:
        TeamError: Account is absent or the identity is incomplete.
    """
    removed = Accounts(config).unlink(user_id, issuer, subject)
    return {"user_id": user_id, "unlinked": removed}


def _configured_identity_issuer(config):
    if config.identity is not None:
        return config.identity.issuer
    if config.nebius_identity is not None:
        return NEBIUS_ISSUER
    return None
