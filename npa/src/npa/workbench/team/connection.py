"""Resolve explicit credentials or a saved login for the same authenticated API."""

import os
from pathlib import Path

from .client import TeamClient, load_bearer_token
from .errors import TeamError
from .nebius_login import fresh_token
from .sessions import SavedSession, resolve_session, verify_session_identity


def connection_settings(
    *,
    endpoint=None,
    token_env="NPA_TEAM_TOKEN",
    token_file=None,
    profile=None,
    ca_file=None,
):
    """Resolve a connection without forwarding saved credentials to another endpoint.

    Args:
        endpoint, token_env, token_file: Existing explicit credential options.
        profile: Optional named saved Workbench connection.
        ca_file: Optional explicit private CA certificate.
    Returns:
        Connection settings and its current bearer token, for in-memory use only.
    Raises:
        TeamError: Credentials, endpoint, saved login, or provider refresh fails.
    """
    token = None
    if token_file is not None or os.environ.get(token_env):
        token = load_bearer_token(token_env, token_file)
    if endpoint and token is not None:
        session = SavedSession(
            endpoint=endpoint, ca_file=str(ca_file) if ca_file else None
        )
        return session, token
    session, token = resolve_session(
        profile=profile, endpoint=endpoint, token=token, token_supplier=fresh_token
    )
    if ca_file:
        session = session.model_copy(
            update={"ca_file": str(Path(ca_file).expanduser().absolute())}
        )
    return session, token


def open_connection(*, client_factory=TeamClient, **options):
    """Open an authenticated connection with its verified saved placement settings.

    Args:
        client_factory: Shared client constructor, injectable for CLI tests.
        options: Explicit credentials or saved profile accepted by connection_settings.
    Returns:
        TeamClient and its resolved connection settings; caller closes the client.
    Raises:
        TeamError: Connection resolution fails or refreshed human identity changes.
    """
    session, token = connection_settings(**options)
    tls = {"ca_file": session.ca_file} if session.ca_file else {}
    client = client_factory(session.endpoint, token, **tls)
    if session.verified_subject:
        try:
            verify_session_identity(session, client.whoami())
        except TeamError:
            client.close()
            raise
    return client, session


def connect(*, profile=None):
    """Use an npa login connection from an agent or Python caller.

    Args:
        profile: Optional named Workbench connection; otherwise use the active one.
    Returns:
        Authenticated TeamClient; caller must close it after use.
    Raises:
        TeamError: Login is absent, invalid, expired, or resolves to another person.
    """
    return open_connection(profile=profile)[0]
