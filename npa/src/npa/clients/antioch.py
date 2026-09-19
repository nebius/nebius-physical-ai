"""Resolve NPA credentials exclusively for local Antioch client subprocesses."""

from __future__ import annotations

import os

from npa.clients.credentials import ANTIOCH_TOKEN_KEY, CredentialStoreError, load_credentials


def antioch_environment() -> dict[str, str]:
    """Build a child environment with the operator's resolved Antioch token.

    Args:
        None.
    Returns:
        A new environment mapping; an explicit token overrides credentials.yaml.
        With neither present, the Antioch CLI retains its native browser login.
    Raises:
        CredentialStoreError: The configured NPA credential file is unusable.
    """
    environment = dict(os.environ)
    try:
        credentials = load_credentials(environ=environment)
    except CredentialStoreError as error:
        # Standalone operator entrypoints must not print YAML parser causes,
        # which can contain the malformed secret scalar.
        raise error from None
    if credentials.antioch_token:
        environment[ANTIOCH_TOKEN_KEY] = credentials.antioch_token
    return environment
