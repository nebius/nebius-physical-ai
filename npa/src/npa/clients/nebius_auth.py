"""Nebius CLI profile selection and stale ambient-token scrubbing."""

from __future__ import annotations

import os
from collections.abc import Mapping

# Ambient token env vars that make the bare CLI skip a real token exchange.
AMBIENT_TOKEN_ENVS = ("NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE")
# Profile selectors, in priority order.
PROFILE_ENVS = ("NPA_NEBIUS_PROFILE", "NEBIUS_PROFILE")


def strip_ambient_token_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of ``env`` with ambient Nebius token vars removed.

    A stale/ambient ``NEBIUS_IAM_TOKEN`` poisons *any* nebius-authenticated
    subprocess, not just token minting: ``kubectl`` and ``sky`` use the nebius
    exec-credential plugin, which prefers the env token and fails with
    "Invalid token" when it is stale. Run those subprocesses with this cleaned
    env so the plugin re-authenticates via the configured profile / SA metadata.
    """

    source = os.environ if env is None else env
    return {k: v for k, v in source.items() if k not in AMBIENT_TOKEN_ENVS}


def nebius_profile(env: Mapping[str, str] | None = None) -> str:
    """Return the configured Nebius CLI profile name, or ``""`` when unset."""

    source = os.environ if env is None else env
    for name in PROFILE_ENVS:
        value = (source.get(name) or "").strip()
        if value:
            return value
    return ""
