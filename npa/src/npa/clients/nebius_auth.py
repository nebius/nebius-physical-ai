"""Canonical, reusable Nebius IAM token minting for registry pulls and API auth.

Any workflow that needs a fresh short-lived Nebius IAM token — for example to
build a Kubernetes image-pull secret for ``cr.*.nebius.cloud`` before a
SkyPilot / Kubernetes job — should call :func:`mint_nebius_iam_token` instead of
shelling out to ``nebius iam get-access-token`` directly.

The bare CLI is *not* robust in operator environments: when a stale/ambient
``NEBIUS_IAM_TOKEN`` (or ``NEBIUS_IAM_TOKEN_FILE``) is exported — as happens when
an operator sources a live-e2e env file — ``nebius iam get-access-token`` refuses
to perform a real profile-scoped exchange ("token from NEBIUS_IAM_TOKEN env is
used") and returns no token, so callers silently skip the pull-secret refresh and
Kubernetes pulls later fail with ``403 Forbidden`` / ``ErrImagePull``. This helper
makes the refresh work every time, without a human fixing the token by hand:

1. Perform a profile-scoped exchange (``--profile`` from ``NPA_NEBIUS_PROFILE`` /
   ``NEBIUS_PROFILE`` when set) with the ambient token env vars stripped, so the
   CLI actually mints a fresh token for the configured identity.
2. Only if that exchange fails, fall back to the ambient ``NEBIUS_IAM_TOKEN`` /
   ``NEBIUS_IAM_TOKEN_FILE`` value (best-effort; may be a different identity).

Discoverability: this is the single source of truth. ``registry_auth`` (sim2real
/ BYOF), the SONIC workflow materializer, and the serverless client all delegate
here; see ``skills/tools/nebius-infra/SKILL.md``.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

# Ambient token env vars that make the bare CLI skip a real token exchange.
AMBIENT_TOKEN_ENVS = ("NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE")
# Profile selectors, in priority order.
PROFILE_ENVS = ("NPA_NEBIUS_PROFILE", "NEBIUS_PROFILE")


class NebiusTokenError(RuntimeError):
    """Raised when a fresh Nebius IAM token cannot be obtained."""


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


def mint_nebius_iam_token(
    *,
    nebius_cli: str = "nebius",
    profile: str | None = None,
    allow_env_token: bool = True,
    timeout: int = 30,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return a short-lived Nebius IAM token, robust to an ambient token env.

    Args:
        nebius_cli: ``nebius`` CLI binary (or absolute path).
        profile: explicit profile name; defaults to :func:`nebius_profile`.
        allow_env_token: fall back to the ambient ``NEBIUS_IAM_TOKEN`` /
            ``NEBIUS_IAM_TOKEN_FILE`` value if the CLI exchange fails.
        timeout: subprocess timeout in seconds.
        env: environment to derive the profile / ambient token / exchange env
            from (defaults to ``os.environ``).

    Raises:
        NebiusTokenError: when no token can be obtained.
    """

    base_env = dict(os.environ if env is None else env)
    # Strip ambient tokens so the CLI performs a real profile-scoped exchange
    # instead of echoing (or refusing to use) the inherited token.
    exchange_env = strip_ambient_token_env(base_env)

    resolved_profile = (profile if profile is not None else nebius_profile(base_env)).strip()
    cmd = [nebius_cli]
    if resolved_profile:
        cmd.extend(["--profile", resolved_profile])
    cmd.extend(["iam", "get-access-token"])

    cli_detail = ""
    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=exchange_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        cli_detail = str(exc)
    else:
        token = (result.stdout or "").strip()
        if result.returncode == 0 and token:
            return token
        cli_detail = (
            (result.stderr or "").strip()
            or (result.stdout or "").strip()
            or f"exit {result.returncode}"
        )

    if allow_env_token:
        env_token = (base_env.get("NEBIUS_IAM_TOKEN") or "").strip()
        if env_token:
            return env_token
        token_file = (base_env.get("NEBIUS_IAM_TOKEN_FILE") or "").strip()
        if token_file:
            try:
                data = Path(token_file).read_text(encoding="utf-8").strip()
            except OSError:
                data = ""
            if data:
                return data

    raise NebiusTokenError(
        "Could not mint Nebius registry token with `nebius iam get-access-token`: "
        + (cli_detail or "no token returned")
    )
