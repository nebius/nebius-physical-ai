"""Share embedded and authenticated HTTP invocation across the CLI and SDK."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import httpx

from . import runtime
from .schemas import MjlabError


def invoke(
    operation, request=None, *, endpoint="", token_env="MJLAB_TOKEN", dry_run=False
):
    """Invoke one shared MJLab operation locally or through its service.

    Args:
        operation: train, eval, export, list, status or system-info.
        request: Validated capability request, if required.
        endpoint: HTTPS service URL or loopback port-forward URL.
        token_env: Name of the bearer-token environment variable.
        dry_run: Plan capability execution locally without I/O.
    Returns:
        The shared implementation's JSON result.
    Raises:
        MjlabError: Invalid transport, authentication or HTTP response.
    """
    operations = {
        "train": runtime.train,
        "eval": runtime.evaluate,
        "export": runtime.export,
        "list": runtime.list_tasks,
        "status": runtime.system_info,
        "system-info": runtime.system_info,
    }
    if operation not in operations:
        raise MjlabError("Unknown MJLab operation")
    if endpoint and not dry_run:
        return _remote(endpoint, operation, request, token_env)
    function = operations[operation]
    return function(request, dry_run=dry_run) if request is not None else function()


def _remote(endpoint, operation, request, token_env):
    parsed = urlsplit(endpoint)
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (parsed.scheme != "https" and not (parsed.scheme == "http" and loopback)) or (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise MjlabError(
            "Use HTTPS or a loopback HTTP port forward, without URL credentials"
        )
    token = os.environ.get(token_env, "")
    if not token:
        raise MjlabError(f"Set {token_env} to the service bearer token")
    try:
        response = httpx.request(
            "POST" if request is not None else "GET",
            endpoint.rstrip("/") + "/" + operation,
            json=request.model_dump() if request is not None else None,
            headers={"Authorization": f"Bearer {token}"},
            timeout=None,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        raise MjlabError("MJLab service connection failed") from exc
    if response.status_code != 200:
        raise MjlabError(f"MJLab service returned HTTP {response.status_code}")
    return response.json()
