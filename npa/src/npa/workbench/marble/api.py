"""Call the official World API without exposing keys or replaying generation POSTs."""

import os
import time

import httpx

from npa.clients.credentials import load_credentials

API_BASE = "https://api.worldlabs.ai/marble/v1"


class MarbleError(RuntimeError):
    """Report a Marble operation failure without secret-bearing response bodies.

    Args: A public diagnostic message.
    Returns: An exception instance.
    Raises: None.
    """


def require_api_key():
    """Resolve the operator's World Labs credential before a generation intent.

    Args: None.
    Returns: The configured secret, for authenticated requests only.
    Raises: MarbleError when no credential is configured.
    """
    key = os.environ.get("WLT_API_KEY") or load_credentials().tokens.get("WLT_API_KEY")
    if not key or not key.strip():
        raise MarbleError("Set WLT_API_KEY from https://platform.worldlabs.ai/api-keys")
    return key.strip()


def api_request(method, route, payload=None):
    """Send one authenticated request to the fixed official API origin.

    Args: HTTP method, relative API route, and optional JSON payload.
    Returns: The JSON response.
    Raises: MarbleError on missing credentials or provider failure.
    """
    key = require_api_key()
    try:
        response = httpx.request(
            method,
            f"{API_BASE}/{route}",
            json=payload,
            headers={"WLT-Api-Key": key},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise MarbleError(
            f"World API returned HTTP {exc.response.status_code}"
        ) from None
    except (httpx.HTTPError, ValueError):
        raise MarbleError(
            "World API transport or JSON failure; generation was not retried"
        ) from None


def await_world(operation_id):
    """Poll a previously recorded operation without starting another generation.

    Args: Provider operation identifier.
    Returns: The complete generated world.
    Raises: MarbleError if generation failed or returned no world.
    """
    while True:
        operation = api_request("GET", f"operations/{operation_id}")
        if operation.get("error"):
            raise MarbleError(
                "World generation failed; inspect the saved operation in World Labs"
            )
        if operation.get("done"):
            world = operation.get("response") or {}
            world_id = world.get("id") or (operation.get("metadata") or {}).get(
                "world_id"
            )
            if not world_id:
                raise MarbleError("Completed operation has no world identifier")
            return api_request("GET", f"worlds/{world_id}")["world"]
        time.sleep(5)
