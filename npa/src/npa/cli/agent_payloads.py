"""Small payload builders shared by the agent CLI and shipped backend."""

from __future__ import annotations

from typing import Any

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG


def coerce_cli_list(value: Any) -> list[str]:
    """Return a list for a possibly-unresolved Typer option default."""

    from npa.cli._typer_defaults import resolve_option_default

    try:
        value = resolve_option_default(value)
    except TypeError:  # required option with no default
        return []
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def tool_catalog_payload() -> dict[str, dict[str, Any]]:
    """Return the stable JSON-ready tool catalog embedded in agent bootstrap."""

    return {
        key: {
            "description": entry.description,
            "argv_template": list(entry.argv_template),
        }
        for key, entry in sorted(TOOL_CATALOG.items())
    }


def agent_credentials_payload(creds: dict[str, str]) -> dict[str, str]:
    """Normalize Nebius bootstrap output for persistence on an agent record."""

    return {
        "service_account_id": str(creds.get("service_account_id", "")).strip(),
        "s3_bucket": str(creds.get("s3_bucket", "")).strip(),
        "s3_prefix": str(creds.get("s3_prefix", "")).strip().strip("/"),
        "s3_endpoint": str(creds.get("s3_endpoint", "")).strip(),
        "access_key": str(creds.get("nebius_api_key", "")).strip(),
        "secret_key": str(creds.get("nebius_secret_key", "")).strip(),
    }


__all__ = ["agent_credentials_payload", "coerce_cli_list", "tool_catalog_payload"]
