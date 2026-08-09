"""Small payload builders shared by the agent CLI and shipped backend."""

from __future__ import annotations

from typing import Any

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG


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


__all__ = ["agent_credentials_payload", "tool_catalog_payload"]
