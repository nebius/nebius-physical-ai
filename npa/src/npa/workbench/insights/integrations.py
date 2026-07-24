"""Seam to the LanceDB workbench tool as the optional insights query index.

The insights store is durable on S3 as append-only JSONL; LanceDB is an
*optional* query index the same way the ``dataset`` tool uses it. These
functions are the call-their-service seam and are mocked at the call site in
unit tests. When no endpoint is configured, callers fall back to the JSONL
store so insights works without a running LanceDB.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class InsightsIntegrationError(RuntimeError):
    """Raised when a downstream workbench service call fails."""


def index_metrics_in_lancedb(
    records: list[dict[str, Any]],
    *,
    lancedb_endpoint: str,
    table: str,
    lance_uri: str = "",
    token_env: str = "LANCEDB_TOKEN",
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Register metric records in the LanceDB index.

    No-ops when ``lancedb_endpoint`` is empty so the store remains the durable
    source of truth without a running LanceDB service.
    """
    if not lancedb_endpoint.strip():
        return {"indexed": False, "backend": "jsonl", "table": table}
    payload = {"table": table, "lance_uri": lance_uri, "records": records}
    data = _post(lancedb_endpoint, "/index", payload=payload, token_env=token_env, timeout=timeout)
    return {"indexed": True, "backend": "lancedb", "table": table, **data}


def query_metrics_in_lancedb(
    *,
    lancedb_endpoint: str,
    filter_predicate: dict[str, Any],
    limit: int,
    token_env: str = "LANCEDB_TOKEN",
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Query the LanceDB-backed metric index by the given facet predicate."""
    payload = {"filter": filter_predicate, "limit": limit}
    data = _post(lancedb_endpoint, "/query", payload=payload, token_env=token_env, timeout=timeout)
    records = data.get("records", [])
    if not isinstance(records, list):
        raise InsightsIntegrationError("LanceDB query returned an unexpected response")
    return records


def _post(
    endpoint: str,
    path: str,
    *,
    payload: dict[str, Any],
    token_env: str,
    timeout: float,
) -> dict[str, Any]:
    resolved = endpoint.strip().rstrip("/")
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.post(f"{resolved}{path}", headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise InsightsIntegrationError(f"workbench service call failed ({resolved}{path}): {exc}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise InsightsIntegrationError("workbench service returned non-JSON response") from exc
    if not isinstance(data, dict):
        raise InsightsIntegrationError("workbench service returned an unexpected response")
    return data
