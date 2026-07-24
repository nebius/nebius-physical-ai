"""Seams to the LanceDB (query index) and FiftyOne (curation) workbench tools.

The dataset-of-record composes existing workbench primitives rather than
re-implementing them: LanceDB backs the metadata + embedding query index, and
FiftyOne receives curation/visualization handoffs. These functions are the
call-their-services seam and are mocked at the call site in unit tests.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class DatasetIntegrationError(RuntimeError):
    """Raised when a downstream workbench service call fails."""


def index_in_lancedb(
    records: list[dict[str, Any]],
    *,
    lancedb_endpoint: str,
    table: str,
    lance_uri: str,
    token_env: str = "LANCEDB_TOKEN",
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Register dataset records (metadata + embeddings) in the LanceDB index.

    No-ops when ``lancedb_endpoint`` is empty so ingest works without a running
    LanceDB service; the manifest remains the durable source of truth.
    """
    if not lancedb_endpoint.strip():
        return {"indexed": False, "backend": "manifest", "table": table}
    payload = {"table": table, "lance_uri": lance_uri, "records": records}
    data = _post(lancedb_endpoint, "/index", payload=payload, token_env=token_env, timeout=timeout)
    return {"indexed": True, "backend": "lancedb", "table": table, **data}


def query_lancedb(
    *,
    lancedb_endpoint: str,
    filter_predicate: dict[str, Any],
    limit: int,
    token_env: str = "LANCEDB_TOKEN",
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Query the LanceDB-backed index by the given facet predicate."""
    payload = {"filter": filter_predicate, "limit": limit}
    data = _post(lancedb_endpoint, "/query", payload=payload, token_env=token_env, timeout=timeout)
    records = data.get("records", [])
    if not isinstance(records, list):
        raise DatasetIntegrationError("LanceDB query returned an unexpected response")
    return records


def fiftyone_handoff(
    *,
    fiftyone_endpoint: str,
    manifest_uri: str,
    dataset_id: str,
    version: str,
    token_env: str = "FIFTYONE_TOKEN",
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Hand a registered dataset version to FiftyOne for curation/visualization.

    No-ops when ``fiftyone_endpoint`` is empty.
    """
    if not fiftyone_endpoint.strip():
        return {"handoff": False}
    payload = {"manifest_uri": manifest_uri, "dataset_id": dataset_id, "version": version}
    data = _post(fiftyone_endpoint, "/load-dataset", payload=payload, token_env=token_env, timeout=timeout)
    return {"handoff": True, **data}


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
        raise DatasetIntegrationError(f"workbench service call failed ({resolved}{path}): {exc}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise DatasetIntegrationError("workbench service returned non-JSON response") from exc
    if not isinstance(data, dict):
        raise DatasetIntegrationError("workbench service returned an unexpected response")
    return data
