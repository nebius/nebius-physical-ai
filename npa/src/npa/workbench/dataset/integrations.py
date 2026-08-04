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


#: Facets that are thresholds or metadata about the query, not equality predicates on a row.
_NON_EQUALITY_FACETS = frozenset({"quality_metric", "min_quality"})


def equality_facets(filter_predicate: dict[str, Any]) -> dict[str, Any]:
    """Return only the facets that are equality predicates on an actual column.

    The caller builds a predicate with every facet it knows about, including the ones left
    unset. Sending those verbatim asks the index for `modality = '' AND min_quality = 'None'`,
    which matches nothing — live, a query returned 0 records from a table that held three rows
    matching its event and location (EVIDENCE.md §R41).
    """

    return {
        key: value
        for key, value in filter_predicate.items()
        if key not in _NON_EQUALITY_FACETS and value not in ("", None)
    }


def query_lancedb(
    *,
    lancedb_endpoint: str,
    filter_predicate: dict[str, Any],
    limit: int,
    table: str = "",
    lance_uri: str = "",
    token_env: str = "LANCEDB_TOKEN",
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Query the LanceDB-backed index by the given facet predicate."""
    payload: dict[str, Any] = {"filter": equality_facets(filter_predicate), "limit": limit}
    if table:
        payload["table"] = table
    if lance_uri:
        payload["lance_uri"] = lance_uri
    data = _post(lancedb_endpoint, "/query", payload=payload, token_env=token_env, timeout=timeout)
    records = data.get("records", [])
    if not isinstance(records, list):
        raise DatasetIntegrationError("LanceDB query returned an unexpected response")
    threshold = filter_predicate.get("min_quality")
    if threshold is not None:
        # A threshold is not an equality predicate, so it is applied here rather than pushed
        # into a facet API that has no operators.
        metric = str(filter_predicate.get("quality_metric") or "completeness")
        records = [
            record
            for record in records
            if isinstance(record.get(metric), (int, float)) and record[metric] >= float(threshold)
        ]
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
