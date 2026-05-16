"""npa.workbench.lancedb - LanceDB workbench SDK functions."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import fields
from typing import Any

import httpx

from npa.workbench.lancedb.bdd100k_import import (
    DEFAULT_LANCE_URI,
    DEFAULT_SPLITS,
    DEFAULT_TABLE,
    BDD100KImportError,
    BDD100KImportResult,
    BDD100KValidationError,
    import_bdd100k as _import_bdd100k_local,
)

DEFAULT_TOKEN_ENV = "LANCEDB_TOKEN"


class BDD100KServiceError(BDD100KImportError):
    """Raised when a service-mode BDD100K import request fails."""


def import_bdd100k(
    *,
    source: str = "",
    table: str = DEFAULT_TABLE,
    lance_uri: str = DEFAULT_LANCE_URI,
    synthetic: int | None = None,
    synthetic_seed: int | None = None,
    splits: Iterable[str] | None = None,
    limit: int | None = None,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 600.0,
) -> BDD100KImportResult:
    """Import BDD100K rows into LanceDB.

    Example:
        from npa.sdk.workbench.lancedb import import_bdd100k

        result = import_bdd100k(synthetic=100, synthetic_seed=42)
        print(f"Ingested {result.total_rows} rows; table version {result.table_version}")
    """
    split_values = list(splits) if splits is not None else list(DEFAULT_SPLITS)
    service_mode = _resolve_mode(mode=mode, service=service)
    if service_mode:
        payload = {
            "source": source,
            "table": table,
            "lance_uri": lance_uri,
            "synthetic": synthetic,
            "synthetic_seed": synthetic_seed,
            "splits": split_values,
            "limit": limit,
        }
        return _result_from_payload(
            _post_json(
                endpoint=endpoint or os.environ.get("NPA_LANCEDB_ENDPOINT", ""),
                token_env=token_env,
                payload=payload,
                timeout=timeout,
            )
        )
    return _import_bdd100k_local(
        source=source,
        table=table,
        lance_uri=lance_uri,
        synthetic=synthetic,
        synthetic_seed=synthetic_seed,
        splits=split_values,
        limit=limit,
    )


def _resolve_mode(*, mode: str | None, service: bool) -> bool:
    if mode is None:
        return service
    value = mode.strip().lower()
    if value == "local":
        return False
    if value == "service":
        return True
    raise BDD100KValidationError("mode must be either 'local' or 'service'")


def _post_json(
    *,
    endpoint: str,
    token_env: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    resolved = endpoint.strip().rstrip("/")
    if not resolved:
        raise BDD100KValidationError("endpoint is required for service mode")
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.post(f"{resolved}/import-bdd100k", json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise BDD100KServiceError(f"LanceDB service request failed ({exc.response.status_code}): {detail}") from exc
    except httpx.HTTPError as exc:
        raise BDD100KServiceError(f"Cannot reach LanceDB service {resolved}: {exc}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise BDD100KServiceError("LanceDB service returned non-JSON response") from exc
    if not isinstance(data, dict):
        raise BDD100KServiceError("LanceDB service returned an unexpected response")
    return data


def _result_from_payload(payload: dict[str, Any]) -> BDD100KImportResult:
    names = {field.name for field in fields(BDD100KImportResult)}
    missing = sorted(name for name in names if name not in payload)
    if missing:
        joined = ", ".join(missing)
        raise BDD100KServiceError(f"LanceDB service response is missing: {joined}")
    return BDD100KImportResult(**{name: payload[name] for name in names})


__all__ = ["import_bdd100k"]
