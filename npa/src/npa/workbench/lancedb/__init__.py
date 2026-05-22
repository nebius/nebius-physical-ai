"""npa.workbench.lancedb - LanceDB workbench SDK functions."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import MISSING, fields
from typing import Any

import httpx

from npa.workbench.lancedb.backfill import (
    DEFAULT_DHASH_HAMMING_THRESHOLD,
    DEFAULT_GPU_DEVICE,
    BackfillError,
    BackfillResult,
    BackfillValidationError,
    backfill_column as _backfill_column_local,
)
from npa.workbench.lancedb.bdd100k_import import (
    DEFAULT_LANCE_URI,
    DEFAULT_SPLITS,
    DEFAULT_TABLE,
    BDD100KImportError,
    BDD100KImportResult,
    BDD100KValidationError,
    import_bdd100k as _import_bdd100k_local,
)
from npa.workbench.lancedb.views import (
    DEFAULT_QUERY_LIMIT,
    MVError,
    MVResult,
    MVValidationError,
    QueryResult,
    create_bdd100k_failure_mode_views as _create_bdd100k_failure_mode_views_local,
    create_mv as _create_mv_local,
    query_table as _query_table_local,
    refresh_mv as _refresh_mv_local,
)

DEFAULT_TOKEN_ENV = "LANCEDB_TOKEN"


class BDD100KServiceError(BDD100KImportError):
    """Raised when a service-mode BDD100K import request fails."""


class BackfillServiceError(BackfillError):
    """Raised when a service-mode backfill request fails."""


class MVServiceError(MVError):
    """Raised when a service-mode materialized-view request fails."""


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


def backfill(
    *,
    table: str = DEFAULT_TABLE,
    udf: str,
    lance_uri: str = DEFAULT_LANCE_URI,
    batch_size: int | None = None,
    force: bool = False,
    force_recompute: bool | None = None,
    dhash_hamming_threshold: int = DEFAULT_DHASH_HAMMING_THRESHOLD,
    device: str = DEFAULT_GPU_DEVICE,
    precision: str | None = None,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 600.0,
) -> BackfillResult:
    """Backfill one BDD100K-derived LanceDB column.

    Example:
        from npa.sdk.workbench.lancedb import backfill

        result = backfill(table="bdd100k", udf="has_person")
        print(f"Updated {result.rows_updated} rows; version {result.table_version_after}")
    """
    service_mode = _resolve_mode(mode=mode, service=service)
    payload = {
        "table": table,
        "udf": udf,
        "lance_uri": lance_uri,
        "batch_size": batch_size,
        "force": force,
        "force_recompute": force_recompute,
        "dhash_hamming_threshold": dhash_hamming_threshold,
        "device": device,
        "precision": precision,
    }
    if service_mode:
        return _backfill_result_from_payload(
            _post_json(
                endpoint=endpoint or os.environ.get("NPA_LANCEDB_ENDPOINT", ""),
                token_env=token_env,
                payload=payload,
                timeout=timeout,
                path="/backfill",
                error_cls=BackfillServiceError,
                validation_cls=BackfillValidationError,
            )
        )
    return _backfill_column_local(**payload)


def create_mv(
    *,
    name: str,
    source_table: str = DEFAULT_TABLE,
    filter_sql: str,
    lance_uri: str = DEFAULT_LANCE_URI,
    force: bool = False,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 600.0,
) -> MVResult:
    """Create a LanceDB materialized view.

    Example:
        from npa.sdk.workbench.lancedb import create_mv

        result = create_mv(
            name="my_view",
            source_table="bdd100k",
            filter_sql="weather = 'rainy' AND split = 'train'",
        )
        print(f"Created {result.view_name} with {result.row_count} rows")
    """
    service_mode = _resolve_mode_as(mode=mode, service=service, validation_cls=MVValidationError)
    payload = {
        "name": name,
        "source_table": source_table,
        "filter_sql": filter_sql,
        "lance_uri": lance_uri,
        "force": force,
    }
    if service_mode:
        return _mv_result_from_payload(
            _post_json(
                endpoint=endpoint or os.environ.get("NPA_LANCEDB_ENDPOINT", ""),
                token_env=token_env,
                payload=payload,
                timeout=timeout,
                path="/create-mv",
                error_cls=MVServiceError,
                validation_cls=MVValidationError,
            )
        )
    return _create_mv_local(**payload)


def refresh_mv(
    *,
    name: str,
    lance_uri: str = DEFAULT_LANCE_URI,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 600.0,
) -> MVResult:
    """Refresh a registered LanceDB materialized view."""
    service_mode = _resolve_mode_as(mode=mode, service=service, validation_cls=MVValidationError)
    payload = {"name": name, "lance_uri": lance_uri}
    if service_mode:
        return _mv_result_from_payload(
            _post_json(
                endpoint=endpoint or os.environ.get("NPA_LANCEDB_ENDPOINT", ""),
                token_env=token_env,
                payload=payload,
                timeout=timeout,
                path="/refresh-mv",
                error_cls=MVServiceError,
                validation_cls=MVValidationError,
            )
        )
    return _refresh_mv_local(**payload)


def query_table(
    *,
    table: str,
    lance_uri: str = DEFAULT_LANCE_URI,
    filter_sql: str | None = None,
    select: Iterable[str] | None = None,
    limit: int = DEFAULT_QUERY_LIMIT,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 120.0,
) -> QueryResult:
    """Run a bounded SQL-filtered LanceDB table query."""
    service_mode = _resolve_mode_as(mode=mode, service=service, validation_cls=MVValidationError)
    payload = {
        "table": table,
        "lance_uri": lance_uri,
        "filter_sql": filter_sql,
        "select": list(select) if select is not None else None,
        "limit": limit,
    }
    if service_mode:
        return _query_result_from_payload(
            _post_json(
                endpoint=endpoint or os.environ.get("NPA_LANCEDB_ENDPOINT", ""),
                token_env=token_env,
                payload=payload,
                timeout=timeout,
                path="/query-table",
                error_cls=MVServiceError,
                validation_cls=MVValidationError,
            )
        )
    return _query_table_local(**payload)


def create_bdd100k_failure_mode_views(
    lance_uri: str = DEFAULT_LANCE_URI,
    source_table: str = DEFAULT_TABLE,
    *,
    distant_person_threshold: float = 0.01,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 600.0,
) -> list[MVResult]:
    """Create the three BDD100K failure-mode training subsets.

    Example:
        from npa.sdk.workbench.lancedb import create_bdd100k_failure_mode_views

        results = create_bdd100k_failure_mode_views(source_table="bdd100k")
        for result in results:
            print(f"{result.view_name}: {result.row_count} rows")
    """
    service_mode = _resolve_mode_as(mode=mode, service=service, validation_cls=MVValidationError)
    if service_mode:
        threshold = format(float(distant_person_threshold), ".12g")
        specs = [
            ("bdd100k_rider_train", "has_rider = true AND split = 'train'"),
            ("bdd100k_nighttime_person_train", "timeofday = 'night' AND has_person = true AND split = 'train'"),
            (
                "bdd100k_distant_person_train",
                f"has_person = true AND person_bbox_area_pct < {threshold} AND split = 'train'",
            ),
        ]
        return [
            create_mv(
                name=view_name,
                source_table=source_table,
                filter_sql=filter_sql,
                lance_uri=lance_uri,
                service=True,
                endpoint=endpoint,
                token_env=token_env,
                timeout=timeout,
            )
            for view_name, filter_sql in specs
        ]
    return _create_bdd100k_failure_mode_views_local(
        lance_uri=lance_uri,
        source_table=source_table,
        distant_person_threshold=distant_person_threshold,
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


def _resolve_mode_as(*, mode: str | None, service: bool, validation_cls: type[Exception]) -> bool:
    if mode is None:
        return service
    value = mode.strip().lower()
    if value == "local":
        return False
    if value == "service":
        return True
    raise validation_cls("mode must be either 'local' or 'service'")


def _post_json(
    *,
    endpoint: str,
    token_env: str,
    payload: dict[str, Any],
    timeout: float,
    path: str = "/import-bdd100k",
    error_cls: type[Exception] = BDD100KServiceError,
    validation_cls: type[Exception] = BDD100KValidationError,
) -> dict[str, Any]:
    resolved = endpoint.strip().rstrip("/")
    if not resolved:
        raise validation_cls("endpoint is required for service mode")
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.post(f"{resolved}{path}", json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise error_cls(f"LanceDB service request failed ({exc.response.status_code}): {detail}") from exc
    except httpx.HTTPError as exc:
        raise error_cls(f"Cannot reach LanceDB service {resolved}: {exc}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise error_cls("LanceDB service returned non-JSON response") from exc
    if not isinstance(data, dict):
        raise error_cls("LanceDB service returned an unexpected response")
    return data


def _result_from_payload(payload: dict[str, Any]) -> BDD100KImportResult:
    names = {field.name for field in fields(BDD100KImportResult)}
    missing = sorted(name for name in names if name not in payload)
    if missing:
        joined = ", ".join(missing)
        raise BDD100KServiceError(f"LanceDB service response is missing: {joined}")
    return BDD100KImportResult(**{name: payload[name] for name in names})


def _backfill_result_from_payload(payload: dict[str, Any]) -> BackfillResult:
    return BackfillResult(**_dataclass_payload(payload, BackfillResult, BackfillServiceError))


def _mv_result_from_payload(payload: dict[str, Any]) -> MVResult:
    return MVResult(**_dataclass_payload(payload, MVResult, MVServiceError))


def _query_result_from_payload(payload: dict[str, Any]) -> QueryResult:
    return QueryResult(**_dataclass_payload(payload, QueryResult, MVServiceError))


def _dataclass_payload(payload: dict[str, Any], result_cls: type[Any], error_cls: type[Exception]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    missing: list[str] = []
    for field in fields(result_cls):
        if field.name in payload:
            values[field.name] = payload[field.name]
        elif field.default is MISSING and field.default_factory is MISSING:
            missing.append(field.name)
    if missing:
        joined = ", ".join(missing)
        raise error_cls(f"LanceDB service response is missing: {joined}")
    return values


__all__ = [
    "BDD100KImportResult",
    "BackfillResult",
    "MVResult",
    "QueryResult",
    "backfill",
    "create_bdd100k_failure_mode_views",
    "create_mv",
    "import_bdd100k",
    "query_table",
    "refresh_mv",
]
