"""Materialized views and general queries for the LanceDB workbench tool."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pyarrow as pa

try:
    from .bdd100k_import import DEFAULT_LANCE_URI, DEFAULT_TABLE, validate_lance_uri, validate_table
except ImportError:  # pragma: no cover - used by the copied Docker module.
    from npa_lancedb_bdd100k_import import DEFAULT_LANCE_URI, DEFAULT_TABLE, validate_lance_uri, validate_table


MV_REGISTRY_TABLE = "_mv_registry"
DEFAULT_QUERY_LIMIT = 100
MAX_QUERY_LIMIT = 10000


class MVError(RuntimeError):
    """Base class for materialized-view failures."""


class MVValidationError(MVError, ValueError):
    """Raised when a materialized-view request is invalid."""


class MVConflictError(MVValidationError):
    """Raised when a materialized-view name conflicts with an existing definition."""


class MVTableNotFoundError(MVError, FileNotFoundError):
    """Raised when a requested LanceDB table or materialized view is missing."""


class MVWriteError(MVError):
    """Raised when LanceDB cannot read or write materialized-view data."""


@dataclass(frozen=True)
class MVDefinition:
    """Registered materialized-view definition."""

    name: str
    source_table: str
    filter_sql: str
    definition_hash: str


@dataclass(frozen=True)
class MVResult:
    """Result returned by create and refresh materialized-view calls."""

    view_name: str
    source_table: str
    filter_sql: str
    row_count: int
    view_table_version: int | None
    manifest_sha256: str
    created_at: str | None = None
    row_count_before: int | None = None
    row_count_after: int | None = None
    view_table_version_before: int | None = None
    view_table_version_after: int | None = None
    refreshed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result payload."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class QueryResult:
    """Result returned by generic table queries."""

    rows: list[dict[str, Any]]
    row_count: int
    total_rows_matched: int
    table_version: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result payload."""
        return asdict(self)


def compute_definition_hash(source_table: str, filter_sql: str) -> str:
    """Return a stable SHA-256 hash for a source-table/filter pair."""
    digest = hashlib.sha256()
    digest.update(_canonical_source(source_table).encode("utf-8"))
    digest.update(b"\n")
    digest.update(_canonical_filter(filter_sql).encode("utf-8"))
    digest.update(b"\n")
    return digest.hexdigest()


def ensure_registry(lance_uri: str):
    """Create `_mv_registry` lazily and return its Lance table."""
    db = _connect_lancedb(validate_lance_uri(lance_uri))
    return _ensure_registry(db)


def create_mv(
    *,
    name: str,
    source_table: str,
    filter_sql: str,
    lance_uri: str = DEFAULT_LANCE_URI,
    force: bool = False,
) -> MVResult:
    """Create or recreate a materialized view from a filtered source scan."""
    resolved_uri = validate_lance_uri(lance_uri)
    view_name = _validate_view_name(name)
    resolved_source = validate_table(source_table)
    resolved_filter = _validate_filter(filter_sql)
    if view_name == resolved_source:
        raise MVValidationError("view name must differ from source_table")

    db = _connect_lancedb(resolved_uri)
    source_obj = _open_table(db, resolved_source)
    registry = _ensure_registry(db)
    rows = _registry_rows(registry)
    existing = _registry_row_for(rows, view_name)
    definition_hash = compute_definition_hash(resolved_source, resolved_filter)

    if existing is None and view_name in _list_tables(db):
        raise MVConflictError(f"LanceDB table already exists and is not registered as an MV: {view_name}")

    if existing is not None and str(existing["definition_hash"]) != definition_hash and not force:
        raise MVConflictError("materialized view already exists with a different source_table or filter_sql")

    view_exists = view_name in _list_tables(db)
    if existing is not None and not force and view_exists:
        view_obj = _open_table(db, view_name)
        row_count = _count_rows(view_obj)
        return MVResult(
            view_name=view_name,
            source_table=str(existing["source_table"]),
            filter_sql=str(existing["filter_sql"]),
            row_count=row_count,
            view_table_version=_table_version(view_obj),
            created_at=_format_timestamp(existing.get("created_at")),
            manifest_sha256=_manifest_sha256(view_obj, str(existing["source_table"]), str(existing["filter_sql"])),
        )

    now = _utc_now()
    created_at = now if existing is None or force else _timestamp_value(existing.get("created_at")) or now
    materialized = _filtered_arrow(source_obj, resolved_filter)
    view_obj = _write_view_table(db, view_name, materialized)
    row_count = int(materialized.num_rows)
    _write_registry_rows(
        db,
        _replace_registry_row(
            rows,
            {
                "name": view_name,
                "source_table": resolved_source,
                "filter_sql": resolved_filter,
                "created_at": created_at,
                "last_refreshed": now,
                "row_count_at_last_refresh": row_count,
                "definition_hash": definition_hash,
            },
        ),
    )
    return MVResult(
        view_name=view_name,
        source_table=resolved_source,
        filter_sql=resolved_filter,
        row_count=row_count,
        view_table_version=_table_version(view_obj),
        created_at=_format_timestamp(created_at),
        manifest_sha256=_manifest_sha256(view_obj, resolved_source, resolved_filter),
    )


def refresh_mv(*, name: str, lance_uri: str = DEFAULT_LANCE_URI) -> MVResult:
    """Recompute a registered materialized view from its current source table."""
    resolved_uri = validate_lance_uri(lance_uri)
    view_name = _validate_view_name(name)
    db = _connect_lancedb(resolved_uri)
    registry = _ensure_registry(db)
    rows = _registry_rows(registry)
    existing = _registry_row_for(rows, view_name)
    if existing is None:
        raise MVTableNotFoundError(f"materialized view is not registered: {view_name}")

    source_table = str(existing["source_table"])
    filter_sql = str(existing["filter_sql"])
    source_obj = _open_table(db, source_table)
    if view_name in _list_tables(db):
        before_obj = _open_table(db, view_name)
        row_count_before = _count_rows(before_obj)
        version_before = _table_version(before_obj)
    else:
        row_count_before = 0
        version_before = None

    materialized = _filtered_arrow(source_obj, filter_sql)
    view_obj = _write_view_table(db, view_name, materialized)
    row_count_after = int(materialized.num_rows)
    refreshed_at = _utc_now()
    replacement = dict(existing)
    replacement["last_refreshed"] = refreshed_at
    replacement["row_count_at_last_refresh"] = row_count_after
    _write_registry_rows(db, _replace_registry_row(rows, replacement))
    return MVResult(
        view_name=view_name,
        source_table=source_table,
        filter_sql=filter_sql,
        row_count=row_count_after,
        view_table_version=_table_version(view_obj),
        row_count_before=row_count_before,
        row_count_after=row_count_after,
        view_table_version_before=version_before,
        view_table_version_after=_table_version(view_obj),
        refreshed_at=_format_timestamp(refreshed_at),
        manifest_sha256=_manifest_sha256(view_obj, source_table, filter_sql),
    )


def query_table(
    *,
    table: str,
    lance_uri: str = DEFAULT_LANCE_URI,
    filter_sql: str | None = None,
    select: Iterable[str] | None = None,
    limit: int = DEFAULT_QUERY_LIMIT,
) -> QueryResult:
    """Run a bounded SQL-filtered read against a LanceDB table."""
    resolved_uri = validate_lance_uri(lance_uri)
    table_name = validate_table(table)
    if limit < 1 or limit > MAX_QUERY_LIMIT:
        raise MVValidationError(f"limit must be between 1 and {MAX_QUERY_LIMIT}")
    resolved_filter = _optional_filter(filter_sql)

    db = _connect_lancedb(resolved_uri)
    table_obj = _open_table(db, table_name)
    selected = _resolve_select(table_obj, select)
    try:
        total_rows = table_obj.count_rows(resolved_filter) if resolved_filter else table_obj.count_rows()
        query = table_obj.search()
        if resolved_filter:
            query = query.where(resolved_filter)
        if selected:
            query = query.select(selected)
        arrow_table = query.limit(limit).to_arrow()
    except MVError:
        raise
    except Exception as exc:
        raise MVWriteError(f"failed to query LanceDB table {table_name}: {exc}") from exc
    rows = [_jsonable_row(row) for row in arrow_table.to_pylist()]
    return QueryResult(
        rows=rows,
        row_count=len(rows),
        total_rows_matched=int(total_rows),
        table_version=_table_version(table_obj),
    )


def create_bdd100k_failure_mode_views(
    lance_uri: str = DEFAULT_LANCE_URI,
    source_table: str = DEFAULT_TABLE,
    *,
    distant_person_threshold: float = 0.01,
) -> list[MVResult]:
    """Create the three BDD100K failure-mode training subsets."""
    if distant_person_threshold <= 0:
        raise MVValidationError("distant_person_threshold must be positive")
    threshold = _format_threshold(distant_person_threshold)
    specs = [
        ("bdd100k_rider_train", "has_rider = true AND split = 'train'"),
        ("bdd100k_nighttime_person_train", "timeofday = 'night' AND has_person = true AND split = 'train'"),
        (
            "bdd100k_distant_person_train",
            f"has_person = true AND person_bbox_area_pct < {threshold} AND split = 'train'",
        ),
    ]
    return [
        create_mv(name=view_name, source_table=source_table, filter_sql=filter_sql, lance_uri=lance_uri)
        for view_name, filter_sql in specs
    ]


def _registry_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("source_table", pa.string()),
            pa.field("filter_sql", pa.string()),
            pa.field("created_at", pa.timestamp("ms")),
            pa.field("last_refreshed", pa.timestamp("ms")),
            pa.field("row_count_at_last_refresh", pa.int64()),
            pa.field("definition_hash", pa.string()),
        ]
    )


def _connect_lancedb(lance_uri: str):
    try:
        import lancedb
    except ImportError as exc:
        raise MVWriteError("Materialized views require the lancedb package") from exc
    try:
        return lancedb.connect(lance_uri)
    except Exception as exc:
        raise MVWriteError(f"failed to connect to LanceDB URI {lance_uri}: {exc}") from exc


def _ensure_registry(db: Any):
    try:
        if MV_REGISTRY_TABLE not in _list_tables(db):
            return db.create_table(MV_REGISTRY_TABLE, schema=_registry_schema(), mode="create")
        table_obj = db.open_table(MV_REGISTRY_TABLE)
    except Exception as exc:
        raise MVWriteError(f"failed to open or create {MV_REGISTRY_TABLE}: {exc}") from exc
    schema = _schema(table_obj)
    missing = [name for name in _registry_schema().names if name not in schema.names]
    if missing:
        joined = ", ".join(missing)
        raise MVWriteError(f"{MV_REGISTRY_TABLE} is missing required column(s): {joined}")
    return table_obj


def _open_table(db: Any, table_name: str) -> Any:
    try:
        if table_name not in _list_tables(db):
            raise MVTableNotFoundError(f"LanceDB table not found: {table_name}")
        return db.open_table(table_name)
    except MVTableNotFoundError:
        raise
    except Exception as exc:
        raise MVWriteError(f"failed to open LanceDB table {table_name}: {exc}") from exc


def _list_tables(db: Any) -> list[str]:
    list_tables = getattr(db, "list_tables", None)
    if callable(list_tables):
        try:
            return _normalize_table_names(list_tables(limit=10000))
        except TypeError:
            return _normalize_table_names(list_tables())
    table_names = getattr(db, "table_names", None)
    if callable(table_names):
        try:
            return _normalize_table_names(table_names(limit=10000))
        except TypeError:
            return _normalize_table_names(table_names())
    raise MVWriteError("LanceDB connection does not expose table listing")


def _normalize_table_names(values: Any) -> list[str]:
    names: list[str] = []
    raw_values = getattr(values, "names", None) or getattr(values, "tables", None) or values
    for value in raw_values:
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, tuple | list) and value:
            names.append(str(value[0]))
        elif hasattr(value, "name"):
            names.append(str(value.name))
        else:
            names.append(str(value))
    return names


def _validate_view_name(name: str) -> str:
    value = validate_table(name)
    if value == MV_REGISTRY_TABLE:
        raise MVValidationError(f"{MV_REGISTRY_TABLE} is reserved")
    return value


def _validate_filter(filter_sql: str) -> str:
    value = filter_sql.strip()
    if not value:
        raise MVValidationError("filter_sql is required")
    return value


def _optional_filter(filter_sql: str | None) -> str | None:
    if filter_sql is None:
        return None
    value = filter_sql.strip()
    return value or None


def _canonical_source(source_table: str) -> str:
    return validate_table(source_table)


def _canonical_filter(filter_sql: str) -> str:
    return " ".join(_validate_filter(filter_sql).split())


def _filtered_arrow(source_obj: Any, filter_sql: str) -> pa.Table:
    try:
        return source_obj.search().where(filter_sql).to_arrow()
    except Exception as exc:
        raise MVWriteError(f"failed to scan source table with filter {filter_sql!r}: {exc}") from exc


def _write_view_table(db: Any, view_name: str, arrow_table: pa.Table) -> Any:
    try:
        return db.create_table(view_name, data=arrow_table, mode="overwrite")
    except Exception as exc:
        raise MVWriteError(f"failed to write materialized view {view_name}: {exc}") from exc


def _registry_rows(registry: Any) -> list[dict[str, Any]]:
    try:
        return registry.to_arrow().to_pylist()
    except Exception as exc:
        raise MVWriteError(f"failed to read {MV_REGISTRY_TABLE}: {exc}") from exc


def _registry_row_for(rows: Iterable[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("name")) == name:
            return row
    return None


def _replace_registry_row(rows: Iterable[dict[str, Any]], replacement: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(replacement["name"])
    updated = [dict(row) for row in rows if str(row.get("name")) != name]
    updated.append(replacement)
    return sorted(updated, key=lambda row: str(row["name"]))


def _write_registry_rows(db: Any, rows: list[dict[str, Any]]) -> None:
    try:
        table = pa.Table.from_pylist(rows, schema=_registry_schema())
        db.create_table(MV_REGISTRY_TABLE, data=table, mode="overwrite")
    except Exception as exc:
        raise MVWriteError(f"failed to update {MV_REGISTRY_TABLE}: {exc}") from exc


def _resolve_select(table_obj: Any, select: Iterable[str] | None) -> list[str] | None:
    schema = _schema(table_obj)
    if select:
        selected = []
        for name in select:
            value = str(name).strip()
            if not value:
                continue
            if value not in schema.names:
                raise MVValidationError(f"table is missing selected column: {value}")
            if value not in selected:
                selected.append(value)
        return selected or None
    selected = [name for name in schema.names if name != "image_bytes"]
    return selected or None


def _schema(table_obj: Any) -> pa.Schema:
    schema = getattr(table_obj, "schema", None)
    if callable(schema):
        schema = schema()
    if not isinstance(schema, pa.Schema):
        raise MVWriteError("LanceDB table schema is not a PyArrow schema")
    return schema


def _count_rows(table_obj: Any) -> int:
    try:
        return int(table_obj.count_rows())
    except Exception as exc:
        raise MVWriteError(f"failed to count rows: {exc}") from exc


def _table_version(table_obj: Any) -> int | None:
    value = getattr(table_obj, "version", None)
    if value is None:
        return None
    try:
        return int(value() if callable(value) else value)
    except (TypeError, ValueError):
        return None


def _manifest_sha256(table_obj: Any, source_table: str, filter_sql: str) -> str:
    try:
        arrow_table = table_obj.to_arrow()
    except Exception as exc:
        raise MVWriteError(f"failed to read materialized view manifest rows: {exc}") from exc
    digest = hashlib.sha256()
    digest.update(_canonical_source(source_table).encode("utf-8"))
    digest.update(b"\n")
    digest.update(_canonical_filter(filter_sql).encode("utf-8"))
    digest.update(b"\n")
    digest.update(json.dumps(_schema_summary(arrow_table.schema), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
    for row in sorted(arrow_table.to_pylist(), key=_row_sort_key):
        digest.update(json.dumps(_manifest_jsonable(row), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _schema_summary(schema: pa.Schema) -> dict[str, str]:
    return {field.name: str(field.type) for field in schema}


def _row_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    image_id = row.get("image_id")
    if image_id is not None:
        return ("image_id", str(image_id))
    return ("row", json.dumps(_manifest_jsonable(row), sort_keys=True, separators=(",", ":")))


def _manifest_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _manifest_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_manifest_jsonable(item) for item in value]
    if isinstance(value, bytes | bytearray | memoryview):
        raw = bytes(value)
        return {"__bytes_sha256__": hashlib.sha256(raw).hexdigest(), "__bytes_len__": len(raw)}
    if isinstance(value, datetime):
        return _format_timestamp(value)
    if hasattr(value, "as_py"):
        return _manifest_jsonable(value.as_py())
    return value


def _jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in row.items()}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, bytes | bytearray | memoryview):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, datetime):
        return _format_timestamp(value)
    if hasattr(value, "as_py"):
        return _jsonable_value(value.as_py())
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _timestamp_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    return None


def _format_timestamp(value: Any) -> str | None:
    timestamp = _timestamp_value(value)
    if timestamp is None:
        return None
    return timestamp.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _format_threshold(value: float) -> str:
    return format(float(value), ".12g")


__all__ = [
    "DEFAULT_QUERY_LIMIT",
    "MAX_QUERY_LIMIT",
    "MVDefinition",
    "MVError",
    "MVConflictError",
    "MV_REGISTRY_TABLE",
    "MVResult",
    "MVTableNotFoundError",
    "MVValidationError",
    "MVWriteError",
    "QueryResult",
    "compute_definition_hash",
    "create_bdd100k_failure_mode_views",
    "create_mv",
    "ensure_registry",
    "query_table",
    "refresh_mv",
]
