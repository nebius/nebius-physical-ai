"""FastAPI server for the NPA LanceDB workbench wrapper."""

from __future__ import annotations

import hmac
import json
import logging
import math
import os
from typing import Any, Literal

import lancedb
import pyarrow as pa
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

try:
    from .backfill import (
        DEFAULT_DHASH_HAMMING_THRESHOLD,
        DEFAULT_GPU_DEVICE,
        BackfillError,
        BackfillTableNotFoundError,
        BackfillValidationError,
        MissingDependencyError,
        backfill_column,
    )
    from .bdd100k_import import (
        DEFAULT_LANCE_URI,
        DEFAULT_SPLITS,
        DEFAULT_TABLE,
        BDD100KImportError,
        BDD100KValidationError,
        import_bdd100k,
    )
    from .views import (
        DEFAULT_QUERY_LIMIT,
        MVConflictError,
        MVError,
        MVTableNotFoundError,
        MVValidationError,
        create_mv,
        query_table as query_lance_table,
        refresh_mv,
    )
except ImportError:  # pragma: no cover - used by the copied Docker module.
    from npa_lancedb_backfill import (
        DEFAULT_DHASH_HAMMING_THRESHOLD,
        DEFAULT_GPU_DEVICE,
        BackfillError,
        BackfillTableNotFoundError,
        BackfillValidationError,
        MissingDependencyError,
        backfill_column,
    )
    from npa_lancedb_bdd100k_import import (
        DEFAULT_LANCE_URI,
        DEFAULT_SPLITS,
        DEFAULT_TABLE,
        BDD100KImportError,
        BDD100KValidationError,
        import_bdd100k,
    )
    from npa_lancedb_views import (
        DEFAULT_QUERY_LIMIT,
        MVConflictError,
        MVError,
        MVTableNotFoundError,
        MVValidationError,
        create_mv,
        query_table as query_lance_table,
        refresh_mv,
    )


class CreateTableRequest(BaseModel):
    schema: dict[str, Any] | None = None
    input_path: str = ""
    rows: list[dict[str, Any]] = Field(default_factory=list)
    mode: Literal["create", "overwrite", "append"] = "create"
    vector_column: str = "vector"
    id_column: str = "id"
    source_format: str = ""


class QueryRequest(BaseModel):
    vector: list[float]
    top_k: int = 5
    filter: str = ""
    select: list[str] = Field(default_factory=list)


class BDD100KImportRequest(BaseModel):
    source: str = ""
    table: str = DEFAULT_TABLE
    lance_uri: str = DEFAULT_LANCE_URI
    synthetic: int | None = None
    synthetic_seed: int | None = None
    splits: list[str] = Field(default_factory=lambda: list(DEFAULT_SPLITS))
    limit: int | None = None


class BackfillRequest(BaseModel):
    table: str = DEFAULT_TABLE
    udf: str
    lance_uri: str = DEFAULT_LANCE_URI
    batch_size: int | None = None
    force: bool = False
    force_recompute: bool | None = None
    device: str = DEFAULT_GPU_DEVICE
    precision: str | None = None
    dhash_hamming_threshold: int = DEFAULT_DHASH_HAMMING_THRESHOLD


class CreateMVRequest(BaseModel):
    name: str
    source_table: str
    filter_sql: str
    lance_uri: str = DEFAULT_LANCE_URI
    force: bool = False


class RefreshMVRequest(BaseModel):
    name: str
    lance_uri: str = DEFAULT_LANCE_URI


class QueryTableRequest(BaseModel):
    table: str
    lance_uri: str = DEFAULT_LANCE_URI
    filter_sql: str | None = None
    select: list[str] | None = None
    limit: int = DEFAULT_QUERY_LIMIT


class DatasetIndexRequest(BaseModel):
    """The dataset-of-record's `register` payload.

    The dataset integration has always POSTed `/index` and `/query` while the wrapper exposed
    `/tables/{name}` and `/query-table`: two halves written against different APIs that never
    met, because the service was never deployed anywhere a stage could reach it. Live job 313
    finally reached it and got a 404 (EVIDENCE.md §R41).
    """

    table: str = DEFAULT_TABLE
    lance_uri: str = DEFAULT_LANCE_URI
    records: list[dict[str, Any]] = Field(default_factory=list)


class DatasetQueryRequest(BaseModel):
    """The dataset-of-record's facet query: an equality predicate, not a vector search."""

    filter: dict[str, Any] = Field(default_factory=dict)
    table: str = DEFAULT_TABLE
    lance_uri: str = DEFAULT_LANCE_URI
    limit: int = DEFAULT_QUERY_LIMIT


def _equality_predicate(filter_spec: dict[str, Any]) -> str:
    """Turn `{"location": "san-francisco"}` into a SQL predicate, quoting values safely.

    Only equality, and only on values LanceDB can compare: the dataset facet API offers no
    operators, and accepting arbitrary SQL here would make a query endpoint an injection point.
    """

    clauses: list[str] = []
    for key, value in sorted(filter_spec.items()):
        if not str(key).replace("_", "").replace(".", "").isalnum():
            raise HTTPException(
                status_code=400, detail=f"unsupported filter field: {key}"
            )
        if isinstance(value, bool):
            clauses.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            clauses.append(f"{key} = {value}")
        else:
            escaped = str(value).replace("'", "''")
            clauses.append(f"{key} = '{escaped}'")
    return " AND ".join(clauses)


def _list_tables(db: Any) -> list[str]:
    list_tables = getattr(db, "list_tables", None)
    if callable(list_tables):
        values = list_tables()
        return _normalize_table_names(getattr(values, "tables", values))
    table_names = getattr(db, "table_names", None)
    if callable(table_names):
        return _normalize_table_names(table_names())
    return []


def _normalize_table_names(values: Any) -> list[str]:
    names: list[str] = []
    for value in values:
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, tuple | list) and value:
            names.append(str(value[0]))
        elif hasattr(value, "name"):
            names.append(str(value.name))
        else:
            names.append(str(value))
    return names


def _parse_arrow_type(type_spec: Any) -> pa.DataType:
    if isinstance(type_spec, str):
        try:
            return pa.type_for_alias(type_spec)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"unsupported Arrow type: {type_spec}"
            ) from exc
    if not isinstance(type_spec, dict):
        raise HTTPException(
            status_code=400, detail="field type must be a string or object"
        )
    if set(type_spec) - {"name", "item_type", "list_size"}:
        raise HTTPException(status_code=400, detail="unsupported keys in field type")
    item_type = _parse_arrow_type(type_spec.get("item_type"))
    if type_spec.get("name") == "list":
        if "list_size" in type_spec:
            raise HTTPException(
                status_code=400, detail="list type cannot set list_size"
            )
        return pa.list_(item_type)
    if type_spec.get("name") == "fixed_size_list":
        list_size = type_spec.get("list_size")
        if (
            not isinstance(list_size, int)
            or isinstance(list_size, bool)
            or list_size < 1
        ):
            raise HTTPException(
                status_code=400, detail="fixed_size_list requires a positive list_size"
            )
        return pa.list_(item_type, list_size)
    raise HTTPException(status_code=400, detail="unsupported nested Arrow type")


def _parse_arrow_schema(schema_spec: dict[str, Any] | None) -> pa.Schema | None:
    if schema_spec is None:
        return None
    if set(schema_spec) != {"fields"} or not isinstance(schema_spec["fields"], list):
        raise HTTPException(
            status_code=400, detail="schema must contain only a fields list"
        )
    fields: list[pa.Field] = []
    for field_spec in schema_spec["fields"]:
        if not isinstance(field_spec, dict) or set(field_spec) - {
            "name",
            "type",
            "nullable",
        }:
            raise HTTPException(status_code=400, detail="invalid schema field")
        name = field_spec.get("name")
        nullable = field_spec.get("nullable", True)
        if not isinstance(name, str) or not name or not isinstance(nullable, bool):
            raise HTTPException(status_code=400, detail="invalid schema field")
        fields.append(
            pa.field(name, _parse_arrow_type(field_spec.get("type")), nullable)
        )
    if not fields or len({field.name for field in fields}) != len(fields):
        raise HTTPException(
            status_code=400, detail="schema fields must be non-empty and uniquely named"
        )
    return pa.schema(fields)


def _rows_as_arrow_table(
    rows: list[dict[str, Any]], requested_schema: pa.Schema | None
) -> pa.Table:
    if requested_schema is not None:
        field_names = set(requested_schema.names)
        for row in rows:
            extra_fields = set(row) - field_names
            if extra_fields:
                names = ", ".join(sorted(extra_fields))
                raise HTTPException(
                    status_code=400, detail=f"row has undeclared fields: {names}"
                )
            for field in requested_schema:
                if not field.nullable and row.get(field.name) is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"row is missing non-nullable field: {field.name}",
                    )
    normalized_rows = rows
    if requested_schema is None:
        field_names = list(dict.fromkeys(key for row in rows for key in row))
        normalized_rows = [
            {field_name: row.get(field_name) for field_name in field_names}
            for row in rows
        ]
    try:
        return pa.Table.from_pylist(normalized_rows, schema=requested_schema)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"rows are incompatible with the table schema: {exc}",
        ) from exc


def _normalize_inferred_vector(table: pa.Table, *, vector_column: str) -> pa.Table:
    """Make LanceDB's implicit vector conversion explicit and verifiable.

    LanceDB stores an inferred ``list<double>`` vector as a fixed-size
    ``float32`` vector. Comparing the stored schema with the pre-conversion
    Arrow schema therefore reported HTTP 500 after a successful write. Cast
    the designated vector before mutation so the schema we verify is the
    schema we asked LanceDB to store.
    """

    if not vector_column or vector_column not in table.schema.names:
        return table
    field = table.schema.field(vector_column)
    if not (pa.types.is_list(field.type) or pa.types.is_large_list(field.type)):
        return table
    if not (
        pa.types.is_integer(field.type.value_type)
        or pa.types.is_floating(field.type.value_type)
    ):
        raise HTTPException(
            status_code=400, detail=f"{vector_column} must contain numeric vectors"
        )
    vectors = [
        value for value in table.column(vector_column).to_pylist() if value is not None
    ]
    sizes = {len(value) for value in vectors}
    if not sizes or len(sizes) != 1 or 0 in sizes:
        raise HTTPException(
            status_code=400,
            detail=f"{vector_column} vectors must have one consistent positive size",
        )
    vector_size = sizes.pop()
    fields = [
        pa.field(
            candidate.name,
            pa.list_(pa.float32(), vector_size)
            if candidate.name == vector_column
            else candidate.type,
            nullable=candidate.nullable,
            metadata=candidate.metadata,
        )
        for candidate in table.schema
    ]
    try:
        return table.cast(pa.schema(fields, metadata=table.schema.metadata))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{vector_column} is incompatible with a float32 vector: {exc}",
        ) from exc


def _schemas_equal(actual: pa.Schema, expected: pa.Schema) -> bool:
    return actual.equals(expected, check_metadata=False)


def _verify_stored_table(
    db: Any, table_name: str, *, expected_rows: int, expected_schema: pa.Schema
) -> None:
    stored = db.open_table(table_name)
    if stored.count_rows() != expected_rows:
        raise HTTPException(
            status_code=500, detail="stored table row count verification failed"
        )
    if not _schemas_equal(stored.schema, expected_schema):
        raise HTTPException(
            status_code=500, detail="stored table schema verification failed"
        )


def _canonical_record(record: dict[str, Any]) -> str:
    try:
        return json.dumps(
            record, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"record is not canonical JSON: {exc}"
        ) from exc


def _unique_index_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    unique_records: list[dict[str, Any]] = []
    request_records: dict[str, str] = {}
    reused = 0
    for record in records:
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise HTTPException(
                status_code=400, detail="each record requires a non-empty record_id"
            )
        canonical = _canonical_record(record)
        previous = request_records.get(record_id)
        if previous is None:
            request_records[record_id] = canonical
            unique_records.append(record)
        elif previous == canonical:
            reused += 1
        else:
            raise HTTPException(
                status_code=409,
                detail=f"record_id has conflicting content: {record_id}",
            )
    return unique_records, reused


def _existing_index_records(table: Any) -> dict[str, str]:
    existing: dict[str, str] = {}
    for record in table.to_arrow().to_pylist():
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise HTTPException(
                status_code=409, detail="existing table has an invalid record_id"
            )
        canonical = _canonical_record(record)
        previous = existing.get(record_id)
        if previous is not None and previous != canonical:
            raise HTTPException(
                status_code=409,
                detail=f"existing record_id has conflicting content: {record_id}",
            )
        existing[record_id] = canonical
    return existing


def _plan_index_append(
    table: Any, records: list[dict[str, Any]], request_reused: int
) -> tuple[pa.Table, int]:
    incoming = _rows_as_arrow_table(records, table.schema)
    existing = _existing_index_records(table)
    novel_records: list[dict[str, Any]] = []
    reused = request_reused
    for record in incoming.to_pylist():
        record_id = record["record_id"]
        canonical = _canonical_record(record)
        stored = existing.get(record_id)
        if stored is None:
            novel_records.append(record)
        elif stored == canonical:
            reused += 1
        else:
            raise HTTPException(
                status_code=409,
                detail=f"record_id has conflicting content: {record_id}",
            )
    return pa.Table.from_pylist(novel_records, schema=table.schema), reused


def _mutate_table(
    db: Any, table_name: str, body: CreateTableRequest
) -> tuple[str, int]:
    if body.input_path.startswith("s3://"):
        raise HTTPException(
            status_code=400,
            detail="server-side S3 import is not implemented in the OSS wrapper",
        )
    requested_schema = _parse_arrow_schema(body.schema)
    if not body.rows and requested_schema is None:
        raise HTTPException(
            status_code=400, detail="rows or a usable schema are required"
        )
    if body.mode == "append":
        table = db.open_table(table_name)
        incoming = _rows_as_arrow_table(body.rows, requested_schema or table.schema)
        if not _schemas_equal(table.schema, incoming.schema):
            raise HTTPException(
                status_code=400, detail="append schema does not match table"
            )
        expected_rows = table.count_rows() + len(body.rows)
        if body.rows:
            table.add(incoming)
        status = "appended"
    else:
        incoming = _rows_as_arrow_table(body.rows, requested_schema)
        if requested_schema is None:
            incoming = _normalize_inferred_vector(
                incoming, vector_column=body.vector_column
            )
        mode = "overwrite" if body.mode == "overwrite" else "create"
        db.create_table(table_name, data=incoming, mode=mode)
        expected_rows = len(body.rows)
        status = "overwritten" if mode == "overwrite" else "created"
    _verify_stored_table(
        db, table_name, expected_rows=expected_rows, expected_schema=incoming.schema
    )
    return status, len(body.rows)


def create_app(
    *,
    storage_path: str | None = None,
    auth_mode: str | None = None,
    token: str | None = None,
) -> FastAPI:
    """Create the LanceDB wrapper FastAPI app."""
    resolved_storage = storage_path or os.environ.get(
        "LANCEDB_STORAGE_PATH", "/tmp/npa-lancedb"
    )
    resolved_auth_mode = auth_mode or os.environ.get("LANCEDB_AUTH_MODE", "token")
    resolved_token = token if token is not None else os.environ.get("LANCEDB_TOKEN", "")
    app = FastAPI(title="NPA LanceDB wrapper")
    db = lancedb.connect(resolved_storage)
    known_tables: set[str] = set()
    if resolved_auth_mode == "none":
        LOGGER.warning(
            "LanceDB wrapper started with auth disabled; every endpoint is reachable without a token. "
            "Set LANCEDB_AUTH_MODE=token and LANCEDB_TOKEN before exposing it beyond localhost."
        )

    async def require_auth(
        request: Request, authorization: str = Header(default="")
    ) -> None:
        if resolved_auth_mode == "none":
            return
        if not resolved_token:
            raise HTTPException(
                status_code=500, detail="LANCEDB_TOKEN is not configured"
            )
        if not hmac.compare_digest(authorization, f"Bearer {resolved_token}"):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        """Unauthenticated storage readiness for container and Kubernetes probes."""

        try:
            known_tables.update(_list_tables(db))
        except Exception as exc:
            LOGGER.exception("LanceDB storage readiness check failed")
            raise HTTPException(status_code=503, detail="storage is not ready") from exc
        return {"status": "ok"}

    @app.get("/health")
    async def health(
        request: Request, authorization: str = Header(default="")
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        known_tables.update(_list_tables(db))
        return {
            "status": "ok",
            "storage_path": resolved_storage,
            "tables": len(known_tables),
        }

    @app.get("/tables")
    async def tables(
        request: Request, authorization: str = Header(default="")
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        known_tables.update(_list_tables(db))
        return {"tables": sorted(known_tables)}

    @app.post("/tables/{table_name}")
    async def create_table(
        table_name: str,
        body: CreateTableRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        status, stored_rows = _mutate_table(db, table_name, body)
        known_tables.add(table_name)
        return {"status": status, "table": table_name, "rows": stored_rows}

    @app.post("/tables/{table_name}/query")
    async def query_table(
        table_name: str,
        body: QueryRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        if body.top_k < 1 or body.top_k > 1000:
            raise HTTPException(
                status_code=400, detail="top_k must be between 1 and 1000"
            )
        if not body.vector or any(
            not math.isfinite(float(value)) for value in body.vector
        ):
            raise HTTPException(
                status_code=400, detail="vector must contain finite numbers"
            )
        table = db.open_table(table_name)
        query = table.search(body.vector).limit(body.top_k)
        if body.filter:
            query = query.where(body.filter)
        if body.select:
            query = query.select(body.select)
        rows = query.to_list()
        return {"table": table_name, "results": rows, "count": len(rows)}

    @app.post("/index")
    async def index_records(
        body: DatasetIndexRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        """Insert novel dataset identities and reuse exact existing records."""

        await require_auth(request, authorization)
        if not body.records:
            raise HTTPException(status_code=400, detail="records must not be empty")
        rows = [dict(record) for record in body.records]
        unique_rows, reused = _unique_index_records(rows)
        if body.table in set(_list_tables(db)):
            table = db.open_table(body.table)
            novel_rows, reused = _plan_index_append(table, unique_rows, reused)
            inserted = novel_rows.num_rows
            if inserted:
                table.add(novel_rows)
                status = "appended"
            else:
                status = "replayed"
        else:
            # First write defines the schema, which is what makes `register` idempotent for a
            # fresh dataset id without a separate create step.
            db.create_table(body.table, data=unique_rows, mode="create")
            inserted = len(unique_rows)
            status = "created"
        known_tables.add(body.table)
        return {
            "status": status,
            "table": body.table,
            "lance_uri": body.lance_uri,
            "rows": inserted,
            "inserted": inserted,
            "reused": reused,
        }

    @app.post("/query")
    async def query_records(
        body: DatasetQueryRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        """Facet query by equality predicate, the shape the dataset CLI asks for."""

        await require_auth(request, authorization)
        if body.limit < 1 or body.limit > 10_000:
            raise HTTPException(
                status_code=400, detail="limit must be between 1 and 10000"
            )
        if body.table not in set(_list_tables(db)):
            # An unregistered dataset is an empty result, not an error: a curation step may
            # legitimately query before anything has been indexed.
            return {"table": body.table, "records": [], "count": 0}
        table = db.open_table(body.table)
        query = table.search().limit(body.limit)
        predicate = _equality_predicate(body.filter)
        if predicate:
            query = query.where(predicate)
        rows = query.to_list()
        return {"table": body.table, "records": rows, "count": len(rows)}

    @app.post("/import-bdd100k")
    async def import_bdd100k_endpoint(
        body: BDD100KImportRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        try:
            result = import_bdd100k(
                source=body.source,
                table=body.table,
                lance_uri=body.lance_uri,
                synthetic=body.synthetic,
                synthetic_seed=body.synthetic_seed,
                splits=body.splits,
                limit=body.limit,
            )
        except BDD100KValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BDD100KImportError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        known_tables.add(result.table)
        return result.to_dict()

    @app.post("/backfill")
    async def backfill_endpoint(
        body: BackfillRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        try:
            result = backfill_column(
                table=body.table,
                udf=body.udf,
                lance_uri=body.lance_uri,
                batch_size=body.batch_size,
                force=body.force,
                force_recompute=body.force_recompute,
                device=body.device,
                precision=body.precision,
                dhash_hamming_threshold=body.dhash_hamming_threshold,
            )
        except BackfillValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BackfillTableNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MissingDependencyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except BackfillError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        known_tables.add(result.table)
        return result.to_dict()

    @app.post("/create-mv")
    async def create_mv_endpoint(
        body: CreateMVRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        try:
            result = create_mv(
                name=body.name,
                source_table=body.source_table,
                filter_sql=body.filter_sql,
                lance_uri=body.lance_uri,
                force=body.force,
            )
        except MVConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except MVValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MVTableNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MVError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        known_tables.add(result.view_name)
        return result.to_dict()

    @app.post("/refresh-mv")
    async def refresh_mv_endpoint(
        body: RefreshMVRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        try:
            result = refresh_mv(name=body.name, lance_uri=body.lance_uri)
        except MVValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MVTableNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MVError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        known_tables.add(result.view_name)
        return result.to_dict()

    @app.post("/query-table")
    async def query_table_endpoint(
        body: QueryTableRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        try:
            result = query_lance_table(
                table=body.table,
                lance_uri=body.lance_uri,
                filter_sql=body.filter_sql,
                select=body.select,
                limit=body.limit,
            )
        except MVValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MVTableNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MVError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    return app


app = create_app()
