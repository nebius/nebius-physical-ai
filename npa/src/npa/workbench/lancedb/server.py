"""FastAPI server for the NPA LanceDB workbench wrapper."""

from __future__ import annotations

import hmac
import logging
import math
import os
from typing import Any

import lancedb
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
    mode: str = "create"
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


def _list_tables(db: Any) -> list[str]:
    table_names = getattr(db, "table_names", None)
    if callable(table_names):
        return _normalize_table_names(table_names())
    return _normalize_table_names(db.list_tables())


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


def create_app(
    *,
    storage_path: str | None = None,
    auth_mode: str | None = None,
    token: str | None = None,
) -> FastAPI:
    """Create the LanceDB wrapper FastAPI app."""
    resolved_storage = storage_path or os.environ.get("LANCEDB_STORAGE_PATH", "/tmp/npa-lancedb")
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

    async def require_auth(request: Request, authorization: str = Header(default="")) -> None:
        if resolved_auth_mode == "none":
            return
        if not resolved_token:
            raise HTTPException(status_code=500, detail="LANCEDB_TOKEN is not configured")
        if not hmac.compare_digest(authorization, f"Bearer {resolved_token}"):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/health")
    async def health(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
        await require_auth(request, authorization)
        known_tables.update(_list_tables(db))
        return {"status": "ok", "storage_path": resolved_storage, "tables": len(known_tables)}

    @app.get("/tables")
    async def tables(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
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
        rows = body.rows
        if not rows and body.input_path.startswith("s3://"):
            raise HTTPException(status_code=400, detail="server-side S3 import is not implemented in the OSS wrapper")
        if not rows:
            rows = [{body.id_column: "empty", body.vector_column: [0.0]}]
        if body.mode == "append":
            table = db.open_table(table_name)
            table.add(rows)
            status = "appended"
        else:
            mode = "overwrite" if body.mode == "overwrite" else "create"
            table = db.create_table(table_name, data=rows, mode=mode)
            status = "created" if mode == "create" else "overwritten"
        known_tables.add(table_name)
        return {"status": status, "table": table_name, "rows": len(rows)}

    @app.post("/tables/{table_name}/query")
    async def query_table(
        table_name: str,
        body: QueryRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        if body.top_k < 1 or body.top_k > 1000:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 1000")
        if not body.vector or any(not math.isfinite(float(value)) for value in body.vector):
            raise HTTPException(status_code=400, detail="vector must contain finite numbers")
        table = db.open_table(table_name)
        query = table.search(body.vector).limit(body.top_k)
        if body.filter:
            query = query.where(body.filter)
        if body.select:
            query = query.select(body.select)
        rows = query.to_list()
        return {"table": table_name, "results": rows, "count": len(rows)}

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
