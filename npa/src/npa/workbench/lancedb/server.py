"""FastAPI server for the NPA LanceDB workbench wrapper."""

from __future__ import annotations

import math
import os
from typing import Any

import lancedb
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

try:
    from .bdd100k_import import (
        DEFAULT_LANCE_URI,
        DEFAULT_SPLITS,
        DEFAULT_TABLE,
        BDD100KImportError,
        BDD100KValidationError,
        import_bdd100k,
    )
except ImportError:  # pragma: no cover - used by the copied Docker module.
    from npa_lancedb_bdd100k_import import (
        DEFAULT_LANCE_URI,
        DEFAULT_SPLITS,
        DEFAULT_TABLE,
        BDD100KImportError,
        BDD100KValidationError,
        import_bdd100k,
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

    async def require_auth(request: Request, authorization: str = Header(default="")) -> None:
        if resolved_auth_mode == "none":
            return
        if not resolved_token:
            raise HTTPException(status_code=500, detail="LANCEDB_TOKEN is not configured")
        if authorization != f"Bearer {resolved_token}":
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

    return app


app = create_app()
