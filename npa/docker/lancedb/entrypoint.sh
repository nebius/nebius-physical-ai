#!/usr/bin/env bash
set -euo pipefail

export LANCEDB_STORAGE_PATH="${LANCEDB_STORAGE_PATH:-/data/lancedb}"
export LANCEDB_PORT="${LANCEDB_PORT:-8686}"
export LANCEDB_AUTH_MODE="${LANCEDB_AUTH_MODE:-token}"

cat >/tmp/npa_lancedb_server.py <<'PY'
from __future__ import annotations

import math
import os
from typing import Any

import lancedb
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

STORAGE_PATH = os.environ.get("LANCEDB_STORAGE_PATH", "/data/lancedb")
AUTH_MODE = os.environ.get("LANCEDB_AUTH_MODE", "token")
TOKEN = os.environ.get("LANCEDB_TOKEN", "")

app = FastAPI(title="NPA LanceDB wrapper")
db = lancedb.connect(STORAGE_PATH)


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


async def _require_auth(request: Request, authorization: str = Header(default="")) -> None:
    if AUTH_MODE == "none":
        return
    expected = TOKEN
    if not expected:
        raise HTTPException(status_code=500, detail="LANCEDB_TOKEN is not configured")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/health")
async def health(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
    await _require_auth(request, authorization)
    return {"status": "ok", "storage_path": STORAGE_PATH, "tables": len(db.table_names())}


@app.get("/tables")
async def tables(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
    await _require_auth(request, authorization)
    return {"tables": db.table_names()}


@app.post("/tables/{table_name}")
async def create_table(
    table_name: str,
    body: CreateTableRequest,
    request: Request,
    authorization: str = Header(default=""),
) -> dict[str, Any]:
    await _require_auth(request, authorization)
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
    return {"status": status, "table": table_name, "rows": len(rows)}


@app.post("/tables/{table_name}/query")
async def query_table(
    table_name: str,
    body: QueryRequest,
    request: Request,
    authorization: str = Header(default=""),
) -> dict[str, Any]:
    await _require_auth(request, authorization)
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
PY

exec uvicorn npa_lancedb_server:app --app-dir /tmp --host 0.0.0.0 --port "${LANCEDB_PORT}"
