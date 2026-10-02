"""Create-table command for LanceDB Workbench services."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer

from .helpers import (
    DEFAULT_TOKEN_ENV,
    OutputFormat,
    auth_headers,
    emit,
    fail,
    load_rows,
    load_schema,
    request_json,
    resolve_endpoint,
    validate_table_name,
)


class CreateMode(str, Enum):
    create = "create"
    overwrite = "overwrite"
    append = "append"


def create_table_cmd(
    endpoint: str = typer.Option("", "--endpoint", help="LanceDB wrapper endpoint."),
    table: str = typer.Option(..., "--table", help="Table name."),
    schema: Path | None = typer.Option(
        None, "--schema", exists=False, help="Optional Arrow JSON schema path."
    ),
    input_path: str = typer.Option(
        "", "--input-path", help="Local parquet, JSON, or JSONL source path."
    ),
    mode: CreateMode = typer.Option(CreateMode.create, "--mode", help="Create mode."),
    vector_column: str = typer.Option(
        "vector", "--vector-column", help="Vector column name."
    ),
    id_column: str = typer.Option("id", "--id-column", help="Identifier column name."),
    token_env: str = typer.Option(
        DEFAULT_TOKEN_ENV,
        "--token-env",
        help="Environment variable containing wrapper token.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Create a table from local rows or an Arrow JSON fields schema; S3 import is unsupported."""
    resolved = resolve_endpoint(endpoint)
    table_name = validate_table_name(table)
    schema_payload = load_schema(schema)
    rows = load_rows(input_path)
    if not rows and schema_payload is None:
        fail("--input-path with rows or --schema is required")
    headers = auth_headers(token_env=token_env)
    payload = {
        "schema": schema_payload,
        "input_path": input_path,
        "rows": rows,
        "mode": mode.value,
        "vector_column": vector_column,
        "id_column": id_column,
    }
    result = request_json(
        "POST",
        resolved,
        f"/tables/{table_name}",
        headers=headers,
        payload=payload,
        timeout=120.0,
    )
    result.setdefault("table", table_name)
    result.setdefault("rows", len(rows))
    emit(
        result,
        output=output,
        text=f"table: {table_name}\nstatus: {result.get('status', 'created')}",
    )
