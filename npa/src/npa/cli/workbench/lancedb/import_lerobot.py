"""Import LeRobot datasets into LanceDB tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from .create_table import CreateMode
from .helpers import (
    DEFAULT_TOKEN_ENV,
    OutputFormat,
    auth_headers,
    emit,
    fail,
    load_rows,
    request_json,
    resolve_endpoint,
    validate_table_name,
)


def resolve_lerobot_dataset_files(dataset_path: str) -> list[Path]:
    if dataset_path.startswith("s3://"):
        return []
    root = Path(dataset_path)
    if not root.exists():
        fail(f"--dataset-path does not exist: {dataset_path}")
    if root.is_file():
        return [root]
    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        files = sorted(root.rglob("*.parquet"))
    if not files:
        fail(f"No parquet files found under LeRobot dataset path: {dataset_path}")
    return files


def _rows_from_lerobot_files(
    files: list[Path],
    *,
    vector_column: str,
    id_column: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in files:
        for index, row in enumerate(load_rows(str(file))):
            row.setdefault(id_column, f"{file.stem}:{index}")
            if vector_column not in row:
                row[vector_column] = _numeric_vector(row)
            rows.append(row)
            if limit and len(rows) >= limit:
                return rows
    return rows


def _numeric_vector(row: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for value in row.values():
        if isinstance(value, (int, float)):
            values.append(float(value))
        if len(values) >= 8:
            break
    if not values:
        values = [0.0]
    return values


def import_lerobot_cmd(
    endpoint: str = typer.Option("", "--endpoint", help="LanceDB wrapper endpoint."),
    dataset_path: str = typer.Option(..., "--dataset-path", help="Local LeRobot dataset path or s3:// prefix."),
    table: str = typer.Option(..., "--table", help="Destination table name."),
    mode: CreateMode = typer.Option(CreateMode.create, "--mode", help="Create mode."),
    vector_column: str = typer.Option("vector", "--vector-column", help="Vector column to create or reuse."),
    id_column: str = typer.Option("id", "--id-column", help="Identifier column name."),
    limit: int = typer.Option(0, "--limit", help="Maximum rows to import; 0 means all."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing wrapper token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Import a LeRobot dataset into a LanceDB table."""
    if limit < 0:
        fail("--limit must be greater than or equal to 0")
    resolved = resolve_endpoint(endpoint)
    table_name = validate_table_name(table)
    files = resolve_lerobot_dataset_files(dataset_path)
    rows = [] if dataset_path.startswith("s3://") else _rows_from_lerobot_files(
        files,
        vector_column=vector_column,
        id_column=id_column,
        limit=limit,
    )
    payload = {
        "schema": None,
        "input_path": dataset_path,
        "rows": rows,
        "mode": mode.value,
        "vector_column": vector_column,
        "id_column": id_column,
        "source_format": "lerobot",
    }
    result = request_json(
        "POST",
        resolved,
        f"/tables/{table_name}",
        headers=auth_headers(token_env=token_env),
        payload=payload,
        timeout=180.0,
    )
    result.setdefault("table", table_name)
    result.setdefault("rows", len(rows))
    emit(result, output=output, text=f"imported: {result.get('rows', len(rows))}\ntable: {table_name}")
