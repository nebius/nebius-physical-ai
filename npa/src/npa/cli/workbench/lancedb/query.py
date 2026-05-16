"""Vector query command for LanceDB Workbench services."""

from __future__ import annotations

from pathlib import Path

import typer

from .helpers import (
    DEFAULT_TOKEN_ENV,
    OutputFormat,
    auth_headers,
    emit,
    parse_vector,
    request_json,
    resolve_endpoint,
    validate_table_name,
    validate_top_k,
)


def query_cmd(
    endpoint: str = typer.Option("", "--endpoint", help="LanceDB wrapper endpoint."),
    table: str = typer.Option(..., "--table", help="Table name."),
    vector: str = typer.Option("", "--vector", help="JSON array query vector."),
    vector_file: Path | None = typer.Option(None, "--vector-file", exists=False, help="Path to JSON query vector."),
    top_k: int = typer.Option(5, "--top-k", help="Number of neighbors to return."),
    filter_expr: str = typer.Option("", "--filter", help="Optional scalar filter expression."),
    select: list[str] = typer.Option([], "--select", help="Column to include; repeatable."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing wrapper token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Run a vector search query."""
    resolved = resolve_endpoint(endpoint)
    table_name = validate_table_name(table)
    query_vector = parse_vector(vector, vector_file)
    k = validate_top_k(top_k)
    headers = auth_headers(token_env=token_env)
    payload = {
        "vector": query_vector,
        "top_k": k,
        "filter": filter_expr,
        "select": select,
    }
    result = request_json("POST", resolved, f"/tables/{table_name}/query", headers=headers, payload=payload)
    rows = result.get("results", [])
    text = "\n".join(str(row) for row in rows) if rows else "No results."
    emit(result, output=output, text=text)
