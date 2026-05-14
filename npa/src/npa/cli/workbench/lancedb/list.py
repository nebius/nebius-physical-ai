"""List tables command for LanceDB Workbench services."""

from __future__ import annotations

import typer

from .helpers import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_TOKEN_ENV,
    OutputFormat,
    auth_headers,
    emit,
    request_json,
    resolve_endpoint,
    validate_limit,
)


def list_cmd(
    endpoint: str = typer.Option("", "--endpoint", help="LanceDB wrapper or Cloud endpoint."),
    limit: int = typer.Option(100, "--limit", help="Maximum number of tables to return."),
    prefix: str = typer.Option("", "--prefix", help="Optional table-name prefix filter."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing wrapper token."),
    api_key_env: str = typer.Option(DEFAULT_API_KEY_ENV, "--api-key-env", help="Environment variable containing LanceDB Cloud API key."),
    cloud: bool = typer.Option(False, "--cloud", help="Use LanceDB Cloud auth headers."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List tables in a LanceDB instance."""
    validate_limit(limit)
    resolved = resolve_endpoint(endpoint)
    headers = auth_headers(token_env=token_env, api_key_env=api_key_env, cloud=cloud)
    payload = request_json(
        "GET",
        resolved,
        "/tables",
        headers=headers,
    )
    tables = payload.get("tables", [])
    if prefix:
        tables = [table for table in tables if str(table).startswith(prefix)]
    tables = tables[:limit]
    result = {"endpoint": resolved, "tables": tables, "count": len(tables)}
    emit(result, output=output, text="\n".join(str(table) for table in tables) or "No tables found.")
