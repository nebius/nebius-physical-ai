"""Status command for LanceDB Workbench services."""

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
)


def status_cmd(
    endpoint: str = typer.Option("", "--endpoint", help="LanceDB wrapper or Cloud endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing wrapper token."),
    api_key_env: str = typer.Option(DEFAULT_API_KEY_ENV, "--api-key-env", help="Environment variable containing LanceDB Cloud API key."),
    database: str = typer.Option("", "--database", help="LanceDB Cloud database name."),
    cloud_region: str = typer.Option("", "--cloud-region", help="LanceDB Cloud region."),
    cloud: bool = typer.Option(False, "--cloud", help="Use LanceDB Cloud auth headers."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Check endpoint reachability."""
    resolved = resolve_endpoint(endpoint)
    headers = auth_headers(
        token_env=token_env,
        api_key_env=api_key_env,
        database=database,
        cloud_region=cloud_region,
        cloud=cloud,
    )
    payload = request_json("GET", resolved, "/health", headers=headers)
    payload.setdefault("endpoint", resolved)
    emit(payload, output=output, text=f"status: {payload.get('status', 'unknown')}")
