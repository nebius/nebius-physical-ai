"""Materialized-view commands for the LanceDB Workbench CLI."""

from __future__ import annotations

import typer

from npa.solutions.workbench.lancedb.bdd100k_import import (
    DEFAULT_LANCE_URI,
    DEFAULT_TABLE,
)
from npa.solutions.workbench.lancedb.views import (
    DEFAULT_QUERY_LIMIT,
    MVError,
    create_mv,
    query_table,
    refresh_mv,
)

from .helpers import (
    DEFAULT_TOKEN_ENV,
    OutputFormat,
    auth_headers,
    emit,
    fail,
    request_json,
    resolve_endpoint,
)


def create_mv_cmd(
    name: str = typer.Option(..., "--name", help="Materialized view table name."),
    source_table: str = typer.Option(
        DEFAULT_TABLE, "--source", "--source-table", help="Source LanceDB table name."
    ),
    filter_sql: str = typer.Option(
        ..., "--filter", "--filter-sql", help="SQL WHERE clause for the source table."
    ),
    lance_uri: str = typer.Option(
        DEFAULT_LANCE_URI,
        "--lance-uri",
        "--input-path",
        "--output-path",
        help="LanceDB URI containing the source and view tables.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Recompute and overwrite an existing registered view."
    ),
    service: bool = typer.Option(
        False, "--service", help="Call a deployed LanceDB workbench endpoint."
    ),
    endpoint: str = typer.Option(
        "", "--endpoint", help="LanceDB wrapper endpoint for --service."
    ),
    token_env: str = typer.Option(
        DEFAULT_TOKEN_ENV,
        "--token-env",
        help="Environment variable containing wrapper token.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Create a filtered materialized view through local mode or a deployed service."""
    payload = {
        "name": name,
        "source_table": source_table,
        "filter_sql": filter_sql,
        "lance_uri": lance_uri,
        "force": force,
    }
    if service:
        result = request_json(
            "POST",
            resolve_endpoint(endpoint),
            "/create-mv",
            headers=auth_headers(token_env=token_env),
            payload=payload,
            timeout=600.0,
        )
    else:
        try:
            result = create_mv(**payload).to_dict()
        except MVError as exc:
            fail(str(exc))
    text = (
        f"view_name: {result.get('view_name')}\n"
        f"source_table: {result.get('source_table')}\n"
        f"row_count: {result.get('row_count')}\n"
        f"manifest_sha256: {result.get('manifest_sha256')}"
    )
    emit(result, output=output, text=text)


def refresh_mv_cmd(
    name: str = typer.Option(..., "--name", help="Materialized view table name."),
    lance_uri: str = typer.Option(
        DEFAULT_LANCE_URI,
        "--lance-uri",
        "--input-path",
        "--output-path",
        help="LanceDB URI containing the registry and view table.",
    ),
    service: bool = typer.Option(
        False, "--service", help="Call a deployed LanceDB workbench endpoint."
    ),
    endpoint: str = typer.Option(
        "", "--endpoint", help="LanceDB wrapper endpoint for --service."
    ),
    token_env: str = typer.Option(
        DEFAULT_TOKEN_ENV,
        "--token-env",
        help="Environment variable containing wrapper token.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Refresh a registered materialized view through local mode or a deployed service."""
    payload = {"name": name, "lance_uri": lance_uri}
    if service:
        result = request_json(
            "POST",
            resolve_endpoint(endpoint),
            "/refresh-mv",
            headers=auth_headers(token_env=token_env),
            payload=payload,
            timeout=600.0,
        )
    else:
        try:
            result = refresh_mv(**payload).to_dict()
        except MVError as exc:
            fail(str(exc))
    text = (
        f"view_name: {result.get('view_name')}\n"
        f"row_count_before: {result.get('row_count_before')}\n"
        f"row_count_after: {result.get('row_count_after')}\n"
        f"manifest_sha256: {result.get('manifest_sha256')}"
    )
    emit(result, output=output, text=text)


def query_table_cmd(
    table: str = typer.Option(..., "--table", help="LanceDB table name to query."),
    filter_sql: str = typer.Option(
        "", "--filter", "--filter-sql", help="Optional SQL WHERE clause."
    ),
    select: list[str] = typer.Option(
        [], "--select", help="Column to return; repeatable."
    ),
    limit: int = typer.Option(
        DEFAULT_QUERY_LIMIT, "--limit", help="Maximum rows to return."
    ),
    lance_uri: str = typer.Option(
        DEFAULT_LANCE_URI,
        "--lance-uri",
        "--input-path",
        help="LanceDB URI containing the table.",
    ),
    service: bool = typer.Option(
        False, "--service", help="Call a deployed LanceDB workbench endpoint."
    ),
    endpoint: str = typer.Option(
        "", "--endpoint", help="LanceDB wrapper endpoint for --service."
    ),
    token_env: str = typer.Option(
        DEFAULT_TOKEN_ENV,
        "--token-env",
        help="Environment variable containing wrapper token.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Run a bounded SQL-filtered LanceDB table query."""
    payload = {
        "table": table,
        "lance_uri": lance_uri,
        "filter_sql": filter_sql or None,
        "select": select or None,
        "limit": limit,
    }
    if service:
        result = request_json(
            "POST",
            resolve_endpoint(endpoint),
            "/query-table",
            headers=auth_headers(token_env=token_env),
            payload=payload,
            timeout=120.0,
        )
    else:
        try:
            result = query_table(**payload).to_dict()
        except MVError as exc:
            fail(str(exc))
    text = f"row_count: {result.get('row_count')}\ntotal_rows_matched: {result.get('total_rows_matched')}"
    emit(result, output=output, text=text)
