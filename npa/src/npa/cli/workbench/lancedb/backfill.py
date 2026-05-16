"""Backfill BDD100K-derived columns in LanceDB."""

from __future__ import annotations

import typer

from npa.workbench.lancedb.backfill import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DHASH_HAMMING_THRESHOLD,
    BackfillError,
    backfill_column,
)
from npa.workbench.lancedb.bdd100k_import import DEFAULT_LANCE_URI, DEFAULT_TABLE

from .helpers import (
    DEFAULT_TOKEN_ENV,
    OutputFormat,
    auth_headers,
    emit,
    fail,
    request_json,
    resolve_endpoint,
)


def backfill_cmd(
    udf: str = typer.Option(..., "--udf", help="BDD100K UDF to backfill."),
    table: str = typer.Option(DEFAULT_TABLE, "--table", help="LanceDB table name."),
    lance_uri: str = typer.Option(
        DEFAULT_LANCE_URI,
        "--lance-uri",
        "--input-path",
        "--output-path",
        help="LanceDB URI containing the table; input and output are the same table.",
    ),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, "--batch-size", help="Rows per CPU batch."),
    force: bool = typer.Option(False, "--force", help="Recompute rows that already have non-null values."),
    dhash_hamming_threshold: int = typer.Option(
        DEFAULT_DHASH_HAMMING_THRESHOLD,
        "--dhash-hamming-threshold",
        help="Near-duplicate threshold for udf=is_duplicate.",
    ),
    service: bool = typer.Option(False, "--service", help="Call a deployed LanceDB workbench endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="LanceDB wrapper endpoint for --service."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing wrapper token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Backfill one BDD100K UDF column through local mode or a deployed service."""
    payload = {
        "table": table,
        "udf": udf,
        "lance_uri": lance_uri,
        "batch_size": batch_size,
        "force": force,
        "dhash_hamming_threshold": dhash_hamming_threshold,
    }
    if service:
        resolved = resolve_endpoint(endpoint)
        result = request_json(
            "POST",
            resolved,
            "/backfill",
            headers=auth_headers(token_env=token_env),
            payload=payload,
            timeout=600.0,
        )
    else:
        try:
            result = backfill_column(**payload).to_dict()
        except BackfillError as exc:
            fail(str(exc))
    text = (
        f"table: {result.get('table')}\n"
        f"udf: {result.get('udf')}\n"
        f"rows_updated: {result.get('rows_updated')}\n"
        f"rows_skipped: {result.get('rows_skipped')}\n"
        f"manifest_sha256: {result.get('manifest_sha256')}"
    )
    emit(result, output=output, text=text)
