"""Import BDD100K detection data into LanceDB."""

from __future__ import annotations

import typer

from npa.workbench.lancedb.bdd100k_import import (
    DEFAULT_LANCE_URI,
    DEFAULT_SPLITS,
    DEFAULT_TABLE,
    BDD100KImportError,
    import_bdd100k,
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


def import_bdd100k_cmd(
    source: str = typer.Option("", "--source", "--input-path", help="Local BDD100K directory or s3:// bundle prefix."),
    table: str = typer.Option(DEFAULT_TABLE, "--table", help="Destination LanceDB table name."),
    lance_uri: str = typer.Option(DEFAULT_LANCE_URI, "--lance-uri", "--output-path", help="Target LanceDB URI."),
    synthetic: int | None = typer.Option(None, "--synthetic", help="Generate N synthetic rows instead of reading source."),
    synthetic_seed: int | None = typer.Option(None, "--synthetic-seed", help="Synthetic generator seed."),
    split: list[str] = typer.Option([], "--split", help="BDD100K split to ingest; repeatable."),
    limit: int | None = typer.Option(None, "--limit", help="Maximum real-source rows per split."),
    service: bool = typer.Option(False, "--service", help="Call a deployed LanceDB workbench endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="LanceDB wrapper endpoint for --service."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing wrapper token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Import BDD100K rows through local mode or a deployed service endpoint."""
    splits = split or list(DEFAULT_SPLITS)
    payload = {
        "source": source,
        "table": table,
        "lance_uri": lance_uri,
        "synthetic": synthetic,
        "synthetic_seed": synthetic_seed,
        "splits": splits,
        "limit": limit,
    }
    if service:
        resolved = resolve_endpoint(endpoint)
        result = request_json(
            "POST",
            resolved,
            "/import-bdd100k",
            headers=auth_headers(token_env=token_env),
            payload=payload,
            timeout=600.0,
        )
    else:
        try:
            result = import_bdd100k(**payload).to_dict()
        except BDD100KImportError as exc:
            fail(str(exc))
    text = (
        f"table: {result.get('table')}\n"
        f"rows: {result.get('total_rows')}\n"
        f"version: {result.get('table_version')}\n"
        f"manifest_sha256: {result.get('manifest_sha256')}"
    )
    emit(result, output=output, text=text)
