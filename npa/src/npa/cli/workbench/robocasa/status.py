"""Fetch RoboCasa run status."""

from __future__ import annotations

import typer

from npa.workbench.robocasa.schemas import DEFAULT_TOKEN_ENV

from npa.cli.workbench.robocasa.helpers import OutputFormat, emit, request_json, resolve_endpoint


def status_cmd(
    run_id: str = typer.Option(..., "--run-id", help="RoboCasa run ID."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="RoboCasa service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Fetch RoboCasa run status."""
    if service:
        result = request_json(
            "GET",
            resolve_endpoint(endpoint),
            "/status",
            params={"run_id": run_id},
            token_env=token_env,
            timeout=30.0,
        )
    else:
        from npa.sdk.workbench.robocasa import status

        result = status(run_id=run_id).model_dump(mode="json")
    emit(result, output=output, text=f"status: {result.get('status')}\ncapability: {result.get('capability')}")
