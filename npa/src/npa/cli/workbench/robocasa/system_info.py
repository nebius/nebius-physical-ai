"""Show RoboCasa runtime information."""

from __future__ import annotations

import typer

from npa.workbench.robocasa.schemas import DEFAULT_TOKEN_ENV

from npa.cli.workbench.robocasa.helpers import OutputFormat, emit, request_json, resolve_endpoint


def system_info_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="RoboCasa service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show RoboCasa runtime information."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/system-info", token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.robocasa.capabilities import system_info

        result = system_info().model_dump(mode="json")
    emit(result, output=output)
