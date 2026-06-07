"""Workbench Cosmos3 commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from npa.workflows.cosmos_split import (
    Cosmos3ReasonConfig,
    build_cosmos3_reason_manifest,
    write_manifest,
)

app = typer.Typer(
    name="cosmos3",
    help="Cosmos3 reasoning workflow contracts.",
    no_args_is_help=True,
)


@app.command("reason")
def reason_cmd(
    input_uri: str = typer.Option(..., "--input-uri", help="Input rollout or frame URI."),
    output_uri: str = typer.Option(..., "--output-uri", help="Output prefix for reasoning JSON."),
    model: str = typer.Option("npa-cosmos3-reason", "--model", help="Reasoning model id."),
    image: str = typer.Option("", "--image", help="BYO Cosmos3 reason image."),
    prompt: str = typer.Option("", "--prompt", help="Optional reasoning prompt."),
    run_id: str = typer.Option("", "--run-id", help="Run id carried into the manifest."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Write manifest JSON locally."),
) -> None:
    """Build the Cosmos3 reason stage manifest."""

    payload = build_cosmos3_reason_manifest(
        Cosmos3ReasonConfig(
            input_uri=input_uri,
            output_uri=output_uri,
            model=model,
            image=image,
            prompt=prompt,
            run_id=run_id,
        )
    )
    if output_json is not None:
        payload = write_manifest(payload, output_json)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
