"""Workbench Cosmos2 commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from npa.workflows.cosmos_split import (
    Cosmos2TransferConfig,
    build_cosmos2_transfer_manifest,
    write_manifest,
)

app = typer.Typer(
    name="cosmos2",
    help="Cosmos2 transfer workflow contracts.",
    no_args_is_help=True,
)


@app.command("transfer")
def transfer_cmd(
    input_uri: str = typer.Option(..., "--input-uri", help="Input frames, assets, or rollout URI."),
    output_uri: str = typer.Option(..., "--output-uri", help="Output prefix for transferred frames."),
    assets_uri: str = typer.Option("", "--assets-uri", help="Optional sim asset source path."),
    scene_spec_uri: str = typer.Option("", "--scene-spec-uri", help="Optional SceneSpec path."),
    image: str = typer.Option("", "--image", help="BYO Cosmos2 transfer image."),
    run_id: str = typer.Option("", "--run-id", help="Run id carried into the manifest."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Write manifest JSON locally."),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Run the real Cosmos-Transfer2.5 model (requires the transfer image/GPU).",
    ),
    spec: str = typer.Option(
        "", "--spec", help="controlnet_spec path (relative to the transfer repo) for --execute."
    ),
) -> None:
    """Build the Cosmos2 transfer stage manifest (or run the real model with --execute)."""

    payload = build_cosmos2_transfer_manifest(
        Cosmos2TransferConfig(
            input_uri=input_uri,
            output_uri=output_uri,
            assets_uri=assets_uri,
            scene_spec_uri=scene_spec_uri,
            image=image,
            run_id=run_id,
        )
    )
    if execute:
        from npa.workbench.cosmos.transfer import cosmos_transfer_available, run_cosmos_transfer

        if not cosmos_transfer_available():
            raise typer.BadParameter(
                "--execute needs the cosmos-transfer2.5 runtime "
                "(run inside the npa-cosmos2-transfer image on a GPU)."
            )
        transfer = run_cosmos_transfer(run_id=run_id, spec=spec or None)
        payload["status"] = "executed"
        payload["mode"] = "cosmos_transfer2.5"
        payload["output_video"] = transfer["video_path"]
        payload["video_bytes"] = transfer["video_bytes"]
        payload["control_spec"] = transfer["spec"]
    if output_json is not None:
        payload = write_manifest(payload, output_json)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
