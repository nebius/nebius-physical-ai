"""CLI for the stateless Encord SaaS transport."""

from __future__ import annotations

import json
from enum import Enum

import typer

from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.encord.schemas import (
    DEFAULT_POLL_TIMEOUT_SECONDS,
    EncordToolError,
)

app = typer.Typer(
    name="encord",
    help="Register S3 media with Encord SaaS and materialize curated results.",
    no_args_is_help=True,
)


class TransferMode(str, Enum):
    register = "register"
    upload = "upload"


class MediaFilter(str, Enum):
    videos_images = "videos-images"
    mcap = "mcap"
    all = "all"


class PullSource(str, Enum):
    collection = "collection"
    dataset = "dataset"
    project = "project"


class LabelExport(str, Enum):
    none = "none"
    initialize = "initialize"


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def _emit(model: object, *, output_json: bool, text: str) -> None:
    if output_json:
        payload = model.model_dump(by_alias=True)  # type: ignore[attr-defined]
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text)


@app.command("push")
@json_stdout_contract
def push_cmd(
    input_path: str = typer.Option(..., "--input-path", help="S3 media prefix."),
    integration: str = typer.Option(
        "",
        "--integration",
        help="Encord cloud-integration title or UUID. Required for register mode.",
    ),
    folder: str = typer.Option(..., "--folder", help="Encord folder title or UUID."),
    output_path: str = typer.Option(
        ..., "--output-path", help="S3 prefix or JSON URI for the durable receipt."
    ),
    dataset: str = typer.Option(
        "", "--dataset", help="Optional dataset title or hash to link exact items into."
    ),
    media: MediaFilter = typer.Option(
        MediaFilter.videos_images, "--media", help="Media suffix filter."
    ),
    transfer: TransferMode = typer.Option(
        TransferMode.register,
        "--transfer",
        help="register retains S3 as source of record; upload creates an Encord copy.",
    ),
    poll_timeout_seconds: int = typer.Option(
        DEFAULT_POLL_TIMEOUT_SECONDS,
        "--poll-timeout-seconds",
        min=1,
        help="Per-batch Encord registration poll timeout.",
    ),
    workflow_run: str = typer.Option(
        "", "--workflow-run", help="Workflow run identifier recorded in the receipt."
    ),
    identity_sidecar_uri: str = typer.Option(
        "",
        "--identity-sidecar",
        help="Optional S3 URI of exact source-to-Encord identity assertions.",
    ),
    output_json: bool = typer.Option(False, "--json", help="Emit one JSON document."),
) -> None:
    """Register or explicitly upload S3 media and write a durable receipt."""

    from npa.cli.path_contract import (
        PathContractError,
        validate_read_path,
        validate_write_path,
    )
    from npa.sdk.workbench.encord import push

    try:
        validate_read_path(input_path, tool="encord push", allow_hf=False)
        validate_write_path(output_path, tool="encord push", required=True)
        if identity_sidecar_uri:
            validate_read_path(
                identity_sidecar_uri,
                tool="encord push",
                option="--identity-sidecar",
                allow_hf=False,
            )
        receipt = push(
            input_path=input_path,
            integration=integration,
            folder=folder,
            output_path=output_path,
            dataset=dataset,
            media=media.value,
            transfer=transfer.value,
            poll_timeout_seconds=poll_timeout_seconds,
            workflow_run=workflow_run,
            identity_sidecar_uri=identity_sidecar_uri,
        )
    except (PathContractError, EncordToolError) as exc:
        _fail(str(exc))
    _emit(
        receipt,
        output_json=output_json,
        text=(
            f"Encord push {receipt.status}: {receipt.counts.successful}/"
            f"{receipt.counts.discovered} successful; receipt: {receipt.receipt_uri}"
        ),
    )


@app.command("pull")
@json_stdout_contract
def pull_cmd(
    source: PullSource = typer.Option(
        ..., "--source", help="Encord source kind: collection, dataset, or project."
    ),
    source_id: str = typer.Option(
        ..., "--source-id", help="Exact source identifier or unique exact title."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="S3 output prefix for media and manifest."
    ),
    workflow_run: str = typer.Option(
        "", "--workflow-run", help="Workflow run identifier recorded in the manifest."
    ),
    label_export: LabelExport = typer.Option(
        LabelExport.none,
        "--label-export",
        help=(
            "none is read-only. initialize explicitly allows Encord label "
            "initialization, which may create or change remote label-row state."
        ),
    ),
    output_json: bool = typer.Option(False, "--json", help="Emit one JSON document."),
) -> None:
    """Materialize an Encord source to S3 with an exact lineage manifest."""

    from npa.cli.path_contract import PathContractError, validate_write_path
    from npa.sdk.workbench.encord import pull

    try:
        validate_write_path(output_path, tool="encord pull", required=True)
        manifest = pull(
            source=source.value,
            source_id=source_id,
            output_path=output_path,
            workflow_run=workflow_run,
            label_export=label_export.value,
        )
    except (PathContractError, EncordToolError) as exc:
        _fail(str(exc))
    _emit(
        manifest,
        output_json=output_json,
        text=(
            f"Encord pull {manifest.status}: {manifest.counts.successful}/"
            f"{manifest.counts.discovered} successful; manifest: {manifest.manifest_uri}"
        ),
    )


@app.command("verify-roundtrip")
@json_stdout_contract
def verify_roundtrip_cmd(
    receipt_uri: str = typer.Option(
        ..., "--receipt-uri", help="S3 URI of a final Encord push receipt."
    ),
    manifest_uri: str = typer.Option(
        ..., "--manifest-uri", help="S3 URI of a final Encord pull manifest."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="S3 URI for the roundtrip verification report."
    ),
    workflow_run: str = typer.Option(
        "", "--workflow-run", help="Workflow run identifier recorded in the report."
    ),
    output_json: bool = typer.Option(False, "--json", help="Emit one JSON document."),
) -> None:
    """Verify identity, destination existence, size, and compatible checksums."""

    from npa.cli.path_contract import (
        PathContractError,
        validate_read_path,
        validate_write_path,
    )
    from npa.sdk.workbench.encord import verify_roundtrip

    try:
        validate_read_path(
            receipt_uri, tool="encord verify-roundtrip", option="--receipt-uri", allow_hf=False
        )
        validate_read_path(
            manifest_uri, tool="encord verify-roundtrip", option="--manifest-uri", allow_hf=False
        )
        validate_write_path(output_path, tool="encord verify-roundtrip", required=True)
        report = verify_roundtrip(
            receipt_uri=receipt_uri,
            manifest_uri=manifest_uri,
            output_path=output_path,
            workflow_run=workflow_run,
        )
    except (PathContractError, EncordToolError) as exc:
        _fail(str(exc))
    _emit(
        report,
        output_json=output_json,
        text=(
            f"Encord roundtrip {report.status}: {report.matched}/{report.expected} matched; "
            f"report: {report.report_uri}"
        ),
    )
