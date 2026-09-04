"""Typer CLI for `npa workbench encord`.

Encord labeling/curation SaaS integration:

- ``push``  register S3 media in place into an Encord storage folder (and
  optionally link a dataset) through a cloud integration created once in the
  Encord app. Bytes stay in the bucket.
- ``curate``  headless curation: declare quality filters (brightness, width,
  ...) from workbench; Encord evaluates them server-side into a Collection —
  no human in the app.
- ``pull``  materialize a curated Collection, a Dataset, or a Project's labels
  back to an S3 prefix as media + item JSON + a lineage manifest.
- ``verify``  join a push receipt to a pull manifest by exact identity and
  fail closed on anything missing, resized, or checksum-mismatched.
- ``cleanup``  tear down run-scoped Encord state by title prefix.
- ``seed-demo``  stage the packaged starter clip for the augment demo workflow.
- ``system-info``  the tool's SDK pin, domain, and configured credentials.

Every verb is a thin client of ``npa.sdk.workbench.encord``: the CLI validates
the path contract, calls the SDK, and renders the returned model.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, TypeVar

import typer

from npa.cli.workbench.lancedb.helpers import OutputFormat, emit, fail
from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.encord.schemas import (
    DEFAULT_CURATE_POLL_SECONDS,
    DEFAULT_POLL_TIMEOUT_SECONDS,
)

app = typer.Typer(
    name="encord",
    help="Encord curation SaaS: register-in-place push and curated pull.",
    no_args_is_help=True,
)

T = TypeVar("T")


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


def _call(operation: Callable[[], T]) -> T:
    """Run one path check or SDK call at the CLI boundary.

    A domain error (``EncordToolError`` and subclasses, including auth) or a
    path-contract violation is the user's problem to fix: print the remedy and
    exit 1. Anything else is a bug and propagates to ``app_entry`` — exit 2,
    with the ``NPA_DEBUG`` traceback path intact — instead of being reworded
    into a client error.
    """

    from npa.cli.path_contract import PathContractError
    from npa.workbench.encord.schemas import EncordToolError

    try:
        return operation()
    except (EncordToolError, PathContractError) as exc:
        fail(str(exc))
        raise  # unreachable: fail() exits


OUTPUT_OPTION = typer.Option(OutputFormat.json, "--output", help="Output format.")
WORKFLOW_RUN_HELP = "Run id recorded in the durable artifact."


@app.command("push")
@json_stdout_contract
def push_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        help="s3:// prefix of media to register in place (bytes stay in the bucket).",
    ),
    integration: str = typer.Option(
        "",
        "--integration",
        help="Encord cloud-integration title or uuid (created once in the Encord "
        "app). Required for --transfer register; unused for upload.",
    ),
    folder: str = typer.Option(
        ...,
        "--folder",
        help="Encord storage folder title or uuid; a title is created if absent.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="s3:// destination prefix (or .json URI) for the push receipt.",
    ),
    dataset: str = typer.Option(
        "",
        "--dataset",
        help="Optional Encord dataset hash or title to link registered items into; "
        "a title is created if absent.",
    ),
    transfer: TransferMode = typer.Option(
        TransferMode.register,
        "--transfer",
        help="register: bytes stay in the bucket, Encord references objectUrls. "
        "upload: bytes are copied into Encord-hosted storage.",
    ),
    media: MediaFilter = typer.Option(
        MediaFilter.videos_images,
        "--media",
        help="Which media suffixes to register. 'mcap'/'all' enable the "
        "experimental MCAP path.",
    ),
    poll_timeout_seconds: int = typer.Option(
        DEFAULT_POLL_TIMEOUT_SECONDS,
        "--poll-timeout-seconds",
        help="Per-batch registration poll timeout.",
    ),
    workflow_run: str = typer.Option("", "--workflow-run", help=WORKFLOW_RUN_HELP),
    output_format: OutputFormat = OUTPUT_OPTION,
) -> None:
    """Register S3 media in Encord and optionally link a dataset."""

    from npa.cli.path_contract import validate_read_path, validate_write_path
    from npa.sdk.workbench.encord import push as sdk_push

    _call(lambda: validate_read_path(input_path, tool="encord push", allow_hf=False))
    _call(lambda: validate_write_path(output_path, tool="encord push", required=True))
    receipt = _call(
        lambda: sdk_push(
            input_path=input_path,
            integration=integration,
            folder=folder,
            output_path=output_path,
            dataset=dataset,
            media=media.value,
            transfer=transfer.value,
            poll_timeout_seconds=poll_timeout_seconds,
            workflow_run=workflow_run,
        )
    )
    emit(
        receipt.model_dump(by_alias=True),
        output=output_format,
        text=(
            f"pushed {receipt.units_done}/{receipt.files_discovered} item(s) to "
            f"Encord folder {receipt.folder_name!r} "
            f"(linked {receipt.linked_count}); receipt: {receipt.receipt_uri}"
        ),
    )


@app.command("curate")
@json_stdout_contract
def curate_cmd(
    folder: str = typer.Option(
        ...,
        "--folder",
        help="Encord storage folder title or uuid to curate (never created).",
    ),
    filters: list[str] = typer.Option(
        [],
        "--filter",
        help="Quality filter metric:min:max (repeatable, or comma-separated in "
        "one value), e.g. brightness:0.2:0.8. Supported metrics: width, "
        "height, area, aspect-ratio, brightness, sharpness, file-size.",
    ),
    collection: str = typer.Option(
        ...,
        "--collection",
        help="Target Encord Collection title or uuid; a title is created if "
        "absent. Pull the result with --source collection.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="s3:// destination prefix (or .json URI) for the curate receipt.",
    ),
    poll_seconds: float = typer.Option(
        DEFAULT_CURATE_POLL_SECONDS,
        "--poll-seconds",
        help="How long to wait for Encord's async server-side selection.",
    ),
    workflow_run: str = typer.Option("", "--workflow-run", help=WORKFLOW_RUN_HELP),
    output_format: OutputFormat = OUTPUT_OPTION,
) -> None:
    """Headlessly curate a folder into a Collection via Encord quality filters."""

    from npa.cli.path_contract import validate_write_path
    from npa.sdk.workbench.encord import curate as sdk_curate

    _call(lambda: validate_write_path(output_path, tool="encord curate", required=True))
    receipt = _call(
        lambda: sdk_curate(
            folder=folder,
            filters=filters,
            collection=collection,
            output_path=output_path,
            workflow_run=workflow_run,
            poll_seconds=poll_seconds,
        )
    )
    emit(
        receipt.model_dump(by_alias=True),
        output=output_format,
        text=(
            f"curated {receipt.items_selected} of {receipt.items_total} item(s) from "
            f"folder {receipt.folder_name!r} into collection "
            f"{receipt.collection_name!r}; receipt: {receipt.receipt_uri}"
        ),
    )


@app.command("pull")
@json_stdout_contract
def pull_cmd(
    source: PullSource = typer.Option(
        ...,
        "--source",
        help="Which Encord container to pull: collection, dataset, or project.",
    ),
    source_id: str = typer.Option(
        ...,
        "--source-id",
        help="Collection uuid / dataset hash / project hash, or a unique title.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="s3:// output prefix for media/, items/, labels/, and manifest.json.",
    ),
    workflow_run: str = typer.Option("", "--workflow-run", help=WORKFLOW_RUN_HELP),
    output_format: OutputFormat = OUTPUT_OPTION,
) -> None:
    """Pull curated media + labels + lineage manifest back to S3."""

    from npa.cli.path_contract import validate_write_path
    from npa.sdk.workbench.encord import pull as sdk_pull

    _call(lambda: validate_write_path(output_path, tool="encord pull", required=True))
    manifest = _call(
        lambda: sdk_pull(
            source=source.value,
            source_id=source_id,
            output_path=output_path,
            workflow_run=workflow_run,
        )
    )
    emit(
        manifest.model_dump(by_alias=True),
        output=output_format,
        text=(
            f"pulled {manifest.items_total} item(s) "
            f"({manifest.media_copied} copied, {manifest.media_downloaded} "
            f"downloaded, {manifest.label_rows} label rows) from "
            f"{manifest.source_kind} {manifest.source_name!r}; manifest: "
            f"{manifest.manifest_uri}"
        ),
    )


@app.command("verify")
@json_stdout_contract
def verify_cmd(
    receipt_uri: str = typer.Option(
        ...,
        "--receipt-uri",
        help="s3:// URI of the push receipt (push_receipt.json).",
    ),
    manifest_uri: str = typer.Option(
        ...,
        "--manifest-uri",
        help="s3:// URI of the pull manifest (manifest.json).",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="s3:// destination prefix (or .json URI) for the roundtrip report.",
    ),
    workflow_run: str = typer.Option("", "--workflow-run", help=WORKFLOW_RUN_HELP),
    output_format: OutputFormat = OUTPUT_OPTION,
) -> None:
    """Verify a push receipt against a pull manifest by exact identity."""

    from npa.cli.path_contract import validate_read_path, validate_write_path
    from npa.sdk.workbench.encord import verify as sdk_verify

    _call(
        lambda: validate_read_path(
            receipt_uri, tool="encord verify", option="--receipt-uri", allow_hf=False
        )
    )
    _call(
        lambda: validate_read_path(
            manifest_uri, tool="encord verify", option="--manifest-uri", allow_hf=False
        )
    )
    _call(lambda: validate_write_path(output_path, tool="encord verify", required=True))
    report = _call(
        lambda: sdk_verify(
            receipt_uri=receipt_uri,
            manifest_uri=manifest_uri,
            output_path=output_path,
            workflow_run=workflow_run,
        )
    )
    emit(
        report.model_dump(by_alias=True),
        output=output_format,
        text=(
            f"roundtrip {report.status}: {report.matched}/{report.expected} matched, "
            f"{report.checksum_verified} checksum-verified, "
            f"{report.checksum_unavailable} unavailable; report: {report.report_uri}"
        ),
    )


@app.command("cleanup")
@json_stdout_contract
def cleanup_cmd(
    title_prefix: str = typer.Option(
        ...,
        "--title-prefix",
        help="Delete Encord folders/collections/presets whose title starts "
        "with this run-scoped prefix (e.g. npa-e2e- or npa-demo-src-). "
        "Minimum 4 characters.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List what would be deleted without deleting."
    ),
    output_format: OutputFormat = OUTPUT_OPTION,
) -> None:
    """Tear down run-scoped Encord state created by push/curate/seed-demo."""

    from npa.sdk.workbench.encord import cleanup as sdk_cleanup

    summary = _call(lambda: sdk_cleanup(title_prefix=title_prefix, dry_run=dry_run))
    verb = "would delete" if dry_run else "deleted"
    emit(
        summary,
        output=output_format,
        text=(
            f"{verb} {len(summary['folders_deleted'])} folder(s) "
            f"({summary['items_deleted']} item(s)), "
            f"{len(summary['collections_deleted'])} collection(s), "
            f"{len(summary['presets_deleted'])} preset(s); "
            f"{len(summary['datasets_undeletable'])} dataset(s) need app-side "
            "removal (the SDK cannot delete datasets)"
        ),
    )


@app.command("seed-demo")
@json_stdout_contract
def seed_demo_cmd(
    media_uri: str = typer.Option(
        ...,
        "--media-uri",
        help="s3:// prefix to stage the packaged demo starter clip under.",
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="Run-scoped demo dataset title to create and push into.",
    ),
    active_source_id: str = typer.Option(
        ...,
        "--active-source-id",
        help="The workflow's configured source id; when it differs from "
        "--dataset the operator supplied a curated source and seeding no-ops.",
    ),
    transfer: TransferMode = typer.Option(
        TransferMode.register,
        "--transfer",
        help="Push mode for the demo clip (same default as push: register).",
    ),
    integration: str = typer.Option(
        "", "--integration", help="Cloud integration title/uuid (register mode only)."
    ),
    output_format: OutputFormat = OUTPUT_OPTION,
) -> None:
    """Seed the demo source dataset for encord-cosmos3-augment, or no-op."""

    from npa.cli.path_contract import validate_write_path
    from npa.sdk.workbench.encord import seed_demo as sdk_seed_demo

    _call(
        lambda: validate_write_path(
            media_uri, tool="encord seed-demo", option="--media-uri", required=True
        )
    )
    summary = _call(
        lambda: sdk_seed_demo(
            media_uri=media_uri,
            dataset=dataset,
            active_source_id=active_source_id,
            transfer=transfer.value,
            integration=integration,
        )
    )
    skipped = summary.get("skipped")
    emit(
        summary,
        output=output_format,
        text=(
            f"seed skipped: {skipped}"
            if skipped
            else f"seeded demo dataset {summary.get('dataset')!r} from the packaged "
            "starter clip"
        ),
    )


@app.command("system-info")
@json_stdout_contract
def system_info_cmd(
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Show the Encord tool's SDK pin, API domain, and configured credentials."""

    from npa.sdk.workbench.encord import system_info as sdk_system_info

    emit(_call(sdk_system_info), output=output_format)
