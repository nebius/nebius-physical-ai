"""Typer CLI for `npa workbench encord`.

Encord labeling/curation SaaS integration:

- ``push``  register S3 media in place into an Encord storage folder (and
  optionally link a dataset) through a cloud integration created once in the
  Encord app. Bytes stay in the bucket.
- ``pull``  materialize a curated Collection, a Dataset, or a Project's labels
  back to an S3 prefix as media + item JSON + a lineage manifest.
- ``cleanup``  tear down run-scoped Encord state by title prefix.

Every verb is a thin client of ``npa.sdk.workbench.encord``: the CLI validates
the path contract, calls the SDK, and renders the returned model.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, TypeVar

import typer

from npa.cli.workbench.lancedb.helpers import OutputFormat, emit, fail
from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.encord.schemas import DEFAULT_POLL_TIMEOUT_SECONDS

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


@app.command("cleanup")
@json_stdout_contract
def cleanup_cmd(
    title_prefix: str = typer.Option(
        ...,
        "--title-prefix",
        help="Delete Encord folders/collections/presets whose title starts "
        "with this run-scoped prefix (e.g. npa-e2e-). "
        "Minimum 4 characters.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List what would be deleted without deleting."
    ),
    output_format: OutputFormat = OUTPUT_OPTION,
) -> None:
    """Tear down run-scoped Encord state created by push."""

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
