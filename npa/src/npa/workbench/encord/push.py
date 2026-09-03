"""Register-in-place or upload push of S3 media into an Encord storage folder.

In register mode bytes never move: the tool lists the S3 prefix, builds public
objectUrls against the configured endpoint, and registers them with Encord
through a cloud integration. In upload mode the bytes are copied into
Encord-hosted storage instead. Either way, items are then explicitly linked to
a dataset (Encord never links automatically), and the receipt is written before
any failure exit so lineage survives fail-closed runs.

Identity is exact (adopted from PR #363): every item is registered with
namespaced ``npa.source_uri`` clientMetadata, and receipt lineage resolves
through that metadata or the item's normalized objectUrl — never through
display names. A write-ahead receipt lands before the first Encord mutation so
even a crash mid-mutation leaves a durable record of intent.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, cast
from urllib.parse import quote

from npa.clients.storage import StorageError, parse_bucket_uri
from npa.workbench.encord.client import (
    ResolvedRef,
    default_user_client,
    resolve_dataset,
    resolve_domain,
    resolve_folder,
    resolve_integration,
    resolve_public_endpoint,
)
from npa.workbench.encord.identity import (
    IdentityIndex,
    canonical_s3_uri,
    identity_metadata,
)
from npa.workbench.encord.integrity import etag_checksum, hash_file
from npa.workbench.encord.schemas import (
    DEFAULT_MEDIA_FILTER,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    DEFAULT_TRANSFER,
    PUSH_RECEIPT_FILENAME,
    TRANSFER_MODES,
    EncordToolError,
    IdentitySignal,
    MediaFilter,
    PushedItem,
    PushReceipt,
    PushStatus,
    TransferMode,
)
from npa.workbench.encord.storage import artifact_uri_for, error_text, finalize_artifact

MEDIA_CATEGORIES: dict[str, str] = {
    ".mp4": "videos",
    ".png": "images",
    ".jpg": "images",
    ".jpeg": "images",
}
# encord==0.1.201 has no cloud-registration category for a raw MCAP file: the
# upload format's `scenes` entries require a per-stream SceneBuilder document,
# not an objectUrl. Until a live spike pins a supported path, .mcap keys are
# discovered and accounted for in the receipt as experimental errors rather
# than sent with a guessed schema.
MCAP_SUFFIX = ".mcap"
MCAP_UNSUPPORTED_ERROR = (
    "MCAP cloud registration is not supported by the pinned encord SDK upload "
    "format (scenes require per-stream assets, not an objectUrl). Tracked as an "
    "experimental follow-up; push videos/images with --media videos-images."
)
FILTER_CATEGORIES: dict[str, dict[str, str]] = {
    "videos-images": MEDIA_CATEGORIES,
    "mcap": {MCAP_SUFFIX: "mcap"},
    "all": {**MEDIA_CATEGORIES, MCAP_SUFFIX: "mcap"},
}
BATCH_SIZE = 500
# One short re-list absorbs folder-listing lag right after registration.
IDENTITY_RELIST_DELAY_SECONDS = 5.0


class S3Object(NamedTuple):
    """One discovered source object: the listing facts push carries forward."""

    key: str
    category: str
    size: int
    etag: str


def require_transfer_integration(transfer: str, integration: str) -> None:
    """Register mode needs a cloud integration; say so before any I/O happens.

    Encord can only reference bytes that stay in the bucket through an
    integration that can read them, so an empty ``--integration`` in register
    mode is a deterministic argument error — it must not surface only after
    the S3 listing, the Encord handshake, or (for seed-demo) a clip upload.
    """

    if transfer == "register" and not integration.strip():
        raise EncordToolError(
            "--integration is required with --transfer register (bytes stay in the "
            "bucket, so Encord needs a cloud integration that can read them). Pass "
            "--integration <title-or-uuid> created once in the Encord app, or use "
            "--transfer upload."
        )


def push_receipt_uri_for(output_path: str) -> str:
    """The exact receipt URI a given --output-path resolves to."""

    return artifact_uri_for(output_path, PUSH_RECEIPT_FILENAME)


def object_url_for(endpoint_url: str, bucket: str, key: str) -> str:
    """Path-style public URL for one object, matching the Encord integration."""

    return f"{endpoint_url.rstrip('/')}/{bucket}/{quote(key, safe='/')}"


def discover_objects(
    storage_client: Any, input_uri: str, media: str
) -> tuple[list[S3Object], list[str]]:
    """List the supported objects under the prefix plus the skipped keys."""

    allowed = FILTER_CATEGORIES.get(media)
    if allowed is None:
        raise EncordToolError(
            f"Unknown --media value {media!r}. Choices: "
            f"{', '.join(FILTER_CATEGORIES)}."
        )
    bucket, prefix = parse_bucket_uri(input_uri)
    entries: list[S3Object] = []
    skipped: list[str] = []
    paginator = storage_client.s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            category = allowed.get(Path(key).suffix.lower())
            if category:
                entries.append(
                    S3Object(
                        key, category, int(obj.get("Size") or 0), str(obj.get("ETag") or "")
                    )
                )
            else:
                skipped.append(key)
    if not entries:
        raise EncordToolError(
            f"No supported media found under {input_uri} (media filter {media!r})."
        )
    return entries, skipped


def _pushed_item(
    obj: S3Object, *, bucket: str, endpoint_url: str, transfer: str
) -> PushedItem:
    """The receipt row for one discovered object, before anything is sent."""

    checksum, checksum_kind = etag_checksum(obj.etag)
    is_mcap = obj.category == "mcap"
    return PushedItem(
        key=obj.key,
        source_etag=obj.etag.strip().strip('"'),
        source_uri=canonical_s3_uri(bucket, obj.key),
        # objectUrls (and therefore a public endpoint) exist only in register
        # mode; upload mode moves the bytes themselves.
        object_url=(
            object_url_for(endpoint_url, bucket, obj.key)
            if transfer == "register" and not is_mcap
            else ""
        ),
        category=obj.category,
        source_size=obj.size,
        source_checksum=checksum,
        source_checksum_kind=checksum_kind,
        status="experimental_error" if is_mcap else "registered",
        error=MCAP_UNSUPPORTED_ERROR if is_mcap else "",
    )


def build_upload_json(items: list[PushedItem]) -> dict[str, Any]:
    """Encord upload-format JSON for one registration batch.

    Every entry carries the namespaced npa.source_uri clientMetadata and the
    full object key as title, so identity never rests on a display name.
    """

    payload: dict[str, Any] = {"skip_duplicate_urls": True}
    for item in items:
        payload.setdefault(item.category, []).append(
            {
                "objectUrl": item.object_url,
                "title": item.key,
                "clientMetadata": identity_metadata(item.source_uri),
            }
        )
    return payload


def _chunks(items: list[PushedItem], size: int) -> list[list[PushedItem]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _folder_inventory(folder_obj: Any) -> list[Any]:
    return list(folder_obj.list_items(page_size=1000, include_client_metadata=True))


def _resolve_identities(
    folder_obj: Any,
    items: list[PushedItem],
    *,
    attempts: int = 2,
    fail_unresolved: bool = False,
) -> None:
    """Resolve each registered item's Encord uuid by exact identity only.

    The folder inventory exposes each item's clientMetadata (npa.source_uri)
    and registered objectUrl; resolution matches on those and nothing else —
    display names, and in particular basenames, are never identity. Items the
    old code registered without metadata still resolve through their
    objectUrl. One short re-list absorbs listing lag right after registration.

    Outcomes are recorded on the items themselves (uuid + signal, or an error
    row). With ``fail_unresolved`` (the post-registration pass) an item that
    still has no exact identity is an error, not a silent gap: an unattributed
    row could never be linked, pulled, or verified.
    """

    for attempt in range(attempts):
        pending = [
            item
            for item in items
            if item.status == "registered" and not item.item_uuid
        ]
        if not pending:
            return
        if attempt:
            time.sleep(IDENTITY_RELIST_DELAY_SECONDS)
        index = IdentityIndex(_folder_inventory(folder_obj))
        for item in pending:
            resolution = index.resolve(
                source_uri=item.source_uri, submitted_object_url=item.object_url
            )
            if resolution.resolved:
                item.item_uuid = resolution.item_uuid
                item.identity_signal = cast(IdentitySignal, resolution.signal)
            elif resolution.error_code == "identity_conflict":
                item.status = "error"
                item.error = resolution.error
    if fail_unresolved:
        for item in items:
            if item.status == "registered" and not item.item_uuid:
                item.status = "error"
                item.error = (
                    "registered but no exact metadata or object URL identity "
                    "matched in the folder inventory"
                )


def _register_items(
    folder_obj: Any,
    items: list[PushedItem],
    *,
    integration_id: str,
    poll_timeout_seconds: int,
) -> PushStatus:
    """Register objectUrls in batches and poll each job; record unit errors.

    Per-item outcomes live on the items: unit errors Encord attributes to an
    objectUrl are recorded there, and lineage (the Encord uuid per item) is
    attached afterwards by exact identity in ``_resolve_identities``, never
    from the job result's display names. The receipt's counts are derived from
    item statuses, so Encord's aggregate counters are never a second source.
    """

    by_url = {item.object_url: item for item in items}
    status: PushStatus = "done"
    for batch in _chunks(items, BATCH_SIZE):
        job_id = folder_obj.add_private_data_to_folder_start(
            integration_id=integration_id,
            private_files=build_upload_json(batch),
            ignore_errors=True,
        )
        result = folder_obj.add_private_data_to_folder_get_result(
            job_id, timeout_seconds=poll_timeout_seconds
        )
        state = getattr(result.status, "name", str(result.status)).upper()
        for unit_error in getattr(result, "unit_errors", None) or []:
            for url in getattr(unit_error, "object_urls", None) or []:
                matched = by_url.get(str(url))
                if matched is not None:
                    matched.status = "error"
                    matched.error = str(getattr(unit_error, "error", "") or "unit error")
        if state == "PENDING":
            status = "timeout"
            break
        if state in ("ERROR", "CANCELLED"):
            status = "failed"
            break
    return status


def _upload_items(
    folder_obj: Any,
    items: list[PushedItem],
    *,
    storage_client: Any,
    source_bucket: str,
) -> None:
    """Copy each object's bytes into Encord-hosted storage.

    Uploads are synchronous per file (the SDK returns the new item uuid), so
    unlike register mode there is no job polling; failures are recorded per
    item and the run still fails closed after the receipt is written.
    """

    for item in items:
        try:
            with tempfile.TemporaryDirectory(prefix="npa-encord-upload-") as tmp:
                local = Path(tmp) / Path(item.key).name
                storage_client.download_file(f"s3://{source_bucket}/{item.key}", str(local))
                # The bytes are local anyway: record their content digest so
                # the roundtrip verifier can compare pulled bytes exactly.
                digest = hash_file(local)
                item.source_size = digest.size
                item.source_checksum = digest.sha256
                item.source_checksum_kind = "sha256"
                upload = (
                    folder_obj.upload_video
                    if item.category == "videos"
                    else folder_obj.upload_image
                )
                # The identity metadata always travels with the bytes: an
                # upload without it would be an unattributable item.
                item_uuid = upload(
                    str(local),
                    title=item.key,
                    client_metadata=identity_metadata(item.source_uri),
                )
            item.item_uuid = str(item_uuid)
            item.identity_signal = "uploaded"
            item.status = "uploaded"
        except Exception as exc:  # noqa: BLE001 - recorded per item, run fails closed
            item.status = "error"
            item.error = str(exc)


def _link_dataset(dataset_obj: Any, items: list[PushedItem]) -> int:
    """Link this push's identity-resolved items into the dataset.

    Register mode with skip_duplicate_urls reports only newly added items, so
    a re-push would otherwise link nothing — but by link time every item
    (fresh or pre-existing) carries the uuid exact identity resolved, and
    link_items skips already-linked uuids server-side, so this is idempotent.
    """

    uuids = sorted({item.item_uuid for item in items if item.item_uuid})
    if not uuids:
        return 0
    dataset_obj.link_items(uuids)
    return len(uuids)


def run_push(
    *,
    input_path: str,
    integration: str,
    folder: str,
    output_path: str,
    dataset: str = "",
    media: str = DEFAULT_MEDIA_FILTER,
    transfer: str = DEFAULT_TRANSFER,
    poll_timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
    workflow_run: str = "",
    user_client: Any = None,
    storage_client: Any = None,
    environ: dict[str, str] | None = None,
) -> PushReceipt:
    """Push the prefix into Encord, link a dataset, and write the receipt."""

    if transfer not in TRANSFER_MODES:
        raise EncordToolError(
            f"Unknown --transfer value {transfer!r}. Choices: {', '.join(TRANSFER_MODES)}."
        )
    transfer_mode = cast(TransferMode, transfer)
    require_transfer_integration(transfer, integration)
    try:
        bucket, _ = parse_bucket_uri(input_path)
    except StorageError:
        raise EncordToolError(
            "--input-path must be an s3:// prefix: Encord media is pushed from "
            "object storage, not local paths."
        ) from None
    from npa.clients.storage import StorageClient

    active_storage = storage_client or StorageClient.from_environment()
    endpoint_url = resolve_public_endpoint(environ) if transfer == "register" else ""

    discovered, skipped = discover_objects(active_storage, input_path, media)
    media_filter = cast(MediaFilter, media)  # validated by discover_objects
    items = [
        _pushed_item(obj, bucket=bucket, endpoint_url=endpoint_url, transfer=transfer)
        for obj in discovered
    ]
    registrable = [item for item in items if item.status == "registered"]

    client = user_client if user_client is not None else default_user_client(environ)
    integration_ref: ResolvedRef | None = None
    if transfer == "register":
        integration_ref = resolve_integration(client, integration)
    integration_id = integration_ref.id if integration_ref else ""
    integration_title = integration_ref.title if integration_ref else ""

    # Write-ahead receipt: land the plan before the first Encord mutation, so
    # even a crash mid-mutation leaves a durable record of what was attempted.
    receipt_uri = push_receipt_uri_for(output_path)
    encord_domain = resolve_domain(environ)
    planned = PushReceipt(
        generated_at=datetime.now(timezone.utc).isoformat(),
        workflow_run=workflow_run,
        input_uri=input_path,
        endpoint_url=endpoint_url,
        encord_domain=encord_domain,
        transfer=transfer_mode,
        integration_id=integration_id,
        integration_title=integration_title,
        folder_name=folder.strip(),
        media_filter=media_filter,
        status="planned",
        files_discovered=len(items),
        receipt_uri=receipt_uri,
        items=items,
        skipped_unsupported=skipped,
    )
    finalize_artifact(
        planned,
        result_uri=receipt_uri,
        filename=PUSH_RECEIPT_FILENAME,
        storage_client=active_storage,
        run_error=None,
        failure_prefix="Encord push failed",
    )
    # From this point on Encord may be mutated (folder/dataset creation). Every
    # such operation stays inside the receipt-finalization path below: a failed
    # dataset create must not strand a newly-created folder without lineage.
    folder_ref: ResolvedRef | None = None
    dataset_ref: ResolvedRef | None = None
    status: PushStatus = "failed"
    linked_count = 0
    run_error: Exception | None = None
    try:
        folder_ref = resolve_folder(client, folder)
        if dataset.strip():
            dataset_ref = resolve_dataset(client, dataset)
        # Caller-side idempotency: resolve exact identities BEFORE transferring
        # anything, so a retried stage re-sends only what does not already
        # exist. This is our invariant, not the SaaS's skip_duplicate_urls —
        # and it makes upload-mode re-pushes no-ops instead of duplicate
        # byte copies.
        _resolve_identities(folder_ref.obj, registrable, attempts=1)
        pending = [
            item
            for item in registrable
            if item.status == "registered" and not item.item_uuid
        ]
        if transfer == "upload":
            for item in registrable:
                if item.item_uuid:
                    item.status = "uploaded"
            _upload_items(
                folder_ref.obj,
                pending,
                storage_client=active_storage,
                source_bucket=bucket,
            )
            status = "done"
        elif pending:
            status = _register_items(
                folder_ref.obj,
                pending,
                integration_id=integration_id,
                poll_timeout_seconds=poll_timeout_seconds,
            )
            # Exact identity resolution (metadata/objectUrl only) attaches the
            # Encord uuid for the freshly registered items; an item Encord
            # accepted but exact identity cannot attribute fails closed.
            _resolve_identities(folder_ref.obj, pending, fail_unresolved=status == "done")
        else:
            status = "done"

        if dataset_ref is not None:
            linked_count = _link_dataset(dataset_ref.obj, items)
    except Exception as exc:  # noqa: BLE001 - recorded in the receipt, re-raised below
        run_error = exc
        status = "failed"

    # The receipt's counts are one function of the per-item outcomes, so an
    # item can never be counted twice (once by Encord's aggregate, once by
    # identity resolution) or be reported done without an attributable uuid.
    units_done = sum(
        1 for item in items if item.status in ("registered", "uploaded") and item.item_uuid
    )
    units_error = sum(1 for item in items if item.status in ("error", "experimental_error"))
    if status == "done" and units_error > 0:
        status = "failed"

    receipt = PushReceipt(
        generated_at=datetime.now(timezone.utc).isoformat(),
        workflow_run=workflow_run,
        input_uri=input_path,
        endpoint_url=endpoint_url,
        encord_domain=encord_domain,
        transfer=transfer_mode,
        integration_id=integration_id,
        integration_title=integration_title,
        folder_uuid=folder_ref.id if folder_ref else "",
        # Preserve the requested title when Encord failed before it could be
        # resolved, while recording the canonical title once it exists.
        folder_name=folder_ref.title if folder_ref else folder.strip(),
        folder_created=bool(folder_ref and folder_ref.created),
        dataset_hash=dataset_ref.id if dataset_ref else "",
        dataset_title=dataset_ref.title if dataset_ref else "",
        dataset_created=bool(dataset_ref and dataset_ref.created),
        linked_count=linked_count,
        media_filter=media_filter,
        status=status,
        files_discovered=len(items),
        units_done=units_done,
        units_error=units_error,
        error=error_text(run_error),
        receipt_uri=receipt_uri,
        items=items,
        skipped_unsupported=skipped,
    )
    finalize_artifact(
        receipt,
        result_uri=receipt_uri,
        filename=PUSH_RECEIPT_FILENAME,
        storage_client=active_storage,
        run_error=run_error,
        failure_prefix="Encord push failed",
    )
    if status != "done":
        raise EncordToolError(
            f"Encord push {status}: {units_error} unit error(s), {units_done} "
            f"registered. Receipt written to {receipt_uri}."
        )
    return receipt
