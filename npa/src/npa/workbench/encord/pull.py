"""Pull curated Encord data (media + item JSON + labels) back to S3.

For register-in-place data the common case is a zero-egress server-side copy:
each item's signed URL is parsed, and when it points back at the configured
endpoint the object is copied bucket-to-bucket instead of round-tripping bytes.
The manifest is always written before a failure exit.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, cast
from urllib.parse import unquote, urlparse

from npa.clients.storage import parse_bucket_uri
from npa.workbench.encord.client import (
    default_user_client,
    resolve_collection,
    resolve_dataset,
    resolve_domain,
    resolve_project,
    resolve_public_endpoint,
)
from npa.workbench.encord.identity import metadata_identity
from npa.workbench.encord.integrity import etag_checksum, write_hashed_stream
from npa.workbench.encord.schemas import (
    PULL_MANIFEST_FILENAME,
    PULL_SOURCES,
    EncordToolError,
    PulledItem,
    PullManifest,
    PullSourceKind,
)
from npa.workbench.encord.storage import error_text, finalize_artifact, write_json

LABEL_BUNDLE_SIZE = 100
# Media transfers are independent I/O; a bounded pool keeps thousands of
# curated clips from paying a serial round-trip each.
PULL_TRANSFER_WORKERS = 8

_LOG = logging.getLogger(__name__)


def pull_manifest_uri_for(output_path: str) -> str:
    """The exact manifest URI a given --output-path resolves to."""

    return output_path.rstrip("/") + f"/{PULL_MANIFEST_FILENAME}"


def _sanitize_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in name.strip())
    return cleaned or "item"


class PullSource(NamedTuple):
    source_id: str
    source_name: str
    items: list[Any]
    # Project pulls carry the resolved project and its (already listed) label
    # rows so export_labels never re-fetches the listing.
    project: Any = None
    label_rows: tuple[Any, ...] = ()


def _backing_item_uuid(row: Any) -> Any:
    # The SDK exposes this as a property that RAISES NotImplementedError on
    # legacy rows without Storage-API backing, so getattr's default alone
    # cannot guard it.
    try:
        return getattr(row, "backing_item_uuid", None)
    except NotImplementedError:
        return None


def enumerate_items(user_client: Any, *, source: str, source_id: str) -> PullSource:
    """Resolve the source and list its storage items, signed in bulk."""

    if source == "collection":
        collection = resolve_collection(user_client, source_id)
        uuids = [item.uuid for item in collection.obj.list_items(page_size=1000)]
        items = user_client.get_storage_items(uuids, sign_url=True) if uuids else []
        return PullSource(collection.id, collection.title, list(items))
    if source == "dataset":
        dataset = resolve_dataset(user_client, source_id, create=False)
        backing = [
            uuid for row in dataset.obj.data_rows if (uuid := _backing_item_uuid(row))
        ]
        items = user_client.get_storage_items(backing, sign_url=True) if backing else []
        return PullSource(dataset.id, dataset.title, list(items))
    if source == "project":
        project = resolve_project(user_client, source_id)
        label_rows = list(project.obj.list_label_rows_v2())
        backing = [
            uuid for row in label_rows if (uuid := _backing_item_uuid(row))
        ]
        items = user_client.get_storage_items(backing, sign_url=True) if backing else []
        return PullSource(
            project.id, project.title, list(items), project.obj, tuple(label_rows)
        )
    raise EncordToolError(
        f"Unknown --source value {source!r}. Choices: {', '.join(PULL_SOURCES)}."
    )


def _same_endpoint_source(signed_url: str, endpoint_url: str) -> tuple[str, str] | None:
    """(bucket, key) when the signed URL points at our own endpoint, else None."""

    signed = urlparse(signed_url)
    endpoint = urlparse(endpoint_url)
    if not signed.netloc or not endpoint.netloc:
        return None
    path = unquote(signed.path.lstrip("/"))
    if signed.netloc == endpoint.netloc:
        # Path-style: /<bucket>/<key>
        bucket, _, key = path.partition("/")
        return (bucket, key) if bucket and key else None
    if signed.netloc.endswith(f".{endpoint.netloc}"):
        # Virtual-hosted: <bucket>.<endpoint>/<key>
        bucket = signed.netloc[: -len(endpoint.netloc) - 1]
        return (bucket, path) if bucket and path else None
    return None


def transfer_item(
    item: Any,
    *,
    storage_client: Any,
    output_uri: str,
    endpoint_url: str,
) -> PulledItem:
    """Copy or download one storage item into the output prefix."""

    pulled = PulledItem(
        item_uuid=str(item.uuid),
        name=str(item.name),
        source_uri=metadata_identity(item),
        item_type=str(getattr(item, "item_type", "") or ""),
        mime_type=str(getattr(item, "mime_type", "") or ""),
        file_size=int(getattr(item, "file_size", 0) or 0),
    )
    dest_uri = (
        output_uri.rstrip("/")
        + f"/media/{pulled.item_uuid}__{_sanitize_name(pulled.name)}"
    )
    dest_bucket, dest_key = parse_bucket_uri(dest_uri)

    try:
        # The SDK refetches the item when its cached URL is missing: a real
        # network call that must fail this item, not the whole pull.
        signed_url = item.get_signed_url()
    except Exception as exc:  # noqa: BLE001 - recorded per item, run fails closed
        pulled.transfer = "error"
        pulled.error = f"could not fetch a signed URL: {exc}"
        return pulled
    if not signed_url:
        pulled.transfer = "error"
        pulled.error = "item has no signed URL (composite items are not supported)"
        return pulled

    source = _same_endpoint_source(signed_url, endpoint_url)
    if source is not None:
        try:
            response = storage_client.s3.copy_object(
                Bucket=dest_bucket,
                Key=dest_key,
                CopySource={"Bucket": source[0], "Key": source[1]},
            )
            pulled.media_uri = dest_uri
            pulled.transfer = "copy"
            # A server-side copy exposes no bytes to hash; the destination
            # ETag (md5 when single-part) is the content evidence.
            etag = str(((response or {}).get("CopyObjectResult") or {}).get("ETag") or "")
            pulled.checksum, pulled.checksum_kind = etag_checksum(etag)
            return pulled
        except Exception as exc:  # noqa: BLE001 - fall back to the download path
            # The manifest carries the reason: an operator must be able to see
            # why a same-bucket item paid for egress instead of a copy.
            pulled.copy_error = f"{type(exc).__name__}: {exc}"
            _LOG.warning(
                "server-side copy of %s failed (%s); falling back to download",
                pulled.name,
                exc,
            )

    import httpx

    def _download(url: str) -> None:
        with tempfile.TemporaryDirectory(prefix="npa-encord-pull-") as tmp:
            local = Path(tmp) / _sanitize_name(pulled.name)
            with httpx.stream("GET", url, timeout=600.0, follow_redirects=True) as response:
                response.raise_for_status()
                digest = write_hashed_stream(response.iter_bytes(8 * 1024 * 1024), local)
            storage_client.upload_file(str(local), dest_uri)
            pulled.observed_size = digest.size
            pulled.checksum = digest.sha256
            pulled.checksum_kind = "sha256"

    try:
        try:
            _download(signed_url)
        except httpx.HTTPStatusError:
            # The signed URL may simply have expired mid-run; refetch once.
            _download(item.get_signed_url(refetch=True) or signed_url)
        pulled.media_uri = dest_uri
        pulled.transfer = "download"
    except Exception as exc:  # noqa: BLE001 - recorded per item, run fails closed
        pulled.transfer = "error"
        pulled.error = str(exc)
    return pulled


def export_labels(
    project: Any,
    label_rows: list[Any],
    *,
    output_uri: str,
    storage_client: Any,
) -> tuple[int, list[str]]:
    """Export every label row as Encord JSON under labels/."""

    if not label_rows:
        return 0, []
    with project.create_bundle(bundle_size=LABEL_BUNDLE_SIZE) as bundle:
        for row in label_rows:
            row.initialise_labels(bundle=bundle)
    label_uris: list[str] = []
    for row in label_rows:
        name = str(row.label_hash or row.data_hash or len(label_uris))
        label_uri = output_uri.rstrip("/") + f"/labels/{name}.json"
        write_json(
            row.to_encord_dict(),
            result_uri=label_uri,
            filename=f"{name}.json",
            storage_client=storage_client,
        )
        label_uris.append(label_uri)
    return len(label_rows), label_uris


def run_pull(
    *,
    source: str,
    source_id: str,
    output_path: str,
    workflow_run: str = "",
    user_client: Any = None,
    storage_client: Any = None,
    environ: dict[str, str] | None = None,
) -> PullManifest:
    """Materialize a curated Encord source into the S3 output prefix."""

    if not output_path.startswith("s3://"):
        raise EncordToolError("--output-path must be an s3:// prefix.")
    from npa.clients.storage import StorageClient

    active_storage = storage_client or StorageClient.from_environment()
    endpoint_url = resolve_public_endpoint(environ)
    client = user_client if user_client is not None else default_user_client(environ)

    found = enumerate_items(client, source=source, source_id=source_id)

    # Everything past this point mutates the output prefix; the manifest must
    # land even when a step throws, so lineage survives a mid-run crash.
    pulled: list[PulledItem] = []
    label_rows = 0
    label_uris: list[str] = []
    run_error: Exception | None = None
    try:

        def _transfer_and_record(item: Any) -> PulledItem:
            try:
                record = transfer_item(
                    item,
                    storage_client=active_storage,
                    output_uri=output_path,
                    endpoint_url=endpoint_url,
                )
            except Exception as exc:  # noqa: BLE001 - one item must not erase the others
                # Anything transfer_item did not classify itself still becomes
                # an error row, so the manifest keeps every item that landed.
                return PulledItem(
                    item_uuid=str(getattr(item, "uuid", "") or ""),
                    name=str(getattr(item, "name", "") or ""),
                    transfer="error",
                    error=error_text(exc),
                )
            try:
                write_json(
                    {
                        "item_uuid": record.item_uuid,
                        "name": record.name,
                        "item_type": record.item_type,
                        "mime_type": record.mime_type,
                        "file_size": record.file_size,
                        "client_metadata": getattr(item, "client_metadata", None) or {},
                    },
                    result_uri=output_path.rstrip("/") + f"/items/{record.item_uuid}.json",
                    filename=f"{record.item_uuid}.json",
                    storage_client=active_storage,
                )
            except Exception as exc:  # noqa: BLE001 - recorded per item, run fails closed
                record.transfer = "error"
                record.error = f"item metadata write failed: {exc}"
            return record

        if found.items:
            from concurrent.futures import ThreadPoolExecutor

            workers = min(PULL_TRANSFER_WORKERS, len(found.items))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pulled = list(pool.map(_transfer_and_record, found.items))

        if found.project is not None:
            label_rows, label_uris = export_labels(
                found.project,
                list(found.label_rows),
                output_uri=output_path,
                storage_client=active_storage,
            )
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest, re-raised below
        run_error = exc

    manifest_uri = pull_manifest_uri_for(output_path)
    manifest = PullManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        workflow_run=workflow_run,
        encord_domain=resolve_domain(environ),
        source_kind=cast(PullSourceKind, source),  # validated by enumerate_items
        source_id=found.source_id,
        source_name=found.source_name,
        output_uri=output_path,
        manifest_uri=manifest_uri,
        items_total=len(pulled),
        media_copied=sum(1 for record in pulled if record.transfer == "copy"),
        media_downloaded=sum(1 for record in pulled if record.transfer == "download"),
        media_failed=sum(1 for record in pulled if record.transfer == "error"),
        label_rows=label_rows,
        media_bytes=sum(
            record.file_size
            for record in pulled
            if record.transfer in ("copy", "download")
        ),
        error=error_text(run_error),
        label_uris=label_uris,
        items=pulled,
    )
    finalize_artifact(
        manifest,
        result_uri=manifest_uri,
        filename=PULL_MANIFEST_FILENAME,
        storage_client=active_storage,
        run_error=run_error,
        failure_prefix=(
            f"Encord pull failed after {len(pulled)} of {len(found.items)} item(s)"
        ),
        artifact_noun="Manifest",
    )
    if manifest.media_failed > 0:
        raise EncordToolError(
            f"Encord pull failed for {manifest.media_failed} of "
            f"{manifest.items_total} item(s). Manifest written to {manifest_uri}."
        )
    if manifest.items_total == 0:
        raise EncordToolError(
            f"Encord {source} {source_id!r} contains no storage items; nothing "
            f"was pulled. Manifest written to {manifest_uri}."
        )
    return manifest
