"""Durable Encord media pull with explicit label-initialization posture."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlparse

from npa.workbench.encord.client import (
    _default_user_client,
    find_dataset,
    resolve_collection,
    resolve_domain,
    resolve_project,
    resolve_public_endpoint,
)
from npa.workbench.encord.identity import canonical_s3_uri, metadata_identity
from npa.workbench.encord.integrity import HttpDownloader, HttpxDownloader, retry_signed_download
from npa.workbench.encord.schemas import (
    PULL_MANIFEST_FILENAME,
    EncordToolError,
    LabelArtifact,
    OutcomeCounts,
    PullItem,
    PullManifest,
    RunStatus,
)
from npa.workbench.encord.storage import (
    ArtifactStore,
    ConditionalArtifactStore,
    ObjectStorageGateway,
    S3ObjectStorageGateway,
    TransferDigest,
    write_json_object,
)

PULL_SOURCES = ("collection", "dataset", "project")
LABEL_EXPORT_MODES = ("none", "initialize")


class _CheckpointFailure(EncordToolError):
    pass


def pull_manifest_uri_for(output_path: str) -> str:
    return output_path.rstrip("/") + f"/{PULL_MANIFEST_FILENAME}"


@dataclass
class PullSource:
    source_id: str
    source_name: str
    items: Iterable[Any]
    project: Any = None
    hydrate_items: bool = False
    labels_follow_items: bool = False


def enumerate_items(user_client: Any, *, source: str, source_id: str) -> PullSource:
    if source == "collection":
        collection, resolved_id, name = resolve_collection(user_client, source_id)

        def collection_rows() -> Iterable[Any]:
            try:
                rows = collection.list_items(
                    include_client_metadata=True, page_size=1000
                )
            except TypeError:
                rows = collection.list_items(page_size=1000)
            yield from rows

        return PullSource(resolved_id, name, collection_rows())
    if source == "dataset":
        found = find_dataset(user_client, source_id)
        if found is None:
            raise EncordToolError(f"dataset {source_id!r} was not found")
        _, dataset_hash, title = found
        try:
            dataset = user_client.get_dataset(dataset_hash, include_data_rows=True)
        except TypeError:
            dataset = user_client.get_dataset(dataset_hash)

        def dataset_rows() -> Iterable[Any]:
            rows = (
                dataset.list_data_rows()
                if hasattr(dataset, "list_data_rows")
                else (getattr(dataset, "data_rows", None) or [])
            )
            yield from rows

        return PullSource(
            dataset_hash, title, dataset_rows(), hydrate_items=True
        )
    if source == "project":
        project, project_hash, title = resolve_project(user_client, source_id)

        def project_rows() -> Iterable[Any]:
            yield from project.list_label_rows_v2(include_client_metadata=True)

        return PullSource(
            project_hash,
            title,
            project_rows(),
            project,
            hydrate_items=True,
            labels_follow_items=True,
        )
    raise EncordToolError(f"unknown pull source {source!r}")


def run_pull(
    *,
    source: str,
    source_id: str,
    output_path: str,
    workflow_run: str = "",
    label_export: str = "none",
    user_client: Any = None,
    storage_client: Any = None,
    artifact_store: ArtifactStore | None = None,
    downloader: HttpDownloader | None = None,
    clock: Callable[[], str] | None = None,
    environ: dict[str, str] | None = None,
) -> PullManifest:
    """Pull media; label initialization is an explicit remote mutation choice."""

    if source not in PULL_SOURCES:
        raise EncordToolError(f"unknown pull source {source!r}")
    if label_export not in LABEL_EXPORT_MODES:
        raise EncordToolError(f"unknown label export mode {label_export!r}")
    if label_export == "initialize" and source != "project":
        raise EncordToolError("label initialization is available only for project pulls")
    if not output_path.startswith("s3://"):
        raise EncordToolError("output_path must be an s3:// prefix")
    from npa.clients.storage import StorageClient

    active_storage = storage_client or StorageClient.from_environment()
    active_artifacts = artifact_store or ConditionalArtifactStore(active_storage)
    object_store = S3ObjectStorageGateway(active_storage)
    active_downloader = downloader or HttpxDownloader()
    client = user_client if user_client is not None else _default_user_client(environ)
    now = clock or _utc_now
    encord_domain = resolve_domain(environ)
    try:
        endpoint_url = resolve_public_endpoint(environ)
    except EncordToolError:
        endpoint_url = ""

    found = enumerate_items(client, source=source, source_id=source_id)
    descriptors: list[Any] = []
    label_descriptors: list[Any] = []
    items: list[PullItem] = []
    label_artifacts: list[LabelArtifact] = []
    manifest_uri = pull_manifest_uri_for(output_path)
    generated_at = now()
    revision = 0

    def make_manifest(
        phase: str,
        status: RunStatus,
        *,
        error_code: str = "",
        error: str = "",
    ) -> PullManifest:
        return PullManifest(
            phase=phase,  # type: ignore[arg-type]
            status=status,
            revision=revision,
            generated_at=generated_at,
            updated_at=now(),
            workflow_run=workflow_run,
            encord_domain=encord_domain,
            source_kind=source,  # type: ignore[arg-type]
            source_id=found.source_id,
            source_name=found.source_name,
            label_export=label_export,  # type: ignore[arg-type]
            label_export_remote_mutation=label_export == "initialize",
            label_export_posture=(
                "Encord label initialization was explicitly enabled and may create "
                "or change remote label-row state."
                if label_export == "initialize"
                else "Labels were not initialized or exported."
            ),
            output_uri=output_path,
            manifest_uri=manifest_uri,
            counts=OutcomeCounts.from_outcomes([item.outcome for item in items]),
            label_counts=OutcomeCounts.from_outcomes(
                [item.outcome for item in label_artifacts]
            ),
            media_copied=sum(
                item.outcome == "successful" and item.transfer == "copy" for item in items
            ),
            media_downloaded=sum(
                item.outcome == "successful" and item.transfer == "download"
                for item in items
            ),
            media_bytes=sum(
                item.destination_size for item in items if item.outcome == "successful"
            ),
            error_code=error_code,
            error=str(error).strip()[:1000],
            items=items,
            label_artifacts=label_artifacts,
        )

    try:
        version = active_artifacts.create_json(
            manifest_uri,
            make_manifest("provisional", "running").model_dump(by_alias=True),
        )
    except Exception as exc:  # noqa: BLE001
        raise EncordToolError(
            f"could not create provisional pull manifest before writes: {exc}"
        ) from exc

    def checkpoint() -> None:
        nonlocal revision, version
        revision += 1
        try:
            version = active_artifacts.replace_json(
                manifest_uri,
                make_manifest("checkpoint", "running").model_dump(by_alias=True),
                version,
            )
        except Exception as exc:  # noqa: BLE001
            raise _CheckpointFailure(
                "pull checkpoint failed after external mutation; no further writes "
                f"were attempted: {exc}"
            ) from exc

    def fail_with_known_rows(error_code: str, exc: Exception) -> None:
        nonlocal revision, version
        failure_label = error_code.replace("_", " ")
        revision += 1
        failed = make_manifest(
            "final",
            "failed",
            error_code=error_code,
            error=f"{type(exc).__name__}: {exc}",
        )
        try:
            version = active_artifacts.replace_json(
                manifest_uri, failed.model_dump(by_alias=True), version
            )
        except Exception as write_exc:  # noqa: BLE001
            raise EncordToolError(
                f"pull {failure_label} and its final manifest could not be persisted: "
                f"{write_exc}"
            ) from write_exc
        raise EncordToolError(
            f"pull {failure_label}; manifest written to {manifest_uri}"
        ) from exc

    try:
        for descriptor in found.items:
            descriptors.append(descriptor)
            items.append(_preallocate_media(descriptor, output_path))
            if found.labels_follow_items and label_export == "initialize":
                label_descriptors.append(descriptor)
                label_artifacts.append(_preallocate_label(descriptor))
            checkpoint()
    except _CheckpointFailure:
        raise
    except Exception as exc:  # noqa: BLE001 - preserve incremental enumeration evidence
        fail_with_known_rows("enumeration_failed", exc)

    if found.hydrate_items:
        try:
            descriptors = _hydrate_rows(client, descriptors)
            items[:] = [
                _preallocate_media(descriptor, output_path)
                for descriptor in descriptors
            ]
            checkpoint()
        except _CheckpointFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - known rows remain durable
            fail_with_known_rows("source_hydration_failed", exc)

    for descriptor, row in zip(descriptors, items):
        if not row.item_uuid:
            row.outcome = "unresolved"
            row.error_code = "missing_backing_item_uuid"
            row.error = "source row has no recoverable Encord storage-item UUID"
            checkpoint()
            continue
        if not row.source_uri:
            row.outcome = "unresolved"
            row.error_code = "missing_source_identity"
            row.error = "source row has no stable source URI"
            checkpoint()
            continue
        _transfer_one(
            descriptor,
            row,
            storage_client=active_storage,
            object_store=object_store,
            downloader=active_downloader,
            endpoint_url=endpoint_url,
        )
        checkpoint()
        if row.outcome == "failed":
            break

    if label_export == "initialize":
        for index, (row, artifact) in enumerate(
            zip(label_descriptors, label_artifacts)
        ):
            try:
                row.initialise_labels()
                label_name = artifact.label_hash or artifact.data_hash or artifact.item_uuid
                artifact.artifact_uri = (
                    output_path.rstrip("/") + f"/labels/{_safe_name(label_name)}.json"
                )
                write_json_object(
                    active_storage,
                    artifact.artifact_uri,
                    row.to_encord_dict(),
                    filename=f"{_safe_name(label_name)}.json",
                )
                artifact.outcome = "successful"
            except Exception as exc:  # noqa: BLE001
                artifact.outcome = "failed"
                artifact.error_code = "label_initialization_or_export_failed"
                artifact.error = str(exc).strip()[:1000]
                checkpoint()
                for pending in label_artifacts[index + 1 :]:
                    pending.outcome = "unattempted"
                break
            checkpoint()

    status = _pull_status(items, label_artifacts if label_export == "initialize" else [])
    revision += 1
    final = make_manifest("final", status)
    try:
        active_artifacts.replace_json(
            manifest_uri, final.model_dump(by_alias=True), version
        )
    except Exception as exc:  # noqa: BLE001
        raise EncordToolError(
            "pull final manifest write failed after mutation; durable completion cannot "
            f"be claimed: {exc}"
        ) from exc
    if status != "completed":
        raise EncordToolError(f"Encord pull {status}; manifest written to {manifest_uri}")
    return final


def _hydrate_rows(user_client: Any, rows: list[Any]) -> list[Any]:
    uuids: list[str] = []
    by_uuid: dict[str, Any] = {}
    for row in rows:
        item_uuid = _backing_item_uuid(row)
        if item_uuid:
            value = str(item_uuid)
            uuids.append(value)
            by_uuid[value] = row
    hydrated: dict[str, Any] = {}
    if uuids:
        for item in user_client.get_storage_items(uuids, sign_url=True):
            hydrated[str(getattr(item, "uuid", ""))] = item
    return [hydrated.get(str(_backing_item_uuid(row)), row) for row in rows]


def _backing_item_uuid(row: Any) -> Any:
    if hasattr(row, "uuid"):
        return getattr(row, "uuid")
    try:
        return getattr(row, "backing_item_uuid", None)
    except NotImplementedError:
        return None


def _preallocate_media(item: Any, output_uri: str) -> PullItem:
    item_uuid = str(_backing_item_uuid(item) or "")
    source_uri, record_id = metadata_identity(item)
    name = str(
        getattr(item, "name", "")
        or getattr(item, "data_title", "")
        or getattr(item, "title", "")
        or item_uuid
    )
    destination = (
        output_uri.rstrip("/")
        + f"/media/{item_uuid or 'unresolved'}__{_safe_name(name)}"
    )
    return PullItem(
        item_uuid=item_uuid,
        record_id=record_id,
        source_uri=source_uri,
        name=name,
        item_type=str(getattr(item, "item_type", "") or getattr(item, "data_type", "") or ""),
        mime_type=str(getattr(item, "mime_type", "") or getattr(item, "file_type", "") or ""),
        source_size=int(getattr(item, "file_size", 0) or 0),
        destination_uri=destination,
        metadata_uri=(
            output_uri.rstrip("/") + f"/items/{item_uuid}.json" if item_uuid else ""
        ),
    )


def _preallocate_label(row: Any) -> LabelArtifact:
    return LabelArtifact(
        label_hash=str(getattr(row, "label_hash", "") or ""),
        data_hash=str(getattr(row, "data_hash", "") or ""),
        item_uuid=str(_backing_item_uuid(row) or ""),
    )


def _transfer_one(
    item: Any,
    row: PullItem,
    *,
    storage_client: Any,
    downloader: HttpDownloader,
    endpoint_url: str,
    object_store: ObjectStorageGateway | None = None,
) -> None:
    gateway = object_store or S3ObjectStorageGateway(storage_client)
    signed_url = item.get_signed_url() if hasattr(item, "get_signed_url") else getattr(item, "signed_url", "")
    if not signed_url:
        row.outcome = "failed"
        row.error_code = "signed_url_unavailable"
        row.error = "storage item has no transferable signed URL"
        return
    copied = False
    try:
        source = (
            _same_endpoint_source(str(signed_url), endpoint_url)
            if endpoint_url
            else None
        )
    except EncordToolError as exc:
        row.outcome = "failed"
        row.error_code = "signed_url_identity_invalid"
        row.error = str(exc).strip()[:1000]
        return
    if source is not None:
        row.copy_attempted = True
        try:
            source_uri = canonical_s3_uri(source[0], source[1])
        except EncordToolError as exc:
            row.copy_failure = str(exc).strip()[:1000]
            row.outcome = "failed"
            row.error_code = "signed_url_identity_invalid"
            row.error = row.copy_failure
            return
        if source_uri != row.source_uri:
            row.copy_failure = (
                "signed URL source identity conflicts with the pull row source URI"
            )
            row.outcome = "failed"
            row.error_code = "signed_url_identity_conflict"
            row.error = row.copy_failure
            return
        try:
            source_meta = gateway.head(source_uri)
            destination = gateway.copy(source_uri, row.destination_uri)
            row.transfer = "copy"
            row.destination_exists = True
            row.destination_size = destination.size
            row.source_size = source_meta.size
            row.source_checksum = source_meta.checksum
            row.source_checksum_kind = source_meta.checksum_kind
            row.destination_checksum = destination.checksum
            row.destination_checksum_kind = destination.checksum_kind
            copied = True
        except Exception as exc:  # noqa: BLE001 - download fallback is deliberate
            row.copy_failure = str(exc).strip()[:1000]
    if not copied:
        try:
            with tempfile.TemporaryDirectory(prefix="npa-encord-pull-") as temporary:
                local = Path(temporary) / "media.bin"
                digest = retry_signed_download(
                    downloader,
                    str(signed_url),
                    lambda: str(item.get_signed_url(refetch=True) or ""),
                    local,
                )
                destination = gateway.upload_file(
                    local,
                    row.destination_uri,
                    digest=TransferDigest(size=digest.size, sha256=digest.sha256),
                )
            if destination.size != digest.size:
                raise EncordToolError("download destination size did not verify")
            row.transfer = "download"
            row.destination_exists = True
            row.destination_size = destination.size
            row.source_size = row.source_size or digest.size
            row.source_checksum = digest.sha256
            row.source_checksum_kind = "sha256"
            row.destination_checksum = destination.checksum
            row.destination_checksum_kind = destination.checksum_kind
        except Exception as exc:  # noqa: BLE001
            row.outcome = "failed"
            row.error_code = "media_transfer_failed"
            row.error = str(exc).strip()[:1000]
            return
    try:
        write_json_object(
            storage_client,
            row.metadata_uri,
            {
                "item_uuid": row.item_uuid,
                "source_uri": row.source_uri,
                "record_id": row.record_id,
                "name": row.name,
                "item_type": row.item_type,
                "mime_type": row.mime_type,
                "source_size": row.source_size,
                "destination_uri": row.destination_uri,
            },
            filename=f"{row.item_uuid}.json",
        )
        row.metadata_state = "written"
        row.outcome = "successful"
    except Exception as exc:  # noqa: BLE001
        row.metadata_state = "failed"
        row.outcome = "failed"
        row.error_code = "metadata_write_failed"
        row.error = str(exc).strip()[:1000]


def _same_endpoint_source(signed_url: str, endpoint_url: str) -> tuple[str, str] | None:
    signed = urlparse(signed_url)
    endpoint = urlparse(endpoint_url)
    signed_host = (signed.hostname or "").lower()
    endpoint_host = (endpoint.hostname or "").lower()
    if not signed_host or not endpoint_host or signed.port != endpoint.port:
        return None
    path = _decode_signed_path_once(signed.path.lstrip("/"))
    if signed_host == endpoint_host:
        bucket, separator, key = path.partition("/")
        return (bucket, key) if separator and bucket and key else None
    suffix = f".{endpoint_host}"
    if signed_host.endswith(suffix):
        bucket = signed_host[: -len(suffix)]
        return (bucket, path) if bucket and path else None
    return None


def _decode_signed_path_once(path: str) -> str:
    hexadecimal = frozenset("0123456789abcdefABCDEF")
    for index, character in enumerate(path):
        if character != "%":
            continue
        if index + 2 >= len(path) or not {
            path[index + 1],
            path[index + 2],
        } <= hexadecimal:
            raise EncordToolError("signed URL contains an invalid percent escape")
    try:
        return unquote(path, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EncordToolError("signed URL path is not valid UTF-8") from exc


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned or "item"


def _pull_status(items: list[PullItem], labels: list[LabelArtifact]) -> RunStatus:
    outcomes = [item.outcome for item in items] + [item.outcome for item in labels]
    if outcomes and all(outcome == "successful" for outcome in outcomes):
        return "completed"
    if any(outcome in {"successful", "unresolved"} for outcome in outcomes):
        return "partial"
    return "failed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
