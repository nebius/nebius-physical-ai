"""Fail-closed register-in-place and explicit-upload transport into Encord."""

from __future__ import annotations

import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote, urlsplit, urlunsplit

from npa.workbench.encord.client import (
    _default_user_client,
    create_dataset,
    create_folder,
    find_dataset,
    find_folder,
    resolve_domain,
    resolve_integration,
    resolve_public_endpoint,
)
from npa.workbench.encord.identity import (
    IdentityResolution,
    canonical_s3_uri,
    resolve_exact_identity,
)
from npa.workbench.encord.schemas import (
    DEFAULT_MEDIA_FILTER,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    DEFAULT_TRANSFER,
    PUSH_RECEIPT_FILENAME,
    EncordToolError,
    IdentitySidecar,
    IdentitySidecarRow,
    OutcomeCounts,
    PushItem,
    PushReceipt,
    RunStatus,
)
from npa.workbench.encord.storage import (
    ArtifactStore,
    ArtifactVersion,
    ConditionalArtifactStore,
    ObjectStorageGateway,
    S3ObjectStorageGateway,
)
from npa.workbench.storage_scope import StorageAuthorizationError, authorize_uri

MEDIA_CATEGORIES = {
    ".mp4": "videos",
    ".png": "images",
    ".jpg": "images",
    ".jpeg": "images",
}
FILTER_CATEGORIES = {
    "videos-images": MEDIA_CATEGORIES,
    "mcap": {".mcap": "mcap"},
    "all": {**MEDIA_CATEGORIES, ".mcap": "mcap"},
}
TRANSFER_MODES = ("register", "upload")
BATCH_SIZE = 500


def push_receipt_uri_for(output_path: str) -> str:
    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{PUSH_RECEIPT_FILENAME}"


def object_url_for(endpoint_url: str, bucket: str, key: str) -> str:
    return f"{endpoint_url.rstrip('/')}/{bucket}/{quote(key, safe='/')}"


def discover_objects(
    object_store: ObjectStorageGateway, input_uri: str, media: str
) -> tuple[list[PushItem], list[str]]:
    allowed = FILTER_CATEGORIES.get(media)
    if allowed is None:
        raise EncordToolError(f"unknown media filter {media!r}")
    items: list[PushItem] = []
    skipped: list[str] = []
    for metadata in object_store.list_objects(input_uri):
        target = authorize_uri(metadata.uri, operation="identify discovered object")
        if target.kind != "s3" or not target.bucket or not target.key:
            raise EncordToolError("discovered object did not have an exact S3 identity")
        bucket, key = target.bucket, target.key
        category = allowed.get(Path(key).suffix.lower())
        if category is None:
            skipped.append(key)
            continue
        items.append(
            PushItem(
                source_uri=canonical_s3_uri(bucket, key),
                bucket=bucket,
                object_key=key,
                category=category,
                source_size=metadata.size,
                source_etag=metadata.etag,
                source_etag_kind=metadata.etag_kind,
                source_checksum=metadata.checksum,
                source_checksum_kind=metadata.checksum_kind,
            )
        )
    if not items:
        raise EncordToolError(f"no supported media found under {input_uri}")
    return items, skipped


def build_upload_json(items: Iterable[PushItem]) -> dict[str, Any]:
    payload: dict[str, Any] = {"skip_duplicate_urls": True}
    for item in items:
        entry: dict[str, Any] = {
            "objectUrl": item.submitted_object_url,
            "clientMetadata": {
                "npa": {
                    "source_uri": item.source_uri,
                    "record_id": item.record_id,
                }
            },
        }
        payload.setdefault(item.category, []).append(entry)
    return payload


def build_registration_payload(items: Iterable[PushItem]) -> Any:
    rows = list(items)
    try:
        from encord.orm.storage import DataUploadImage, DataUploadItems, DataUploadVideo
    except ModuleNotFoundError:
        return build_upload_json(rows)
    images = []
    videos = []
    for item in rows:
        metadata = {
            "npa": {"source_uri": item.source_uri, "record_id": item.record_id}
        }
        if item.category == "images":
            images.append(
                DataUploadImage(
                    objectUrl=item.submitted_object_url,
                    title=item.object_key,
                    clientMetadata=metadata,
                )
            )
        elif item.category == "videos":
            videos.append(
                DataUploadVideo(
                    objectUrl=item.submitted_object_url,
                    title=item.object_key,
                    clientMetadata=metadata,
                )
            )
    return DataUploadItems(
        images=images,
        videos=videos,
        skipDuplicateUrls=True,
        upsertMetadata=False,
    )


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
    identity_sidecar_uri: str = "",
    user_client: Any = None,
    storage_client: Any = None,
    artifact_store: ArtifactStore | None = None,
    clock: Callable[[], str] | None = None,
    environ: dict[str, str] | None = None,
) -> PushReceipt:
    """Push media and return only after a validated final receipt is durable."""

    if transfer not in TRANSFER_MODES:
        raise EncordToolError(f"unknown transfer mode {transfer!r}")
    try:
        target = authorize_uri(input_path.rstrip("/"), operation="discover Encord media")
        if target.kind != "s3" or not target.bucket:
            raise StorageAuthorizationError("input is not an S3 prefix")
    except StorageAuthorizationError:
        raise EncordToolError("input_path must be an s3:// prefix") from None
    from npa.clients.storage import StorageClient

    active_storage = storage_client or StorageClient.from_environment()
    active_artifacts = artifact_store or ConditionalArtifactStore(active_storage)
    object_store = S3ObjectStorageGateway(active_storage)
    now = clock or _utc_now
    endpoint_url = resolve_public_endpoint(environ) if transfer == "register" else ""
    encord_domain = resolve_domain(environ)

    items, skipped = discover_objects(object_store, input_path, media)
    for item in items:
        item.transfer_mode = transfer  # type: ignore[assignment]
        item.link_state = "unattempted" if dataset.strip() else "not_requested"
        item.registration_state = "unattempted" if transfer == "register" else "not_applicable"
        if item.category == "mcap":
            item.registration_state = "failed" if transfer == "register" else "not_applicable"
            item.outcome = "failed"
            item.error_code = "unsupported_media"
            item.error = "MCAP requires a structured Encord scene definition"
        if transfer == "register":
            item.submitted_object_url = object_url_for(
                endpoint_url, item.bucket, item.object_key
            )

    sidecars = _load_sidecar(active_artifacts, identity_sidecar_uri, items)
    client = user_client if user_client is not None else _default_user_client(environ)
    integration_id = integration_title = ""
    if transfer == "register":
        integration_id, integration_title = resolve_integration(client, integration)

    # All discovery and resolution through this point is read-only.
    folder_obj = find_folder(client, folder)
    found_dataset = find_dataset(client, dataset) if dataset.strip() else None
    dataset_obj = found_dataset[0] if found_dataset else None
    dataset_hash = found_dataset[1] if found_dataset else ""
    dataset_title = found_dataset[2] if found_dataset else dataset.strip()
    folder_created = False
    dataset_created = False
    receipt_uri = push_receipt_uri_for(output_path)
    generated_at = now()
    revision = 0

    def make_receipt(
        phase: str,
        status: RunStatus,
        *,
        error_code: str = "",
        error: str = "",
    ) -> PushReceipt:
        return PushReceipt(
            phase=phase,  # type: ignore[arg-type]
            status=status,
            revision=revision,
            generated_at=generated_at,
            updated_at=now(),
            workflow_run=workflow_run,
            input_uri=input_path,
            endpoint_url=endpoint_url,
            encord_domain=encord_domain,
            transfer_mode=transfer,  # type: ignore[arg-type]
            idempotency="exact_identity" if transfer == "register" else "not_guaranteed",
            integration_id=integration_id,
            integration_title=integration_title,
            folder_uuid=str(getattr(folder_obj, "uuid", "") or ""),
            folder_name=str(getattr(folder_obj, "name", "") or folder.strip()),
            folder_created=folder_created,
            dataset_hash=dataset_hash,
            dataset_title=dataset_title,
            dataset_created=dataset_created,
            dataset_requested=bool(dataset.strip()),
            linked_count=sum(
                item.link_state in {"linked", "already_linked"} for item in items
            ),
            media_filter=media,
            counts=OutcomeCounts.from_outcomes([item.outcome for item in items]),
            receipt_uri=receipt_uri,
            receipt_store_kind="s3" if receipt_uri.startswith("s3://") else "local",
            identity_sidecar_uri=identity_sidecar_uri,
            error_code=error_code,
            error=_sanitize_error(error),
            skipped_unsupported=skipped,
            items=items,
        )

    # Creating this object is the output preflight. No Encord mutation precedes it.
    try:
        version = active_artifacts.create_json(
            receipt_uri, make_receipt("provisional", "running").model_dump(by_alias=True)
        )
    except Exception as exc:  # noqa: BLE001 - store implementations vary
        raise EncordToolError(
            f"could not create provisional push receipt before mutation: {_sanitize_error(str(exc))}"
        ) from exc

    def checkpoint() -> None:
        nonlocal revision, version
        revision += 1
        try:
            version = active_artifacts.replace_json(
                receipt_uri,
                make_receipt("checkpoint", "running").model_dump(by_alias=True),
                version,
            )
        except Exception as exc:  # noqa: BLE001 - durability boundary
            raise EncordToolError(
                "push checkpoint failed after external mutation; no further mutation "
                f"was attempted: {_sanitize_error(str(exc))}"
            ) from exc

    if folder_obj is None:
        try:
            folder_obj = create_folder(client, folder)
            folder_created = True
        except Exception as exc:  # noqa: BLE001 - SDK exceptions vary
            return _finalize_push_failure(
                active_artifacts,
                version,
                receipt_uri,
                make_receipt,
                revision,
                "folder_create_failed",
                exc,
            )
        checkpoint()

    if dataset.strip() and dataset_obj is None:
        try:
            dataset_hash, dataset_title = create_dataset(client, dataset)
            dataset_created = True
        except Exception as exc:  # noqa: BLE001
            return _finalize_push_failure(
                active_artifacts,
                version,
                receipt_uri,
                make_receipt,
                revision,
                "dataset_create_failed",
                exc,
            )
        checkpoint()
        try:
            dataset_obj = client.get_dataset(dataset_hash)
        except Exception as exc:  # noqa: BLE001 - mutation is already checkpointed
            return _finalize_push_failure(
                active_artifacts,
                version,
                receipt_uri,
                make_receipt,
                revision,
                "dataset_hydration_failed",
                exc,
            )

    if transfer == "register":
        _run_register(
            items,
            folder_obj,
            client,
            integration_id=integration_id,
            poll_timeout_seconds=poll_timeout_seconds,
            sidecars=sidecars,
            checkpoint=checkpoint,
        )
    else:
        _run_upload(
            items,
            folder_obj,
            object_store=object_store,
            checkpoint=checkpoint,
        )

    _fail_duplicate_resolutions(items)
    if dataset_obj is not None:
        linkable = [
            item for item in items if item.identity_state == "resolved" and item.item_uuid
        ]
        if linkable:
            before = _dataset_membership(client, dataset_obj, dataset_hash)
            missing = [item for item in linkable if item.item_uuid not in before]
            for item in linkable:
                if item.item_uuid in before:
                    item.link_state = "already_linked"
            try:
                if missing:
                    dataset_obj.link_items(sorted(item.item_uuid for item in missing))
                    after = _dataset_membership(client, dataset_obj, dataset_hash)
                    for item in missing:
                        if item.item_uuid in after:
                            item.link_state = "linked"
                            item.outcome = "successful"
                        else:
                            item.link_state = "failed"
                            item.outcome = "unresolved"
                            item.error_code = "dataset_link_unverified"
                            item.error = "Encord did not expose exact post-link membership"
            except Exception as exc:  # noqa: BLE001
                for item in missing:
                    item.link_state = "failed"
                    item.outcome = "failed"
                    item.error_code = "dataset_link_failed"
                    item.error = _sanitize_error(str(exc))
            if missing:
                checkpoint()
    else:
        for item in items:
            if item.identity_state == "resolved" and item.outcome != "failed":
                item.outcome = "successful"

    status = _final_status(items)
    revision += 1
    final = make_receipt("final", status)
    try:
        active_artifacts.replace_json(
            receipt_uri, final.model_dump(by_alias=True), version
        )
    except Exception as exc:  # noqa: BLE001 - final durability failure
        raise EncordToolError(
            "push final receipt write failed after mutation; durable completion cannot "
            f"be claimed: {_sanitize_error(str(exc))}"
        ) from exc
    if status != "completed":
        raise EncordToolError(
            f"Encord push {status}; receipt written to {receipt_uri}"
        )
    return final


def _run_register(
    items: list[PushItem],
    folder_obj: Any,
    user_client: Any,
    *,
    integration_id: str,
    poll_timeout_seconds: int,
    sidecars: Mapping[str, IdentitySidecarRow],
    checkpoint: Callable[[], None],
) -> None:
    inventory = _folder_inventory(folder_obj, user_client)
    unmatched: list[PushItem] = []
    for item in items:
        if item.outcome == "failed":
            continue
        resolution = resolve_exact_identity(
            source_uri=item.source_uri,
            record_id=item.record_id,
            submitted_object_url=item.submitted_object_url,
            candidates=inventory,
            sidecar=sidecars.get(item.source_uri),
        )
        if resolution.resolved:
            _apply_resolution(item, resolution, state="existing")
        elif resolution.error_code in {
            "identity_conflict",
            "identity_sidecar_mismatch",
        }:
            item.registration_state = "unresolved"
            item.identity_state = "unresolved"
            item.outcome = "unresolved"
            item.error_code = resolution.error_code
            item.error = resolution.error
        else:
            unmatched.append(item)

    for batch in _chunks(unmatched, BATCH_SIZE):
        for item in batch:
            item.registration_state = "submitted"
        try:
            job_id = folder_obj.add_private_data_to_folder_start(
                integration_id=integration_id,
                private_files=build_registration_payload(batch),
                ignore_errors=False,
            )
            result = folder_obj.add_private_data_to_folder_get_result(
                job_id, timeout_seconds=poll_timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001
            for item in batch:
                item.registration_state = "failed"
                item.outcome = "failed"
                item.error_code = "registration_failed"
                item.error = _sanitize_error(str(exc))
            checkpoint()
            break

        unit_errors = _registration_errors(result)
        returned = list(getattr(result, "items_with_names", None) or [])
        inventory = _folder_inventory(folder_obj, user_client)
        candidates = [*_hydrate_returned(user_client, returned), *inventory]
        state = getattr(
            getattr(result, "status", None), "name", getattr(result, "status", "")
        )
        state = str(state).upper()
        pending = state == "PENDING"
        result_failed = state in {"ERROR", "CANCELLED"}
        global_error = "; ".join(
            str(value) for value in (getattr(result, "errors", None) or []) if value
        )
        units = sum(
            int(getattr(result, field, 0) or 0)
            for field in (
                "units_done_count",
                "units_error_count",
                "units_pending_count",
                "units_cancelled_count",
            )
        )
        count_mismatch = units != len(batch)
        for item in batch:
            unit_error = unit_errors.get(item.submitted_object_url)
            if unit_error:
                item.registration_state = "failed"
                item.outcome = "failed"
                item.error_code = "registration_unit_error"
                item.error = _sanitize_error(unit_error)
                continue
            if result_failed:
                item.registration_state = "failed"
                item.outcome = "failed"
                item.error_code = "registration_job_failed"
                item.error = _sanitize_error(global_error or f"registration job {state.lower()}")
                continue
            resolution = resolve_exact_identity(
                source_uri=item.source_uri,
                record_id=item.record_id,
                submitted_object_url=item.submitted_object_url,
                candidates=candidates,
                sidecar=sidecars.get(item.source_uri),
            )
            if resolution.resolved:
                _apply_resolution(item, resolution, state="registered")
            else:
                item.registration_state = "unresolved"
                item.identity_state = "unresolved"
                item.outcome = "unresolved"
                item.error_code = "registration_timeout" if pending else resolution.error_code
                item.error = (
                    "registration polling remained pending; exact identity is unknown"
                    if pending
                    else (
                        "registration unit counts did not reconcile with the submitted batch"
                        if count_mismatch
                        else resolution.error
                    )
                )
        checkpoint()
        if pending:
            break


def _run_upload(
    items: list[PushItem],
    folder_obj: Any,
    *,
    object_store: ObjectStorageGateway,
    checkpoint: Callable[[], None],
) -> None:
    for item in items:
        if item.category == "mcap":
            item.outcome = "failed"
            item.error_code = "unsupported_media"
            item.error = "MCAP upload requires a structured Encord scene definition"
            checkpoint()
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="npa-encord-upload-") as temporary:
                local = Path(temporary) / "source.bin"
                digest = object_store.download_to_file(item.source_uri, local)
                client_metadata = {
                    "npa": {
                        "source_uri": item.source_uri,
                        "record_id": item.record_id,
                    }
                }
                if item.category == "videos":
                    item_uuid = folder_obj.upload_video(
                        str(local),
                        title=item.object_key,
                        client_metadata=client_metadata,
                    )
                else:
                    item_uuid = folder_obj.upload_image(
                        str(local),
                        title=item.object_key,
                        client_metadata=client_metadata,
                    )
            item.source_size = digest.size
            item.source_checksum = digest.sha256
            item.source_checksum_kind = "sha256"
            item.item_uuid = str(item_uuid)
            item.identity_state = "resolved"
            item.registration_state = "not_applicable"
            item.outcome = "successful"
        except Exception as exc:  # noqa: BLE001
            item.outcome = "failed"
            item.error_code = "upload_failed"
            item.error = _sanitize_error(str(exc))
        checkpoint()


def _folder_inventory(folder_obj: Any, user_client: Any) -> list[Any]:
    try:
        rows = list(folder_obj.list_items(page_size=1000, include_client_metadata=True))
    except TypeError:
        rows = list(folder_obj.list_items(page_size=1000))
    uuids = [str(getattr(row, "uuid", "") or "") for row in rows]
    if uuids and hasattr(user_client, "get_storage_items"):
        try:
            hydrated = list(user_client.get_storage_items(uuids, sign_url=False))
        except TypeError:
            hydrated = list(user_client.get_storage_items(uuids))
        if hydrated:
            return [*rows, *hydrated]
    return rows


def _dataset_membership(user_client: Any, dataset_obj: Any, dataset_hash: str) -> set[str]:
    current = dataset_obj
    if dataset_hash and hasattr(user_client, "get_dataset"):
        try:
            current = user_client.get_dataset(dataset_hash, include_data_rows=True)
        except TypeError:
            current = user_client.get_dataset(dataset_hash)
    rows = (
        list(current.list_data_rows())
        if hasattr(current, "list_data_rows")
        else list(getattr(current, "data_rows", None) or [])
    )
    membership: set[str] = set()
    for row in rows:
        try:
            item_uuid = getattr(row, "backing_item_uuid", None)
        except NotImplementedError:
            item_uuid = None
        if item_uuid:
            membership.add(str(item_uuid))
    return membership


def _registration_errors(result: Any) -> dict[str, str]:
    errors: dict[str, str] = {}
    for unit in getattr(result, "unit_errors", None) or []:
        message = str(getattr(unit, "error", "") or "registration unit error")
        for url in getattr(unit, "object_urls", None) or []:
            errors[str(url)] = message
    return errors


def _hydrate_returned(user_client: Any, returned: list[Any]) -> list[Any]:
    uuids = [str(getattr(row, "item_uuid", "") or "") for row in returned]
    uuids = [value for value in uuids if value]
    if not uuids or not hasattr(user_client, "get_storage_items"):
        return []
    try:
        return list(user_client.get_storage_items(uuids, sign_url=False))
    except TypeError:
        return list(user_client.get_storage_items(uuids))


def _apply_resolution(
    item: PushItem, resolution: IdentityResolution, *, state: str
) -> None:
    item.item_uuid = resolution.item_uuid
    item.identity_state = "resolved"
    item.registration_state = state  # type: ignore[assignment]
    item.outcome = "successful"
    item.error_code = ""
    item.error = ""


def _fail_duplicate_resolutions(items: list[PushItem]) -> None:
    by_uuid: dict[str, list[PushItem]] = {}
    for item in items:
        if item.item_uuid:
            by_uuid.setdefault(item.item_uuid, []).append(item)
    for duplicates in by_uuid.values():
        if len(duplicates) < 2:
            continue
        for item in duplicates:
            item.item_uuid = ""
            item.identity_state = "unresolved"
            item.registration_state = "unresolved"
            item.outcome = "unresolved"
            item.error_code = "duplicate_exact_identity"
            item.error = "multiple source objects resolved to one Encord UUID"


def _load_sidecar(
    artifacts: ArtifactStore,
    uri: str,
    items: list[PushItem],
) -> dict[str, IdentitySidecarRow]:
    if not uri:
        return {}
    sidecar = IdentitySidecar.model_validate(artifacts.read_json(uri))
    discovered = {item.source_uri for item in items}
    unknown = {row.source_uri for row in sidecar.items} - discovered
    if unknown:
        raise EncordToolError("identity sidecar contains undiscovered source URIs")
    by_source = {row.source_uri: row for row in sidecar.items}
    for item in items:
        assertion = by_source.get(item.source_uri)
        if assertion is not None and assertion.record_id:
            item.record_id = assertion.record_id
    return by_source


def _chunks(items: list[PushItem], size: int) -> list[list[PushItem]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _final_status(items: list[PushItem]) -> RunStatus:
    outcomes = [item.outcome for item in items]
    if outcomes and all(outcome == "successful" for outcome in outcomes):
        return "completed"
    if any(outcome in {"successful", "unresolved"} for outcome in outcomes):
        return "partial"
    return "failed"


def _finalize_push_failure(
    artifacts: ArtifactStore,
    version: ArtifactVersion,
    receipt_uri: str,
    make_receipt: Callable[..., PushReceipt],
    revision: int,
    error_code: str,
    exc: Exception,
) -> PushReceipt:
    receipt = make_receipt(
        "final", "failed", error_code=error_code, error=str(exc)
    )
    payload = receipt.model_dump(by_alias=True)
    payload["revision"] = revision + 1
    receipt = PushReceipt.model_validate(payload)
    try:
        artifacts.replace_json(receipt_uri, receipt.model_dump(by_alias=True), version)
    except Exception as write_exc:  # noqa: BLE001
        raise EncordToolError(
            f"{error_code}; final receipt also failed: {_sanitize_error(str(write_exc))}"
        ) from write_exc
    raise EncordToolError(
        f"{error_code}; receipt written to {receipt_uri}: {_sanitize_error(str(exc))}"
    ) from exc


def _sanitize_error(message: str) -> str:
    def strip_query(match: re.Match[str]) -> str:
        parsed = urlsplit(match.group(0))
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))

    return re.sub(r"https?://[^\s]+", strip_query, message).strip()[:1000]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
