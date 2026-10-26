"""Select Encord storage items into a run-scoped Collection without the app."""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from npa.workbench.encord.client import (
    _default_user_client,
    find_folder,
    resolve_collection,
    resolve_domain,
)
from npa.workbench.encord.schemas import (
    CURATE_RECEIPT_FILENAME,
    CurateReceipt,
    EncordToolError,
    PushReceipt,
)
from npa.workbench.encord.storage import ArtifactStore, ConditionalArtifactStore

# Encord preset metric IDs for the supported range filters.
METRICS = {
    "width": "metric_width",
    "height": "metric_height",
    "area": "metric_area",
    "aspect-ratio": "metric_aspect_ratio",
    "brightness": "metric_brightness",
    "sharpness": "metric_sharpness",
    "file-size": "metric_file_size",
}
COMPUTED_METRICS = {"brightness", "sharpness", "file-size"}


def curate_receipt_uri_for(output_path: str) -> str:
    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{CURATE_RECEIPT_FILENAME}"


def parse_filters(specs: list[str]) -> tuple[list[str], dict[str, Any]]:
    parts = [part.strip() for spec in specs for part in spec.split(",") if part.strip()]
    if not parts:
        raise EncordToolError("at least one --filter metric:min:max is required")
    canonical: list[str] = []
    filters: list[dict[str, Any]] = []
    for part in parts:
        pieces = [piece.strip() for piece in part.split(":")]
        if len(pieces) != 3 or pieces[0] not in METRICS:
            raise EncordToolError(
                f"invalid --filter {part!r}; supported metrics: {', '.join(METRICS)}"
            )
        try:
            low, high = float(pieces[1]), float(pieces[2])
        except ValueError as exc:
            raise EncordToolError(
                f"invalid numeric bounds in --filter {part!r}"
            ) from exc
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise EncordToolError(f"invalid range in --filter {part!r}")
        canonical.append(f"{pieces[0]}:{low:g}:{high:g}")
        filters.append(
            {
                "include": True,
                "values": [low, high],
                "domain": "item" if pieces[0] == "file-size" else "data",
                "metric": METRICS[pieces[0]],
                "type": "metric",
            }
        )
    return canonical, {"global_filters": {"filters": filters}}


def _item_ids(collection: Any) -> list[str]:
    return sorted(str(item.uuid) for item in collection.list_items(page_size=1000))


def _poll_selection(collection: Any, poll_seconds: float) -> tuple[list[str], bool]:
    # ponytail: Encord exposes no completion handle; exact UUIDs are a snapshot,
    # and the downstream subset verifier catches additions before pull completes.
    deadline = time.monotonic() + poll_seconds
    previous: list[str] = []
    while True:
        selected = _item_ids(collection)
        if selected and selected == previous:
            return selected, True
        now = time.monotonic()
        if now >= deadline:
            return selected, False
        previous = selected
        time.sleep(min(1 if selected else 5, deadline - now))


def run_curate(
    *,
    folder: str,
    filters: list[str],
    collection: str,
    output_path: str,
    source_receipt_uri: str = "",
    workflow_run: str = "",
    poll_seconds: float = 300,
    user_client: Any = None,
    storage_client: Any = None,
    artifact_store: ArtifactStore | None = None,
    environ: dict[str, str] | None = None,
) -> CurateReceipt:
    """Evaluate Encord filters, checkpoint remote mutations, and return a receipt."""

    canonical, preset_json = parse_filters(filters)
    if (
        not folder.strip()
        or not collection.strip()
        or not math.isfinite(poll_seconds)
        or poll_seconds <= 0
    ):
        raise EncordToolError(
            "folder, collection, and positive poll-seconds are required"
        )
    from npa.clients.storage import StorageClient

    artifacts = artifact_store or ConditionalArtifactStore(
        storage_client or StorageClient.from_environment()
    )
    client = user_client if user_client is not None else _default_user_client(environ)
    uri = curate_receipt_uri_for(output_path)
    now = datetime.now(timezone.utc).isoformat()
    receipt = CurateReceipt(
        phase="provisional",
        status="running",
        revision=0,
        generated_at=now,
        updated_at=now,
        workflow_run=workflow_run,
        source_receipt_uri=source_receipt_uri,
        encord_domain=resolve_domain(environ),
        folder_name=folder.strip(),
        collection_name=collection.strip(),
        preset_name=f"npa-curate-{workflow_run or 'adhoc'}-{uuid.uuid4().hex[:8]}",
        filters=canonical,
        filter_preset_json=preset_json,
        receipt_uri=uri,
    )
    version = artifacts.create_json(uri, receipt.model_dump(by_alias=True))
    checkpoint_failed = False

    def save(**changes: Any) -> None:
        nonlocal receipt, version, checkpoint_failed
        payload = receipt.model_dump(by_alias=True)
        payload.update(changes)
        payload["revision"] += 1
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated = CurateReceipt.model_validate(payload)
        try:
            version = artifacts.replace_json(
                uri, updated.model_dump(by_alias=True), version
            )
        except Exception:
            checkpoint_failed = True
            raise
        receipt = updated

    error: Exception | None = None
    preset_uuid = ""
    preset_deleted = False
    try:
        source: PushReceipt | None = None
        if source_receipt_uri:
            source = PushReceipt.model_validate(artifacts.read_json(source_receipt_uri))
            if source.phase != "final" or source.status != "completed":
                raise EncordToolError("source push receipt is not final and completed")
            if source.encord_domain != receipt.encord_domain:
                raise EncordToolError(
                    "source push receipt uses a different Encord domain"
                )
        folder_obj = find_folder(client, folder)
        if folder_obj is None:
            raise EncordToolError(f"Encord folder {folder!r} does not exist")
        folder_uuid = str(folder_obj.uuid)
        if source and source.folder_uuid != folder_uuid:
            raise EncordToolError("source push receipt belongs to a different folder")
        folder_ids = set(_item_ids(folder_obj))
        total = len(folder_ids)
        if total == 0:
            raise EncordToolError("Encord folder has no storage items")
        save(phase="checkpoint", folder_uuid=folder_uuid, items_total=total)

        collection_obj, collection_uuid, collection_name = resolve_collection(
            client, collection, create_in_folder_uuid=folder_uuid
        )
        save(collection_uuid=collection_uuid, collection_name=collection_name)
        if _item_ids(collection_obj):
            raise EncordToolError(
                "collection already contains items; use a fresh title"
            )

        preset = client.create_preset(
            name=receipt.preset_name,
            description="Created by npa workbench encord curate",
            filter_preset_json=preset_json,
        )
        preset_uuid = str(preset.uuid)
        if not preset_uuid:
            raise EncordToolError("Encord preset creation returned no UUID")
        save(preset_uuid=preset_uuid)
        collection_obj.add_preset_items(preset_uuid)
        selected, settled = _poll_selection(collection_obj, poll_seconds)
        save(selected_item_uuids=selected, items_selected=len(selected))
        if not settled and selected:
            raise EncordToolError("Encord selection was still changing")
        if not selected:
            note = (
                "Computed quality metrics may need to be started in Encord for this folder. "
                if any(name.split(":", 1)[0] in COMPUTED_METRICS for name in canonical)
                else ""
            )
            raise EncordToolError(
                f"Encord curate selected no items. {note}Check filter ranges."
            )
        if not set(selected) <= folder_ids:
            raise EncordToolError(
                "curated Collection contains items outside its folder snapshot"
            )
        if source:
            pushed = {
                item.item_uuid for item in source.items if item.outcome == "successful"
            }
            if not set(selected) <= pushed:
                raise EncordToolError(
                    "curated Collection contains items outside the source push"
                )
    except Exception as exc:  # noqa: BLE001 - persist failure after remote mutation
        error = exc
    finally:
        if preset_uuid:
            try:
                client.delete_preset(preset_uuid)
                preset_deleted = True
            except Exception as exc:  # noqa: BLE001 - cleanup failure is recorded
                if error is None:
                    error = EncordToolError(f"Encord preset cleanup failed: {exc}")

    if checkpoint_failed:
        raise EncordToolError(
            f"Encord curate artifact checkpoint failed; last receipt at {uri}"
        ) from error
    save(
        phase="final",
        status="failed" if error else "completed",
        preset_deleted=preset_deleted,
        error_code=type(error).__name__ if error else "",
        error=str(error) if error else "",
    )
    if error:
        raise EncordToolError(
            f"Encord curate failed: {error}; receipt at {uri}"
        ) from error
    return receipt
