"""Glue stage for the Encord → Cosmos 3 augmentation workflows.

The Encord pull stage names media by item uuid (``media/<uuid>__<name>``), which
a workflow spec cannot know in advance, while ``cosmos3 generate`` conditions on
one exact video URI. ``stage_media_for_augment`` bridges the two: it reads the
pull manifest and server-side-copies the selected item to a deterministic URI
the spec can template. (Demo-source seeding lives in the tool itself:
``npa workbench encord seed-demo``.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from npa.workbench.encord.storage import object_location


class EncordLoopError(RuntimeError):
    """Raised when the pull manifest cannot supply the requested media item."""


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _is_video_item(item: dict[str, Any]) -> bool:
    """Whether an Encord pull-manifest row is safe for Cosmos video2video."""

    mime_type = str(item.get("mime_type", "")).lower()
    item_type = str(item.get("item_type", "")).lower()
    suffix = Path(str(item.get("name", ""))).suffix.lower()
    return mime_type.startswith("video/") or item_type == "video" or suffix in VIDEO_SUFFIXES


def stage_media_for_augment(
    manifest_uri: str,
    dest_uri: str,
    index: str = "0",
    storage_client: Any = None,
) -> dict[str, Any]:
    """Copy the pull manifest's ``index``-th media item to ``dest_uri``.

    Fails closed when the manifest is missing, the index is out of range, or the
    selected item did not transfer successfully — a spec must never condition
    Cosmos on a file that is not really there.
    """

    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    got = client.read_bytes_with_etag(manifest_uri)
    if got is None:
        raise EncordLoopError(f"pull manifest not found at {manifest_uri}")
    manifest = json.loads(got[0])
    items = [
        item
        for item in manifest.get("items", [])
        if item.get("transfer") in ("copy", "download") and item.get("media_uri")
    ]
    position = int(index)
    if not items or position >= len(items) or position < 0:
        raise EncordLoopError(
            f"pull manifest has {len(items)} transferred media item(s); "
            f"index {position} is unavailable"
        )
    selected = items[position]
    if not _is_video_item(selected):
        raise EncordLoopError(
            "selected Encord item is not a video; Cosmos video2video requires a "
            f"video item (name={selected.get('name', '')!r}, "
            f"mime_type={selected.get('mime_type', '')!r})"
        )
    source_bucket, source_key = object_location(
        str(selected["media_uri"]), error_type=EncordLoopError, require_key=True
    )
    dest_bucket, dest_key = object_location(
        dest_uri, error_type=EncordLoopError, require_key=True
    )
    client.s3.copy_object(
        Bucket=dest_bucket,
        Key=dest_key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
    )
    summary = {
        "stage": "stage_media_for_augment",
        "item_uuid": selected.get("item_uuid", ""),
        "name": selected.get("name", ""),
        "source_uri": selected["media_uri"],
        "staged_uri": dest_uri,
        "items_available": len(items),
        "index": position,
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":  # pragma: no cover - exercised through the spec argv
    stage_media_for_augment(*sys.argv[1:])
