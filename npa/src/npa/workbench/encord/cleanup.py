"""Teardown for run-scoped Encord state (npa-e2e-*, npa-demo-src-*, ...).

Everything the tool creates in Encord is namespaced with a run-scoped title
(the run id embeds a UTC timestamp, so age is readable from the name). This
verb deletes what the SDK can delete — storage folders (items first),
Collections, and filter presets — for a given title prefix, and reports the
datasets it cannot delete because the Encord SDK exposes no dataset deletion.
"""

from __future__ import annotations

from typing import Any

from npa.workbench.encord.client import default_user_client
from npa.workbench.encord.schemas import EncordToolError

# A too-short prefix is a foot-gun: "n" would match every npa-* artifact and
# anything else that starts with it.
MIN_PREFIX_LENGTH = 4


def run_cleanup(
    *,
    title_prefix: str,
    dry_run: bool = False,
    user_client: Any = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Delete prefix-matched folders/collections/presets; report the rest."""

    prefix = title_prefix.strip()
    if len(prefix) < MIN_PREFIX_LENGTH:
        raise EncordToolError(
            f"--title-prefix must be at least {MIN_PREFIX_LENGTH} characters "
            "to avoid matching unrelated Encord artifacts."
        )
    client = user_client if user_client is not None else default_user_client(environ)

    summary: dict[str, Any] = {
        "stage": "cleanup",
        "title_prefix": prefix,
        "dry_run": dry_run,
        "collections_deleted": [],
        "presets_deleted": [],
        "folders_deleted": [],
        "items_deleted": 0,
        # The pinned Encord SDK exposes no dataset deletion; run-scoped
        # datasets are reported so an operator can remove them in the app.
        "datasets_undeletable": [],
    }

    for collection in list(client.list_collections(page_size=500)):
        if str(collection.name).startswith(prefix):
            if not dry_run:
                client.delete_collection(collection.uuid)
            summary["collections_deleted"].append(str(collection.name))

    for preset in list(client.list_presets(page_size=500)):
        if str(preset.name).startswith(prefix):
            if not dry_run:
                client.delete_preset(preset.uuid)
            summary["presets_deleted"].append(str(preset.name))

    for folder in list(client.list_storage_folders(search=prefix, page_size=500)):
        if not str(folder.name).startswith(prefix):
            continue
        item_uuids = [str(item.uuid) for item in folder.list_items(page_size=1000)]
        if not dry_run:
            if item_uuids:
                folder.delete_storage_items(item_uuids, remove_unused_frames=True)
            folder.delete()
        summary["items_deleted"] += len(item_uuids)
        summary["folders_deleted"].append(str(folder.name))

    for row in client.get_datasets():
        info = row.get("dataset") if isinstance(row, dict) else row
        title = str(getattr(info, "title", "") or "")
        if title.startswith(prefix):
            summary["datasets_undeletable"].append(title)

    return summary
