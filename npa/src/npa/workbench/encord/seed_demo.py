"""Seed the demo source dataset for the encord-cosmos3-augment workflow.

The committed spec defaults ``encord_source_id`` to a run-scoped demo dataset
title; when an operator overrides it with a real curated Collection/Dataset id
this stage becomes a no-op, so the default run works out of the box without
side effects on curated runs. The demo clip is the packaged PAIDF starter
asset: a pinned, SHA-256-verified public sample (CC-BY-4.0).
"""

from __future__ import annotations

import sys
from typing import Any

from npa.workbench.encord.schemas import DEFAULT_TRANSFER

STARTER_CLIP_NAME = "starter-clip.mp4"


def run_seed_demo(
    *,
    media_uri: str,
    dataset: str,
    active_source_id: str,
    transfer: str = DEFAULT_TRANSFER,
    integration: str = "",
    storage_client: Any = None,
    user_client: Any = None,
) -> dict[str, Any]:
    """Stage the starter clip under ``media_uri`` and push it, or skip.

    Seeding runs only when the workflow's active source id *is* the demo
    dataset title; any other id means the operator supplied curated data.
    """

    if active_source_id.strip() != dataset.strip():
        return {
            "stage": "seed_demo_source",
            "skipped": "operator supplied a curated source id",
            "source_id": active_source_id,
        }

    from npa.clients.storage import StorageClient
    from npa.workbench.encord.push import require_transfer_integration, run_push
    from npa.workflows.data_factory_input import _fetch_starter, load_starter_contract

    require_transfer_integration(transfer, integration)
    contract = load_starter_contract()
    # Progress lines go to stderr: stdout is reserved for the one JSON document
    # the CLI emits.
    local_path, cache_state = _fetch_starter(
        contract,
        cache_dir=None,
        offline=None,
        reporter=lambda line: print(line, file=sys.stderr),
    )
    client = storage_client or StorageClient.from_environment()
    prefix = media_uri.rstrip("/")
    clip_uri = f"{prefix}/{STARTER_CLIP_NAME}"
    client.upload_file(str(local_path), clip_uri)

    receipt = run_push(
        input_path=f"{prefix}/",
        integration=integration,
        folder=dataset,
        dataset=dataset,
        output_path=f"{prefix}/push/",
        transfer=transfer,
        workflow_run=dataset,
        storage_client=client,
        user_client=user_client,
    )
    return {
        "stage": "seed_demo_source",
        "cache": cache_state,
        "clip_uri": clip_uri,
        "dataset": dataset,
        "transfer": transfer,
        "units_done": receipt.units_done,
        "attribution": str((contract.get("license") or {}).get("name", "")),
        "asset_sha256": str(contract["integrity"]["sha256"]),
    }
