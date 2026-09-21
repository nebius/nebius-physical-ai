"""Observe durable case progress without changing or scoring a campaign."""

from __future__ import annotations

import json

from npa.clients.storage import StorageClient

from .campaign import validate_panel
from .case_store import CaseStore


def panel_status(storage: StorageClient, panel: dict, output_path: str) -> dict:
    """Read every prescribed case from its exact panel ledger.

    Args:
        storage: Client authorized to read the campaign state prefix.
        panel: Immutable, validated policy panel.
        output_path: Stable campaign state prefix used by the workers.
    Returns:
        Case states and counts, without scores or artifact-verification claims.
    Raises:
        ValueError: Panel or stored case identity is inconsistent.
        StorageError: The state cannot be read; absence is never inferred from errors.
    """
    validate_panel(panel)
    store = CaseStore(storage, output_path, panel["panel_id"])
    counts = dict.fromkeys(("unclaimed", "claimed", "started", "complete"), 0)
    cases = []
    for case in panel["cases"]:
        version = store.read(case)
        record = version.record if version else {}
        state = record.get("state", "unclaimed")
        counts[state] += 1
        cases.append(
            {
                "case": case,
                "state": state,
                **{
                    key: record[key]
                    for key in ("worker_id", "claimed_at", "started_at", "completed_at")
                    if key in record
                },
            }
        )
    return {
        "schema": "npa.behavior.campaign-status.v1",
        "panel_id": panel["panel_id"],
        "case_count": len(cases),
        "counts": counts,
        "cases": cases,
        "all_case_records_complete": counts["complete"] == len(cases),
        "artifact_bytes_verified": False,
        "live_worker_state_verified": False,
        "aggregation_required_before_comparison": True,
    }


def observe_campaign(args) -> dict:
    """Read an immutable panel and report its durable state through the stage CLI.

    Args:
        args: Parsed panel URI and campaign output prefix.
    Returns:
        Read-only panel status.
    Raises:
        ValueError: The panel is missing or has an invalid identity.
        StorageError: A required object cannot be read.
    """
    storage = StorageClient.from_environment()
    saved = storage.read_bytes_with_etag(args.panel_uri)
    if saved is None:
        raise ValueError("Campaign panel does not exist at the supplied URI")
    return panel_status(storage, json.loads(saved[0]), args.output_path)
