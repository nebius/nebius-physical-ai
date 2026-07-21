"""Artifact-backed Stages timeline helpers for the NPA agent.

Pure, side-effect-free helpers embedded into the agent VM backend (same
mechanism as ``agent_chat`` / ``agent_routing``). No network I/O.

These decide when a session workflow draft may overlay unmatched stages as
``pending`` onto an artifact-backed run, and how S3 keys map into stage rows.
"""

from __future__ import annotations

import re
from typing import Any


def _slug(value: str, *, fallback: str = "default") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return cleaned or fallback


def run_owns_workflow_stage_overlay(state: dict[str, Any], run_id: str) -> bool:
    """True when unmatched draft stages should show as pending for this run.

    Historical capture runs must not inherit an unrelated session draft as a
    wall of pending stages. Overlay only for the active submit, an explicitly
    tracked sim2real run with ``submitted_at``, or the draft's own run_id.
    """
    rid = str(run_id or "").strip()
    if not rid:
        return False
    latest = state.get("latest_submit")
    if isinstance(latest, dict) and str(latest.get("run_id") or "").strip() == rid:
        return True
    details_map = state.get("sim2real_runs")
    if isinstance(details_map, dict):
        existing = details_map.get(rid)
        if isinstance(existing, dict) and str(existing.get("submitted_at") or "").strip():
            return True
    draft = state.get("workflow_draft")
    if not isinstance(draft, dict):
        draft = {}
    plan = draft.get("plan") if isinstance(draft.get("plan"), dict) else {}
    if str(plan.get("run_id") or "").strip() == rid:
        return True
    if str(draft.get("name") or "").strip() and str(draft.get("run_id") or "").strip() == rid:
        return True
    return False


def artifact_stage_key(key: str, run_id: str, prefix: str) -> str:
    """Return the first path segment (or known compound key) under a run prefix."""
    value = str(key or "").strip("/")
    for lead in (str(prefix or "").strip("/"), ""):
        scoped = value
        if lead and scoped.startswith(lead + "/"):
            scoped = scoped[len(lead) + 1 :]
        if run_id and scoped.startswith(run_id + "/"):
            scoped = scoped[len(run_id) + 1 :]
            break
    parts = [part for part in scoped.split("/") if part]
    if not parts:
        return "artifacts"
    first = parts[0]
    if first == "reports":
        return "reports"
    if first == "eval" and len(parts) > 1:
        return "eval/" + parts[1]
    if first in {"actions", "vlm_eval", "training_signal", "envs"} and len(parts) > 1:
        return first + "/" + parts[1]
    return first


def artifact_stage_label(stage_key: str) -> str:
    labels = {
        "stage_01_trigger": "Trigger",
        "stage_02_assets": "Assets",
        "stage_12_external_validation": "External validation",
        "stage_13_retrigger": "Retrigger",
        "eval/heldout": "Held-out eval",
        "actions/train": "Policy rollouts",
        "vlm_eval/train": "VLM eval",
        "training_signal/train": "Training signal",
        "envs/raw": "Raw envs",
        "envs/train": "Train envs",
        "outer_loop": "Decision / outer loop",
        "reports": "Reports / visualization",
        "isaac-capture": "Isaac capture",
    }
    if stage_key in labels:
        return labels[stage_key]
    cleaned = stage_key.replace("_", " ").replace("/", " / ").replace("-", " ").strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else "Artifacts"


def build_artifact_backed_stages(
    keys: list[str],
    *,
    run_id: str,
    prefix: str,
    workflow_stage_defs: list[tuple[str, str, list[str]]],
    overlay_unmatched: bool,
) -> list[dict[str, Any]]:
    """Build Stages rows from artifact keys + optional workflow draft defs.

    When ``overlay_unmatched`` is false, draft states with zero matching
    artifacts are omitted (browse historical capture runs truthfully).
    """
    stages: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    if workflow_stage_defs:
        for stage_id, label, patterns in workflow_stage_defs:
            matched = [
                key for key in keys if any(pattern and pattern in key for pattern in patterns)
            ]
            used_keys.update(matched)
            count = len(matched)
            if count == 0 and not overlay_unmatched:
                continue
            # stage_key: the artifact stage of a matched key so the UI timeline row
            # is clickable and scopes the artifact browser to it (empty when unmatched).
            stage_key = artifact_stage_key(matched[0], run_id, prefix) if matched else ""
            stages.append(
                {
                    "id": stage_id,
                    "label": label,
                    "stage_key": stage_key,
                    "status": "succeeded" if count else "pending",
                    "started_at": "",
                    "finished_at": "",
                    "summary": (
                        f"{count} artifact{'s' if count != 1 else ''} matched workflow state '{label}'."
                        if count
                        else "No artifact matched this workflow state yet."
                    ),
                }
            )
    grouped: dict[str, list[str]] = {}
    for key in keys:
        stage_key = artifact_stage_key(key, run_id, prefix)
        grouped.setdefault(stage_key, []).append(key)
    for stage_key, matched in sorted(grouped.items()):
        if workflow_stage_defs and all(key in used_keys for key in matched):
            continue
        count = len(matched)
        stages.append(
            {
                "id": _slug(stage_key, fallback="artifacts"),
                "label": artifact_stage_label(stage_key),
                "stage_key": stage_key,
                "status": "succeeded",
                "started_at": "",
                "finished_at": "",
                "summary": f"{count} artifact{'s' if count != 1 else ''} found under '{stage_key}'.",
            }
        )
    return stages
