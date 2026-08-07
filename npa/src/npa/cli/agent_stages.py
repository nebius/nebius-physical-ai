"""Artifact-backed Stages timeline helpers for the NPA agent.

Pure, side-effect-free helpers embedded into the agent VM backend (same
mechanism as ``agent_chat`` / ``agent_routing``). No network I/O.

These decide when a session workflow draft may overlay unmatched stages as
``pending`` onto an artifact-backed run, and how S3 keys map into stage rows.
"""

from __future__ import annotations

import re
from typing import Any


def resolve_run_source(payload: dict[str, Any], existing: dict[str, Any], run_id: str) -> tuple[str, str]:
    """Resolve stable provenance for a viewer-history entry."""
    source_type = str(payload.get("source_type") or existing.get("source_type") or "").strip()
    if not source_type:
        if run_id == "franka-demo":
            source_type = "local_demo"
        elif str(payload.get("artifact_uri") or existing.get("artifact_uri") or "").startswith("s3://"):
            source_type = "artifact_storage"
        else:
            source_type = "workflow_history"
    source_label = {
        "local_demo": "Local demo",
        "artifact_storage": "S3 artifacts",
        "workflow_history": "Workflow history",
    }.get(source_type, source_type.replace("_", " ").title())
    return source_type, source_label


def build_available_sim_viz_runs(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Project viewer history into source-aware selector summaries."""
    return [
        {
            "run_id": str(item.get("run_id") or "").strip(),
            # Viewer activity must not override artifact recency in the merged list.
            "activity_at": str(
                item.get("rrd_updated_at")
                or item.get("updated_at")
                or item.get("submitted_at")
                or ""
            ).strip(),
            "started_at": str(item.get("submitted_at") or "").strip(),
            "last_modified": "",
            "stage": str(item.get("stage") or "").strip(),
            "source_type": str(item.get("source_type") or "workflow_history").strip(),
            "source_label": str(item.get("source_label") or "Workflow history").strip(),
        }
        for item in runs
        if str(item.get("run_id") or "").strip()
    ]


def local_demo_run_details(
    state: dict[str, Any], run_id: str, recorded: dict[str, Any], now_iso: str
) -> dict[str, Any] | None:
    """Return deterministic local-demo stage details, or ``None`` for real runs."""
    if str(recorded.get("source_type") or "") != "local_demo" and run_id != "franka-demo":
        return None
    updated_at = str(recorded.get("rrd_updated_at") or now_iso)
    ready = bool(recorded.get("rerun_ready"))
    return {
        "run_id": "franka-demo",
        "source_type": "local_demo",
        "source_label": "Local demo",
        "status": "completed",
        "result": "rerun_ready" if ready else "recording_unavailable",
        "updated_at": updated_at,
        "selection": state.get("selection") if isinstance(state.get("selection"), dict) else {},
        "stages": [
            {
                "id": "local_demo",
                "label": "Local Franka demo",
                "status": "succeeded" if ready else "pending",
                "summary": "Deterministically generated on this agent VM.",
            }
        ],
        "logs": [
            {
                "timestamp": updated_at,
                "level": "info",
                "message": "Local Franka demo recording regenerated from stock assets.",
            }
        ],
        "artifacts": [],
    }


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


def _scoped_after_run(key: str, run_id: str, prefix: str) -> str:
    """Return the path under ``<prefix>/<run_id>/`` (or ``<run_id>/``)."""
    value = str(key or "").strip("/")
    for lead in (str(prefix or "").strip("/"), ""):
        scoped = value
        if lead and scoped.startswith(lead + "/"):
            scoped = scoped[len(lead) + 1 :]
        if run_id and scoped.startswith(run_id + "/"):
            scoped = scoped[len(run_id) + 1 :]
            break
    return scoped


def run_stage_wrapper(keys: list[str], run_id: str, prefix: str) -> str:
    """Common workflow-name wrapper dir(s) between the run and its real stages.

    Some workflows nest artifacts as ``<run_id>/<workflow-name>/<stage>/...``
    (e.g. ``.../<run>/tokenfactory-cosmos-gate/augment/frame.png``), so the first
    segment after the run id is a wrapper, not a stage — deriving stages from it
    collapses the whole pipeline into one row. This returns the leading seg
    (possibly multi-level) that ALL keys share AND that still has a deeper
    directory for every key, so it is safe to strip. The actual stage level —
    whose children are files — is never stripped, so flat ``<run>/<stage>/file``
    layouts are unaffected.
    """
    scoped = [s for s in (_scoped_after_run(k, run_id, prefix) for k in keys if k) if s]
    if not scoped:
        return ""
    wrapper_parts: list[str] = []
    tails = scoped
    while True:
        firsts = {t.split("/", 1)[0] for t in tails}
        if len(firsts) != 1:
            break
        seg = next(iter(firsts))
        new_tails: list[str] = []
        strip_ok = True
        for tail in tails:
            rest = tail[len(seg) + 1 :] if tail.startswith(seg + "/") else ""
            if "/" not in rest:  # seg is the stage (its children are files) → stop
                strip_ok = False
                break
            new_tails.append(rest)
        if not strip_ok:
            break
        wrapper_parts.append(seg)
        tails = new_tails
    return "/".join(wrapper_parts)


def artifact_stage_key(key: str, run_id: str, prefix: str, wrapper: str = "") -> str:
    """Return the first path segment (or known compound key) under a run prefix.

    ``wrapper`` (see :func:`run_stage_wrapper`) strips common workflow-name
    nesting so the stage is the real pipeline stage, not a wrapper directory.
    """
    scoped = _scoped_after_run(key, run_id, prefix)
    wrap = str(wrapper or "").strip("/")
    if wrap and scoped.startswith(wrap + "/"):
        scoped = scoped[len(wrap) + 1 :]
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
    # Strip common workflow-name nesting so runs stored as
    # <run>/<workflow-name>/<stage>/... expose their real pipeline stages instead
    # of collapsing into a single wrapper row.
    wrapper = run_stage_wrapper(keys, run_id, prefix)
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
            stage_key = artifact_stage_key(matched[0], run_id, prefix, wrapper) if matched else ""
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
        stage_key = artifact_stage_key(key, run_id, prefix, wrapper)
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
