"""Evidence-backed Stages timeline helpers for the NPA agent.

Pure, side-effect-free helpers embedded into the agent VM backend (same
mechanism as ``agent_chat`` / ``agent_routing``). No network I/O.

Every stage row uses one conservative evidence contract.  Authoritative
workflow state may establish an execution status; an artifact can only establish
that output was observed.  Missing output establishes nothing.
"""

from __future__ import annotations

import re
from typing import Any


_SUCCEEDED = {"succeeded", "success", "successful", "ok", "done", "complete", "completed", "passed"}
_FAILED = {"failed", "failure", "error", "errored", "blocked", "cancelled", "canceled"}
_RUNNING = {"running", "active", "in_progress", "in-progress", "executing"}
_SKIPPED = {"skipped", "skip"}
_NOT_RUN = {"not_run", "not-run", "not run", "not_launched", "not-launched", "not launched"}
_PENDING = {"pending", "planned", "queued", "submitted", "created", "waiting"}
_AUTHORITATIVE_STATUSES = {"succeeded", "failed", "running", "skipped", "not_run", "pending"}


def normalize_explicit_stage_status(value: Any, *, returncode: Any = None) -> tuple[str, str]:
    """Normalize an explicit workflow/status value without guessing from artifacts."""
    raw = str(value or "").strip().lower()
    if raw in _SUCCEEDED:
        return "succeeded", "Succeeded"
    if raw in _FAILED:
        return "failed", "Failed"
    if raw in _RUNNING:
        return "running", "Running"
    if raw in _SKIPPED:
        return "skipped", "Skipped"
    if raw in _NOT_RUN:
        return "not_run", "Not run"
    if raw in _PENDING:
        return "pending", raw.replace("_", " ").replace("-", " ").title()
    if returncode not in (None, ""):
        try:
            return ("succeeded", "Succeeded") if int(returncode) == 0 else ("failed", "Failed")
        except (TypeError, ValueError):
            pass
    return "status_unavailable", "Status unavailable"


def stage_evidence_record(
    *,
    stage_id: str,
    label: str,
    status: str,
    status_label: str,
    evidence_type: str,
    evidence_source: str,
    authority: str,
    confidence: str,
    reason: str,
    stage_key: str = "",
    raw_status: str = "",
    artifact_count: int = 0,
    started_at: str = "",
    finished_at: str = "",
    observed_at: str = "",
    summary: str = "",
) -> dict[str, Any]:
    """Build the versioned stage-evidence shape consumed by both API and UI."""
    evidence = {
        "type": str(evidence_type or "unknown"),
        "source": str(evidence_source or ""),
        "authority": str(authority or "unknown"),
        "confidence": str(confidence or "unknown"),
        "reason": str(reason or ""),
        "observed_at": str(observed_at or ""),
    }
    return {
        "evidence_version": "npa.stage-evidence/v1",
        "id": _slug(stage_id, fallback="stage"),
        "label": str(label or stage_id or "Stage"),
        "stage_key": str(stage_key or ""),
        "status": str(status or "status_unavailable"),
        "status_label": str(status_label or "Status unavailable"),
        "raw_status": str(raw_status or ""),
        "started_at": str(started_at or ""),
        "finished_at": str(finished_at or ""),
        "artifact_count": max(0, int(artifact_count or 0)),
        "summary": str(summary or reason or "Status unavailable."),
        "evidence": evidence,
        # Flat aliases keep the API convenient and make the provenance contract
        # explicit even for clients that do not inspect nested objects.
        "evidence_type": evidence["type"],
        "evidence_source": evidence["source"],
        "authority": evidence["authority"],
        "confidence": evidence["confidence"],
        "diagnostic_reason": evidence["reason"],
    }


def summarize_stage_evidence(stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return counts/text that cannot imply unsupported execution outcomes."""
    rows = [item for item in stages if isinstance(item, dict)]
    counts = {name: 0 for name in (
        "succeeded", "failed", "running", "skipped", "not_run", "pending",
        "observed_output", "status_unavailable",
    )}
    for item in rows:
        status = str(item.get("status") or "status_unavailable")
        counts[status if status in counts else "status_unavailable"] += 1
    observed_count = sum(1 for item in rows if int(item.get("artifact_count") or 0) > 0)
    authoritative_count = sum(
        1
        for item in rows
        if str(item.get("authority") or (item.get("evidence") or {}).get("authority") or "")
        == "authoritative"
    )
    outcome_count = sum(counts[name] for name in _AUTHORITATIVE_STATUSES)
    if rows and outcome_count == 0 and observed_count:
        text = (
            f"{observed_count} observed group{'s' if observed_count != 1 else ''}"
            " · execution status unavailable"
        )
    elif rows and outcome_count:
        ordered = [
            ("succeeded", "succeeded"),
            ("failed", "failed"),
            ("running", "running"),
            ("skipped", "skipped"),
            ("not_run", "not run"),
            ("pending", "pending"),
        ]
        parts = [f"{counts[key]} {label}" for key, label in ordered if counts[key]]
        if counts["observed_output"]:
            parts.append(f"{counts['observed_output']} observed output")
        if counts["status_unavailable"]:
            parts.append(f"{counts['status_unavailable']} status unavailable")
        text = " · ".join(parts)
    elif rows:
        text = f"{len(rows)} stage{'s' if len(rows) != 1 else ''} · execution status unavailable"
    else:
        text = "No stage evidence available"
    return {
        "evidence_version": "npa.stage-evidence/v1",
        "text": text,
        "displayed_stage_count": len(rows),
        "observed_stage_count": observed_count,
        "authoritative_stage_count": authoritative_count,
        "execution_status_available": outcome_count > 0,
        **{f"{name}_count": value for name, value in counts.items()},
    }


def merge_stage_evidence(
    primary: list[dict[str, Any]], secondary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge observations into authoritative rows without weakening their status."""
    merged = [dict(item) for item in primary if isinstance(item, dict)]
    positions: dict[str, int] = {}
    for index, item in enumerate(merged):
        for key in (item.get("id"), item.get("stage_key")):
            normalized = _slug(str(key or ""), fallback="")
            if normalized:
                positions[normalized] = index
    for incoming_raw in secondary:
        if not isinstance(incoming_raw, dict):
            continue
        incoming = dict(incoming_raw)
        candidates = [
            _slug(str(incoming.get("id") or ""), fallback=""),
            _slug(str(incoming.get("stage_key") or ""), fallback=""),
        ]
        index = next((positions[key] for key in candidates if key and key in positions), None)
        if index is None:
            merged.append(incoming)
            new_index = len(merged) - 1
            for key in candidates:
                if key:
                    positions[key] = new_index
            continue
        current = merged[index]
        current_authority = str(current.get("authority") or (current.get("evidence") or {}).get("authority") or "")
        incoming_authority = str(incoming.get("authority") or (incoming.get("evidence") or {}).get("authority") or "")
        if incoming_authority == "authoritative" and current_authority != "authoritative":
            replacement = dict(incoming)
            replacement["artifact_count"] = max(
                int(current.get("artifact_count") or 0), int(incoming.get("artifact_count") or 0)
            )
            merged[index] = replacement
            continue
        current["artifact_count"] = max(
            int(current.get("artifact_count") or 0), int(incoming.get("artifact_count") or 0)
        )
        observations = list(current.get("observations") or [])
        incoming_evidence = incoming.get("evidence")
        if isinstance(incoming_evidence, dict) and incoming_evidence.get("type") == "artifact_observation":
            observations.append(dict(incoming_evidence))
        observations.extend(
            dict(item) for item in incoming.get("observations") or [] if isinstance(item, dict)
        )
        if observations:
            current["observations"] = observations
        if current_authority != "authoritative" and incoming_authority:
            merged[index] = incoming
    return merged


def coerce_authoritative_stage_evidence(
    stages: list[dict[str, Any]], *, source: str
) -> list[dict[str, Any]]:
    """Upgrade legacy persisted agent stage rows to the evidence-v1 contract."""
    upgraded: list[dict[str, Any]] = []
    for item in stages:
        if not isinstance(item, dict):
            continue
        if str(item.get("evidence_version") or "") == "npa.stage-evidence/v1":
            upgraded.append(dict(item))
            continue
        raw_status = str(item.get("status") or "").strip()
        status, label = normalize_explicit_stage_status(raw_status)
        reason = str(item.get("summary") or "").strip() or (
            f"Persisted authoritative agent state reports '{raw_status}'."
            if raw_status
            else "Persisted agent state declares this stage without an execution status."
        )
        upgraded.append(
            stage_evidence_record(
                stage_id=str(item.get("id") or item.get("stage_key") or "stage"),
                label=str(item.get("label") or item.get("id") or "Stage"),
                stage_key=str(item.get("stage_key") or ""),
                status=status,
                status_label=label,
                raw_status=raw_status,
                evidence_type="workflow_status" if raw_status else "workflow_graph",
                evidence_source=source,
                authority="authoritative",
                confidence="high" if raw_status else "medium",
                reason=reason,
                artifact_count=int(item.get("artifact_count") or 0),
                started_at=str(item.get("started_at") or ""),
                finished_at=str(item.get("finished_at") or ""),
                summary=reason,
            )
        )
    return upgraded


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
            "bucket": str(item.get("bucket") or "").strip(),
            "project_id": str(item.get("project_id") or "").strip(),
            "resolved_prefix": str(item.get("resolved_prefix") or "").strip(),
            "run_ref": str(item.get("artifact_run_ref") or "").strip(),
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
    stage = stage_evidence_record(
        stage_id="local_demo",
        label="Local Franka demo",
        status="succeeded" if ready else "status_unavailable",
        status_label="Succeeded" if ready else "Status unavailable",
        evidence_type="demo_fixture",
        evidence_source="local_franka_demo",
        authority="fixture",
        confidence="high",
        reason=(
            "The deterministic local demo generator published its recording."
            if ready
            else "The local demo fixture has not published a recording."
        ),
        observed_at=updated_at,
        artifact_count=1 if ready else 0,
        summary="Deterministically generated on this agent VM.",
    )
    return {
        "run_id": "franka-demo",
        "source_type": "local_demo",
        "source_label": "Local demo",
        "status": "completed",
        "result": "rerun_ready" if ready else "recording_unavailable",
        "updated_at": updated_at,
        "selection": state.get("selection") if isinstance(state.get("selection"), dict) else {},
        "stages": [stage],
        "stage_summary": summarize_stage_evidence([stage]),
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


def _explicit_stage_record(
    stage_id: str,
    payload: dict[str, Any],
    *,
    source: str,
    evidence_type: str,
    graph_only: bool = False,
) -> dict[str, Any]:
    raw_status = str(
        payload.get("status")
        or payload.get("outcome")
        or payload.get("result")
        or (payload.get("state") if evidence_type == "workflow_status" else "")
        or ""
    ).strip()
    status, status_label = normalize_explicit_stage_status(
        raw_status, returncode=payload.get("returncode")
    )
    explicit = bool(raw_status) or payload.get("returncode") not in (None, "")
    if graph_only and not explicit:
        status, status_label = "status_unavailable", "Status unavailable"
    reason = (
        f"Authoritative workflow evidence reports '{raw_status}'."
        if explicit
        else "The authoritative workflow graph declares this stage, but no execution status was recorded."
    )
    label = str(payload.get("label") or payload.get("name") or stage_id).strip()
    return stage_evidence_record(
        stage_id=stage_id,
        label=label or artifact_stage_label(stage_id),
        stage_key=stage_id,
        status=status,
        status_label=status_label,
        raw_status=raw_status,
        evidence_type=evidence_type if explicit else "workflow_graph",
        evidence_source=source,
        authority="authoritative",
        confidence="high" if explicit else "medium",
        reason=reason,
        started_at=str(payload.get("start_time") or payload.get("start") or payload.get("started_at") or ""),
        finished_at=str(payload.get("end_time") or payload.get("end") or payload.get("finished_at") or ""),
        observed_at=str(payload.get("updated_at") or payload.get("end_time") or payload.get("end") or ""),
        summary=str(payload.get("summary") or payload.get("error_summary") or reason),
    )


def parse_stage_evidence_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse known manifest/status/report shapes into authoritative stage evidence.

    The parser is deliberately typed: arbitrary JSON and artifact paths do not
    create execution outcomes.  Callers inject already-decoded candidate
    documents, which keeps S3/network behavior out of this reusable model.
    """
    stages_by_id: dict[str, dict[str, Any]] = {}
    ranks: dict[str, int] = {}
    order: list[str] = []
    workflow_name = ""
    graph_source = ""
    run_status = ""
    run_status_label = ""
    run_status_source = ""
    updated_at = ""
    consumed_sources: list[str] = []

    def upsert(record: dict[str, Any], rank: int) -> None:
        key = str(record.get("id") or "").strip()
        if not key:
            return
        existing = stages_by_id.get(key)
        attempts = int((existing or {}).get("attempt_count") or 0) + 1
        if existing is None:
            order.append(key)
        if existing is None or rank >= ranks.get(key, -1):
            record["attempt_count"] = attempts
            stages_by_id[key] = record
            ranks[key] = rank
        elif existing is not None:
            existing["attempt_count"] = attempts

    for document in documents:
        if not isinstance(document, dict):
            continue
        source = str(document.get("key") or document.get("source") or "").strip()
        payload = document.get("payload")
        if not isinstance(payload, dict):
            continue
        source_lower = source.lower()
        schema = str(payload.get("schema_version") or payload.get("apiVersion") or "").lower()
        is_npa_manifest = source_lower.endswith("/npa-workflow/manifest.json") or schema == "npa.workflow.run.v1"
        is_durable_manifest = bool(
            source_lower.endswith("/manifest.json")
            and isinstance(payload.get("stages"), dict)
            and (payload.get("workflow_name") or payload.get("run_prefix_uri"))
        )
        is_status_doc = source_lower.endswith("/status.json")
        is_report_doc = source_lower.endswith("report.json") or "/reports/" in source_lower
        recognized = is_npa_manifest or is_durable_manifest or is_status_doc or is_report_doc
        consumed = is_npa_manifest or is_durable_manifest

        candidate_workflow = str(payload.get("workflow") or payload.get("workflow_name") or "").strip()
        if candidate_workflow and recognized:
            workflow_name = candidate_workflow
        candidate_updated = str(payload.get("updated_at") or payload.get("end_time") or "").strip()
        if candidate_updated > updated_at:
            updated_at = candidate_updated

        top_status = str(payload.get("status") or "").strip()
        if top_status and (is_npa_manifest or is_status_doc or is_durable_manifest):
            consumed = True
            normalized, label = normalize_explicit_stage_status(top_status)
            run_status = normalized if normalized != "status_unavailable" else top_status.lower()
            run_status_label = label if normalized != "status_unavailable" else top_status.replace("_", " ").title()
            run_status_source = source

        steps = payload.get("steps")
        if is_npa_manifest and isinstance(steps, list):
            graph_source = graph_source or source
            for step in steps:
                if not isinstance(step, dict):
                    continue
                stage_id = str(step.get("state") or step.get("stage") or step.get("id") or "").strip()
                if not stage_id:
                    continue
                raw = str(step.get("status") or "").strip()
                rank = 40 if raw or step.get("returncode") not in (None, "") else 20
                upsert(
                    _explicit_stage_record(
                        stage_id,
                        step,
                        source=source,
                        evidence_type="manifest_status",
                        graph_only=True,
                    ),
                    rank,
                )

        declared = payload.get("stages")
        if isinstance(declared, dict) and (is_durable_manifest or is_report_doc):
            graph_source = graph_source or source
            for stage_id, raw_info in declared.items():
                info = dict(raw_info) if isinstance(raw_info, dict) else {"name": str(stage_id)}
                raw = str(info.get("state") or info.get("status") or info.get("outcome") or "").strip()
                if is_report_doc and not raw:
                    continue
                consumed = True
                upsert(
                    _explicit_stage_record(
                        str(stage_id),
                        info,
                        source=source,
                        evidence_type="manifest_status" if is_durable_manifest else "report_status",
                        graph_only=is_durable_manifest,
                    ),
                    45 if raw else 20,
                )
        elif isinstance(declared, list) and is_report_doc:
            for item in declared:
                if not isinstance(item, dict):
                    continue
                stage_id = str(item.get("stage") or item.get("id") or item.get("name") or "").strip()
                raw = str(item.get("state") or item.get("status") or item.get("outcome") or "").strip()
                if stage_id and raw:
                    consumed = True
                    upsert(
                        _explicit_stage_record(
                            stage_id, item, source=source, evidence_type="report_status"
                        ),
                        45,
                    )

        outcomes = payload.get("stage_outcomes")
        if isinstance(outcomes, dict) and (is_report_doc or is_status_doc):
            consumed = bool(outcomes) or consumed
            for stage_id, raw_info in outcomes.items():
                info = dict(raw_info) if isinstance(raw_info, dict) else {"status": raw_info}
                upsert(
                    _explicit_stage_record(
                        str(stage_id), info, source=source, evidence_type="report_status"
                    ),
                    45,
                )
        elif isinstance(outcomes, list) and (is_report_doc or is_status_doc):
            for item in outcomes:
                if not isinstance(item, dict):
                    continue
                stage_id = str(item.get("stage") or item.get("id") or item.get("name") or "").strip()
                if stage_id:
                    consumed = True
                    upsert(
                        _explicit_stage_record(
                            stage_id, item, source=source, evidence_type="report_status"
                        ),
                        45,
                    )

        # Durable per-stage status documents use logs/<stage>/status.json and
        # carry an explicit stage field.  They outrank manifest snapshots.
        status_stage = str(payload.get("stage") or "").strip()
        if is_status_doc and status_stage:
            consumed = True
            upsert(
                _explicit_stage_record(
                    status_stage, payload, source=source, evidence_type="workflow_status"
                ),
                60,
            )
        if consumed and source and source not in consumed_sources:
            consumed_sources.append(source)

    return {
        "evidence_version": "npa.stage-evidence/v1",
        "workflow_name": workflow_name,
        "graph_source": graph_source,
        "run_status": run_status,
        "run_status_label": run_status_label,
        "run_status_source": run_status_source,
        "updated_at": updated_at,
        "consumed_sources": consumed_sources,
        "stages": [stages_by_id[key] for key in order if key in stages_by_id],
    }


def build_artifact_backed_stages(
    keys: list[str],
    *,
    run_id: str,
    prefix: str,
    workflow_stage_defs: list[tuple[str, str, list[str]]],
    overlay_unmatched: bool,
    authoritative_stages: list[dict[str, Any]] | None = None,
    evidence_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build stage rows from explicit workflow evidence plus artifact observations.

    Artifacts are never promoted to execution success.  Missing artifacts are
    never promoted to ``not_run`` or failure.
    """
    stages: list[dict[str, Any]] = []
    # Manifest/status documents establish provenance but are not workload output
    # groups of their own.
    used_keys: set[str] = {str(key) for key in (evidence_keys or []) if str(key)}
    artifact_keys = [str(key) for key in keys if str(key) not in used_keys]
    # Strip common workflow-name nesting so runs stored as
    # <run>/<workflow-name>/<stage>/... expose their real pipeline stages instead
    # of collapsing into a single wrapper row.
    wrapper = run_stage_wrapper(artifact_keys, run_id, prefix)
    authoritative = [item for item in (authoritative_stages or []) if isinstance(item, dict)]
    if authoritative:
        for source_stage in authoritative:
            stage = dict(source_stage)
            stage_id = str(stage.get("id") or stage.get("stage_key") or "").strip()
            patterns = [stage_id, stage_id.replace("_", "-"), stage_id.replace("-", "_")]
            matched = [
                key for key in artifact_keys if any(pattern and pattern in key for pattern in patterns)
            ]
            used_keys.update(matched)
            stage["artifact_count"] = len(matched)
            if matched and not str(stage.get("stage_key") or "").strip():
                stage["stage_key"] = artifact_stage_key(matched[0], run_id, prefix, wrapper)
            if matched:
                observation = {
                    "type": "artifact_observation",
                    "source": "artifact_listing",
                    "authority": "observed",
                    "confidence": "high",
                    "reason": f"Observed {len(matched)} artifact{'s' if len(matched) != 1 else ''} for this stage.",
                    "observed_at": "",
                }
                stage["observations"] = [observation]
                if str(stage.get("status") or "") == "status_unavailable":
                    graph_evidence = dict(stage.get("evidence") or {})
                    stage["evidence_chain"] = [graph_evidence, observation]
                    stage["status"] = "observed_output"
                    stage["status_label"] = "Observed output"
                    stage["evidence"] = observation
                    stage["evidence_type"] = observation["type"]
                    stage["evidence_source"] = observation["source"]
                    stage["authority"] = observation["authority"]
                    stage["confidence"] = observation["confidence"]
                    stage["diagnostic_reason"] = observation["reason"]
                    stage["summary"] = observation["reason"] + " Execution status is unavailable."
            stages.append(stage)
    elif workflow_stage_defs:
        for stage_id, label, patterns in workflow_stage_defs:
            matched = [
                key for key in artifact_keys if any(pattern and pattern in key for pattern in patterns)
            ]
            used_keys.update(matched)
            count = len(matched)
            if count == 0 and not overlay_unmatched:
                continue
            # stage_key: the artifact stage of a matched key so the UI timeline row
            # is clickable and scopes the artifact browser to it (empty when unmatched).
            stage_key = artifact_stage_key(matched[0], run_id, prefix, wrapper) if matched else ""
            stages.append(
                stage_evidence_record(
                    stage_id=stage_id,
                    label=label,
                    stage_key=stage_key,
                    status="observed_output" if count else "status_unavailable",
                    status_label="Observed output" if count else "Status unavailable",
                    evidence_type="artifact_observation" if count else "workflow_graph",
                    evidence_source="artifact_listing" if count else "active_workflow_plan",
                    authority="observed" if count else "authoritative",
                    confidence="high" if count else "medium",
                    reason=(
                        f"Observed {count} artifact{'s' if count != 1 else ''} matching workflow state '{label}'; execution status is unavailable."
                        if count
                        else "Declared by the active workflow plan; execution status is unavailable."
                    ),
                    artifact_count=count,
                )
            )
    grouped: dict[str, list[str]] = {}
    for key in artifact_keys:
        stage_key = artifact_stage_key(key, run_id, prefix, wrapper)
        grouped.setdefault(stage_key, []).append(key)
    for stage_key, matched in sorted(grouped.items()):
        if all(key in used_keys for key in matched):
            continue
        count = len(matched)
        stages.append(
            stage_evidence_record(
                stage_id=_slug(stage_key, fallback="artifacts"),
                label=artifact_stage_label(stage_key),
                stage_key=stage_key,
                status="observed_output",
                status_label="Observed output",
                evidence_type="artifact_observation",
                evidence_source="artifact_listing",
                authority="observed",
                confidence="high",
                reason=(
                    f"Observed {count} artifact{'s' if count != 1 else ''} under '{stage_key}'; "
                    "execution status is unavailable."
                ),
                artifact_count=count,
            )
        )
    return stages
