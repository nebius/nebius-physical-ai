"""Runtime adapters for evidence-backed agent stage responses.

This source is embedded into the generated agent backend after
``agent_stages.py``.  The evidence semantics stay pure in ``agent_stages``;
this module performs the run-scoped S3 reads and state integration.
"""

from __future__ import annotations

# This source is embedded into backend.py, where these adapter dependencies are
# defined by the surrounding generated module.
# ruff: noqa: F821

import json
import re
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError


def _workflow_stage_defs_from_state(state: dict) -> list[tuple[str, str, list[str]]]:
    draft = _workflow_draft_from_state(state)
    stages: list[tuple[str, str, list[str]]] = []
    plan = draft.get("plan") if isinstance(draft.get("plan"), dict) else {}
    for source in (plan.get("steps"), plan.get("states"), draft.get("states")):
        if not isinstance(source, list):
            continue
        for item in source:
            if isinstance(item, dict):
                raw_id = str(item.get("state") or item.get("id") or item.get("name") or "").strip()
                label = str(item.get("label") or item.get("description") or raw_id).strip() or raw_id
            else:
                raw_id = str(item or "").strip()
                label = raw_id
            if not raw_id:
                continue
            stage_id = _slug(raw_id, fallback="stage")
            patterns = [raw_id, raw_id.replace("_", "-"), raw_id.replace("-", "_")]
            if (stage_id, label, patterns) not in stages:
                stages.append((stage_id, label, patterns))
        if stages:
            break
    return stages


def _stage_evidence_documents(s3, bucket: str, artifacts: list) -> list:
    # Read only typed manifest/status/report candidates from one resolved run.
    documents = []
    for artifact in artifacts:
        key = str(getattr(artifact, "key", "") or "")
        lower = key.lower()
        leaf = Path(key).name.lower()
        candidate = (
            lower.endswith("/npa-workflow/manifest.json")
            or lower.endswith("/npa-workflow/status.json")
            or ("/logs/" in lower and lower.endswith("/status.json"))
            or leaf == "manifest.json"
            or leaf.endswith("report.json")
        )
        if not candidate:
            continue
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            payload = json.loads(body)
        except (
            ClientError,
            BotoCoreError,
            OSError,
            KeyError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            continue
        if isinstance(payload, dict):
            documents.append({"key": key, "payload": payload})
    return documents


_SENSITIVE_WORKFLOW_ARG = re.compile(
    r"(?i)(?:authorization|credential|password|passwd|private[_-]?key|secret|token|api[_-]?key|access[_-]?key)"
)


def _public_workflow_command(argv) -> str:
    # Render manifest argv without reflecting embedded or following secrets.
    values = argv if isinstance(argv, list) else [argv]
    public = []
    redact_next = False
    for raw in values:
        value = str(raw or "")
        if redact_next:
            public.append("<redacted>")
            redact_next = False
            continue
        key, separator, _assigned = value.partition("=")
        if _SENSITIVE_WORKFLOW_ARG.search(key.lstrip("-")):
            if separator and key.strip():
                public.append(key + "=<redacted>")
            else:
                public.append(value)
                redact_next = True
            continue
        if value.lower().startswith("bearer "):
            public.append("<redacted>")
            continue
        public.append(value)
    return " ".join(public)[:2000]


def _workflow_run_steps(documents: list) -> list:
    # Project the npa.workflow run manifest into a backward-compatible execution
    # log surface. Stage cards themselves come from parse_stage_evidence_documents.
    manifest = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        if str(document.get("key") or "").endswith("/npa-workflow/manifest.json"):
            payload = document.get("payload")
            if isinstance(payload, dict):
                manifest = payload
                break
    if not manifest:
        return []
    out = []
    for step in manifest.get("steps", []) if isinstance(manifest, dict) else []:
        if not isinstance(step, dict):
            continue
        command = _public_workflow_command(step.get("argv") or [])
        outputs = step.get("outputs") or []
        output_uri = ""
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            output_uri = str(outputs[0].get("uri") or "").split("?", 1)[0].split("#", 1)[0]
        out.append(
            {
                "stage": str(step.get("state") or ""),
                "status": str(step.get("status") or ""),
                "returncode": step.get("returncode"),
                "iteration": step.get("iteration"),
                "command": command,
                "output_uri": output_uri,
            }
        )
    return out


def _artifact_backed_run_details(
    state: dict,
    run_id: str,
    prefix: str = "",
    *,
    resource_bucket: str = "",
    project_id: str = "",
    resolved_prefix: str = "",
) -> dict | None:
    if not run_id:
        return None
    try:
        s3, settings = _agent_s3_client()
        artifacts = []
        run_bucket = settings["bucket"]
        access_report = _agent_access_report()
        bucket_projects = artifact_bucket_projects(access_report)
        if resource_bucket:
            allowed_buckets, _selected_scope = _agent_artifact_list_scope(
                access_report, resource_bucket, project_id
            )
            run_bucket = str(resource_bucket).strip()
            if run_bucket not in allowed_buckets:
                raise HTTPException(
                    status_code=403,
                    detail="artifact bucket is outside effective agent access",
                )
            exact_prefix = str(resolved_prefix or "").strip().strip("/")
            if any(part in {"", ".", ".."} for part in exact_prefix.split("/")):
                if exact_prefix:
                    raise HTTPException(status_code=400, detail="invalid resolved artifact prefix")
            if exact_prefix:
                artifacts = list_artifacts(
                    run_bucket,
                    validate_run_id(run_id),
                    prefix=exact_prefix,
                    s3=s3,
                )
            if not artifacts:
                artifacts = find_run_artifacts(
                    run_bucket,
                    base_prefix=settings.get("prefix", ""),
                    run_id=validate_run_id(run_id),
                    s3=s3,
                )
        elif prefix:
            effective_prefix = _artifact_discovery_prefix(settings, prefix)
            artifacts = list_artifacts(
                settings["bucket"],
                validate_run_id(run_id),
                prefix=effective_prefix,
                s3=s3,
            )
        if not artifacts and not resource_bucket:
            run_bucket, artifacts = find_run_artifacts_across_buckets(
                _agent_s3_buckets(s3, settings),
                base_prefix=settings.get("prefix", ""),
                run_id=validate_run_id(run_id),
                s3=s3,
            )
    except HTTPException:
        raise
    except (
        ArtifactDiscoveryError,
        ClientError,
        BotoCoreError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None
    if not artifacts:
        return None
    keys = [str(item.key or "") for item in artifacts]
    marker = "/" + str(run_id) + "/"
    effective_prefix = (
        keys[0].split(marker, 1)[0]
        if marker in keys[0]
        else settings.get("prefix", "")
    )
    evidence_documents = _stage_evidence_documents(s3, run_bucket, artifacts)
    parsed_evidence = parse_stage_evidence_documents(evidence_documents)
    workflow_steps = _workflow_run_steps(evidence_documents)
    stages = build_artifact_backed_stages(
        keys,
        run_id=run_id,
        prefix=effective_prefix,
        workflow_stage_defs=_workflow_stage_defs_from_state(state),
        overlay_unmatched=run_owns_workflow_stage_overlay(state, run_id),
        authoritative_stages=parsed_evidence.get("stages", []),
        evidence_keys=[str(key) for key in parsed_evidence.get("consumed_sources", [])],
    )
    preferred = select_preferred_artifact(artifacts)
    report_note = ""
    report_artifact = next(
        (item for item in artifacts if item.key.endswith("/reports/sim2real-report.json")),
        None,
    )
    if report_artifact:
        local_report = RECORDINGS_DIR / (_artifact_filename(report_artifact.key) + ".json")
        try:
            download_s3_uri(report_artifact.s3_uri, local_report, s3=s3)
            report = json.loads(local_report.read_text(encoding="utf-8"))
            viz = report.get("visualization") if isinstance(report.get("visualization"), dict) else {}
            outer_loop = report.get("outer_loop", {})
            decision = outer_loop.get("latest_decision", {}) if isinstance(outer_loop, dict) else {}
            source = str(viz.get("source") or "").strip()
            success_rate = decision.get("success_rate")
            if source or success_rate is not None:
                report_note = (
                    "Report summary: visualization source="
                    + (source or "unknown")
                    + (f", success_rate={success_rate}" if success_rate is not None else "")
                    + "."
                )
        except Exception:
            report_note = ""
    stage_summary = summarize_stage_evidence(stages)
    authoritative_run_status = str(parsed_evidence.get("run_status") or "").strip()
    return {
        "run_id": run_id,
        "source_type": "artifact_storage",
        "source_label": "S3 artifacts",
        "project_id": str(bucket_projects.get(run_bucket) or project_id or ""),
        "bucket": run_bucket,
        "resolved_prefix": effective_prefix,
        "workflow_name": str(parsed_evidence.get("workflow_name") or ""),
        "workflow_graph_source": str(parsed_evidence.get("graph_source") or ""),
        "status": authoritative_run_status or "status_unavailable",
        "status_label": str(parsed_evidence.get("run_status_label") or "Status unavailable"),
        "status_source": str(parsed_evidence.get("run_status_source") or ""),
        "result": "artifacts_available",
        "submitted_at": "",
        "updated_at": str(parsed_evidence.get("updated_at") or "")
        or max((str(item.last_modified or "") for item in artifacts), default=_now_iso()),
        "selection": {},
        "stages": stages,
        "stage_summary": stage_summary,
        "artifact_count": len(artifacts),
        "workflow_steps": workflow_steps,
        "logs": [
            {
                "timestamp": _now_iso(),
                "level": "info",
                "message": (
                    f"Observed {len(artifacts)} S3 artifacts across "
                    f"{stage_summary.get('observed_stage_count', 0)} logical groups; "
                    "artifact presence does not establish execution success."
                ),
            },
            *[
                {
                    "timestamp": _now_iso(),
                    "level": "info"
                    if str(step.get("status") or "") in ("ok", "succeeded", "")
                    and step.get("returncode") in (0, None)
                    else "error",
                    "message": (
                        f"[{step.get('stage') or '?'}"
                        + (
                            f" #{step.get('iteration')}"
                            if step.get("iteration") not in (None, "")
                            else ""
                        )
                        + f"] rc={step.get('returncode')} ({step.get('status') or 'n/a'}) "
                        + f"$ {step.get('command') or ''}"
                    ),
                }
                for step in workflow_steps
            ],
            {
                "timestamp": _now_iso(),
                "level": "info",
                "message": (
                    f"Preferred viewable artifact: {preferred.key}"
                    if preferred
                    else "No preferred viewable artifact was observed."
                ),
            },
            {
                "timestamp": _now_iso(),
                "level": "info",
                "message": report_note or "No structured run report summary was available.",
            },
        ],
        "artifacts": [item.to_dict() for item in artifacts[:25]],
    }


def _sim2real_run_details(
    state: dict,
    run_id: str = "",
    prefix: str = "",
    *,
    resource_bucket: str = "",
    project_id: str = "",
    resolved_prefix: str = "",
) -> dict:
    latest = state.get("latest_submit", {})
    if not isinstance(latest, dict):
        latest = {}
    sim_viz = state.get("sim_viz", {})
    if not isinstance(sim_viz, dict):
        sim_viz = {}
    resolved_run_id = str(
        run_id
        or latest.get("run_id")
        or sim_viz.get("run_id")
        or state.get("active_run_id")
        or ""
    ).strip()
    history = state.get("sim_viz_runs") if isinstance(state.get("sim_viz_runs"), dict) else {}
    recorded = history.get(resolved_run_id) if isinstance(history.get(resolved_run_id), dict) else {}
    run_viz = dict(recorded)
    if not run_viz and str(sim_viz.get("run_id") or "").strip() == resolved_run_id:
        run_viz = dict(sim_viz)
    local_demo = local_demo_run_details(state, resolved_run_id, run_viz, _now_iso())
    if local_demo:
        return local_demo
    details_map = state.get("sim2real_runs")
    if not isinstance(details_map, dict):
        details_map = {}
    existing = details_map.get(resolved_run_id, {}) if resolved_run_id else {}
    details = dict(existing) if isinstance(existing, dict) else {}
    if details and isinstance(details.get("stages"), list):
        details["stages"] = coerce_authoritative_stage_evidence(
            details["stages"], source="agent_session_workflow_status"
        )
        details["stage_summary"] = summarize_stage_evidence(details["stages"])
    artifact_details = _artifact_backed_run_details(
        state,
        resolved_run_id,
        prefix=prefix,
        resource_bucket=resource_bucket,
        project_id=project_id,
        resolved_prefix=resolved_prefix,
    )
    if artifact_details:
        authoritative_existing = [
            item
            for item in details.get("stages", [])
            if isinstance(item, dict)
            and str(
                item.get("authority")
                or (item.get("evidence") or {}).get("authority")
                or ""
            )
            == "authoritative"
        ]
        artifact_stages = [
            item for item in artifact_details.get("stages", []) if isinstance(item, dict)
        ]
        if authoritative_existing:
            artifact_details["stages"] = merge_stage_evidence(
                authoritative_existing, artifact_stages
            )
            artifact_details["stage_summary"] = summarize_stage_evidence(
                artifact_details["stages"]
            )
            if str(artifact_details.get("status") or "") == "status_unavailable":
                artifact_details["status"] = str(details.get("status") or "status_unavailable")
                artifact_details["status_label"] = str(
                    details.get("status_label")
                    or artifact_details.get("status_label")
                    or "Status unavailable"
                )
        details = _merge_sim2real_run_details(details, artifact_details)
    if not details:
        source_type, source_label = resolve_run_source(recorded, {}, resolved_run_id)
        details = {
            "run_id": resolved_run_id,
            "source_type": source_type,
            "source_label": source_label,
            "status": "status_unavailable",
            "status_label": "Status unavailable",
            "result": "unavailable",
            "submitted_at": "",
            "updated_at": str(recorded.get("rrd_updated_at") or ""),
            "selection": {},
            "stages": [],
            "stage_summary": summarize_stage_evidence([]),
            "logs": [
                {
                    "timestamp": _now_iso(),
                    "level": "info",
                    "message": (
                        "No authoritative workflow status or artifact-stage evidence "
                        "is available for this run."
                    ),
                }
            ],
            "artifacts": [],
        }
    details["run_id"] = resolved_run_id
    if run_viz.get("rrd_uri"):
        if str(details.get("result") or "") in {"", "unavailable", "recorded_not_launched"}:
            details["result"] = "recording_observed"
        for item in details.get("stages", []):
            if isinstance(item, dict) and item.get("id") == "stage_14_rerun_viz":
                item["artifact_count"] = max(1, int(item.get("artifact_count") or 0))
                observation = {
                    "type": "artifact_observation",
                    "source": "sim_viz_recording",
                    "authority": "observed",
                    "confidence": "high",
                    "reason": (
                        "A Rerun recording was observed; this alone does not establish "
                        "stage success."
                    ),
                    "observed_at": str(run_viz.get("rrd_updated_at") or ""),
                }
                item["observations"] = [
                    *[
                        entry
                        for entry in item.get("observations", [])
                        if isinstance(entry, dict)
                    ],
                    observation,
                ]
                if str(item.get("authority") or "") != "authoritative":
                    item["status"] = "observed_output"
                    item["status_label"] = "Observed output"
                    item["evidence"] = observation
                    item["evidence_type"] = observation["type"]
                    item["evidence_source"] = observation["source"]
                    item["authority"] = observation["authority"]
                    item["confidence"] = observation["confidence"]
                    item["diagnostic_reason"] = observation["reason"]
                    item["summary"] = observation["reason"]
    details["stage_summary"] = summarize_stage_evidence(
        [item for item in details.get("stages", []) if isinstance(item, dict)]
    )
    return details
