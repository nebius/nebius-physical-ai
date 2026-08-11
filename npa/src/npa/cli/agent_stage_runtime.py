"""Runtime adapters for evidence-backed agent stage responses.

This source is embedded into the generated agent backend after
``agent_stages.py``.  The evidence semantics stay pure in ``agent_stages``;
this module performs the run-scoped S3 reads and state integration.
"""

from __future__ import annotations

# This source is embedded into backend.py, where these adapter dependencies are
# defined by the surrounding generated module.
import json
import re
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

# NPA_EMBED_STANDALONE_START
# These adapter globals are intentionally supplied by the rendered backend. Use
# explicit standalone sentinels for direct helper tests instead of suppressing
# F821 for the entire embedded module.
if __name__ == "npa.cli.agent_stage_runtime":
    (
        ArtifactDiscoveryError,
        HTTPException,
        _agent_access_report,
        _agent_artifact_list_scope,
        _agent_s3_buckets,
        _agent_s3_client,
        _artifact_discovery_prefix,
        _discovery_exclude_roots,
        _load_selected_run_artifacts,
        _merge_sim2real_run_details,
        _now_iso,
        _slug,
        _validated_resolved_prefix,
        _workflow_draft_from_state,
        artifact_bucket_projects,
        build_artifact_backed_stages,
        coerce_authoritative_stage_evidence,
        find_run_artifacts,
        find_run_artifacts_across_buckets,
        list_artifacts,
        local_demo_run_details,
        merge_stage_evidence,
        parse_stage_evidence_documents,
        resolve_run_source,
        run_owns_workflow_stage_overlay,
        select_preferred_artifact,
        summarize_stage_evidence,
        validate_run_id,
    ) = (None,) * 28
# NPA_EMBED_STANDALONE_END


_MAX_STAGE_EVIDENCE_DOCUMENTS = 8
_MAX_STAGE_EVIDENCE_BYTES = 65_536


def _read_bounded_json_object(
    s3, bucket: str, key: str, *, max_bytes: int = _MAX_STAGE_EVIDENCE_BYTES
):
    """Read one JSON object with a hard byte bound and deterministic cleanup."""
    body = None
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        raw = body.read(max_bytes + 1)
        encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        if len(encoded) > max_bytes:
            return None
        payload = json.loads(encoded)
        return payload if isinstance(payload, dict) else None
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            try:
                close()
            except (OSError, RuntimeError, ValueError):
                # Cleanup must not turn an otherwise safely bounded read into a
                # request failure. StreamingBody.close() is best-effort here.
                pass


def _stage_evidence_candidate_rank(key: str) -> int | None:
    lower = str(key or "").lower()
    leaf = Path(lower).name
    if lower.endswith("/npa-workflow/status.json") or (
        "/logs/" in lower and lower.endswith("/status.json")
    ):
        return 0
    if lower.endswith("/npa-workflow/manifest.json"):
        return 1
    if leaf == "manifest.json":
        return 2
    if leaf.endswith("report.json"):
        return 3
    return None


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
    # Select the most authoritative typed candidates before any S3 GET. Reads are
    # bounded by both object count and bytes; parser order remains low-to-high
    # authority so status documents deterministically override snapshots.
    candidates = []
    for artifact in artifacts:
        key = str(getattr(artifact, "key", "") or "")
        rank = _stage_evidence_candidate_rank(key)
        if rank is None:
            continue
        size = int(getattr(artifact, "size", 0) or 0)
        if size > _MAX_STAGE_EVIDENCE_BYTES:
            continue
        candidates.append((rank, key))
    candidates.sort(key=lambda item: (item[0], item[1]))

    loaded = []
    for rank, key in candidates[:_MAX_STAGE_EVIDENCE_DOCUMENTS]:
        try:
            payload = _read_bounded_json_object(s3, bucket, key)
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
            loaded.append((rank, key, {"key": key, "payload": payload}))
    loaded.sort(key=lambda item: (-item[0], item[1]))
    return [document for _rank, _key, document in loaded]


_SENSITIVE_PUBLIC_MARKER = (
    r"(?:authorization|credentials?|password|passwd|private[_-]?key|"
    r"secret(?:[_-]?access)?(?:[_-]?key)?(?:[_-]?(?:id|value))?|"
    r"tokens?(?:[_-]?(?:id|value))?|api[_-]?key(?:[_-]?id)?|"
    r"access[_-]?key(?:[_-]?id)?)"
)
_SENSITIVE_PUBLIC_NAME = re.compile(
    rf"(?i)(?<![A-Za-z0-9]){_SENSITIVE_PUBLIC_MARKER}(?![A-Za-z0-9])"
)
_SENSITIVE_PUBLIC_VALUE = re.compile(
    rf"(?i)(?:authorization\s*:|bearer\s+|{_SENSITIVE_PUBLIC_MARKER}\s*(?:=|:|\s))"
)
_SENSITIVE_PUBLIC_NAME_TOKEN = (
    rf"(?:--?)?(?:[A-Za-z0-9]+[_.-])*{_SENSITIVE_PUBLIC_MARKER}"
)
_SENSITIVE_INLINE_ASSIGNMENT = re.compile(
    rf"(?i)(?P<name>(?<![A-Za-z0-9]){_SENSITIVE_PUBLIC_NAME_TOKEN})"
    r"(?P<separator>\s*(?:=|:)\s*|\s+)"
    r"(?P<secret>(?:bearer\s+)?[^\s]+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[^\s]+")
_BARE_SENSITIVE_ARG = re.compile(
    rf"(?i)^{_SENSITIVE_PUBLIC_NAME_TOKEN}$"
)
_EMPTY_SENSITIVE_SEPARATOR = re.compile(
    rf"(?i)^(?P<name>{_SENSITIVE_PUBLIC_NAME_TOKEN})(?P<separator>\s*(?:=|:)\s*)$"
)


def _redact_inline_workflow_secret(value: str) -> str:
    def replace_assignment(match: re.Match) -> str:
        secret = str(match.group("secret") or "")
        replacement = "Bearer <redacted>" if secret.lower().startswith("bearer ") else "<redacted>"
        return str(match.group("name")) + str(match.group("separator")) + replacement

    redacted = _SENSITIVE_INLINE_ASSIGNMENT.sub(replace_assignment, value)
    return _BEARER_SECRET.sub("Bearer <redacted>", redacted)


def _public_workflow_command(argv) -> str:
    # Render manifest argv without reflecting embedded or following secrets.
    # A bare sensitive option consumes its next argv item even when the secret
    # begins with "-"; completed inline assignments never create pending state.
    values = argv if isinstance(argv, list) else [argv]
    public = []
    pending = ""
    for raw in values:
        value = str(raw or "")
        if pending == "authorization" and value.lower() == "bearer":
            public.append("Bearer")
            pending = "secret"
            continue
        elif pending:
            public.append("<redacted>")
            pending = ""
            continue
        redacted = _redact_inline_workflow_secret(value)
        if redacted != value:
            public.append(redacted)
            continue
        empty_separator = _EMPTY_SENSITIVE_SEPARATOR.fullmatch(value)
        if empty_separator:
            public.append(value)
            marker = str(empty_separator.group("name")).lower().lstrip("-")
            separator = str(empty_separator.group("separator"))
            # A standalone name followed by ':' is commonly split from its
            # value by argv construction. An empty '=' assignment is already a
            # complete (empty) value and must not consume an unrelated positional.
            pending = (
                "authorization"
                if marker == "authorization" and ":" in separator
                else "secret"
                if ":" in separator
                else ""
            )
            continue
        if _BARE_SENSITIVE_ARG.fullmatch(value):
            public.append(value)
            pending = (
                "authorization" if value.lower().lstrip("-") == "authorization" else "secret"
            )
            continue
        if value.lower() == "bearer":
            public.append("Bearer")
            pending = "secret"
            continue
        public.append(value)
    return " ".join(public)[:2000]


def _public_url_without_credentials(value: str) -> str:
    return re.sub(
        r"(?i)(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@",
        r"\g<scheme><redacted>@",
        str(value or ""),
    )


def _public_workflow_output_uri(value: str) -> str:
    """Remove URL credentials and secret-bearing suffixes from public evidence."""
    public = _public_url_without_credentials(value).split("?", 1)[0].split("#", 1)[0]
    return _redact_inline_workflow_secret(public)


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
            output_uri = _public_workflow_output_uri(outputs[0].get("uri") or "")
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
    source_selected: bool = False,
) -> dict | None:
    if not run_id:
        return None
    exact_prefix = _validated_resolved_prefix(resolved_prefix)
    try:
        s3, settings = _agent_s3_client()
        artifacts = []
        run_bucket = settings["bucket"]
        access_report = _agent_access_report()
        bucket_projects = artifact_bucket_projects(access_report)
        if resource_bucket:
            run_bucket, selected_project, exact_prefix, artifacts = (
                _load_selected_run_artifacts(
                    s3=s3,
                    settings=settings,
                    run_id=run_id,
                    resource_bucket=resource_bucket,
                    project_id=project_id,
                    resolved_prefix=exact_prefix,
                    source_selected=source_selected,
                    exclude=_discovery_exclude_roots(),
                )
            )
            if selected_project:
                bucket_projects[run_bucket] = selected_project
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
    effective_prefix = exact_prefix if resource_bucket else (
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
        try:
            report = _read_bounded_json_object(s3, run_bucket, report_artifact.key)
        except (ClientError, BotoCoreError, OSError, KeyError, TypeError, ValueError):
            report = None
        if report:
            viz = report.get("visualization")
            viz = viz if isinstance(viz, dict) else {}
            outer_loop = report.get("outer_loop", {})
            decision = (
                outer_loop.get("latest_decision", {})
                if isinstance(outer_loop, dict)
                else {}
            )
            source = str(viz.get("source") or "").strip()
            success_rate = decision.get("success_rate")
            if source or success_rate is not None:
                report_note = (
                    "Report summary: visualization source="
                    + (source or "unknown")
                    + (f", success_rate={success_rate}" if success_rate is not None else "")
                    + "."
                )
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
    source_selected: bool = False,
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
    # A session-owned run already has its authoritative stage graph in local
    # state. Do not turn the default status poll into an all-bucket S3 search for
    # a just-submitted run that cannot have artifacts yet. Explicit source
    # selection still asks for artifact evidence, and artifact-only run IDs (no
    # local graph) continue through bounded discovery below.
    has_session_graph = bool(
        details
        and isinstance(details.get("stages"), list)
        and details.get("stages")
    )
    explicit_artifact_source = bool(
        prefix or resource_bucket or project_id or resolved_prefix or source_selected
    )
    artifact_details = None
    if explicit_artifact_source or not has_session_graph:
        artifact_details = _artifact_backed_run_details(
            state,
            resolved_run_id,
            prefix=prefix,
            resource_bucket=resource_bucket,
            project_id=project_id,
            resolved_prefix=resolved_prefix,
            source_selected=source_selected,
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
