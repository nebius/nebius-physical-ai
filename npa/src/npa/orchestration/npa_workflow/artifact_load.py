"""Restart-safe PAIDF final-artifact handoff to a configured NPA agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Callable
from urllib.parse import quote, urlencode


FINAL_RERUN_KEY = "reports/sim2real.rrd"
logger = logging.getLogger(__name__)


class ArtifactLoadError(RuntimeError):
    """Raised when the exact run artifact cannot be discovered or verified."""


@dataclass(frozen=True)
class ArtifactLoadResult:
    status: str
    artifact_uri: str = ""
    artifact_render: str = ""
    agent_name: str = ""
    verified: bool = False
    detail: str = ""
    retry_command: str = ""
    posted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _AgentArtifactSelection:
    run_id: str
    run_ref: str
    key: str
    artifact_uri: str
    project_id: str
    resource_bucket: str
    resolved_prefix: str

    def load_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_ref": self.run_ref,
            "key": self.key,
            "project_id": self.project_id,
            "resource_bucket": self.resource_bucket,
            "resolved_prefix": self.resolved_prefix,
            "source_selected": True,
        }


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    value = str(uri or "").strip().rstrip("/")
    if not value.startswith("s3://"):
        raise ArtifactLoadError(f"Expected an s3:// run prefix, got {value or '<empty>'}")
    bucket_and_key = value.removeprefix("s3://").split("/", 1)
    if len(bucket_and_key) != 2 or not all(bucket_and_key):
        raise ArtifactLoadError(f"Expected s3://<bucket>/<run-prefix>, got {value}")
    return bucket_and_key[0], bucket_and_key[1]


def discover_final_rerun_artifact(run_prefix_uri: str, *, client: Any) -> str:
    """Discover a final Rerun object strictly below one exact workflow prefix."""

    bucket, prefix = _parse_s3_uri(run_prefix_uri)
    exact_key = f"{prefix.rstrip('/')}/{FINAL_RERUN_KEY}"
    try:
        client.s3.head_object(Bucket=bucket, Key=exact_key)
        return f"s3://{bucket}/{exact_key}"
    except Exception:  # noqa: BLE001 - fall back to final-report discovery
        logger.debug("Exact PAIDF Rerun object is unavailable; listing reports", exc_info=True)
    report_prefix = f"{prefix.rstrip('/')}/reports/"
    try:
        paginator = client.s3.get_paginator("list_objects_v2")
        keys = sorted(
            str(item.get("Key") or "")
            for page in paginator.paginate(Bucket=bucket, Prefix=report_prefix)
            for item in page.get("Contents", [])
            if str(item.get("Key") or "").endswith(".rrd")
        )
    except Exception as exc:  # noqa: BLE001 - include provider detail, never credentials
        raise ArtifactLoadError(
            f"Could not discover a final Rerun artifact below "
            f"s3://{bucket}/{report_prefix}: {exc}"
        ) from exc
    if not keys:
        raise ArtifactLoadError(
            f"Workflow succeeded but no .rrd artifact exists below "
            f"s3://{bucket}/{report_prefix}"
        )
    return f"s3://{bucket}/{keys[-1]}"


def _status_matches(
    payload: object,
    artifact_uri: str,
    selection: _AgentArtifactSelection | None = None,
) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, ""
    render = str(payload.get("artifact_render") or payload.get("render") or "").lower()
    actual = str(payload.get("artifact_uri") or "")
    ready = bool(payload.get("rerun_ready"))
    exact_source = True
    if selection is not None:
        exact_source = bool(
            str(payload.get("run_id") or "") == selection.run_id
            and str(payload.get("artifact_key") or "") == selection.key
            and str(payload.get("artifact_run_ref") or "") == selection.run_ref
            and str(payload.get("project_id") or "") == selection.project_id
            and str(payload.get("bucket") or "") == selection.resource_bucket
            and "resolved_prefix" in payload
            and str(payload.get("resolved_prefix") or "")
            == selection.resolved_prefix
        )
    return (
        actual == artifact_uri and render == "rerun" and ready and exact_source,
        render,
    )


def _request_agent_json(
    request: Callable[..., Any],
    method: str,
    url: str,
    *,
    auth: tuple[str, str],
    verify: bool,
    operation: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "auth": auth,
        "timeout": timeout,
        "verify": verify,
    }
    if payload is not None:
        kwargs["json"] = payload
    try:
        response = request(method, url, **kwargs)
    except Exception as exc:  # noqa: BLE001 - sanitize URL/auth-bearing client errors
        raise ArtifactLoadError(f"agent {operation} request failed") from exc
    status_code = int(getattr(response, "status_code", 0))
    if status_code != 200:
        raise ArtifactLoadError(
            f"agent {operation} returned HTTP {status_code or 'unknown'}"
        )
    try:
        response_payload = response.json()
    except Exception as exc:  # noqa: BLE001 - third-party response object
        raise ArtifactLoadError(f"agent {operation} returned invalid JSON") from exc
    if not isinstance(response_payload, dict):
        raise ArtifactLoadError(f"agent {operation} returned a non-object response")
    return response_payload


def _agent_source_tuple(
    payload: dict[str, Any], *, run_id: str
) -> tuple[str, str, str, str]:
    run_ref = str(payload.get("run_ref") or "").strip()
    project_id = str(payload.get("project_id") or "").strip()
    bucket = str(payload.get("resource_bucket") or payload.get("bucket") or "").strip()
    if (
        str(payload.get("run_id") or "").strip() != run_id
        or not run_ref
        or not project_id
        or not bucket
        or "resolved_prefix" not in payload
        or payload.get("source_selected") is not True
    ):
        raise ArtifactLoadError(
            "agent artifact lookup did not return one exact server-selected source"
        )
    if str(payload.get("bucket") or "").strip() not in {"", bucket}:
        raise ArtifactLoadError("agent artifact lookup returned conflicting buckets")
    return run_ref, project_id, bucket, str(payload.get("resolved_prefix") or "").strip()


def _discover_agent_selection(
    *,
    request: Callable[..., Any],
    base_url: str,
    auth: tuple[str, str],
    verify: bool,
    run_id: str,
    artifact_uri: str,
) -> _AgentArtifactSelection:
    expected_bucket, expected_key = _parse_s3_uri(artifact_uri)
    run_selector = quote(run_id, safe="")
    page = _request_agent_json(
        request,
        "GET",
        f"{base_url}/api/artifacts/run/{run_selector}",
        auth=auth,
        verify=verify,
        operation="artifact lookup",
        timeout=60.0,
    )
    source = _agent_source_tuple(page, run_id=run_id)
    run_ref, project_id, resource_bucket, resolved_prefix = source
    if resource_bucket != expected_bucket:
        raise ArtifactLoadError(
            "agent artifact lookup resolved a different storage source"
        )

    matches: list[dict[str, Any]] = []
    seen_cursors: set[str] = set()
    while True:
        if _agent_source_tuple(page, run_id=run_id) != source:
            raise ArtifactLoadError(
                "agent artifact pagination changed the selected source"
            )
        for item in page.get("artifacts") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("key") or "") != expected_key:
                continue
            if str(item.get("s3_uri") or "") != artifact_uri:
                raise ArtifactLoadError(
                    "agent inventory key has conflicting storage provenance"
                )
            matches.append(item)
        cursor = str(page.get("next_cursor") or "")
        if not cursor:
            if page.get("truncated") is True:
                raise ArtifactLoadError(
                    "agent artifact inventory ended before pagination completed"
                )
            break
        if cursor in seen_cursors:
            raise ArtifactLoadError("agent artifact inventory repeated its cursor")
        seen_cursors.add(cursor)
        query = urlencode(
            {
                "project_id": project_id,
                "resource_bucket": resource_bucket,
                "resolved_prefix": resolved_prefix,
                "source_selected": "1",
                "cursor": cursor,
            }
        )
        page = _request_agent_json(
            request,
            "GET",
            f"{base_url}/api/artifacts/run/{quote(run_ref, safe='')}?{query}",
            auth=auth,
            verify=verify,
            operation="artifact inventory continuation",
            timeout=60.0,
        )
    if len(matches) != 1:
        raise ArtifactLoadError(
            "agent inventory did not contain one exact final Rerun artifact"
        )
    return _AgentArtifactSelection(
        run_id=run_id,
        run_ref=run_ref,
        key=expected_key,
        artifact_uri=artifact_uri,
        project_id=project_id,
        resource_bucket=resource_bucket,
        resolved_prefix=resolved_prefix,
    )


def _retry_command(run_id: str, project: str, agent_name: str) -> str:
    return (
        f"npa workbench workflow load-artifact {run_id}"
        + (f" --project {project}" if project else "")
        + (f" --agent-name {agent_name}" if agent_name else "")
    )


def load_final_artifact_into_agent(
    *,
    project: str,
    run_id: str,
    run_prefix_uri: str,
    storage_client: Any,
    agent_name: str = "",
    http_request: Callable[..., Any] | None = None,
) -> ArtifactLoadResult:
    """Load and verify the final PAIDF recording, returning a partial on agent errors.

    A successful data workflow remains successful even when the optional agent is
    absent or unreachable.  Credentials are loaded only for HTTP auth and never
    enter the returned payload or the durable submission state.
    """

    from npa.cli.agent import (
        _agent_record,
        _load_auth_secret,
        _record_tls_verify,
        resolve_project_agents,
    )
    from npa.orchestration.npa_workflow.submission_state import update_submission_state

    retry = _retry_command(run_id, project, agent_name)
    try:
        artifact_uri = discover_final_rerun_artifact(run_prefix_uri, client=storage_client)
    except ArtifactLoadError as exc:
        result = ArtifactLoadResult(status="partial", detail=str(exc), retry_command=retry)
        update_submission_state(project or "default", run_id, {"artifact_load": result.to_dict()})
        return result

    agents = resolve_project_agents(project) if project else {}
    selected = str(agent_name or "").strip()
    if not selected:
        if "agent" in agents:
            selected = "agent"
        elif len(agents) == 1:
            selected = str(next(iter(agents)))
    record = _agent_record(project, selected) if selected else {}
    base_url = str(record.get("agent_url") or "").rstrip("/")
    if not selected or not base_url:
        retry = _retry_command(run_id, project, selected)
        result = ArtifactLoadResult(
            status="partial",
            artifact_uri=artifact_uri,
            agent_name=selected,
            detail="workflow succeeded; no configured agent is available for artifact loading",
            retry_command=retry,
        )
        update_submission_state(project or "default", run_id, {"artifact_load": result.to_dict()})
        return result

    try:
        auth = _load_auth_secret(str(record.get("auth_secret_path") or ""))
        verify = _record_tls_verify(record)
        if http_request is None:
            import httpx

            request = httpx.request
        else:
            request = http_request
        selection = _discover_agent_selection(
            request=request,
            base_url=base_url,
            auth=auth,
            verify=verify,
            run_id=run_id,
            artifact_uri=artifact_uri,
        )
        status_url = f"{base_url}/api/sim-viz/status"
        status_payload = _request_agent_json(
            request,
            "GET",
            status_url,
            auth=auth,
            verify=verify,
            operation="sim-viz status",
            timeout=10.0,
        )
        matches, render = _status_matches(
            status_payload, artifact_uri, selection
        )
        if matches:
            result = ArtifactLoadResult(
                status="verified",
                artifact_uri=artifact_uri,
                artifact_render=render,
                agent_name=selected,
                verified=True,
                retry_command=retry,
                posted=False,
            )
            update_submission_state(
                project or "default", run_id, {"artifact_load": result.to_dict()}
            )
            return result
        loaded = _request_agent_json(
            request,
            "POST",
            f"{base_url}/api/sim-viz/load-artifact",
            payload=selection.load_payload(),
            auth=auth,
            verify=verify,
            operation="load-artifact",
            timeout=60.0,
        )
        loaded_sim_viz = loaded.get("sim_viz")
        load_matches, _load_render = _status_matches(
            loaded_sim_viz, artifact_uri, selection
        )
        if loaded.get("ok") is not True or not load_matches:
            raise ArtifactLoadError(
                "agent load-artifact did not confirm the exact selected recording"
            )
        verify_payload = _request_agent_json(
            request,
            "GET",
            status_url,
            auth=auth,
            verify=verify,
            operation="sim-viz verification",
            timeout=10.0,
        )
        matches, render = _status_matches(
            verify_payload, artifact_uri, selection
        )
        if not matches:
            raise ArtifactLoadError(
                "agent sim-viz status did not verify the exact selected recording"
            )
        result = ArtifactLoadResult(
            status="verified",
            artifact_uri=artifact_uri,
            artifact_render=render,
            agent_name=selected,
            verified=True,
            retry_command=retry,
            posted=True,
        )
    except Exception as exc:  # noqa: BLE001 - optional post-success side effect
        result = ArtifactLoadResult(
            status="partial",
            artifact_uri=artifact_uri,
            agent_name=selected,
            detail=f"workflow succeeded; artifact load/verification is incomplete: {exc}",
            retry_command=retry,
        )
    update_submission_state(project or "default", run_id, {"artifact_load": result.to_dict()})
    return result
