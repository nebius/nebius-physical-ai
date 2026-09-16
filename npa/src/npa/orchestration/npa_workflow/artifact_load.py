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
        """Build the exact server-selected source payload for artifact loading.

        Args:
            None.

        Returns:
            The run, artifact key, and complete source tuple accepted by the agent.

        Raises:
            None.
        """
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
        raise ArtifactLoadError(
            f"Expected an s3:// run prefix, got {value or '<empty>'}"
        )
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
        logger.debug(
            "Exact PAIDF Rerun object is unavailable; listing reports", exc_info=True
        )
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
            and str(payload.get("resolved_prefix") or "") == selection.resolved_prefix
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
    return (
        run_ref,
        project_id,
        bucket,
        str(payload.get("resolved_prefix") or "").strip(),
    )


def _initial_agent_artifact_page(
    *,
    request: Callable[..., Any],
    base_url: str,
    auth: tuple[str, str],
    verify: bool,
    run_id: str,
) -> dict[str, Any]:
    run_selector = quote(run_id, safe="")
    return _request_agent_json(
        request,
        "GET",
        f"{base_url}/api/artifacts/run/{run_selector}",
        auth=auth,
        verify=verify,
        operation="artifact lookup",
        timeout=60.0,
    )


def _expected_agent_source(
    page: dict[str, Any], run_id: str, expected_bucket: str
) -> tuple[str, str, str, str]:
    source = _agent_source_tuple(page, run_id=run_id)
    if source[2] != expected_bucket:
        raise ArtifactLoadError(
            "agent artifact lookup resolved a different storage source"
        )
    return source


def _matching_agent_artifacts(
    page: dict[str, Any], expected_key: str, artifact_uri: str
) -> list[dict[str, Any]]:
    matches = []
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
    return matches


def _next_agent_artifact_cursor(page: dict[str, Any], seen: set[str]) -> str:
    cursor = str(page.get("next_cursor") or "")
    if not cursor:
        if page.get("truncated") is True:
            raise ArtifactLoadError(
                "agent artifact inventory ended before pagination completed"
            )
        return ""
    if cursor in seen:
        raise ArtifactLoadError("agent artifact inventory repeated its cursor")
    seen.add(cursor)
    return cursor


def _continued_agent_artifact_page(
    *,
    request: Callable[..., Any],
    base_url: str,
    auth: tuple[str, str],
    verify: bool,
    run_ref: str,
    project_id: str,
    resource_bucket: str,
    resolved_prefix: str,
    cursor: str,
) -> dict[str, Any]:
    query = urlencode(
        {
            "project_id": project_id,
            "resource_bucket": resource_bucket,
            "resolved_prefix": resolved_prefix,
            "source_selected": "1",
            "cursor": cursor,
        }
    )
    return _request_agent_json(
        request,
        "GET",
        f"{base_url}/api/artifacts/run/{quote(run_ref, safe='')}?{query}",
        auth=auth,
        verify=verify,
        operation="artifact inventory continuation",
        timeout=60.0,
    )


def _agent_artifact_matches(
    *,
    request: Callable[..., Any],
    base_url: str,
    auth: tuple[str, str],
    verify: bool,
    run_id: str,
    artifact_uri: str,
    expected_key: str,
    source: tuple[str, str, str, str],
    first_page: dict[str, Any],
) -> list[dict[str, Any]]:
    run_ref, project_id, resource_bucket, resolved_prefix = source
    matches: list[dict[str, Any]] = []
    seen_cursors: set[str] = set()
    page = first_page
    while True:
        if _agent_source_tuple(page, run_id=run_id) != source:
            raise ArtifactLoadError(
                "agent artifact pagination changed the selected source"
            )
        matches.extend(_matching_agent_artifacts(page, expected_key, artifact_uri))
        cursor = _next_agent_artifact_cursor(page, seen_cursors)
        if not cursor:
            return matches
        page = _continued_agent_artifact_page(
            request=request,
            base_url=base_url,
            auth=auth,
            verify=verify,
            run_ref=run_ref,
            project_id=project_id,
            resource_bucket=resource_bucket,
            resolved_prefix=resolved_prefix,
            cursor=cursor,
        )


def _final_agent_artifact_selection(
    *,
    run_id: str,
    expected_key: str,
    artifact_uri: str,
    source: tuple[str, str, str, str],
    matches: list[dict[str, Any]],
) -> _AgentArtifactSelection:
    if len(matches) != 1:
        raise ArtifactLoadError(
            "agent inventory did not contain one exact final Rerun artifact"
        )
    run_ref, project_id, resource_bucket, resolved_prefix = source
    return _AgentArtifactSelection(
        run_id=run_id,
        run_ref=run_ref,
        key=expected_key,
        artifact_uri=artifact_uri,
        project_id=project_id,
        resource_bucket=resource_bucket,
        resolved_prefix=resolved_prefix,
    )


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
    page = _initial_agent_artifact_page(
        request=request,
        base_url=base_url,
        auth=auth,
        verify=verify,
        run_id=run_id,
    )
    source = _expected_agent_source(page, run_id, expected_bucket)
    matches = _agent_artifact_matches(
        request=request,
        base_url=base_url,
        auth=auth,
        verify=verify,
        run_id=run_id,
        artifact_uri=artifact_uri,
        expected_key=expected_key,
        source=source,
        first_page=page,
    )
    return _final_agent_artifact_selection(
        run_id=run_id,
        expected_key=expected_key,
        artifact_uri=artifact_uri,
        source=source,
        matches=matches,
    )


def _retry_command(run_id: str, project: str, agent_name: str) -> str:
    return (
        f"npa workbench workflow load-artifact {run_id}"
        + (f" --project {project}" if project else "")
        + (f" --agent-name {agent_name}" if agent_name else "")
    )


@dataclass(frozen=True)
class _ArtifactLoadRuntime:
    agent_record: Callable[..., dict[str, Any]]
    load_auth_secret: Callable[[str], tuple[str, str]]
    record_tls_verify: Callable[[dict[str, Any]], bool]
    resolve_project_agents: Callable[[str], dict[str, Any]]
    update_submission_state: Callable[[str, str, dict[str, Any]], Any]


@dataclass(frozen=True)
class _AgentConnection:
    base_url: str
    auth: tuple[str, str]
    verify: bool
    request: Callable[..., Any]


def _artifact_load_runtime() -> _ArtifactLoadRuntime:
    from npa.cli.agent import (
        _agent_record,
        _load_auth_secret,
        _record_tls_verify,
        resolve_project_agents,
    )
    from npa.orchestration.npa_workflow.submission_state import update_submission_state

    return _ArtifactLoadRuntime(
        agent_record=_agent_record,
        load_auth_secret=_load_auth_secret,
        record_tls_verify=_record_tls_verify,
        resolve_project_agents=resolve_project_agents,
        update_submission_state=update_submission_state,
    )


def _persist_artifact_load_result(
    runtime: _ArtifactLoadRuntime,
    project: str,
    run_id: str,
    result: ArtifactLoadResult,
) -> ArtifactLoadResult:
    runtime.update_submission_state(
        project or "default", run_id, {"artifact_load": result.to_dict()}
    )
    return result


def _discover_handoff_artifact(
    run_prefix_uri: str, storage_client: Any, retry: str
) -> tuple[str, ArtifactLoadResult | None]:
    try:
        return discover_final_rerun_artifact(
            run_prefix_uri, client=storage_client
        ), None
    except ArtifactLoadError as exc:
        return "", ArtifactLoadResult(
            status="partial", detail=str(exc), retry_command=retry
        )


def _selected_agent_endpoint(
    runtime: _ArtifactLoadRuntime, project: str, agent_name: str
) -> tuple[str, dict[str, Any], str]:
    agents = runtime.resolve_project_agents(project) if project else {}
    selected = str(agent_name or "").strip()
    if not selected and "agent" in agents:
        selected = "agent"
    if not selected and len(agents) == 1:
        selected = str(next(iter(agents)))
    record = runtime.agent_record(project, selected) if selected else {}
    return selected, record, str(record.get("agent_url") or "").rstrip("/")


def _unavailable_agent_result(
    project: str, run_id: str, artifact_uri: str, agent_name: str
) -> ArtifactLoadResult:
    return ArtifactLoadResult(
        status="partial",
        artifact_uri=artifact_uri,
        agent_name=agent_name,
        detail="workflow succeeded; no configured agent is available for artifact loading",
        retry_command=_retry_command(run_id, project, agent_name),
    )


def _agent_connection(
    runtime: _ArtifactLoadRuntime,
    record: dict[str, Any],
    base_url: str,
    http_request: Callable[..., Any] | None,
) -> _AgentConnection:
    auth = runtime.load_auth_secret(str(record.get("auth_secret_path") or ""))
    verify = runtime.record_tls_verify(record)
    if http_request is None:
        import httpx

        request = httpx.request
    else:
        request = http_request
    return _AgentConnection(base_url, auth, verify, request)


def _sim_viz_status(connection: _AgentConnection, operation: str) -> dict[str, Any]:
    return _request_agent_json(
        connection.request,
        "GET",
        f"{connection.base_url}/api/sim-viz/status",
        auth=connection.auth,
        verify=connection.verify,
        operation=operation,
        timeout=10.0,
    )


def _verified_artifact_result(
    artifact_uri: str,
    render: str,
    agent_name: str,
    retry: str,
    *,
    posted: bool,
) -> ArtifactLoadResult:
    return ArtifactLoadResult(
        status="verified",
        artifact_uri=artifact_uri,
        artifact_render=render,
        agent_name=agent_name,
        verified=True,
        retry_command=retry,
        posted=posted,
    )


def _existing_agent_artifact_result(
    connection: _AgentConnection,
    selection: _AgentArtifactSelection,
    agent_name: str,
    retry: str,
) -> ArtifactLoadResult | None:
    payload = _sim_viz_status(connection, "sim-viz status")
    matches, render = _status_matches(payload, selection.artifact_uri, selection)
    if not matches:
        return None
    return _verified_artifact_result(
        selection.artifact_uri, render, agent_name, retry, posted=False
    )


def _load_selected_agent_artifact(
    connection: _AgentConnection, selection: _AgentArtifactSelection
) -> None:
    loaded = _request_agent_json(
        connection.request,
        "POST",
        f"{connection.base_url}/api/sim-viz/load-artifact",
        payload=selection.load_payload(),
        auth=connection.auth,
        verify=connection.verify,
        operation="load-artifact",
        timeout=60.0,
    )
    matches, _render = _status_matches(
        loaded.get("sim_viz"), selection.artifact_uri, selection
    )
    if loaded.get("ok") is not True or not matches:
        raise ArtifactLoadError(
            "agent load-artifact did not confirm the exact selected recording"
        )


def _verified_agent_render(
    connection: _AgentConnection, selection: _AgentArtifactSelection
) -> str:
    payload = _sim_viz_status(connection, "sim-viz verification")
    matches, render = _status_matches(payload, selection.artifact_uri, selection)
    if not matches:
        raise ArtifactLoadError(
            "agent sim-viz status did not verify the exact selected recording"
        )
    return render


def _complete_agent_handoff(
    connection: _AgentConnection,
    run_id: str,
    artifact_uri: str,
    agent_name: str,
    retry: str,
) -> ArtifactLoadResult:
    selection = _discover_agent_selection(
        request=connection.request,
        base_url=connection.base_url,
        auth=connection.auth,
        verify=connection.verify,
        run_id=run_id,
        artifact_uri=artifact_uri,
    )
    existing = _existing_agent_artifact_result(connection, selection, agent_name, retry)
    if existing is not None:
        return existing
    _load_selected_agent_artifact(connection, selection)
    render = _verified_agent_render(connection, selection)
    return _verified_artifact_result(
        artifact_uri, render, agent_name, retry, posted=True
    )


def _partial_agent_handoff(
    artifact_uri: str, agent_name: str, retry: str, error: Exception
) -> ArtifactLoadResult:
    return ArtifactLoadResult(
        status="partial",
        artifact_uri=artifact_uri,
        agent_name=agent_name,
        detail=f"workflow succeeded; artifact load/verification is incomplete: {error}",
        retry_command=retry,
    )


def _load_final_artifact_into_agent(
    *,
    project: str,
    run_id: str,
    run_prefix_uri: str,
    storage_client: Any,
    agent_name: str,
    http_request: Callable[..., Any] | None,
    runtime: _ArtifactLoadRuntime,
) -> ArtifactLoadResult:
    retry = _retry_command(run_id, project, agent_name)
    artifact_uri, discovery_failure = _discover_handoff_artifact(
        run_prefix_uri, storage_client, retry
    )
    if discovery_failure is not None:
        return _persist_artifact_load_result(
            runtime, project, run_id, discovery_failure
        )
    selected, record, base_url = _selected_agent_endpoint(runtime, project, agent_name)
    if not selected or not base_url:
        result = _unavailable_agent_result(project, run_id, artifact_uri, selected)
        return _persist_artifact_load_result(runtime, project, run_id, result)
    try:
        connection = _agent_connection(runtime, record, base_url, http_request)
        result = _complete_agent_handoff(
            connection, run_id, artifact_uri, selected, retry
        )
    except Exception as exc:  # noqa: BLE001 - optional post-success side effect
        result = _partial_agent_handoff(artifact_uri, selected, retry, exc)
    return _persist_artifact_load_result(runtime, project, run_id, result)


def load_final_artifact_into_agent(
    *,
    project: str,
    run_id: str,
    run_prefix_uri: str,
    storage_client: Any,
    agent_name: str = "",
    http_request: Callable[..., Any] | None = None,
) -> ArtifactLoadResult:
    """Load the final workflow recording, returning partial on agent errors.

    Args:
        project: The configured project alias that owns the workflow run.
        run_id: The completed workflow run identifier.
        run_prefix_uri: The exact durable S3 prefix for the workflow run.
        storage_client: The authorized storage client used for final discovery.
        agent_name: An optional configured agent name to select explicitly.
        http_request: An optional HTTP transport used for agent requests.

    Returns:
        A verified result, or a partial result describing recoverable handoff failure.

    Raises:
        Exception: If configuration lookup or submission-state persistence fails.
    """
    return _load_final_artifact_into_agent(
        project=project,
        run_id=run_id,
        run_prefix_uri=run_prefix_uri,
        storage_client=storage_client,
        agent_name=agent_name,
        http_request=http_request,
        runtime=_artifact_load_runtime(),
    )
