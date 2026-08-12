"""Restart-safe PAIDF final-artifact handoff to a configured NPA agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Callable


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


def _status_matches(payload: object, artifact_uri: str) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, ""
    render = str(payload.get("artifact_render") or payload.get("render") or "").lower()
    actual = str(payload.get("artifact_uri") or "")
    ready = bool(payload.get("rerun_ready"))
    return actual == artifact_uri and render == "rerun" and ready, render


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
        status_url = f"{base_url}/api/sim-viz/status"
        status_response = request("GET", status_url, auth=auth, timeout=10.0, verify=verify)
        if int(getattr(status_response, "status_code", 0)) == 200:
            matches, render = _status_matches(status_response.json(), artifact_uri)
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
        load_response = request(
            "POST",
            f"{base_url}/api/sim-viz/load-artifact",
            json={"s3_uri": artifact_uri},
            auth=auth,
            timeout=60.0,
            verify=verify,
        )
        if int(getattr(load_response, "status_code", 0)) >= 400:
            raise ArtifactLoadError(
                f"agent load-artifact returned HTTP {getattr(load_response, 'status_code', 'unknown')}"
            )
        verify_response = request("GET", status_url, auth=auth, timeout=10.0, verify=verify)
        if int(getattr(verify_response, "status_code", 0)) != 200:
            raise ArtifactLoadError(
                f"agent sim-viz status returned HTTP {getattr(verify_response, 'status_code', 'unknown')}"
            )
        matches, render = _status_matches(verify_response.json(), artifact_uri)
        if not matches:
            raise ArtifactLoadError(
                "agent sim-viz status did not verify the exact artifact URI as a ready Rerun recording"
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
