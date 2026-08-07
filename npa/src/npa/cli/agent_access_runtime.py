"""Runtime cloud adapters and read authorization for the embedded NPA agent.

This source is embedded into the generated backend, where its dependencies are
defined by the surrounding backend module. Keeping the adapter here prevents
the deployment CLI/bootstrap template from becoming the access domain model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

# These names are supplied by the generated backend into which this module is
# embedded. They intentionally remain injectable/mocked at that boundary.
# ruff: noqa: F821


_AGENT_ACCESS_CACHE = {"report": None, "expires_at": 0.0}
_AGENT_ACCESS_LOCK = threading.Lock()


def _access_probe_error(operation: str, detail: str = ""):
    lowered = str(detail or "").lower()
    denied = any(
        marker in lowered
        for marker in (
            "permissiondenied",
            "permission_denied",
            "permission denied",
            "forbidden",
            "unauthorized",
            "accessdenied",
            "access denied",
        )
    )
    return AccessProbeError("denied" if denied else "unavailable", operation)


def _agent_nebius_json(args: list[str], *, operation: str) -> dict:
    nebius_bin = shutil.which("nebius") or "/usr/local/bin/nebius"
    if not Path(nebius_bin).exists() and shutil.which(nebius_bin) is None:
        raise AccessProbeError("unavailable", operation)
    command = [nebius_bin]
    # Prefer the attached service-account metadata profile. Unlike the staged
    # bootstrap token, metadata credentials rotate and reflect the running VM's
    # current tenant/project grants.
    if Path("/mnt/cloud-metadata/token").is_file():
        command.extend(["--profile", "cursor-sa"])
    command.extend([*args, "--all", "--format", "json"])
    proc = subprocess.run(
        command,
        env=_agent_command_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise _access_probe_error(operation, proc.stderr)
    try:
        payload = json.loads(proc.stdout or "{}")
    except (TypeError, ValueError) as exc:
        raise AccessProbeError("unavailable", operation) from exc
    if not isinstance(payload, dict):
        raise AccessProbeError("unavailable", operation)
    return payload


def _agent_list_tenant_projects(tenant_id: str) -> list:
    payload = _agent_nebius_json(
        ["iam", "project", "list", "--parent-id", tenant_id],
        operation="list tenant projects",
    )
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def _agent_list_project_buckets(project_id: str) -> list:
    payload = _agent_nebius_json(
        ["storage", "bucket", "list", "--parent-id", project_id],
        operation="list project object storage resources",
    )
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def _agent_probe_bucket(s3, bucket: str) -> "BucketProbe":
    if s3 is None:
        raise AccessProbeError("unavailable", "probe object storage bucket")
    try:
        page = s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
    except Exception as exc:
        raise _access_probe_error("list objects in bucket", str(exc)) from exc
    contents = page.get("Contents", []) if isinstance(page, dict) else []
    if not contents:
        return BucketProbe(
            list_status="available",
            read_status="unverified",
            reason="Object listing succeeded; the empty bucket has no object available for a read probe.",
        )
    first = contents[0] if isinstance(contents[0], dict) else {}
    key = str(first.get("Key") or "").strip()
    if not key:
        return BucketProbe(
            list_status="available",
            read_status="unverified",
            reason="Object listing succeeded; object read access could not be verified.",
        )
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        status = _access_probe_error("read object metadata", str(exc)).status
        return BucketProbe(
            list_status="available",
            read_status=status,
            reason=(
                "Object listing succeeded, but object read permission was denied."
                if status == "denied"
                else "Object listing succeeded, but object read access is unavailable."
            ),
        )
    return BucketProbe(
        list_status="available",
        read_status="available",
        reason="Object listing and object read access were verified.",
    )


def _agent_access_report(*, refresh: bool = False) -> "AgentAccessReport":
    now_mono = time.monotonic()
    with _AGENT_ACCESS_LOCK:
        cached = _AGENT_ACCESS_CACHE.get("report")
        expires_at = float(_AGENT_ACCESS_CACHE.get("expires_at") or 0.0)
        if not refresh and isinstance(cached, AgentAccessReport) and expires_at > now_mono:
            return cached
    s3, settings = _agent_s3_client_optional()
    primary = str(settings.get("bucket") or "").strip()
    configured = [primary] if primary else []
    for item in str(os.environ.get("NPA_AGENT_S3_BUCKETS", "")).split(","):
        name = item.strip()
        if name and name not in configured:
            configured.append(name)
    report = discover_agent_access(
        tenant_id=str(os.environ.get("NEBIUS_TENANT_ID") or "").strip(),
        deployment_project_id=str(os.environ.get("NEBIUS_PROJECT_ID") or "").strip(),
        deployment_project_name=str(
            os.environ.get("NPA_AGENT_PROJECT_ALIAS") or NPA_PROJECT_ALIAS
        ).strip(),
        fallback_buckets=configured,
        list_projects=_agent_list_tenant_projects,
        list_buckets=_agent_list_project_buckets,
        probe_bucket=lambda bucket: _agent_probe_bucket(s3, bucket),
    )
    with _AGENT_ACCESS_LOCK:
        _AGENT_ACCESS_CACHE["report"] = report
        _AGENT_ACCESS_CACHE["expires_at"] = time.monotonic() + 30.0
    return report


def _agent_access_diagnostics(report: "AgentAccessReport") -> dict:
    payload = report.to_dict()
    unavailable = [
        {
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "status": item.get("status", ""),
        }
        for item in payload.get("projects", [])
        if isinstance(item, dict) and item.get("status") != "available"
    ]
    return {
        "status": payload.get("status", "unavailable"),
        "scope": payload.get("scope", "single_project"),
        "searched_projects": [
            {"id": item.get("id", ""), "name": item.get("name", "")}
            for item in payload.get("projects", [])
            if isinstance(item, dict)
            and (
                item.get("capabilities", {})
                .get("artifact_discovery", {})
                .get("status")
                == "available"
            )
        ],
        "unavailable_projects": unavailable,
    }


def _agent_artifact_list_scope(report, resource_bucket: str = "", project_id: str = ""):
    """Resolve list filters through the effective-access model for API routes."""
    try:
        buckets = scoped_artifact_buckets(
            report,
            resource_bucket=resource_bucket,
            project_id=project_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return buckets, {
        "project_id": str(project_id or "").strip(),
        "bucket": str(resource_bucket or "").strip(),
    }


def _agent_access_api_response(refresh: bool = False):
    try:
        report = _agent_access_report(refresh=bool(refresh))
        if refresh:
            _run_list_cache_clear()
        return {"ok": True, **report.to_dict()}
    except Exception:
        # Raw cloud or credential errors are never part of the public contract.
        return JSONResponse(
            status_code=503,
            content={"ok": False, "error": "Agent access discovery is unavailable."},
        )


def _agent_s3_buckets(s3, settings) -> list:
    # Search scope comes from the project-aware effective-access report, not an
    # unowned ListBuckets result or a hardcoded workflow/bucket list.
    primary = str(settings.get("bucket") or "")
    try:
        return accessible_artifact_buckets(_agent_access_report())
    except Exception:
        return [primary] if primary else []


def _configured_agent_s3_buckets(settings) -> set:
    return configured_agent_s3_buckets(
        str((settings or {}).get("bucket") or ""),
        str(os.environ.get("NPA_AGENT_S3_BUCKETS", "")).strip(),
    )


def _assert_s3_uri_in_agent_bucket(uri: str, settings) -> None:
    # Bucket-only gate (configured primary + explicit NPA_AGENT_S3_BUCKETS).
    # Prefix is intentionally not enforced: runs live under multiple category
    # roots inside the same configured bucket.
    ok, reason = s3_uri_in_configured_buckets(
        uri,
        primary=str((settings or {}).get("bucket") or ""),
        extras_csv=str(os.environ.get("NPA_AGENT_S3_BUCKETS", "")).strip(),
        prefix="",
    )
    if not ok:
        raise HTTPException(
            status_code=400,
            detail=reason or "s3_uri bucket is not the configured agent bucket",
        )


def _resolve_accessible_run_artifact(
    *,
    s3,
    settings,
    run_id: str,
    key: str,
    bucket: str = "",
):
    # Authorize one discovered artifact by exact run membership. This is the
    # read-only bridge for cross-project artifacts; it deliberately does not
    # make every URI in a discovered bucket caller-addressable.
    normalized_run = validate_run_id(run_id)
    normalized_key = _safe_artifact_key(key)
    accessible = _agent_s3_buckets(s3, settings)
    requested_bucket = str(bucket or "").strip()
    # Membership is structural and exact: the key must sit below this run's
    # directory, then a metadata read must prove that exact object exists in an
    # effectively accessible bucket. This avoids enumerating a 10k-object run
    # merely to authorize one already-discovered key.
    if not (
        normalized_key.startswith(normalized_run + "/")
        or f"/{normalized_run}/" in normalized_key
    ):
        raise HTTPException(
            status_code=404,
            detail="artifact is not a discovered object for this run",
        )
    if requested_bucket:
        if requested_bucket not in accessible:
            raise HTTPException(
                status_code=400,
                detail="artifact bucket is outside effective agent access",
            )
        candidates = [requested_bucket]
    else:
        candidates = accessible
    run_bucket = ""
    for candidate in candidates:
        try:
            s3.head_object(Bucket=candidate, Key=normalized_key)
        except Exception:
            continue
        run_bucket = candidate
        break
    if not run_bucket:
        raise HTTPException(
            status_code=404,
            detail="artifact is not a discovered object for this run",
        )
    return run_bucket, normalized_key, normalized_run


def _authorize_agent_artifact_uri(*, s3, settings, uri: str, run_id: str = ""):
    bucket, key = parse_s3_uri(uri)
    if bucket in _configured_agent_s3_buckets(settings):
        return bucket, _safe_artifact_key(key), str(run_id or "").strip()
    if not str(run_id or "").strip():
        raise HTTPException(
            status_code=400,
            detail="cross-project s3_uri requires a run_id and exact discovered artifact",
        )
    return _resolve_accessible_run_artifact(
        s3=s3,
        settings=settings,
        run_id=run_id,
        key=key,
        bucket=bucket,
    )
