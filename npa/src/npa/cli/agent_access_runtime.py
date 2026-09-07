"""Runtime cloud adapters and read authorization for the embedded NPA agent.

This source is embedded into the generated backend, where its dependencies are
defined by the surrounding backend module. Keeping the adapter here prevents
the deployment CLI/bootstrap template from becoming the access domain model.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

# These names are supplied by the generated backend into which this module is
# embedded. They intentionally remain injectable/mocked at that boundary.
if TYPE_CHECKING:
    from npa.cli.agent_access import AccessProbeError, AgentAccessReport, BucketProbe

# NPA_EMBED_STANDALONE_START
# Standalone imports make this adapter directly unit-testable. The embedded
# source runs under backend.py's module name, so this block does not overwrite
# the intentionally injected backend globals.
if __name__ == "npa.cli.agent_access_runtime":
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    from npa.cli.agent_access import (
        AccessProbeError,
        AgentAccessReport,
        BucketProbe,
        _bucket_identity,
        _project_identity,
        accessible_artifact_buckets,
        artifact_bucket_projects,
        discover_agent_access,
        normalize_configured_artifact_sources,
        scoped_artifact_buckets,
    )
    from npa.cli.agent_s3_guard import (
        configured_agent_s3_buckets,
        s3_uri_in_configured_buckets,
    )
    from npa.workflows.artifacts import (
        decode_run_ref,
        find_run_sources_across_buckets,
        list_artifacts,
        parse_s3_uri,
        validate_run_id,
    )

    NPA_PROJECT_ALIAS = ""
    _agent_command_env: Any = None
    _agent_artifact_s3_client_optional: Any = None
    _run_list_cache_clear: Any = None
    _safe_artifact_key: Any = None
# NPA_EMBED_STANDALONE_END


# Tenant inventory runs on a FastAPI request worker: 15 seconds allows a normal
# CLI auth/list round-trip while guaranteeing one stalled project probe cannot
# pin that worker indefinitely.
_AGENT_NEBIUS_TIMEOUT_SECONDS = 15.0
_AGENT_ACCESS_CACHE_TTL_SECONDS = 30.0
_AGENT_EXACT_SOURCE_ACCESS_TTL_SECONDS = 30.0
_AGENT_RUN_CURSOR_TTL_SECONDS = 300.0
_AGENT_RUN_CURSOR_MAX_SNAPSHOTS = 64
_MAX_ARTIFACT_MEMBERSHIP_BUCKETS = 32
_AGENT_ACCESS_CACHE: dict[str, Any] = {
    "report": None,
    "expires_at": 0.0,
    "refreshing": False,
    # Waiters must distinguish a completed refresh from a failed one. Keeping
    # an expired report is useful for ordinary stale-while-revalidate status
    # reads, but an explicit refresh may never report that stale value as a
    # newly refreshed authorization snapshot.
    "last_refresh_succeeded": None,
    # Artifact discovery takes a lease on the effective-access report. A
    # successful refresh waits for those readers before publishing and clearing
    # derived indexes, so a request that began with the old source set cannot
    # repopulate a new-generation cache after the refresh.
    "artifact_readers": 0,
}
_AGENT_EXACT_SOURCE_ACCESS_CACHE: dict[tuple[str, ...], float] = {}
_AGENT_RUN_CURSOR_SNAPSHOTS: dict[str, dict[str, Any]] = {}
_AGENT_RUN_CURSOR_GENERATION = 0
_AGENT_ACCESS_LOCK = threading.Lock()
_AGENT_ACCESS_CONDITION = threading.Condition(_AGENT_ACCESS_LOCK)
_AGENT_RUN_CURSOR_LOCK = threading.Lock()
_AMBIENT_NEBIUS_TOKEN_KEYS = frozenset(
    {
        "NEBIUS_IAM_TOKEN",
        "NPA_NEBIUS_IAM_TOKEN",
        "TF_VAR_iam_token",
        "NPA_REUSE_IAM_TOKEN",
        "IAM_TOKEN",
    }
)


def _artifact_run_cursor(snapshot_id: str, offset: int) -> str:
    payload = json.dumps(
        {"v": 1, "s": str(snapshot_id), "o": max(0, int(offset))},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _artifact_run_cursor_parts(cursor: str) -> tuple[str, int]:
    value = str(cursor or "").strip()
    if not value:
        return "", 0
    try:
        padded = value + ("=" * (-len(value) % 4))
        payload = json.loads(
            base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        )
        if not isinstance(payload, dict) or payload.get("v") != 1:
            raise ValueError("unsupported cursor")
        snapshot_id = str(payload.get("s") or "").strip()
        offset = int(payload.get("o"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid run-list cursor") from exc
    if (
        not snapshot_id
        or len(snapshot_id) > 128
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in snapshot_id
        )
        or offset < 0
    ):
        raise HTTPException(status_code=400, detail="invalid run-list cursor")
    return snapshot_id, offset


def _artifact_run_snapshot_context(value: Any) -> str:
    """Return a stable, non-reversible identity for one authorized list scope."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clear_artifact_run_cursor_snapshots() -> None:
    """Invalidate pagination whenever effective source authorization changes."""
    global _AGENT_RUN_CURSOR_GENERATION
    with _AGENT_RUN_CURSOR_LOCK:
        _AGENT_RUN_CURSOR_GENERATION += 1
        _AGENT_RUN_CURSOR_SNAPSHOTS.clear()


def _artifact_run_snapshot_page(
    *,
    cursor: str,
    context: str,
    limit: int,
    items: list[Any] | tuple[Any, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[Any], str, dict[str, Any]]:
    """Page through an immutable, bounded server-side discovery snapshot.

    A cursor is deliberately tied to both the original query/source tuple and
    the access generation. This prevents a refreshed or reordered source index
    from silently skipping or duplicating runs halfway through traversal.
    """
    page_size = max(1, min(int(limit), 500))
    normalized_context = str(context or "").strip()
    if not normalized_context:
        raise ValueError("run-list snapshot context is required")
    now_mono = time.monotonic()
    with _AGENT_RUN_CURSOR_LOCK:
        expired = [
            key
            for key, snapshot in _AGENT_RUN_CURSOR_SNAPSHOTS.items()
            if float(snapshot.get("expires_at") or 0.0) <= now_mono
        ]
        for key in expired:
            _AGENT_RUN_CURSOR_SNAPSHOTS.pop(key, None)

        if cursor:
            snapshot_id, offset = _artifact_run_cursor_parts(cursor)
            snapshot = _AGENT_RUN_CURSOR_SNAPSHOTS.get(snapshot_id)
            if (
                not snapshot
                or int(snapshot.get("generation", -1)) != _AGENT_RUN_CURSOR_GENERATION
                or snapshot.get("context") != normalized_context
            ):
                raise HTTPException(
                    status_code=409,
                    detail="run-list cursor is stale for the current access scope; restart pagination",
                )
        else:
            if items is None:
                raise ValueError("run-list snapshot items are required")
            snapshot_id = secrets.token_urlsafe(18)
            offset = 0
            snapshot = {
                "generation": _AGENT_RUN_CURSOR_GENERATION,
                "context": normalized_context,
                "items": tuple(items),
                "metadata": dict(metadata or {}),
                "expires_at": now_mono + _AGENT_RUN_CURSOR_TTL_SECONDS,
            }
            _AGENT_RUN_CURSOR_SNAPSHOTS[snapshot_id] = snapshot
            while len(_AGENT_RUN_CURSOR_SNAPSHOTS) > _AGENT_RUN_CURSOR_MAX_SNAPSHOTS:
                oldest = min(
                    _AGENT_RUN_CURSOR_SNAPSHOTS,
                    key=lambda key: float(
                        _AGENT_RUN_CURSOR_SNAPSHOTS[key].get("expires_at") or 0.0
                    ),
                )
                _AGENT_RUN_CURSOR_SNAPSHOTS.pop(oldest, None)

        frozen_items = list(snapshot.get("items") or ())
        if offset > len(frozen_items):
            raise HTTPException(status_code=400, detail="invalid run-list cursor")
        end = min(offset + page_size, len(frozen_items))
        next_cursor = (
            _artifact_run_cursor(snapshot_id, end) if end < len(frozen_items) else ""
        )
        return (
            frozen_items[offset:end],
            next_cursor,
            dict(snapshot.get("metadata") or {}),
        )


def _artifact_search_scope_complete(report) -> bool:
    payload = report.to_dict() if hasattr(report, "to_dict") else report
    if not isinstance(payload, dict):
        return False
    capabilities = payload.get("capabilities") or {}
    if (
        payload.get("status") != "available"
        or (capabilities.get("project_discovery") or {}).get("status") != "available"
        or (capabilities.get("artifact_discovery") or {}).get("status") != "available"
    ):
        return False
    projects = payload.get("projects") or []
    if not isinstance(projects, list):
        return False
    for project in projects:
        if not isinstance(project, dict) or project.get("status") != "available":
            return False
        project_capabilities = project.get("capabilities") or {}
        for capability in (
            "storage_resource_discovery",
            "artifact_discovery",
            "artifact_read",
        ):
            if (project_capabilities.get(capability) or {}).get(
                "status"
            ) != "available":
                return False
        resources = project.get("resources") or []
        if not isinstance(resources, list):
            return False
        for resource in resources:
            if not isinstance(resource, dict):
                return False
            resource_capabilities = resource.get("capabilities") or {}
            if (resource_capabilities.get("artifact_discovery") or {}).get(
                "status"
            ) != "available" or (resource_capabilities.get("artifact_read") or {}).get(
                "status"
            ) != "available":
                return False
    return True


def _artifact_selected_scope_complete(report, scope: dict[str, str]) -> bool:
    """Say whether a caller-selected project/source was exhaustively verified."""
    selected = dict(scope or {})
    project_id = str(selected.get("project_id") or "").strip()
    bucket = str(selected.get("bucket") or "").strip()
    if not project_id:
        return _artifact_search_scope_complete(report)
    payload = report.to_dict() if hasattr(report, "to_dict") else report
    if not isinstance(payload, dict):
        return False
    project = next(
        (
            item
            for item in payload.get("projects") or []
            if isinstance(item, dict)
            and str(item.get("id") or "").strip() == project_id
        ),
        None,
    )
    if not isinstance(project, dict):
        return False
    capabilities = project.get("capabilities") or {}
    if any(
        (capabilities.get(name) or {}).get("status") != "available"
        for name in (
            "storage_resource_discovery",
            "artifact_discovery",
            "artifact_read",
        )
    ):
        return False
    resources = [
        item
        for item in project.get("resources") or []
        if isinstance(item, dict)
        and (not bucket or str(item.get("name") or "").strip() == bucket)
    ]
    if bucket and not resources:
        return False
    return all(
        (resource.get("capabilities") or {}).get("artifact_discovery", {}).get("status")
        == "available"
        and (resource.get("capabilities") or {}).get("artifact_read", {}).get("status")
        == "available"
        for resource in resources
    )


def _agent_inventory_credential_context() -> tuple[dict[str, str], str, str, str]:
    """Return a deterministic metadata-profile environment for read inventory."""
    base = _agent_command_env() if callable(_agent_command_env) else dict(os.environ)
    env = {str(key): str(value) for key, value in dict(base or {}).items()}
    for key in _AMBIENT_NEBIUS_TOKEN_KEYS:
        env.pop(key, None)
    config_path = str(
        os.environ.get("NPA_NEBIUS_CONFIG") or "/root/.nebius/config.yaml"
    ).strip()
    profile = str(os.environ.get("NPA_NEBIUS_PROFILE") or "cursor-sa").strip()
    env["HOME"] = str(Path(config_path).parent.parent) if config_path else "/root"
    env["NEBIUS_PROFILE"] = profile
    try:
        metadata_available = Path("/mnt/cloud-metadata/token").is_file()
    except OSError:
        metadata_available = False
    source = "instance_metadata" if metadata_available else "configured_profile"
    return env, profile, config_path, source


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
    env, profile, config_path, _source = _agent_inventory_credential_context()
    try:
        config_available = bool(config_path and Path(config_path).is_file())
    except OSError:
        config_available = False
    if config_available:
        command.extend(["--config", config_path])
    if profile:
        command.extend(["--profile", profile])
    command.extend([*args, "--all", "--format", "json"])
    try:
        proc = subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=_AGENT_NEBIUS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # TimeoutExpired may retain captured stdout/stderr or the command. Do not
        # reflect any of it into the public access report.
        raise AccessProbeError("unavailable", operation) from None
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


def _discover_agent_access_report() -> "AgentAccessReport":
    s3, settings = _agent_artifact_s3_client_optional()
    primary = str(settings.get("bucket") or "").strip()
    configured = [primary] if primary else []
    for item in str(os.environ.get("NPA_AGENT_S3_BUCKETS", "")).split(","):
        name = item.strip()
        if name and name not in configured:
            configured.append(name)
    _inventory_env, inventory_profile, inventory_config, credential_source = (
        _agent_inventory_credential_context()
    )
    return discover_agent_access(
        tenant_id=str(os.environ.get("NEBIUS_TENANT_ID") or "").strip(),
        deployment_project_id=str(os.environ.get("NEBIUS_PROJECT_ID") or "").strip(),
        deployment_project_name=str(
            os.environ.get("NPA_AGENT_PROJECT_ALIAS") or NPA_PROJECT_ALIAS
        ).strip(),
        fallback_buckets=configured,
        configured_sources=_configured_agent_artifact_sources(),
        list_projects=_agent_list_tenant_projects,
        list_buckets=_agent_list_project_buckets,
        probe_bucket=lambda bucket: _agent_probe_bucket(s3, bucket),
        service_account_id=str(
            os.environ.get("NEBIUS_SERVICE_ACCOUNT_ID") or ""
        ).strip(),
        credential_source=credential_source,
        credential_profile=inventory_profile,
        credential_config=inventory_config,
    )


def _configured_agent_artifact_sources() -> tuple[dict[str, str], ...]:
    """Decode the owner-staged exact read sources from the service environment."""
    encoded = str(os.environ.get("NPA_AGENT_ARTIFACT_SOURCES_B64") or "").strip()
    if not encoded:
        return ()
    try:
        padded = encoded + ("=" * (-len(encoded) % 4))
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except Exception as exc:
        raise ValueError("configured artifact sources are invalid") from exc
    return normalize_configured_artifact_sources(payload)


def _configured_agent_artifact_source_matches(
    *, project_id: str, bucket: str, resolved_prefix: str
) -> bool:
    identity = (
        str(project_id or "").strip(),
        str(bucket or "").strip(),
        _validated_resolved_prefix(resolved_prefix),
    )
    return any(
        identity == (source["project_id"], source["bucket"], source["resolved_prefix"])
        for source in _configured_agent_artifact_sources()
    )


def _find_configured_exact_run_sources(s3, run_id: str, *, sources=None):
    """Probe only durable exact sources for one validated run identifier."""
    normalized_run = validate_run_id(run_id)
    matches = []
    source_errors: list[dict[str, str]] = []
    complete = True
    effective_sources = (
        _configured_agent_artifact_sources() if sources is None else tuple(sources)
    )
    for source in effective_sources:
        bucket = source["bucket"]
        project = source["project_id"]
        found, errors, source_complete = find_run_sources_across_buckets(
            [bucket],
            base_prefix="",
            run_id=normalized_run,
            exact_prefix=source["resolved_prefix"],
            bucket_projects={bucket: project},
            s3=s3,
        )
        matches.extend(
            item
            for item in found
            if (
                item.run_id == normalized_run
                and item.bucket == bucket
                and item.project_id == project
                and item.resolved_prefix == source["resolved_prefix"]
            )
        )
        source_errors.extend(dict(item) for item in errors)
        complete = complete and bool(source_complete) and not bool(errors)
    unique = {
        (item.project_id, item.bucket, item.resolved_prefix, item.run_id): item
        for item in matches
    }
    return list(unique.values()), tuple(source_errors), complete


def _invalidate_agent_artifact_discovery() -> None:
    """Drop every artifact view derived from an older effective-access report."""
    if callable(_run_list_cache_clear):
        _run_list_cache_clear()
    _clear_artifact_run_cursor_snapshots()
    _clear_exact_run_ref_source_authorizations()
    clear_foxglove_inventory = globals().get(
        "_clear_foxglove_exact_artifact_inventory"
    )
    if callable(clear_foxglove_inventory):
        clear_foxglove_inventory()


def _finish_agent_access_refresh(report: "AgentAccessReport | None") -> None:
    if report is None:
        with _AGENT_ACCESS_CONDITION:
            _AGENT_ACCESS_CACHE["last_refresh_succeeded"] = False
            # A failed explicit refresh must not leave a still-live report
            # available to authorization-sensitive readers. Ordinary access
            # status may continue to display the stale report, but the next
            # artifact operation must re-probe and fail closed on error.
            _AGENT_ACCESS_CACHE["expires_at"] = 0.0
            _AGENT_ACCESS_CACHE["refreshing"] = False
            _AGENT_ACCESS_CONDITION.notify_all()
        return

    # Keep ``refreshing`` true while existing artifact readers drain. New
    # readers wait in ``_begin_agent_artifact_access``; old readers finish using
    # the report they leased. Only then invalidate and publish as one generation
    # transition. This closes the unbounded stale-request race that two best-
    # effort cache clears cannot close.
    with _AGENT_ACCESS_CONDITION:
        _AGENT_ACCESS_CONDITION.wait_for(
            lambda: int(_AGENT_ACCESS_CACHE.get("artifact_readers") or 0) == 0
        )
    _invalidate_agent_artifact_discovery()
    with _AGENT_ACCESS_CONDITION:
        _AGENT_ACCESS_CACHE["report"] = report
        _AGENT_ACCESS_CACHE["expires_at"] = (
            time.monotonic() + _AGENT_ACCESS_CACHE_TTL_SECONDS
        )
        _AGENT_ACCESS_CACHE["last_refresh_succeeded"] = True
        _AGENT_ACCESS_CACHE["refreshing"] = False
        _AGENT_ACCESS_CONDITION.notify_all()


def _begin_agent_artifact_access() -> "AgentAccessReport":
    """Lease one current report for a complete artifact operation.

    Ordinary access polling may serve a stale report while it refreshes in the
    background. Artifact discovery is authorization-sensitive, so it waits for
    that refresh and pins the resulting report until its S3 index/snapshot has
    been produced.
    """
    while True:
        with _AGENT_ACCESS_CONDITION:
            while bool(_AGENT_ACCESS_CACHE.get("refreshing")):
                _AGENT_ACCESS_CONDITION.wait()
            report = _AGENT_ACCESS_CACHE.get("report")
            expires_at = float(_AGENT_ACCESS_CACHE.get("expires_at") or 0.0)
            if isinstance(report, AgentAccessReport) and expires_at > time.monotonic():
                _AGENT_ACCESS_CACHE["artifact_readers"] = (
                    int(_AGENT_ACCESS_CACHE.get("artifact_readers") or 0) + 1
                )
                return report
        # Artifact operations never treat a stale report as current. Perform a
        # synchronous refresh outside the condition lock; failures propagate and
        # fail closed instead of leasing the expired report.
        _agent_access_report(refresh=True)


def _end_agent_artifact_access() -> None:
    """Release a report lease acquired by ``_begin_agent_artifact_access``."""
    with _AGENT_ACCESS_CONDITION:
        readers = int(_AGENT_ACCESS_CACHE.get("artifact_readers") or 0)
        if readers <= 0:
            raise RuntimeError("artifact access lease is not active")
        _AGENT_ACCESS_CACHE["artifact_readers"] = readers - 1
        _AGENT_ACCESS_CONDITION.notify_all()


def _refresh_agent_access_in_background() -> None:
    try:
        report = _discover_agent_access_report()
    except BaseException:
        report = None
    _finish_agent_access_refresh(report)


def _agent_access_report(*, refresh: bool = False) -> "AgentAccessReport":
    now_mono = time.monotonic()
    if refresh:
        _invalidate_agent_artifact_discovery()
    with _AGENT_ACCESS_CONDITION:
        cached = _AGENT_ACCESS_CACHE.get("report")
        expires_at = float(_AGENT_ACCESS_CACHE.get("expires_at") or 0.0)
        if not refresh and isinstance(cached, AgentAccessReport):
            if expires_at <= now_mono and not bool(
                _AGENT_ACCESS_CACHE.get("refreshing")
            ):
                _AGENT_ACCESS_CACHE["refreshing"] = True
                threading.Thread(
                    target=_refresh_agent_access_in_background,
                    name="npa-agent-access-refresh",
                    daemon=True,
                ).start()
            return cached
        if bool(_AGENT_ACCESS_CACHE.get("refreshing")):
            while bool(_AGENT_ACCESS_CACHE.get("refreshing")):
                _AGENT_ACCESS_CONDITION.wait()
            if not bool(_AGENT_ACCESS_CACHE.get("last_refresh_succeeded")):
                raise RuntimeError("agent access refresh did not complete")
            cached = _AGENT_ACCESS_CACHE.get("report")
            if isinstance(cached, AgentAccessReport):
                return cached
        _AGENT_ACCESS_CACHE["refreshing"] = True
    try:
        report = _discover_agent_access_report()
    except BaseException:
        _finish_agent_access_refresh(None)
        raise
    _finish_agent_access_refresh(report)
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
                item.get("capabilities", {}).get("artifact_discovery", {}).get("status")
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


def _resolve_selected_run_source(
    **kwargs,
) -> tuple[str, str, str]:
    """Resolve a selected source while pinning the effective-access report."""
    _begin_agent_artifact_access()
    try:
        return _resolve_selected_run_source_with_access(**kwargs)
    finally:
        _end_agent_artifact_access()


def _resolve_selected_run_source_with_access(
    *,
    s3,
    settings,
    run_id: str,
    resource_bucket: str,
    project_id: str = "",
    resolved_prefix: str = "",
    source_selected: bool = False,
    exclude: "set[str] | None" = None,
) -> tuple[str, str, str]:
    """Authorize and resolve one exact server-discovered artifact source."""
    report = _agent_access_report()
    buckets, _scope = _agent_artifact_list_scope(report, resource_bucket, project_id)
    bucket = str(resource_bucket or "").strip()
    if bucket not in buckets:
        raise HTTPException(
            status_code=403,
            detail="artifact bucket is outside effective agent access",
        )
    prefix = _validated_resolved_prefix(resolved_prefix)
    sources, source_errors, complete = find_run_sources_across_buckets(
        [bucket],
        base_prefix=str((settings or {}).get("prefix") or ""),
        run_id=validate_run_id(run_id),
        exact_prefix=prefix if prefix else "" if source_selected else None,
        exclude=exclude,
        bucket_projects=artifact_bucket_projects(report),
        s3=s3,
    )
    if project_id:
        sources = [item for item in sources if item.project_id == project_id]
    if prefix:
        sources = [item for item in sources if item.resolved_prefix == prefix]
    elif source_selected:
        sources = [item for item in sources if not item.resolved_prefix]
    if len(sources) > 1:
        raise HTTPException(
            status_code=409,
            detail="run id is ambiguous in the selected bucket; select a resolved prefix",
        )
    if sources and (not complete or source_errors) and not (prefix or source_selected):
        raise HTTPException(
            status_code=503,
            detail="selected artifact source discovery was incomplete",
        )
    if not sources:
        raise HTTPException(
            status_code=404 if complete and not source_errors else 503,
            detail="selected artifact source was not discovered",
        )
    selected = sources[0]
    return selected.bucket, selected.project_id, selected.resolved_prefix


def _selected_run_request(body: dict) -> tuple[str, str, str, bool]:
    bucket = str(body.get("resource_bucket") or "").strip()
    project = str(body.get("project_id") or "").strip()
    prefix = _validated_resolved_prefix(str(body.get("resolved_prefix") or ""))
    raw_selected = body.get("source_selected")
    selected = raw_selected is True or str(raw_selected or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    return bucket, project, prefix, selected


def _load_selected_run_artifacts(**kwargs):
    _begin_agent_artifact_access()
    try:
        bucket, project, prefix = _resolve_selected_run_source_with_access(**kwargs)
        artifacts = list_artifacts(
            bucket,
            run_id=validate_run_id(str(kwargs.get("run_id") or "")),
            prefix=prefix,
            s3=kwargs.get("s3"),
        )
        return bucket, project, prefix, artifacts
    finally:
        _end_agent_artifact_access()


def _artifact_source_metadata(report, bucket: str, key: str, run_id: str):
    """Derive persisted source identity from an already-authorized object key."""
    normalized_run = str(run_id or "").strip()
    parts = [part for part in str(key or "").strip().strip("/").split("/") if part]
    try:
        run_index = parts.index(normalized_run)
    except ValueError:
        run_index = 0
    return (
        str(bucket or "").strip(),
        artifact_bucket_projects(report).get(str(bucket or "").strip(), ""),
        "/".join(parts[:run_index]) if run_index else "",
    )


def _agent_access_api_response(refresh: bool = False):
    try:
        report = _agent_access_report(refresh=bool(refresh))
        return {
            "ok": True,
            **report.to_dict(),
            "refresh": {
                "requested": bool(refresh),
                "state": "refreshed" if refresh else "cached_or_current",
            },
        }
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


def _validated_resolved_prefix(value: str) -> str:
    """Validate an opaque S3 parent prefix without path normalization."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    variants = [raw]
    for _attempt in range(2):
        decoded = unquote(variants[-1])
        if decoded == variants[-1]:
            break
        variants.append(decoded)
    for candidate in variants:
        if (
            candidate.startswith("/")
            or candidate.endswith("/")
            or "\\" in candidate
            or any(part in {"", ".", ".."} for part in candidate.split("/"))
        ):
            raise HTTPException(
                status_code=400, detail="invalid resolved artifact prefix"
            )
    return raw


def _authorize_exact_run_ref_source(
    *,
    s3,
    settings,
    run_id: str,
    run_ref: str,
    resource_bucket: str,
    project_id: str,
    resolved_prefix: str,
) -> tuple[str, str, str]:
    """Authorize one server-issued source without a tenant-wide bucket scan.

    Artifact cards carry the exact project, bucket, prefix, and run reference
    returned by discovery. Revalidate that narrow ownership chain and current
    bucket access rather than rebuilding the complete effective-access report,
    which can probe hundreds of unrelated buckets. The run reference remains a
    selector, not an authorization capability: every caller-supplied component
    must agree and the selected bucket must still be discoverable.
    """
    requested_run = validate_run_id(str(run_id or "").strip())
    requested_bucket = str(resource_bucket or "").strip()
    requested_project = str(project_id or "").strip()
    requested_prefix = _validated_resolved_prefix(resolved_prefix)
    if not requested_bucket or not requested_project:
        raise HTTPException(
            status_code=400,
            detail="project and resource bucket are required for exact artifact playback",
        )
    try:
        ref_bucket, ref_prefix, ref_run = decode_run_ref(run_ref)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid run_ref") from exc
    if ref_run != requested_run:
        raise HTTPException(status_code=409, detail="run_ref does not identify run_id")
    if ref_bucket != requested_bucket:
        raise HTTPException(
            status_code=409,
            detail="the selected artifact bucket does not match run_ref",
        )
    if ref_prefix != requested_prefix:
        raise HTTPException(
            status_code=409,
            detail="the selected artifact prefix does not match run_ref",
        )

    cache_key = (
        str(os.environ.get("NEBIUS_TENANT_ID") or "").strip(),
        str(os.environ.get("NEBIUS_PROJECT_ID") or "").strip(),
        requested_project,
        requested_bucket,
        requested_prefix,
        requested_run,
        str(run_ref or "").strip(),
    )
    now_mono = time.monotonic()
    with _AGENT_ACCESS_LOCK:
        expires_at = float(_AGENT_EXACT_SOURCE_ACCESS_CACHE.get(cache_key) or 0.0)
        if expires_at > now_mono:
            return requested_bucket, requested_project, requested_prefix
        for stale_key, stale_expiry in list(_AGENT_EXACT_SOURCE_ACCESS_CACHE.items()):
            if stale_expiry <= now_mono:
                _AGENT_EXACT_SOURCE_ACCESS_CACHE.pop(stale_key, None)

    # A durable owner-configured tuple is the narrow fallback when tenant or
    # project inventory is unavailable. It is a selector, never a capability:
    # re-probe the exact bucket with the service's existing S3 credentials
    # before authorizing any run inventory or object load.
    if _configured_agent_artifact_source_matches(
        project_id=requested_project,
        bucket=requested_bucket,
        resolved_prefix=requested_prefix,
    ):
        try:
            configured_probe = _agent_probe_bucket(s3, requested_bucket)
        except Exception as exc:
            status = getattr(exc, "status", "unavailable")
            raise HTTPException(
                status_code=403 if status == "denied" else 503,
                detail="configured artifact source access could not be verified",
            ) from exc
        if str(getattr(configured_probe, "list_status", "unavailable")) != "available":
            raise HTTPException(
                status_code=(
                    403
                    if str(getattr(configured_probe, "list_status", "")) == "denied"
                    else 503
                ),
                detail="configured artifact source is not currently searchable",
            )
        _remember_exact_run_ref_source_authorization(
            run_id=requested_run,
            run_ref=run_ref,
            resource_bucket=requested_bucket,
            project_id=requested_project,
            resolved_prefix=requested_prefix,
        )
        return requested_bucket, requested_project, requested_prefix

    deployment_project = str(os.environ.get("NEBIUS_PROJECT_ID") or "").strip()
    tenant_id = str(os.environ.get("NEBIUS_TENANT_ID") or "").strip()
    if requested_project != deployment_project:
        if not tenant_id:
            raise HTTPException(
                status_code=403,
                detail="artifact project is outside effective agent access",
            )
        try:
            visible_projects = {
                _project_identity(item)[0]
                for item in _agent_list_tenant_projects(tenant_id)
            }
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="artifact project access could not be verified",
            ) from exc
        if requested_project not in visible_projects:
            raise HTTPException(
                status_code=403,
                detail="artifact project is outside effective agent access",
            )

    inventory_failed = False
    try:
        project_buckets = {
            _bucket_identity(item)[1]
            for item in _agent_list_project_buckets(requested_project)
        }
    except Exception:
        inventory_failed = True
        project_buckets = set()
    configured_deployment_bucket = (
        requested_project == deployment_project
        and requested_bucket in _configured_agent_s3_buckets(settings)
    )
    if requested_bucket not in project_buckets and not configured_deployment_bucket:
        raise HTTPException(
            status_code=503 if inventory_failed else 403,
            detail=(
                "artifact bucket ownership could not be verified"
                if inventory_failed
                else "artifact bucket does not belong to the selected project"
            ),
        )

    try:
        probe = _agent_probe_bucket(s3, requested_bucket)
    except Exception as exc:
        status = getattr(exc, "status", "unavailable")
        raise HTTPException(
            status_code=403 if status == "denied" else 503,
            detail="artifact bucket access could not be verified",
        ) from exc
    if str(getattr(probe, "list_status", "unavailable")) != "available":
        raise HTTPException(
            status_code=(
                403 if str(getattr(probe, "list_status", "")) == "denied" else 503
            ),
            detail="artifact bucket is not currently searchable",
        )

    # Project ownership plus bucket access does not prove a caller-provided
    # parent prefix is a server-issued run source. Re-discover the run through
    # the generic bounded index (without using the requested prefix as a probe)
    # and require the complete source tuple to be present. Configured exact
    # sources took the explicit owner-controlled branch above.
    exclusions = globals().get("_discovery_exclude_roots")
    excluded_roots = exclusions() if callable(exclusions) else set()
    try:
        discovered, source_errors, discovery_complete = (
            find_run_sources_across_buckets(
                [requested_bucket],
                base_prefix=str((settings or {}).get("prefix") or ""),
                run_id=requested_run,
                exact_prefix=None,
                exclude=excluded_roots,
                bucket_projects={requested_bucket: requested_project},
                s3=s3,
            )
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="artifact run-source discovery could not be verified",
        ) from exc
    if source_errors or not discovery_complete:
        raise HTTPException(
            status_code=503,
            detail="artifact run-source discovery was incomplete",
        )
    exact_sources = [
        item
        for item in discovered
        if (
            str(getattr(item, "run_id", "") or "") == requested_run
            and str(getattr(item, "bucket", "") or "") == requested_bucket
            and str(getattr(item, "project_id", "") or "") == requested_project
            and str(getattr(item, "resolved_prefix", "") or "")
            == requested_prefix
        )
    ]
    if not exact_sources:
        raise HTTPException(
            status_code=403,
            detail="artifact run source was not discovered for this access scope",
        )
    if len(exact_sources) > 1:
        raise HTTPException(
            status_code=409,
            detail="artifact run source is ambiguous for this access scope",
        )
    _remember_exact_run_ref_source_authorization(
        run_id=requested_run,
        run_ref=run_ref,
        resource_bucket=requested_bucket,
        project_id=requested_project,
        resolved_prefix=requested_prefix,
    )
    return requested_bucket, requested_project, requested_prefix


def _remember_exact_run_ref_source_authorization(
    *,
    run_id: str,
    run_ref: str,
    resource_bucket: str,
    project_id: str,
    resolved_prefix: str,
) -> None:
    """Keep a just-proven exact scope warm for the immediately following click."""
    cache_key = (
        str(os.environ.get("NEBIUS_TENANT_ID") or "").strip(),
        str(os.environ.get("NEBIUS_PROJECT_ID") or "").strip(),
        str(project_id or "").strip(),
        str(resource_bucket or "").strip(),
        str(resolved_prefix or "").strip(),
        str(run_id or "").strip(),
        str(run_ref or "").strip(),
    )
    with _AGENT_ACCESS_LOCK:
        _AGENT_EXACT_SOURCE_ACCESS_CACHE[cache_key] = (
            time.monotonic() + _AGENT_EXACT_SOURCE_ACCESS_TTL_SECONDS
        )


def _clear_exact_run_ref_source_authorizations() -> None:
    """Clear exact-source access proofs after explicit access refresh/tests."""
    with _AGENT_ACCESS_LOCK:
        _AGENT_EXACT_SOURCE_ACCESS_CACHE.clear()


def _resolve_accessible_run_artifact(
    *,
    s3,
    settings,
    run_id: str,
    key: str,
    bucket: str = "",
):
    # Authorize one discovered artifact by exact server-resolved run membership.
    # This is the read-only bridge for cross-project artifacts; a caller-chosen
    # substring/path shape is never an authorization fact.
    normalized_run = validate_run_id(run_id)
    normalized_key = _safe_artifact_key(key)
    accessible = list(_agent_s3_buckets(s3, settings))[
        :_MAX_ARTIFACT_MEMBERSHIP_BUCKETS
    ]
    requested_bucket = str(bucket or "").strip()
    if requested_bucket:
        if requested_bucket not in accessible:
            raise HTTPException(
                status_code=400,
                detail="artifact bucket is outside effective agent access",
            )
        candidates = [requested_bucket]
    else:
        candidates = accessible
    key_parts = [part for part in normalized_key.strip("/").split("/") if part]
    try:
        run_index = key_parts.index(normalized_run)
    except ValueError:
        run_index = -1
    exact_prefix = "/".join(key_parts[:run_index]) if run_index >= 0 else None
    for candidate in candidates:
        try:
            sources, _source_errors, _discovery_complete = (
                find_run_sources_across_buckets(
                    [candidate],
                    base_prefix=str((settings or {}).get("prefix") or ""),
                    run_id=normalized_run,
                    exact_prefix=exact_prefix if requested_bucket else None,
                    s3=s3,
                )
            )
        except Exception:
            continue
        for source in sources:
            resolved_prefix = str(source.resolved_prefix or "").strip().strip("/")
            discovered_scope = (
                "/".join(part for part in (resolved_prefix, normalized_run) if part)
                + "/"
            )
            if not normalized_key.startswith(discovered_scope):
                continue
            try:
                s3.head_object(Bucket=candidate, Key=normalized_key)
            except Exception:
                continue
            return candidate, normalized_key, normalized_run
    raise HTTPException(
        status_code=404,
        detail="artifact is not a discovered object for this run",
    )


def _authorize_agent_artifact_uri(*, s3, settings, uri: str, run_id: str = ""):
    bucket, key = parse_s3_uri(uri)
    if not str(run_id or "").strip():
        raise HTTPException(
            status_code=400,
            detail="s3_uri requires a run_id and exact discovered artifact membership",
        )
    return _resolve_accessible_run_artifact(
        s3=s3,
        settings=settings,
        run_id=run_id,
        key=key,
        bucket=bucket,
    )
