"""Shared policy for the NPA Agent's S3 artifact discovery routes.

The generated VM backend owns FastAPI registration and supplies its authorized
S3 clients, access report, and opaque cursor store. This shipped module owns the
response policy those routes must use. Keeping the policy importable avoids a
second, dormant route implementation drifting away from what bootstrap serves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


#: Version of the shared artifact-route policy consumed by the rendered backend.
ARTIFACT_ROUTE_POLICY_CONTRACT = "npa.agent.artifact-route-policy/v1"
#: Public response contract for source-qualified artifact discovery.
ARTIFACT_DISCOVERY_CONTRACT = "s3-source-qualified-v1"
#: Namespace that distinguishes workflow artifacts from local maintenance runs.
ARTIFACT_RUN_NAMESPACE = "npa_workflow_artifact_run"
#: Server-issued source fields required by every exact artifact request.
ARTIFACT_EXACT_SOURCE_FIELDS = (
    "run_ref",
    "project_id",
    "resource_bucket",
    "resolved_prefix",
    "source_selected",
)

_INCOMPLETE_MESSAGE = (
    "One or more authorized artifact sources could not be searched completely."
)
_RUN_NOT_DISCOVERED_MESSAGE = (
    "No discovered NPA workflow/artifact run has this identifier. Identifiers "
    "under /home/ubuntu/codex-runs are Codex maintenance job IDs, not NPA run IDs."
)


@dataclass(frozen=True)
class ArtifactRunLookupDecision:
    """Represent the fail-closed result of resolving one run identifier.

    Args:
        status_code: HTTP status for the lookup outcome.
        selected: Exact discovered run that may continue to S3 pagination.
        body: Complete public error or ambiguity response, when applicable.
        source_resolution_complete: Whether every effective source was searched.
    Returns:
        None.
    Raises:
        None.
    """

    status_code: int
    selected: Any | None = None
    body: dict[str, Any] | None = None
    source_resolution_complete: bool = False


def _normalized_snapshot_sources(
    source_tuples: Sequence[Sequence[str]],
) -> list[list[str]]:
    normalized_sources: list[tuple[str, str, str]] = []
    for value in source_tuples:
        if len(value) != 3:
            raise ValueError("artifact source identity must contain three fields")
        project_id, bucket, resolved_prefix = (
            str(item or "").strip() for item in value
        )
        if not project_id or not bucket:
            raise ValueError("artifact source identity is incomplete")
        normalized_sources.append((project_id, bucket, resolved_prefix.strip("/")))
    return [list(item) for item in sorted(set(normalized_sources))]


def _normalized_resource_scope(
    resource_scope: Mapping[str, Any] | None,
) -> dict[str, str]:
    return {
        str(key): str(value or "").strip()
        for key, value in dict(resource_scope or {}).items()
        if str(value or "").strip()
    }


def artifact_run_snapshot_scope(
    *,
    query: str,
    prefix: str,
    resource_scope: Mapping[str, Any] | None,
    source_tuples: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Build the deterministic source scope bound to an opaque cursor.

    Args:
        query: Run-id substring applied to the source index.
        prefix: Requested run-parent prefix.
        resource_scope: Exact project and bucket filters selected by the caller.
        source_tuples: Authorized project, bucket, and run-parent identities.
    Returns:
        A stable snapshot-context mapping.
    Raises:
        ValueError: A source identity is malformed or incomplete.
    """
    return {
        "contract": ARTIFACT_ROUTE_POLICY_CONTRACT,
        "query": str(query or "").strip(),
        "prefix": str(prefix or "").strip().strip("/"),
        "resource_scope": _normalized_resource_scope(resource_scope),
        "sources": _normalized_snapshot_sources(source_tuples),
    }


def _public_source_error(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {
            "code": "artifact_discovery_unavailable",
            "message": "Artifact discovery is unavailable for one authorized source.",
        }
    allowed = ("project_id", "bucket", "resolved_prefix", "code", "message")
    return {
        key: str(value.get(key) or "")
        for key in allowed
        if str(value.get(key) or "").strip()
    }


def _serialize_run(value: Any) -> dict[str, Any]:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    if not isinstance(payload, Mapping):
        raise TypeError("artifact run rows must serialize to mappings")
    return dict(payload)


def _filtered_run_rows(page: Any, query: str) -> list[Any]:
    indexed = list(getattr(page, "runs", ()) or ())
    if not query:
        return indexed
    needle = query.lower()
    return [
        item
        for item in indexed
        if needle in str(getattr(item, "run_id", "") or "").lower()
    ]


def _fresh_run_list_metadata(
    page: Any, indexed: Sequence[Any], effective_scope_complete: bool
) -> dict[str, Any]:
    source_errors = [
        _public_source_error(item)
        for item in (getattr(page, "source_errors", ()) or ())
    ]
    source_complete = bool(
        not bool(getattr(page, "truncated", False))
        and bool(getattr(page, "discovery_complete", False))
        and not source_errors
    )
    query_complete = bool(source_complete and effective_scope_complete)
    has_incomplete_error = any(
        item.get("code") == "artifact_search_incomplete" for item in source_errors
    )
    if not query_complete and not has_incomplete_error:
        source_errors.append(
            {"code": "artifact_search_incomplete", "message": _INCOMPLETE_MESSAGE}
        )
    return {
        "contract": ARTIFACT_ROUTE_POLICY_CONTRACT,
        "observed_run_count": int(getattr(page, "total_runs", len(indexed))),
        "observed_match_count": len(indexed),
        "query_complete": query_complete,
        "source_index_truncated": bool(getattr(page, "truncated", False)),
        "source_errors": source_errors,
    }


def _first_run_snapshot(
    page: Any | None,
    query: str,
    size: int,
    context: str,
    snapshot_page: Callable[..., tuple[list[Any], str, dict[str, Any]]],
    effective_scope_complete: bool,
) -> tuple[list[Any], str, dict[str, Any]]:
    if page is None:
        raise ValueError("the first run-list page requires refreshed discovery")
    indexed = _filtered_run_rows(page, query)
    metadata = _fresh_run_list_metadata(page, indexed, effective_scope_complete)
    return snapshot_page(
        cursor="", context=context, limit=size, items=indexed, metadata=metadata
    )


def _continuation_run_snapshot(
    page: Any | None,
    cursor: str,
    size: int,
    context: str,
    snapshot_page: Callable[..., tuple[list[Any], str, dict[str, Any]]],
) -> tuple[list[Any], str, dict[str, Any]]:
    if page is not None:
        raise ValueError("continuation pages must use the stored run snapshot")
    return snapshot_page(
        cursor=cursor, context=context, limit=size, items=None, metadata=None
    )


def _snapshot_run_page(
    page: Any | None,
    query: str,
    cursor: str,
    size: int,
    context: str,
    snapshot_page: Callable[..., tuple[list[Any], str, dict[str, Any]]],
    effective_scope_complete: bool,
) -> tuple[list[Any], str, dict[str, Any]]:
    if cursor:
        return _continuation_run_snapshot(page, cursor, size, context, snapshot_page)
    return _first_run_snapshot(
        page, query, size, context, snapshot_page, effective_scope_complete
    )


def _run_list_total_scope(query: str, query_complete: bool) -> str:
    if not query_complete:
        return "unavailable"
    if query:
        return "filtered_global"
    return "global"


def _run_list_identity_fields(
    visible: Sequence[Any], query: str, size: int, cursor: str, next_cursor: str
) -> dict[str, Any]:
    return {
        "ok": True,
        "contract": ARTIFACT_DISCOVERY_CONTRACT,
        "query": query,
        "runs": [_serialize_run(item) for item in visible],
        "count": len(visible),
        "count_scope": "page",
        "limit": size,
        "cursor": cursor,
        "next_cursor": str(next_cursor or ""),
    }


def _run_list_completeness_fields(
    metadata: Mapping[str, Any], query: str, next_cursor: str
) -> dict[str, Any]:
    query_complete = bool(metadata.get("query_complete"))
    observed_matches = int(metadata.get("observed_match_count") or 0)
    source_truncated = bool(metadata.get("source_index_truncated"))
    return {
        "total_runs": observed_matches if query_complete else None,
        "total_runs_scope": _run_list_total_scope(query, query_complete),
        "observed_run_count": int(metadata.get("observed_run_count") or 0),
        "observed_match_count": observed_matches,
        "query_complete": query_complete,
        "truncated": bool(next_cursor or source_truncated or not query_complete),
        "pagination_complete": bool(not next_cursor and query_complete),
        "source_errors": [
            _public_source_error(item) for item in (metadata.get("source_errors") or ())
        ],
    }


def _build_run_list_response(
    visible: Sequence[Any],
    query: str,
    size: int,
    cursor: str,
    next_cursor: str,
    metadata: Mapping[str, Any],
    response_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if metadata.get("contract") != ARTIFACT_ROUTE_POLICY_CONTRACT:
        raise ValueError("run-list snapshot uses an incompatible policy contract")
    response = dict(response_context or {})
    response.update(
        _run_list_identity_fields(visible, query, size, cursor, next_cursor)
    )
    response.update(_run_list_completeness_fields(metadata, query, next_cursor))
    return response


def _artifact_run_list_response(
    page: Any | None,
    *,
    query: str,
    page_size: int,
    cursor: str,
    snapshot_context: str,
    snapshot_page: Callable[..., tuple[list[Any], str, dict[str, Any]]],
    effective_scope_complete: bool,
    response_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    size = max(1, min(int(page_size), 500))
    normalized_query = str(query or "").strip()
    normalized_cursor = str(cursor or "").strip()
    visible, next_cursor, metadata = _snapshot_run_page(
        page,
        normalized_query,
        normalized_cursor,
        size,
        str(snapshot_context or ""),
        snapshot_page,
        effective_scope_complete,
    )
    return _build_run_list_response(
        visible,
        normalized_query,
        size,
        normalized_cursor,
        next_cursor,
        metadata,
        response_context,
    )


def build_artifact_run_list_response(
    page: Any | None,
    *,
    query: str,
    page_size: int,
    cursor: str,
    snapshot_context: str,
    snapshot_page: Callable[..., tuple[list[Any], str, dict[str, Any]]],
    effective_scope_complete: bool,
    response_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one immutable run-list page with honest completeness semantics.

    Args:
        page: Refreshed discovery page, or None for cursor continuation.
        query: Run-id substring bound to the snapshot.
        page_size: Maximum rows returned in this page.
        cursor: Opaque continuation cursor.
        snapshot_context: Authorization and query context digest.
        snapshot_page: Snapshot storage callback.
        effective_scope_complete: Whether every effective source was searchable.
        response_context: Additional route response fields.
    Returns:
        One source-qualified run-list response.
    Raises:
        ValueError: Snapshot input or contract state is invalid.
    """
    return _artifact_run_list_response(
        page,
        query=query,
        page_size=page_size,
        cursor=cursor,
        snapshot_context=snapshot_context,
        snapshot_page=snapshot_page,
        effective_scope_complete=effective_scope_complete,
        response_context=response_context,
    )


def _run_source_identity(value: Any) -> tuple[str, str, str, str]:
    def _field(name: str) -> str:
        if isinstance(value, Mapping):
            return str(value.get(name) or "").strip()
        return str(getattr(value, name, "") or "").strip()

    return (
        _field("project_id"),
        _field("bucket"),
        _field("resolved_prefix").strip("/"),
        _field("run_id"),
    )


def _source_payload(value: Any) -> dict[str, Any]:
    payload = _serialize_run(value)
    return {
        key: payload[key]
        for key in (
            "run_id",
            "run_ref",
            "project_id",
            "bucket",
            "resolved_prefix",
            "source_prefix",
            "summary_complete",
        )
        if key in payload
    }


def _selected_artifact_source(selected: Any, project_id: str) -> dict[str, str]:
    selected_payload = _serialize_run(selected)
    selected_project = str(selected_payload.get("project_id") or "").strip()
    resolved_project = str(project_id or selected_project).strip()
    if not resolved_project:
        raise ValueError("artifact run source project_id is required")
    if selected_project and selected_project != resolved_project:
        raise ValueError("artifact run source project_id conflicts with selection")
    bucket = str(selected_payload.get("bucket") or "").strip()
    resolved_prefix = (
        str(
            selected_payload.get("resolved_prefix")
            or selected_payload.get("source_prefix")
            or ""
        )
        .strip()
        .strip("/")
    )
    run_id = str(selected_payload.get("run_id") or "").strip()
    run_ref = str(selected_payload.get("run_ref") or "").strip()
    if not bucket or not run_id or not run_ref:
        raise ValueError("artifact run source identity is incomplete")
    return {
        "bucket": bucket,
        "project_id": resolved_project,
        "resolved_prefix": resolved_prefix,
        "run_id": run_id,
        "run_ref": run_ref,
    }


def _artifact_page_response_fields(artifact_page: Any) -> dict[str, Any]:
    page_payload = (
        artifact_page.to_dict() if hasattr(artifact_page, "to_dict") else artifact_page
    )
    if not isinstance(page_payload, Mapping):
        raise TypeError("artifact page must serialize to a mapping")
    raw_artifacts = list(page_payload.get("artifacts") or ())
    return {
        "artifacts": [_serialize_run(item) for item in raw_artifacts],
        "count": len(raw_artifacts),
        "truncated": bool(page_payload.get("truncated")),
        "next_cursor": str(page_payload.get("next_cursor") or ""),
        "page_size": max(1, min(int(page_payload.get("page_size") or 1000), 1000)),
    }


def _artifact_source_response_fields(
    source: Mapping[str, str], base_prefix: str
) -> dict[str, Any]:
    bucket = source["bucket"]
    prefix = source["resolved_prefix"]
    return {
        "ok": True,
        "contract": ARTIFACT_DISCOVERY_CONTRACT,
        "bucket": bucket,
        "resource_bucket": bucket,
        "project_id": source["project_id"],
        "prefix": prefix,
        "resolved_prefix": prefix,
        "base_prefix": str(base_prefix or "").strip().strip("/"),
        "run_id": source["run_id"],
        "run_ref": source["run_ref"],
        "source_selected": True,
        "pagination": {
            "contract": "one_native_s3_page",
            "max_objects": 1000,
            "continue_with": ["next_cursor", *ARTIFACT_EXACT_SOURCE_FIELDS],
        },
    }


def build_artifact_run_detail_response(
    *,
    selected: Any,
    artifact_page: Any,
    project_id: str,
    base_prefix: str,
    access: Mapping[str, Any] | None,
    preferred: Any | None = None,
    response_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one native S3 artifact page without type filtering.

    Args:
        selected: Server-selected source-qualified run.
        artifact_page: One native S3 artifact page.
        project_id: Authorized project identity for the source.
        base_prefix: Configured artifact discovery root.
        access: Public effective-access summary.
        preferred: Preferred viewable artifact, when one exists.
        response_context: Additional route response fields.
    Returns:
        A source-qualified artifact-page response.
    Raises:
        TypeError: An input page cannot serialize to a mapping.
        ValueError: The selected source identity is incomplete or inconsistent.
    """
    source = _selected_artifact_source(selected, project_id)
    response = dict(response_context or {})
    response.update(_artifact_source_response_fields(source, base_prefix))
    response.update(_artifact_page_response_fields(artifact_page))
    response["preferred"] = _serialize_run(preferred) if preferred is not None else None
    response["access"] = dict(access or {})
    return response


def _lookup_candidates(
    run_id: str, matches: Sequence[Any], errors: list[dict[str, str]]
) -> list[Any]:
    unique: dict[tuple[str, str, str, str], Any] = {}
    for match in matches:
        identity = _run_source_identity(match)
        if identity[3] != run_id:
            continue
        payload = _serialize_run(match)
        source_is_complete = bool(
            identity[0] and identity[1] and str(payload.get("run_ref") or "").strip()
        )
        if source_is_complete:
            unique.setdefault(identity, match)
            continue
        errors.append(
            {
                "code": "artifact_source_identity_incomplete",
                "message": (
                    "A discovered run did not include a complete authorized "
                    "source identity."
                ),
            }
        )
    return [unique[key] for key in sorted(unique)]


def _ambiguous_lookup_decision(
    run_id: str,
    candidates: Sequence[Any],
    errors: Sequence[Mapping[str, str]],
    access: Mapping[str, Any],
) -> ArtifactRunLookupDecision:
    return ArtifactRunLookupDecision(
        status_code=409,
        body={
            "ok": False,
            "error": {
                "code": "ambiguous_run_id",
                "message": (
                    "This run ID exists in multiple artifact sources; select "
                    "a project, bucket, and resolved prefix."
                ),
            },
            "run_id": run_id,
            "namespace": ARTIFACT_RUN_NAMESPACE,
            "sources": [_source_payload(item) for item in candidates],
            "access": dict(access),
            "source_errors": list(errors),
        },
    )


def _lookup_search_complete(
    discovery_complete: bool,
    errors: Sequence[Mapping[str, str]],
    effective_scope_complete: bool,
    exact_source_authorized: bool,
) -> bool:
    return bool(
        discovery_complete
        and not errors
        and (exact_source_authorized or effective_scope_complete)
    )


def _absent_lookup_decision(
    run_id: str,
    search_complete: bool,
    errors: Sequence[Mapping[str, str]],
    access: Mapping[str, Any],
) -> ArtifactRunLookupDecision:
    code = "run_not_discovered" if search_complete else "artifact_search_incomplete"
    message = _RUN_NOT_DISCOVERED_MESSAGE if search_complete else _INCOMPLETE_MESSAGE
    return ArtifactRunLookupDecision(
        status_code=404 if search_complete else 503,
        body={
            "ok": False,
            "error": {"code": code, "message": message},
            "run_id": run_id,
            "namespace": ARTIFACT_RUN_NAMESPACE,
            "access": dict(access),
            "source_errors": list(errors),
        },
    )


def _decide_artifact_run_lookup(
    run_id: str,
    matches: Sequence[Any],
    source_errors: Sequence[Mapping[str, Any]],
    discovery_complete: bool,
    effective_scope_complete: bool,
    exact_source_authorized: bool,
    access: Mapping[str, Any] | None,
) -> ArtifactRunLookupDecision:
    normalized_run = str(run_id or "").strip()
    errors = [_public_source_error(item) for item in source_errors]
    candidates = _lookup_candidates(normalized_run, matches, errors)
    public_access = dict(access or {})
    if len(candidates) > 1:
        return _ambiguous_lookup_decision(
            normalized_run, candidates, errors, public_access
        )
    search_complete = _lookup_search_complete(
        discovery_complete, errors, effective_scope_complete, exact_source_authorized
    )
    if not candidates:
        return _absent_lookup_decision(
            normalized_run, search_complete, errors, public_access
        )
    return ArtifactRunLookupDecision(
        status_code=200,
        selected=candidates[0],
        source_resolution_complete=search_complete,
    )


def decide_artifact_run_lookup(
    *,
    run_id: str,
    matches: Sequence[Any],
    source_errors: Sequence[Mapping[str, Any]],
    discovery_complete: bool,
    effective_scope_complete: bool,
    exact_source_authorized: bool,
    access: Mapping[str, Any] | None = None,
) -> ArtifactRunLookupDecision:
    """Resolve one run without guessing across incomplete source scope.

    Args:
        run_id: Exact run identifier requested by the caller.
        matches: Source-qualified runs observed during bounded discovery.
        source_errors: Sanitized failures from authorized source scans.
        discovery_complete: Whether storage discovery exhausted its bounds.
        effective_scope_complete: Whether effective access covered every source.
        exact_source_authorized: Whether the caller selected an exact source.
        access: Public effective-access summary.
    Returns:
        A fail-closed exact, ambiguous, absent, or incomplete decision.
    Raises:
        TypeError: A discovered run cannot serialize to a mapping.
    """
    return _decide_artifact_run_lookup(
        run_id,
        matches,
        source_errors,
        discovery_complete,
        effective_scope_complete,
        exact_source_authorized,
        access,
    )
