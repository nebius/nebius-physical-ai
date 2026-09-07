"""Shared policy for the NPA Agent's S3 artifact discovery routes.

The generated VM backend owns FastAPI registration and supplies its authorized
S3 clients, access report, and opaque cursor store. This shipped module owns the
response policy those routes must use. Keeping the policy importable avoids a
second, dormant route implementation drifting away from what bootstrap serves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


ARTIFACT_ROUTE_POLICY_CONTRACT = "npa.agent.artifact-route-policy/v1"
ARTIFACT_DISCOVERY_CONTRACT = "s3-source-qualified-v1"
ARTIFACT_RUN_NAMESPACE = "npa_workflow_artifact_run"
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
    """Fail-closed result of resolving one run identifier.

    ``selected`` is populated only when the caller may continue to native S3
    artifact pagination. Otherwise ``body`` is the complete public response and
    ``status_code`` is its HTTP status.
    """

    status_code: int
    selected: Any | None = None
    body: dict[str, Any] | None = None
    source_resolution_complete: bool = False


def artifact_run_snapshot_scope(
    *,
    query: str,
    prefix: str,
    resource_scope: Mapping[str, Any] | None,
    source_tuples: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Build the complete, deterministic scope bound to an opaque cursor.

    Source identity includes project, bucket, and exact run-parent prefix. A
    cursor generated for one query or source selection therefore cannot be
    replayed after the caller changes filters or effective authorization.
    """

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
    selected = {
        str(key): str(value or "").strip()
        for key, value in dict(resource_scope or {}).items()
        if str(value or "").strip()
    }
    return {
        "contract": ARTIFACT_ROUTE_POLICY_CONTRACT,
        "query": str(query or "").strip(),
        "prefix": str(prefix or "").strip().strip("/"),
        "resource_scope": selected,
        "sources": [list(item) for item in sorted(set(normalized_sources))],
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

    On the first request, the fully refreshed bounded index is filtered before
    being handed to ``snapshot_page``. Continuation requests pass ``page=None``
    and resume the stored snapshot, so a stale or reordered discovery cache
    cannot skip or duplicate rows midway through traversal.
    """

    size = max(1, min(int(page_size), 500))
    normalized_query = str(query or "").strip()
    normalized_cursor = str(cursor or "").strip()
    if normalized_cursor:
        if page is not None:
            raise ValueError("continuation pages must use the stored run snapshot")
        visible, next_cursor, metadata = snapshot_page(
            cursor=normalized_cursor,
            context=str(snapshot_context or ""),
            limit=size,
            items=None,
            metadata=None,
        )
    else:
        if page is None:
            raise ValueError("the first run-list page requires refreshed discovery")
        indexed = list(getattr(page, "runs", ()) or ())
        if normalized_query:
            needle = normalized_query.lower()
            indexed = [
                item
                for item in indexed
                if needle in str(getattr(item, "run_id", "") or "").lower()
            ]
        source_errors = [
            _public_source_error(item)
            for item in (getattr(page, "source_errors", ()) or ())
        ]
        source_index_complete = bool(
            not bool(getattr(page, "truncated", False))
            and bool(getattr(page, "discovery_complete", False))
            and not source_errors
        )
        query_complete = bool(source_index_complete and effective_scope_complete)
        if not query_complete and not any(
            item.get("code") == "artifact_search_incomplete" for item in source_errors
        ):
            source_errors.append(
                {
                    "code": "artifact_search_incomplete",
                    "message": _INCOMPLETE_MESSAGE,
                }
            )
        metadata = {
            "contract": ARTIFACT_ROUTE_POLICY_CONTRACT,
            "observed_run_count": int(getattr(page, "total_runs", len(indexed))),
            "observed_match_count": len(indexed),
            "query_complete": query_complete,
            "source_index_truncated": bool(getattr(page, "truncated", False)),
            "source_errors": source_errors,
        }
        visible, next_cursor, metadata = snapshot_page(
            cursor="",
            context=str(snapshot_context or ""),
            limit=size,
            items=indexed,
            metadata=metadata,
        )

    if metadata.get("contract") != ARTIFACT_ROUTE_POLICY_CONTRACT:
        raise ValueError("run-list snapshot uses an incompatible policy contract")
    query_complete = bool(metadata.get("query_complete"))
    observed_match_count = int(metadata.get("observed_match_count") or 0)
    source_index_truncated = bool(metadata.get("source_index_truncated"))
    response = dict(response_context or {})
    response.update(
        {
            "ok": True,
            "contract": ARTIFACT_DISCOVERY_CONTRACT,
            "query": normalized_query,
            "runs": [_serialize_run(item) for item in visible],
            "count": len(visible),
            "count_scope": "page",
            "total_runs": observed_match_count if query_complete else None,
            "total_runs_scope": (
                "filtered_global"
                if normalized_query and query_complete
                else "global"
                if query_complete
                else "unavailable"
            ),
            "observed_run_count": int(metadata.get("observed_run_count") or 0),
            "observed_match_count": observed_match_count,
            "query_complete": query_complete,
            "limit": size,
            "cursor": normalized_cursor,
            "next_cursor": str(next_cursor or ""),
            "truncated": bool(
                next_cursor or source_index_truncated or not query_complete
            ),
            "pagination_complete": bool(not next_cursor and query_complete),
            "source_errors": [
                _public_source_error(item)
                for item in (metadata.get("source_errors") or ())
            ],
        }
    )
    return response


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
    """Serialize one native S3 artifact page without type/path filtering.

    Every object returned by storage remains visible, including unknown formats
    whose classifier chose the download fallback. The response always returns
    the complete server-issued source tuple required for the next page or load.
    """

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

    page_payload = (
        artifact_page.to_dict() if hasattr(artifact_page, "to_dict") else artifact_page
    )
    if not isinstance(page_payload, Mapping):
        raise TypeError("artifact page must serialize to a mapping")
    raw_artifacts = list(page_payload.get("artifacts") or ())
    serialized_artifacts = [_serialize_run(item) for item in raw_artifacts]
    next_cursor = str(page_payload.get("next_cursor") or "")
    page_size = max(1, min(int(page_payload.get("page_size") or 1000), 1000))

    response = dict(response_context or {})
    response.update(
        {
            "ok": True,
            "contract": ARTIFACT_DISCOVERY_CONTRACT,
            "bucket": bucket,
            "resource_bucket": bucket,
            "project_id": resolved_project,
            "prefix": resolved_prefix,
            "resolved_prefix": resolved_prefix,
            "base_prefix": str(base_prefix or "").strip().strip("/"),
            "run_id": run_id,
            "run_ref": run_ref,
            "source_selected": True,
            "pagination": {
                "contract": "one_native_s3_page",
                "max_objects": 1000,
                "continue_with": [
                    "next_cursor",
                    *ARTIFACT_EXACT_SOURCE_FIELDS,
                ],
            },
            "artifacts": serialized_artifacts,
            "count": len(serialized_artifacts),
            "truncated": bool(page_payload.get("truncated")),
            "next_cursor": next_cursor,
            "page_size": page_size,
            "preferred": _serialize_run(preferred) if preferred is not None else None,
            "access": dict(access or {}),
        }
    )
    return response


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
    """Resolve one run without guessing across incomplete or ambiguous scope.

    Known duplicate full source tuples are always a 409 until the caller sends a
    server-issued exact selection. Absence becomes a trustworthy 404 only when
    both storage discovery and effective access scope are exhaustive. A positive
    observation may proceed with its exact tuple while explicitly retaining an
    incomplete wider-scope result.
    """

    normalized_run = str(run_id or "").strip()
    errors = [_public_source_error(item) for item in source_errors]
    unique: dict[tuple[str, str, str, str], Any] = {}
    for match in matches:
        identity = _run_source_identity(match)
        if identity[3] != normalized_run:
            continue
        payload = _serialize_run(match)
        if (
            not identity[0]
            or not identity[1]
            or not str(payload.get("run_ref") or "").strip()
        ):
            errors.append(
                {
                    "code": "artifact_source_identity_incomplete",
                    "message": (
                        "A discovered run did not include a complete authorized "
                        "source identity."
                    ),
                }
            )
            continue
        unique.setdefault(identity, match)
    candidates = [unique[key] for key in sorted(unique)]
    public_access = dict(access or {})

    if len(candidates) > 1:
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
                "run_id": normalized_run,
                "namespace": ARTIFACT_RUN_NAMESPACE,
                "sources": [_source_payload(item) for item in candidates],
                "access": public_access,
                "source_errors": errors,
            },
        )

    search_complete = bool(
        discovery_complete
        and not errors
        and (exact_source_authorized or effective_scope_complete)
    )
    if not candidates:
        code = "run_not_discovered" if search_complete else "artifact_search_incomplete"
        return ArtifactRunLookupDecision(
            status_code=404 if search_complete else 503,
            body={
                "ok": False,
                "error": {
                    "code": code,
                    "message": (
                        _RUN_NOT_DISCOVERED_MESSAGE
                        if search_complete
                        else _INCOMPLETE_MESSAGE
                    ),
                },
                "run_id": normalized_run,
                "namespace": ARTIFACT_RUN_NAMESPACE,
                "access": public_access,
                "source_errors": errors,
            },
        )

    # A positive observation has a server-issued exact source tuple and may be
    # returned even if unrelated tenant sources were inaccessible. This does
    # not turn the wider search into a completeness claim: the caller exposes
    # ``source_resolution_complete=False`` and every subsequent read reuses the
    # selected tuple. Only absence remains blocked behind exhaustive discovery.
    return ArtifactRunLookupDecision(
        status_code=200,
        selected=candidates[0],
        source_resolution_complete=search_complete,
    )
