"""Secure artifact content routes embedded into the deployed agent backend."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

try:
    from agent_backend.publication_reader import canonical_publication_uri
except ModuleNotFoundError:
    from npa.agent_backend.publication_reader import canonical_publication_uri

if TYPE_CHECKING:
    from npa.workflows.artifacts import (
        Artifact,
        artifact_role_for_relative_key,
        is_inline_render,
        render_hint_for_object,
        validate_run_id,
    )

# This module is source-embedded after the backend and artifact helpers are
# defined. Names intentionally resolve in that generated backend namespace.
# ruff: noqa: F821,E501

_artifact_content_logger = logging.getLogger("npa.agent.artifact_content")


def _summary_documents_for_run(s3, bucket: str, artifacts: list) -> dict:
    candidates = {
        "manifest.json",
        "npa-workflow/manifest.json",
        "evidence/training.json",
        "evidence/capacity.json",
        "evidence/collective.json",
    }
    candidates.update(GROOT_ARTIFACT_PATHS["training_manifest"])
    candidates.update(GROOT_ARTIFACT_PATHS["report"])
    documents = {}
    for artifact in artifacts:
        relative = str(getattr(artifact, "relative_key", "") or "").strip().lstrip("/")
        if (
            relative not in candidates
            or int(getattr(artifact, "size", 0) or 0) > INLINE_TEXT_MAX_BYTES
        ):
            continue
        try:
            obj = s3.get_object(Bucket=bucket, Key=str(artifact.key))
            raw = obj["Body"].read(INLINE_TEXT_MAX_BYTES + 1)
            if len(raw) > INLINE_TEXT_MAX_BYTES:
                continue
            documents[relative] = json.loads(raw.decode("utf-8"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ):
            _artifact_content_logger.warning(
                "Ignoring malformed summary document at key %s", artifact.key
            )
            continue
        except Exception:  # contained storage boundary; traceback stays server-side
            _artifact_content_logger.exception(
                "Could not read summary document at key %s", artifact.key
            )
            continue
    return documents


def _publication_bound_artifact_key(
    s3,
    bucket: str,
    run_root_key: str,
    key: str,
) -> str:
    try:
        return _resolve_committed_artifact_key(
            s3,
            bucket,
            run_root_key,
            key,
        )
    except PublicationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="the selected publication generation is not committed",
        ) from exc


def _authorized_artifact_from_head(
    s3,
    *,
    run_id: str,
    bucket: str,
    source_prefix: str,
    key: str,
):
    run_root_key = "/".join(part for part in (source_prefix, run_id) if part)
    resolved_key = _publication_bound_artifact_key(
        s3,
        bucket,
        run_root_key,
        key,
    )
    scope = f"{run_root_key}/"
    if not resolved_key.startswith(scope):
        raise HTTPException(
            status_code=404,
            detail="artifact key is outside the selected run source",
        )
    relative_key = resolved_key[len(scope) :]
    if not relative_key:
        raise HTTPException(
            status_code=404,
            detail="artifact key does not identify an object in the selected run",
        )
    head = s3.head_object(Bucket=bucket, Key=resolved_key)
    modified = head.get("LastModified")
    if hasattr(modified, "isoformat"):
        modified = modified.isoformat()
    render = render_hint_for_object(key=resolved_key)
    return Artifact(
        run_id=run_id,
        key=resolved_key,
        s3_uri=f"s3://{bucket}/{resolved_key}",
        size=int(head.get("ContentLength") or 0),
        last_modified=str(modified or ""),
        render=render,
        inline=is_inline_render(render),
        role=artifact_role_for_relative_key(relative_key),
        namespace=source_prefix or "<bucket-root>",
        relative_key=relative_key,
        source_etag=str(head.get("ETag") or "").strip(),
        source_version_id=str(head.get("VersionId") or "").strip(),
    )


def _resolved_artifact_for_content(
    s3,
    settings,
    *,
    run_id: str,
    key: str,
    requested_bucket: str = "",
    exact_membership: bool = False,
    source_authorized: bool = False,
    resolved_prefix: str = "",
):
    normalized_key = _safe_artifact_key(key)
    if exact_membership and source_authorized:
        # The caller already re-proved the complete server-issued source tuple
        # (run_ref + project + bucket + prefix) through
        # ``_authorize_exact_run_ref_source``. Keep the object lookup on that
        # exact path: validate run-prefix membership and HEAD only this key.
        # Re-enumerating the run here is both slower and less precise when a
        # source contains duplicate basenames or more than one native S3 page.
        normalized_run = validate_run_id(run_id)
        run_bucket = str(requested_bucket or "").strip()
        if not run_bucket:
            raise HTTPException(
                status_code=400,
                detail="resource bucket is required for exact artifact playback",
            )
        source_prefix = _validated_resolved_prefix(resolved_prefix)
        discovered_scope = (
            "/".join(part for part in (source_prefix, normalized_run) if part) + "/"
        )
        if not normalized_key.startswith(discovered_scope):
            raise HTTPException(
                status_code=404,
                detail="artifact key is outside the selected run source",
            )
        artifact = _authorized_artifact_from_head(
            s3,
            run_id=normalized_run,
            bucket=run_bucket,
            source_prefix=source_prefix,
            key=normalized_key,
        )
        return normalized_run, run_bucket, artifact
    if exact_membership:
        # A load request that carries an exact bucket/key tuple must stay on the
        # bounded membership path.  Re-listing the whole run here both loses the
        # caller's source precision for duplicate basenames and can turn a
        # single-object authorization check into a multi-second bucket scan.
        normalized_run = validate_run_id(run_id)
        run_bucket, normalized_key, normalized_run = _resolve_accessible_run_artifact(
            s3=s3,
            settings=settings,
            run_id=normalized_run,
            key=normalized_key,
            bucket=requested_bucket,
        )
        key_parts = [part for part in normalized_key.split("/") if part]
        run_index = key_parts.index(normalized_run)
        source_prefix = "/".join(key_parts[:run_index])
        artifact = _authorized_artifact_from_head(
            s3,
            run_id=normalized_run,
            bucket=run_bucket,
            source_prefix=source_prefix,
            key=normalized_key,
        )
        return normalized_run, run_bucket, artifact
    try:
        normalized_run, run_bucket, artifacts, _ = _resolved_run_artifacts(
            s3, settings, run_id
        )
    except HTTPException:
        raise
    except ArtifactDiscoveryError as exc:
        raise HTTPException(
            status_code=400, detail="invalid run artifact request"
        ) from exc
    supplied_bucket = str(requested_bucket or "").strip()
    if supplied_bucket and supplied_bucket != run_bucket:
        raise HTTPException(
            status_code=400,
            detail="artifact bucket does not match the resolved run bucket",
        )
    try:
        normalized_key = authorize_artifact_inventory_key(
            normalized_run,
            normalized_key,
            [str(item.key) for item in artifacts],
        )
    except ArtifactDiscoveryError as exc:
        raise HTTPException(
            status_code=404,
            detail="artifact key is not present in the authorized run inventory",
        ) from exc
    run_root_key = _run_root_key(artifacts, normalized_run)
    normalized_key = _publication_bound_artifact_key(
        s3,
        run_bucket,
        run_root_key,
        normalized_key,
    )
    artifact = next(
        (item for item in artifacts if str(item.key) == normalized_key),
        None,
    )
    if artifact is None or not (artifact.source_etag or artifact.source_version_id):
        source_prefix = run_root_key[: -(len(normalized_run) + 1)]
        artifact = _authorized_artifact_from_head(
            s3,
            run_id=normalized_run,
            bucket=run_bucket,
            source_prefix=source_prefix,
            key=normalized_key,
        )
    artifact_bucket, artifact_key = parse_s3_uri(str(artifact.s3_uri))
    if artifact_bucket != run_bucket or artifact_key != normalized_key:
        raise HTTPException(
            status_code=400,
            detail="artifact bucket/key does not match the resolved run",
        )
    return normalized_run, run_bucket, artifact


def _exact_artifact_source(
    *,
    s3,
    settings,
    run_id: str,
    run_ref: str,
    project_id: str,
    resource_bucket: str,
    resolved_prefix: str | None,
    source_selected: bool,
    key: str,
):
    """Authorize one server-selected artifact without raw-URI rediscovery."""
    missing = []
    if not str(run_id or "").strip():
        missing.append("run_id")
    if not str(run_ref or "").strip():
        missing.append("run_ref")
    if not str(project_id or "").strip():
        missing.append("project_id")
    if not str(resource_bucket or "").strip():
        missing.append("resource_bucket")
    if resolved_prefix is None:
        missing.append("resolved_prefix")
    if not source_selected:
        missing.append("source_selected")
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "schema": "npa.agent.api_error/v1",
                "code": "exact_artifact_source_required",
                "message": (
                    "Artifact preview/download requires the exact server-selected "
                    "run source. List the run again and use its scoped card action."
                ),
                "missing_fields": missing,
            },
        )
    source_bucket, _source_project, source_prefix = _authorize_exact_run_ref_source(
        s3=s3,
        settings=settings,
        run_id=str(run_id or "").strip(),
        run_ref=str(run_ref or "").strip(),
        resource_bucket=str(resource_bucket or "").strip(),
        project_id=str(project_id or "").strip(),
        resolved_prefix=str(resolved_prefix or ""),
    )
    return _resolved_artifact_for_content(
        s3,
        settings,
        run_id=str(run_id or "").strip(),
        key=key,
        requested_bucket=source_bucket,
        exact_membership=True,
        source_authorized=True,
        resolved_prefix=source_prefix,
    )


def _artifact_stream(body):
    try:
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        try:
            body.close()
        except (AttributeError, OSError, RuntimeError):
            pass


def _artifact_read_conditions(artifact, *, required: bool) -> dict[str, str]:
    """Pin a content read to the exact object version authorized by HEAD."""

    conditions: dict[str, str] = {}
    etag = str(getattr(artifact, "source_etag", "") or "").strip()
    version_id = str(getattr(artifact, "source_version_id", "") or "").strip()
    if etag:
        conditions["IfMatch"] = etag
    if version_id:
        conditions["VersionId"] = version_id
    if required and not conditions:
        raise HTTPException(
            status_code=409,
            detail="artifact object identity is unavailable; list the run again",
        )
    return conditions


def _get_authorized_artifact_object(s3, **kwargs):
    try:
        return s3.get_object(**kwargs)
    except ClientError as exc:
        error = exc.response.get("Error", {})
        code = str(error.get("Code", ""))
        status = int(
            exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0
        )
        if code in {
            "409",
            "412",
            "ConditionalRequestConflict",
            "PreconditionFailed",
        } or status in {409, 412}:
            raise HTTPException(
                status_code=409,
                detail="artifact changed after its exact object version was authorized",
            ) from exc
        raise


def _publication_snapshot_for_artifact(s3, run_bucket: str, artifact):
    artifact_uri = str(artifact.s3_uri or "")
    try:
        canonical_uri = canonical_publication_uri(artifact_uri)
    except ValueError:
        return None, None
    publication = resolve_committed_publication(
        lambda uri: _read_publication_journal(s3, run_bucket, uri),
        canonical_uri,
    )
    if publication.journaled:
        target = publication.target(canonical_uri)
        if target.immutable_uri != artifact_uri:
            raise PublicationConflict(
                "artifact is not part of the current committed publication"
            )
        return publication, target
    if artifact_uri != canonical_uri:
        raise PublicationConflict(
            "immutable publication artifact has no committed journal authority"
        )
    return publication, publication.target(canonical_uri)


def _verified_publication_artifact_body(s3, run_bucket: str, artifact):
    """Stage and verify the exact reserved bytes that the response will serve."""

    publication, target = _publication_snapshot_for_artifact(
        s3,
        run_bucket,
        artifact,
    )
    if publication is None:
        return None
    body = None
    staged = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
    try:
        response = _get_authorized_artifact_object(
            s3,
            Bucket=run_bucket,
            Key=str(artifact.key),
            **_artifact_read_conditions(artifact, required=not publication.journaled),
        )
        body = response["Body"]
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            material = chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
            digest.update(material)
            staged.write(material)
            size += len(material)
        if publication.journaled and (
            size != int(target.size_bytes) or digest.hexdigest() != str(target.sha256)
        ):
            raise PublicationConflict(
                "committed publication object bytes disagree with its journal"
            )
        if not publication.journaled:
            if size != int(artifact.size or 0):
                raise PublicationConflict(
                    "legacy publication alias changed after authorization"
                )
            if (
                _read_publication_journal(
                    s3,
                    run_bucket,
                    publication.journal_uri,
                )
                is not None
            ):
                raise PublicationConflict(
                    "publication journal appeared during a legacy alias read"
                )
        staged.seek(0)
        return staged, size
    except Exception:
        staged.close()
        raise
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()


def _verify_downloaded_publication_artifact(
    s3,
    run_bucket: str,
    artifact,
    path,
) -> None:
    """Bind a completed local download to the current publication snapshot."""

    publication, target = _publication_snapshot_for_artifact(
        s3,
        run_bucket,
        artifact,
    )
    if publication is None:
        return
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if publication.journaled and (
        size != int(target.size_bytes) or digest.hexdigest() != str(target.sha256)
    ):
        raise PublicationConflict(
            "downloaded publication object bytes disagree with its journal"
        )
    if not publication.journaled:
        if size != int(artifact.size or 0):
            raise PublicationConflict(
                "legacy publication alias changed after authorization"
            )
        if (
            _read_publication_journal(
                s3,
                run_bucket,
                publication.journal_uri,
            )
            is not None
        ):
            raise PublicationConflict(
                "publication journal appeared during a legacy alias download"
            )


def _verified_legacy_viewer_body(s3, run_bucket, artifact, publication):
    if publication is not None:
        body, _size = _verified_publication_artifact_body(s3, run_bucket, artifact)
        return body
    response = _get_authorized_artifact_object(
        s3,
        Bucket=run_bucket,
        Key=str(artifact.key),
        **_artifact_read_conditions(artifact, required=True),
    )
    return response["Body"]


def _assert_viewer_download_matches_body(body, expected_digest, expected_size):
    try:
        digest = hashlib.sha256()
        size = 0
        while chunk := body.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        if size != expected_size or digest.digest() != expected_digest:
            raise PublicationConflict(
                "legacy viewer download changed after authorization"
            )
    finally:
        body.close()


def _download_verified_viewer_artifact(
    s3, run_bucket: str, artifact, *, recordings_dir
):
    """Expose immutable cache bytes only after the staged download is verified."""
    root = Path(recordings_dir)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".viewer-download-", dir=root) as directory:
        staged = Path(directory) / "artifact"
        download_s3_uri(artifact.s3_uri, staged, s3=s3)
        _verify_downloaded_publication_artifact(s3, run_bucket, artifact, staged)
        digest = hashlib.sha256()
        size = 0
        with staged.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        publication, _target = _publication_snapshot_for_artifact(
            s3, run_bucket, artifact
        )
        if publication is None or not publication.journaled:
            # Legacy aliases have no journal digest: bind to the authorized
            # conditional read rather than trusting download_file plus size.
            verified = _verified_legacy_viewer_body(
                s3, run_bucket, artifact, publication
            )
            _assert_viewer_download_matches_body(verified, digest.digest(), size)
        source = hashlib.sha256(f"{run_bucket}/{artifact.key}".encode()).hexdigest()
        suffix = Path(str(artifact.key)).suffix[:16]
        verified_path = root / f"{source}-{digest.hexdigest()}{suffix}"
        staged.replace(verified_path)
        return verified_path


def _artifact_file_stream(body, *, start: int = 0, length: int | None = None):
    try:
        body.seek(start)
        remaining = length
        while remaining is None or remaining > 0:
            chunk = body.read(
                1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            )
            if not chunk:
                break
            if remaining is not None:
                remaining -= len(chunk)
            yield chunk
    finally:
        body.close()


def _artifact_content_response(
    request: Request,
    *,
    run_id: str,
    run_ref: str,
    key: str,
    project_id: str,
    resource_bucket: str,
    resolved_prefix: str | None,
    source_selected: bool,
    download: bool = False,
):
    """Authorize and open one artifact against a pinned access generation.

    The returned streaming response already owns an opened S3 body, so the
    lease may be released after this helper returns. Holding it through exact
    source authorization, HEAD validation, and ``get_object`` prevents an old
    request from repopulating an exact-source proof after access refresh has
    published a new generation.
    """
    _begin_agent_artifact_access()
    try:
        return _artifact_content_response_with_access(
            request,
            run_id=run_id,
            run_ref=run_ref,
            key=key,
            project_id=project_id,
            resource_bucket=resource_bucket,
            resolved_prefix=resolved_prefix,
            source_selected=source_selected,
            download=download,
        )
    finally:
        _end_agent_artifact_access()


def _authorized_artifact_content(
    run_id: str,
    run_ref: str,
    key: str,
    project_id: str,
    resource_bucket: str,
    resolved_prefix: str | None,
    source_selected: bool,
):
    s3, settings = _agent_artifact_s3_client()
    selected = _exact_artifact_source(
        s3=s3,
        settings=settings,
        run_id=run_id,
        run_ref=run_ref,
        key=key,
        project_id=project_id,
        resource_bucket=resource_bucket,
        resolved_prefix=resolved_prefix,
        source_selected=source_selected,
    )
    return (s3, *selected)


class _ArtifactContentResponseContext:
    """Keep response metadata together without requiring module registration."""

    __slots__ = ("render", "category", "total", "content_type", "headers")

    def __init__(
        self,
        render: str,
        category: str,
        total: int,
        content_type: str,
        headers: dict[str, str],
    ) -> None:
        self.render = render
        self.category = category
        self.total = total
        self.content_type = content_type
        self.headers = headers


def _artifact_content_response_context(
    normalized_run: str, run_ref: str, artifact, download: bool
) -> _ArtifactContentResponseContext:
    render = str(artifact.render or "download")
    category = artifact_category_for_relative_key(
        str(artifact.relative_key or ""), role=str(artifact.role or "output")
    )
    total = int(artifact.size or 0)
    content_type = artifact_media_type(str(artifact.key))
    attachment = bool(download or render not in {"image", "video"})
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store",
        "Content-Disposition": safe_content_disposition(
            str(artifact.key), attachment=attachment
        ),
        "X-Content-Type-Options": "nosniff",
        "X-NPA-Artifact-Category": category,
        "X-NPA-Artifact-Render": render,
        "X-NPA-Run-Id": normalized_run,
        "X-NPA-Run-Ref": str(run_ref or "").strip(),
        "X-NPA-Source-Selected": "true",
    }
    return _ArtifactContentResponseContext(
        render=render,
        category=category,
        total=total,
        content_type=content_type,
        headers=headers,
    )


def _artifact_head_response(
    s3, run_bucket: str, artifact, context: _ArtifactContentResponseContext
):
    try:
        publication, target = _publication_snapshot_for_artifact(
            s3,
            run_bucket,
            artifact,
        )
        head = s3.head_object(Bucket=run_bucket, Key=str(artifact.key))
        expected_etag = str(getattr(artifact, "source_etag", "") or "").strip()
        expected_version = str(getattr(artifact, "source_version_id", "") or "").strip()
        if (expected_etag and str(head.get("ETag") or "").strip() != expected_etag) or (
            expected_version
            and str(head.get("VersionId") or "").strip() != expected_version
        ):
            raise PublicationConflict(
                "artifact changed after its exact object version was authorized"
            )
        if publication is not None:
            if publication.journaled and int(head.get("ContentLength") or -1) != int(
                target.size_bytes
            ):
                raise PublicationConflict(
                    "committed publication object size disagrees with its journal"
                )
            if (
                not publication.journaled
                and _read_publication_journal(
                    s3,
                    run_bucket,
                    publication.journal_uri,
                )
                is not None
            ):
                raise PublicationConflict(
                    "publication journal appeared during a legacy alias read"
                )
    except PublicationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="the selected publication generation is not committed",
        ) from exc
    total = int(head.get("ContentLength") or 0)
    context.headers["Content-Length"] = str(total)
    return Response(
        status_code=200,
        media_type=context.content_type,
        headers=context.headers,
    )


def _read_artifact_text_preview(s3, run_bucket: str, artifact, total: int) -> bytes:
    if not total:
        return b""
    try:
        verified = _verified_publication_artifact_body(s3, run_bucket, artifact)
    except PublicationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="the selected publication generation is not committed",
        ) from exc
    if verified is not None:
        body, actual_total = verified
        try:
            if actual_total != total:
                raise HTTPException(
                    status_code=409,
                    detail="artifact changed since inventory discovery; list the run again",
                )
            return body.read(INLINE_TEXT_MAX_BYTES + 1)
        finally:
            body.close()
    end = min(total - 1, INLINE_TEXT_MAX_BYTES)
    obj = _get_authorized_artifact_object(
        s3,
        Bucket=run_bucket,
        Key=str(artifact.key),
        Range=f"bytes=0-{end}",
        **_artifact_read_conditions(artifact, required=True),
    )
    raw = obj["Body"].read(INLINE_TEXT_MAX_BYTES + 1)
    content_range = str(obj.get("ContentRange") or "")
    range_match = re.fullmatch(r"bytes \d+-\d+/(\d+)", content_range)
    actual_total = (
        int(range_match.group(1))
        if range_match
        else int(obj.get("ContentLength") or len(raw))
    )
    if actual_total != total:
        raise HTTPException(
            status_code=409,
            detail="artifact changed since inventory discovery; list the run again",
        )
    return raw


def _artifact_text_preview_response(
    s3,
    run_bucket: str,
    artifact,
    normalized_run: str,
    context: _ArtifactContentResponseContext,
):
    raw = _read_artifact_text_preview(s3, run_bucket, artifact, context.total)
    preview = build_text_preview(
        raw,
        total_bytes=context.total,
        render=context.render,
        max_bytes=INLINE_TEXT_MAX_BYTES,
    )
    preview.update(
        {
            "ok": True,
            "run_id": normalized_run,
            "key": str(artifact.key),
            "category": context.category,
            "content_type": context.content_type,
        }
    )
    context.headers["Content-Disposition"] = safe_content_disposition(
        str(artifact.key), attachment=False
    )
    context.headers["X-NPA-Preview-Truncated"] = (
        "true" if preview["truncated"] else "false"
    )
    context.headers["X-NPA-Preview-Redacted"] = (
        "true" if preview["redacted"] else "false"
    )
    return JSONResponse(content=preview, headers=context.headers)


def _artifact_requested_range(range_value: str, total: int):
    try:
        return parse_http_byte_range(range_value, total)
    except ArtifactDiscoveryError as exc:
        raise HTTPException(
            status_code=416,
            detail=str(exc),
            headers={"Content-Range": f"bytes */{total}", "Accept-Ranges": "bytes"},
        ) from exc


def _artifact_stream_request(
    run_bucket: str, artifact, selected_range, total: int, headers
):
    get_kwargs = {
        "Bucket": run_bucket,
        "Key": str(artifact.key),
        **_artifact_read_conditions(artifact, required=True),
    }
    status_code = 200
    if selected_range is not None:
        start, end = selected_range
        get_kwargs["Range"] = f"bytes={start}-{end}"
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{total}"
    return get_kwargs, status_code


def _validated_artifact_stream_length(obj, selected_range, total: int, headers) -> int:
    actual_length = int(obj.get("ContentLength") or 0)
    actual_range = str(obj.get("ContentRange") or "")
    if selected_range is None:
        if actual_length != total:
            obj["Body"].close()
            raise HTTPException(
                status_code=409,
                detail="artifact changed since inventory discovery; list the run again",
            )
        return actual_length
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", actual_range)
    if match is None:
        obj["Body"].close()
        raise HTTPException(
            status_code=502, detail="S3 range response omitted Content-Range"
        )
    actual_start, actual_end, actual_total = (int(value) for value in match.groups())
    if actual_total != total or (actual_start, actual_end) != selected_range:
        obj["Body"].close()
        raise HTTPException(
            status_code=409,
            detail="artifact changed since inventory discovery; list the run again",
        )
    headers["Content-Range"] = actual_range
    return actual_end - actual_start + 1


def _artifact_stream_response(
    s3,
    request: Request,
    run_bucket: str,
    artifact,
    context: _ArtifactContentResponseContext,
):
    range_value = str(request.headers.get("range") or "").strip()
    selected_range = _artifact_requested_range(range_value, context.total)
    try:
        verified = _verified_publication_artifact_body(s3, run_bucket, artifact)
    except PublicationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="the selected publication generation is not committed",
        ) from exc
    if verified is not None:
        body, actual_total = verified
        if actual_total != context.total:
            body.close()
            raise HTTPException(
                status_code=409,
                detail="artifact changed since inventory discovery; list the run again",
            )
        status_code = 200
        start = 0
        length = actual_total
        if selected_range is not None:
            start, end = selected_range
            length = end - start + 1
            status_code = 206
            context.headers["Content-Range"] = f"bytes {start}-{end}/{actual_total}"
        context.headers["Content-Length"] = str(length)
        return StreamingResponse(
            _artifact_file_stream(body, start=start, length=length),
            status_code=status_code,
            media_type=context.content_type,
            headers=context.headers,
        )
    get_kwargs, status_code = _artifact_stream_request(
        run_bucket, artifact, selected_range, context.total, context.headers
    )
    obj = _get_authorized_artifact_object(s3, **get_kwargs)
    content_length = _validated_artifact_stream_length(
        obj, selected_range, context.total, context.headers
    )
    context.headers["Content-Length"] = str(content_length)
    return StreamingResponse(
        _artifact_stream(obj["Body"]),
        status_code=status_code,
        media_type=context.content_type,
        headers=context.headers,
    )


def _artifact_content_response_with_access(
    request: Request,
    *,
    run_id: str,
    run_ref: str,
    key: str,
    project_id: str,
    resource_bucket: str,
    resolved_prefix: str | None,
    source_selected: bool,
    download: bool = False,
):
    s3, normalized_run, run_bucket, artifact = _authorized_artifact_content(
        run_id,
        run_ref,
        key,
        project_id,
        resource_bucket,
        resolved_prefix,
        source_selected,
    )
    context = _artifact_content_response_context(
        normalized_run, run_ref, artifact, download
    )
    if request.method == "HEAD":
        return _artifact_head_response(s3, run_bucket, artifact, context)
    render = context.render
    if render in {"json", "text"} and not download and not request.headers.get("range"):
        return _artifact_text_preview_response(
            s3, run_bucket, artifact, normalized_run, context
        )
    return _artifact_stream_response(s3, request, run_bucket, artifact, context)


@app.get("/artifacts/content", operation_id="artifacts_content_get")
@app.head("/artifacts/content", operation_id="artifacts_content_head")
def artifacts_content(
    request: Request,
    run_id: str = "",
    run_ref: str = "",
    key: str = "",
    project_id: str = "",
    resource_bucket: str = "",
    resolved_prefix: str | None = None,
    source_selected: bool = False,
    download: bool = False,
):
    try:
        return _artifact_content_response(
            request,
            run_id=run_id,
            run_ref=run_ref,
            key=key,
            project_id=project_id,
            resource_bucket=resource_bucket,
            resolved_prefix=resolved_prefix,
            source_selected=source_selected,
            download=download,
        )
    except HTTPException:
        raise
    except Exception:  # contained route boundary; preserve traceback in server logs
        _artifact_content_logger.exception("Artifact content storage request failed")
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "error": "artifact storage request failed",
                "error_code": "artifact_storage_error",
                "source": "s3",
            },
        )


@app.get("/artifacts/file/{filename}", operation_id="artifact_file_get")
@app.head("/artifacts/file/{filename}", operation_id="artifact_file_head")
def artifact_file(filename: str):
    safe_name = Path(str(filename)).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="invalid artifact filename")
    target = RECORDINGS_DIR / safe_name
    if not target.is_file():
        raise HTTPException(
            status_code=404, detail=f"artifact file not found: {filename}"
        )
    if needs_image_transcode(safe_name):
        try:
            import io as _io

            from PIL import Image as _Image

            with _Image.open(target) as _im:
                _buf = _io.BytesIO()
                _im.convert("RGB").save(_buf, format="PNG")
            return Response(
                content=_buf.getvalue(),
                media_type="image/png",
                headers={
                    "Content-Disposition": safe_content_disposition(
                        safe_name, attachment=False
                    ),
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except (OSError, ValueError) as exc:
            _artifact_content_logger.exception(
                "Image transcode failed for local artifact %s", safe_name
            )
            raise HTTPException(
                status_code=500, detail="image transcode failed"
            ) from exc
    local_media_type = artifact_media_type(safe_name)
    local_inline = local_media_type.startswith("image/") or local_media_type.startswith(
        "video/"
    )
    return FileResponse(
        str(target),
        media_type=local_media_type,
        headers={
            "Content-Disposition": safe_content_disposition(
                safe_name, attachment=not local_inline
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


class _ArtifactDownloadSelection:
    """Carry one download request through the contained route boundary."""

    __slots__ = (
        "run_id",
        "run_ref",
        "key",
        "s3_uri",
        "project_id",
        "resource_bucket",
        "resolved_prefix",
        "source_selected",
    )

    def __init__(
        self,
        run_id: str,
        run_ref: str,
        key: str,
        s3_uri: str,
        project_id: str,
        resource_bucket: str,
        resolved_prefix: str | None,
        source_selected: bool,
    ) -> None:
        self.run_id = run_id
        self.run_ref = run_ref
        self.key = key
        self.s3_uri = s3_uri
        self.project_id = project_id
        self.resource_bucket = resource_bucket
        self.resolved_prefix = resolved_prefix
        self.source_selected = source_selected


def _authorized_artifact_download(
    request: Request, selected: _ArtifactDownloadSelection
):
    requested_uri = str(selected.s3_uri or "").strip()
    if requested_uri:
        raise HTTPException(
            status_code=400,
            detail=_raw_artifact_uri_migration_detail("s3_uri"),
        )
    requested_key = str(selected.key or "").strip()
    if not requested_key:
        raise HTTPException(status_code=400, detail="key is required")
    return _artifact_content_response(
        request,
        run_id=selected.run_id,
        run_ref=selected.run_ref,
        key=requested_key,
        project_id=selected.project_id,
        resource_bucket=str(selected.resource_bucket or "").strip(),
        resolved_prefix=selected.resolved_prefix,
        source_selected=selected.source_selected,
        download=True,
    )


def _artifact_download_response(request: Request, selected: _ArtifactDownloadSelection):
    try:
        return _authorized_artifact_download(request, selected)
    except HTTPException:
        raise
    except Exception:  # contained route boundary; preserve traceback in server logs
        _artifact_content_logger.exception("Artifact download storage request failed")
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "error": "artifact storage request failed",
                "error_code": "artifact_storage_error",
                "source": "s3",
            },
        )


@app.get("/artifacts/download", operation_id="artifacts_download_get")
@app.head("/artifacts/download", operation_id="artifacts_download_head")
def artifacts_download(
    request: Request,
    run_id: str = "",
    run_ref: str = "",
    key: str = "",
    s3_uri: str = "",
    project_id: str = "",
    resource_bucket: str = "",
    resolved_prefix: str | None = None,
    source_selected: bool = False,
):
    """Download one artifact from an exact server-selected run source.

    Args:
        request: The authenticated HTTP request, including any Range header.
        run_id: The validated workflow run identifier.
        run_ref: The opaque run reference issued by discovery.
        key: The exact object key issued by the selected run inventory.
        s3_uri: Deprecated raw URI selector, which is always rejected.
        project_id: The project from the server-issued source tuple.
        resource_bucket: The bucket from the server-issued source tuple.
        resolved_prefix: The run-parent prefix from the source tuple.
        source_selected: Whether the caller explicitly selected that source.

    Returns:
        An authorized download response or a sanitized storage-error response.

    Raises:
        HTTPException: If request fields or source authorization are invalid.
    """
    selected = _ArtifactDownloadSelection(
        run_id,
        run_ref,
        key,
        s3_uri,
        project_id,
        resource_bucket,
        resolved_prefix,
        source_selected,
    )
    return _artifact_download_response(request, selected)
