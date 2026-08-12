"""Secure artifact content routes embedded into the deployed agent backend."""

from __future__ import annotations

import logging
import re

# This module is source-embedded after the backend and artifact helpers are
# defined. Names intentionally resolve in that generated backend namespace.
# ruff: noqa: F821,E501

_artifact_content_logger = logging.getLogger("npa.agent.artifact_content")


def _apply_content_artifact(
    *,
    state: dict,
    run_id: str,
    key: str,
    bucket: str,
    s3_uri: str,
    render: str,
) -> dict:
    """Select a non-recording artifact without staging its S3 bytes locally."""
    now = _now_iso()
    sim_viz = dict(DEFAULT_SIM_VIZ)
    current = state.get("sim_viz")
    if isinstance(current, dict):
        sim_viz.update(current)
    query = (
        f"run_id={quote(run_id, safe='')}&key={quote(key, safe='')}&"
        f"bucket={quote(bucket, safe='')}"
    )
    content_url = f"/api/artifacts/content?{query}"
    download_url = f"{content_url}&download=true"
    previewable = render in {"json", "text", "image", "video"}
    sim_viz.update(
        {
            "run_id": run_id,
            "active_run_id": run_id,
            "stage": "artifact-selected",
            "rrd_uri": "",
            "rerun_iframe_url": "/rerun/",
            "rerun_ready": False,
            "artifact_uri": s3_uri,
            "artifact_key": key,
            "artifact_render": render,
            "artifact_preview_url": content_url if previewable else "",
            "artifact_download_url": download_url,
            "preview_status": "artifact_preview" if previewable else "download_only",
            "visualization_note": (
                f"Selected {render} artifact for secure same-origin preview."
                if previewable
                else "Binary/download-only artifact selected; metadata is shown without rendering bytes."
            ),
            "rrd_updated_at": now,
            "mode": "static",
        }
    )
    state["sim_viz"] = sim_viz
    _record_sim_viz_run(state, sim_viz)
    _save_state(state)
    return sim_viz


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
        if relative not in candidates or int(getattr(artifact, "size", 0) or 0) > INLINE_TEXT_MAX_BYTES:
            continue
        try:
            obj = s3.get_object(Bucket=bucket, Key=str(artifact.key))
            raw = obj["Body"].read(INLINE_TEXT_MAX_BYTES + 1)
            if len(raw) > INLINE_TEXT_MAX_BYTES:
                continue
            documents[relative] = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
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


def _resolved_artifact_for_content(
    s3,
    settings,
    *,
    run_id: str,
    key: str,
    requested_bucket: str = "",
):
    normalized_key = _safe_artifact_key(key)
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
            status_code=404, detail="artifact key is not present in the authorized run inventory"
        ) from exc
    artifact = next(item for item in artifacts if str(item.key) == normalized_key)
    artifact_bucket, artifact_key = parse_s3_uri(str(artifact.s3_uri))
    if artifact_bucket != run_bucket or artifact_key != normalized_key:
        raise HTTPException(
            status_code=400,
            detail="artifact bucket/key does not match the resolved run",
        )
    return normalized_run, run_bucket, artifact


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


def _artifact_content_response(
    request: Request,
    *,
    run_id: str,
    key: str,
    bucket: str = "",
    download: bool = False,
):
    s3, settings = _agent_s3_client()
    normalized_run, run_bucket, artifact = _resolved_artifact_for_content(
        s3,
        settings,
        run_id=run_id,
        key=key,
        requested_bucket=bucket,
    )
    render = str(artifact.render or "download")
    category = artifact_category_for_relative_key(
        str(artifact.relative_key or ""), role=str(artifact.role or "output")
    )
    total = int(artifact.size or 0)
    inline_media = render in {"image", "video"}
    attachment = bool(download or not inline_media)
    content_type = artifact_media_type(str(artifact.key))
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
    }
    if request.method == "HEAD":
        head = s3.head_object(Bucket=run_bucket, Key=str(artifact.key))
        total = int(head.get("ContentLength") or 0)
        headers["Content-Length"] = str(total)
        return Response(status_code=200, media_type=content_type, headers=headers)

    # A caller-supplied Range header always requests raw object bytes, including
    # for JSON/text artifacts. Inline structured previews remain the default for
    # ordinary GETs, while standards-compliant range/download clients receive
    # 206 + Content-Range from the common streaming path below.
    if render in {"json", "text"} and not download and not request.headers.get("range"):
        if total:
            end = min(total - 1, INLINE_TEXT_MAX_BYTES)
            obj = s3.get_object(
                Bucket=run_bucket,
                Key=str(artifact.key),
                Range=f"bytes=0-{end}",
            )
            raw = obj["Body"].read(INLINE_TEXT_MAX_BYTES + 1)
            content_range = str(obj.get("ContentRange") or "")
            range_match = re.fullmatch(r"bytes \d+-\d+/(\d+)", content_range)
            actual_total = int(range_match.group(1)) if range_match else int(
                obj.get("ContentLength") or len(raw)
            )
            if actual_total != total:
                raise HTTPException(
                    status_code=409,
                    detail="artifact changed since inventory discovery; list the run again",
                )
        else:
            raw = b""
        preview = build_text_preview(
            raw,
            total_bytes=total,
            render=render,
            max_bytes=INLINE_TEXT_MAX_BYTES,
        )
        preview.update(
            {
                "ok": True,
                "run_id": normalized_run,
                "key": str(artifact.key),
                "category": category,
                "content_type": content_type,
            }
        )
        headers["Content-Disposition"] = safe_content_disposition(
            str(artifact.key), attachment=False
        )
        headers["X-NPA-Preview-Truncated"] = (
            "true" if preview["truncated"] else "false"
        )
        headers["X-NPA-Preview-Redacted"] = (
            "true" if preview["redacted"] else "false"
        )
        return JSONResponse(content=preview, headers=headers)

    range_value = str(request.headers.get("range") or "").strip()
    try:
        selected_range = parse_http_byte_range(range_value, total)
    except ArtifactDiscoveryError as exc:
        raise HTTPException(
            status_code=416,
            detail=str(exc),
            headers={"Content-Range": f"bytes */{total}", "Accept-Ranges": "bytes"},
        ) from exc
    get_kwargs = {"Bucket": run_bucket, "Key": str(artifact.key)}
    status_code = 200
    content_length = total
    if selected_range is not None:
        start, end = selected_range
        get_kwargs["Range"] = f"bytes={start}-{end}"
        status_code = 206
        content_length = end - start + 1
        headers["Content-Range"] = f"bytes {start}-{end}/{total}"
    obj = s3.get_object(**get_kwargs)
    actual_length = int(obj.get("ContentLength") or 0)
    actual_range = str(obj.get("ContentRange") or "")
    if selected_range is not None:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", actual_range)
        if match is None:
            obj["Body"].close()
            raise HTTPException(status_code=502, detail="S3 range response omitted Content-Range")
        actual_start, actual_end, actual_total = (int(value) for value in match.groups())
        if actual_total != total or (actual_start, actual_end) != selected_range:
            obj["Body"].close()
            raise HTTPException(
                status_code=409,
                detail="artifact changed since inventory discovery; list the run again",
            )
        headers["Content-Range"] = actual_range
        content_length = actual_end - actual_start + 1
    else:
        if actual_length != total:
            obj["Body"].close()
            raise HTTPException(
                status_code=409,
                detail="artifact changed since inventory discovery; list the run again",
            )
        content_length = actual_length
    headers["Content-Length"] = str(content_length)
    return StreamingResponse(
        _artifact_stream(obj["Body"]),
        status_code=status_code,
        media_type=content_type,
        headers=headers,
    )


@app.api_route("/artifacts/content", methods=["GET", "HEAD"])
def artifacts_content(
    request: Request,
    run_id: str,
    key: str,
    bucket: str = "",
    download: bool = False,
):
    try:
        return _artifact_content_response(
            request,
            run_id=run_id,
            key=key,
            bucket=bucket,
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


@app.api_route("/artifacts/file/{filename}", methods=["GET", "HEAD"])
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


@app.api_route("/artifacts/download", methods=["GET", "HEAD"])
def artifacts_download(
    request: Request,
    run_id: str = "",
    key: str = "",
    s3_uri: str = "",
    bucket: str = "",
):
    requested_uri = str(s3_uri or "").strip()
    requested_key = str(key or "").strip()
    requested_bucket = str(bucket or "").strip()
    if not str(run_id or "").strip():
        raise HTTPException(status_code=400, detail="run_id is required")
    try:
        if requested_uri:
            uri_bucket, uri_key = parse_s3_uri(requested_uri)
            if requested_key and requested_key != uri_key:
                raise HTTPException(
                    status_code=400, detail="s3_uri and key do not match"
                )
            if requested_bucket and requested_bucket != uri_bucket:
                raise HTTPException(
                    status_code=400, detail="s3_uri and bucket do not match"
                )
            requested_bucket, requested_key = uri_bucket, uri_key
        if not requested_key:
            raise HTTPException(status_code=400, detail="key is required")
        return _artifact_content_response(
            request,
            run_id=run_id,
            key=requested_key,
            bucket=requested_bucket,
            download=True,
        )
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
