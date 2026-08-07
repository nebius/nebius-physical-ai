"""Secure artifact content routes embedded into the deployed agent backend."""

from __future__ import annotations

# This module is source-embedded after the backend and artifact helpers are
# defined. Names intentionally resolve in that generated backend namespace.
# ruff: noqa: F821,E501


def _summary_documents_for_run(s3, bucket: str, artifacts: list) -> dict:
    candidates = {
        "manifest.json",
        "npa-workflow/manifest.json",
        "evidence/training.json",
        "evidence/capacity.json",
        "evidence/collective.json",
        "checkpoints/npa_groot_finetune_manifest.json",
    }
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
        except Exception:
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
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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
    head = s3.head_object(Bucket=run_bucket, Key=str(artifact.key))
    total = int(head.get("ContentLength") or artifact.size or 0)
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
        headers["Content-Length"] = str(total)
        return Response(status_code=200, media_type=content_type, headers=headers)

    if render in {"json", "text"} and not download:
        if total:
            end = min(total - 1, INLINE_TEXT_MAX_BYTES)
            obj = s3.get_object(
                Bucket=run_bucket,
                Key=str(artifact.key),
                Range=f"bytes=0-{end}",
            )
            raw = obj["Body"].read(INLINE_TEXT_MAX_BYTES + 1)
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
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"ok": False, "error": str(exc), "source": "s3"},
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
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"image transcode failed: {exc}"
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
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"ok": False, "error": str(exc), "source": "s3"},
        )
