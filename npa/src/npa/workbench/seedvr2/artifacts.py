"""Probe, independently verify, and render review media for SeedVR2 runs."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

from npa.clients.storage import StorageClient
from npa.workbench.storage_scope import authorize_uri

from .runtime import (
    SeedVR2Error,
    _canonical_json,
    _create_work_directory,
    _decoded_frame_hashes,
    _ensure_artifacts_absent,
    _probe_video,
    _publish_verified,
    _sha256,
    _utc_now,
)
from .schemas import (
    MODEL_REVISION,
    PROBE_SCHEMA,
    RESULT_SCHEMA,
    REVIEW_SCHEMA,
    SOURCE_REVISION,
    VERIFICATION_SCHEMA,
    VideoArtifactRequest,
)


def _validate_artifact_request(
    request: VideoArtifactRequest,
    *,
    input_suffix: str,
    output_kind: str,
) -> None:
    source = authorize_uri(request.input_path, operation="read SeedVR2 artifact")
    destination = authorize_uri(request.output_path, operation="write SeedVR2 artifact")
    if source.kind != "s3" or not source.key.lower().endswith(input_suffix):
        raise SeedVR2Error(f"input_path must be one exact s3://{input_suffix} object")
    if destination.kind != "s3" or not destination.key:
        raise SeedVR2Error("output_path must use the S3 handoff contract")
    if output_kind == "json" and not destination.key.lower().endswith(".json"):
        raise SeedVR2Error("output_path must name one exact s3:// JSON object")


def _load_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedVR2Error("SeedVR2 result is not valid JSON") from exc
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "ok"
        or result.get("source", {}).get("revision") != SOURCE_REVISION
        or result.get("model", {}).get("revision") != MODEL_REVISION
    ):
        raise SeedVR2Error("SeedVR2 result identity or status is invalid")
    return result


def probe(
    request: VideoArtifactRequest,
    *,
    storage_factory: Callable[[], Any] = StorageClient.from_environment,
) -> dict[str, Any]:
    """Decode an S3 input and publish a readback-verified media manifest."""

    _validate_artifact_request(request, input_suffix=".mp4", output_kind="json")
    directory = _create_work_directory(request.run_id)
    storage = storage_factory()
    try:
        source = directory / "input.mp4"
        storage.download_file(request.input_path, str(source))
        hashes = _decoded_frame_hashes(source)
        document = {
            "schema": PROBE_SCHEMA,
            "status": "ok",
            "run_id": request.run_id,
            "created_at": _utc_now(),
            "input": {
                "uri": request.input_path,
                "sha256": _sha256(source),
                "media": _probe_video(source),
                "unique_decoded_frames": len(set(hashes)),
            },
        }
        target = directory / "probe.json"
        target.write_bytes(_canonical_json(document))
        _ensure_artifacts_absent(storage, [request.output_path])
        _publish_verified(storage, target, request.output_path, directory / "readback")
        shutil.rmtree(directory)
        return document
    except Exception:
        raise


def verify(
    request: VideoArtifactRequest,
    *,
    storage_factory: Callable[[], Any] = StorageClient.from_environment,
) -> dict[str, Any]:
    """Recompute the delivered video hash and decode contract from result.json."""

    _validate_artifact_request(request, input_suffix=".json", output_kind="json")
    directory = _create_work_directory(request.run_id)
    storage = storage_factory()
    result_path = directory / "result.json"
    storage.download_file(request.input_path, str(result_path))
    result = _load_result(result_path)
    video_uri = str(result.get("artifacts", {}).get("restored_video", ""))
    authorize_uri(video_uri, operation="verify SeedVR2 restored video")
    video = directory / "restored.mp4"
    storage.download_file(video_uri, str(video))
    media = _probe_video(video)
    digest = _sha256(video)
    if digest != result.get("output", {}).get("sha256"):
        raise SeedVR2Error("restored video hash disagrees with result.json")
    if media != result.get("output", {}).get("media"):
        raise SeedVR2Error("restored video media metadata disagrees with result.json")
    document = {
        "schema": VERIFICATION_SCHEMA,
        "status": "ok",
        "run_id": request.run_id,
        "verified_at": _utc_now(),
        "result_uri": request.input_path,
        "result_sha256": _sha256(result_path),
        "restored_video_uri": video_uri,
        "restored_video_sha256": digest,
        "media": media,
    }
    target = directory / "verification.json"
    target.write_bytes(_canonical_json(document))
    _ensure_artifacts_absent(storage, [request.output_path])
    readback = directory / "readback"
    readback.mkdir()
    _publish_verified(storage, target, request.output_path, readback)
    shutil.rmtree(directory)
    return document


def _comparison_command(
    source: Path, restored: Path, output: Path, *, width: int, height: int
) -> list[str]:
    graph = (
        f"[0:v]scale={width}:{height}:flags=bicubic[baseline];"
        f"[1:v]scale={width}:{height}:flags=lanczos[candidate];"
        "[baseline][candidate]hstack=inputs=2[comparison]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-i",
        str(restored),
        "-filter_complex",
        graph,
        "-map",
        "[comparison]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]


def _selected_frames(frame_count: int) -> list[int]:
    if frame_count <= 9:
        return list(range(frame_count))
    return sorted({round(index * (frame_count - 1) / 8) for index in range(9)})


def _contact_sheet(comparison: Path, output: Path, frame_indices: list[int]) -> None:
    expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    columns = 3
    rows = (len(frame_indices) + columns - 1) // columns
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(comparison),
        "-vf",
        f"select='{expression}',tile={columns}x{rows}:padding=8:margin=8",
        "-frames:v",
        "1",
        str(output),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise SeedVR2Error("failed to render the matched review contact sheet")


def _review_html(document: dict[str, Any]) -> str:
    source_hash = escape(document["source"]["sha256"])
    restored_hash = escape(document["candidate"]["sha256"])
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>SeedVR2 review</title>
<style>body{{font:16px sans-serif;max-width:1200px;margin:auto;padding:2rem}}
video,img{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}</style>
<h1>SeedVR2 matched review</h1>
<p>Left: bicubic enlargement of the exact low-resolution input. Right: SeedVR2.
Generated detail is derived review media, not observed sensor truth.</p>
<video controls src="comparison.mp4"></video>
<h2>Fixed matched views</h2><img src="contact-sheet.png" alt="matched frames">
<h2>Artifact identity</h2>
<p>Input SHA-256: <code>{source_hash}</code><br>
Candidate SHA-256: <code>{restored_hash}</code></p>
<h2>Limits</h2><p>Do not use this comparison to claim recovered metric geometry,
calibration accuracy, or robot-policy success.</p></html>
"""


def _run_comparison(command: list[str], output: Path) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise SeedVR2Error("failed to render the matched baseline/candidate video")


def review(
    request: VideoArtifactRequest,
    *,
    storage_factory: Callable[[], Any] = StorageClient.from_environment,
) -> dict[str, Any]:
    """Build a non-blended comparison MP4, contact sheet, JSON, and HTML page."""

    _validate_artifact_request(request, input_suffix=".json", output_kind="prefix")
    directory = _create_work_directory(request.run_id)
    storage = storage_factory()
    result_path = directory / "result.json"
    storage.download_file(request.input_path, str(result_path))
    result = _load_result(result_path)
    source = directory / "source.mp4"
    candidate = directory / "restored.mp4"
    storage.download_file(result["input"]["uri"], str(source))
    storage.download_file(result["artifacts"]["restored_video"], str(candidate))
    if _sha256(source) != result["input"]["sha256"]:
        raise SeedVR2Error("review source hash disagrees with result.json")
    if _sha256(candidate) != result["output"]["sha256"]:
        raise SeedVR2Error("review candidate hash disagrees with result.json")
    return _render_and_publish_review(
        request, directory, storage, result, source, candidate
    )


def _render_and_publish_review(
    request: VideoArtifactRequest,
    directory: Path,
    storage: Any,
    result: dict[str, Any],
    source: Path,
    candidate: Path,
) -> dict[str, Any]:
    media = result["output"]["media"]
    comparison = directory / "comparison.mp4"
    _run_comparison(
        _comparison_command(
            source,
            candidate,
            comparison,
            width=media["width"],
            height=media["height"],
        ),
        comparison,
    )
    indices = _selected_frames(media["frames"])
    contact_sheet = directory / "contact-sheet.png"
    _contact_sheet(comparison, contact_sheet, indices)
    document = _review_document(request, result, source, candidate, comparison, indices)
    review_json = directory / "review.json"
    review_json.write_bytes(_canonical_json(document))
    page = directory / "index.html"
    page.write_text(_review_html(document), encoding="utf-8")
    _publish_review_files(request, directory, storage)
    shutil.rmtree(directory)
    return document


def _review_document(
    request: VideoArtifactRequest,
    result: dict[str, Any],
    source: Path,
    candidate: Path,
    comparison: Path,
    indices: list[int],
) -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "status": "ok",
        "run_id": request.run_id,
        "created_at": _utc_now(),
        "source": {
            "uri": result["input"]["uri"],
            "sha256": _sha256(source),
            "presentation_transform": "bicubic scale to candidate dimensions",
        },
        "candidate": {
            "uri": result["artifacts"]["restored_video"],
            "sha256": _sha256(candidate),
        },
        "comparison": {
            "sha256": _sha256(comparison),
            "selected_frame_indices": indices,
            "layout": "baseline-left,candidate-right",
            "blending": False,
        },
    }


def _publish_review_files(
    request: VideoArtifactRequest, directory: Path, storage: Any
) -> None:
    names = ("comparison.mp4", "contact-sheet.png", "review.json", "index.html")
    uris = [request.output_path.rstrip("/") + "/" + name for name in names]
    _ensure_artifacts_absent(storage, uris)
    readback = directory / "readback"
    readback.mkdir()
    for name, uri in zip(names, uris, strict=True):
        _publish_verified(storage, directory / name, uri, readback)


__all__ = ["probe", "review", "verify"]
