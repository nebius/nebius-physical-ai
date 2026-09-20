"""Probe, independently verify, and render review media for SeedVR2 runs."""

from __future__ import annotations

from fractions import Fraction
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable

from pydantic import ValidationError

from npa.clients.storage import StorageClient
from npa.workbench.storage_scope import authorize_uri

from .runtime import (
    MIN_H100_MEMORY_MIB,
    SOURCE_REVISION_PATH,
    SeedVR2Error,
    _canonical_json,
    _create_work_directory,
    _decoded_frame_hashes,
    _ensure_artifacts_absent,
    _media_environment,
    _probe_video,
    _publish_verified,
    _sha256,
    _utc_now,
    _validate_source_geometry,
)
from .schemas import (
    MODEL_FILES,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PROBE_SCHEMA,
    RESULT_SCHEMA,
    REVIEW_SCHEMA,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    RestoreRequest,
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
    sections = (
        (
            result.get("source"),
            result.get("model"),
            result.get("input"),
            result.get("output"),
            result.get("artifacts"),
        )
        if isinstance(result, dict)
        else ()
    )
    if (
        not isinstance(result, dict)
        or len(sections) != 5
        or not all(isinstance(section, dict) for section in sections)
        or result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "ok"
        or sections[0].get("revision") != SOURCE_REVISION
        or sections[1].get("revision") != MODEL_REVISION
    ):
        raise SeedVR2Error("SeedVR2 result identity or status is invalid")
    return result


def _authorized_result_uri(
    result: dict[str, Any],
    group: str,
    name: str,
    *,
    operation: str,
    suffix: str,
) -> str:
    section = result.get(group)
    if not isinstance(section, dict):
        raise SeedVR2Error(f"SeedVR2 result {group} is invalid")
    value = section.get(name, "")
    if not isinstance(value, str):
        raise SeedVR2Error(f"SeedVR2 result {group}.{name} is invalid")
    location = authorize_uri(value, operation=operation)
    if location.kind != "s3" or not location.key.lower().endswith(suffix):
        raise SeedVR2Error(
            f"SeedVR2 result {group}.{name} must be an S3 {suffix} object"
        )
    return value


def _validate_result_context(
    result: dict[str, Any],
    request: VideoArtifactRequest,
    storage: Any,
    directory: Path,
) -> None:
    try:
        restore_request = RestoreRequest.model_validate(result["request"])
    except (KeyError, TypeError, ValidationError) as exc:
        raise SeedVR2Error("SeedVR2 result request is invalid") from exc
    if (
        restore_request.dry_run
        or not restore_request.probe_path
        or result.get("run_id") != request.run_id
        or restore_request.run_id != request.run_id
    ):
        raise SeedVR2Error("SeedVR2 result does not bind the workflow run")

    runtime = result.get("runtime")
    gpu = runtime.get("gpu") if isinstance(runtime, dict) else None
    if not isinstance(runtime, dict) or not isinstance(gpu, dict):
        raise SeedVR2Error("SeedVR2 result runtime identity is invalid")
    current_image = os.environ.get("NPA_TASK_IMAGE", "")
    try:
        baked_source = SOURCE_REVISION_PATH.read_text(encoding="utf-8").strip()
        memory_mib = int(gpu.get("memory_mib", ""))
    except (OSError, TypeError, ValueError) as exc:
        raise SeedVR2Error("SeedVR2 result runtime identity is invalid") from exc
    image = runtime.get("image", "")
    valid_runtime = (
        re.fullmatch(r".+@sha256:[0-9a-f]{64}", current_image) is not None
        and image == current_image
        and runtime.get("image_digest") == image.rsplit("@", 1)[-1]
        and re.fullmatch(r"[0-9a-f]{40}", baked_source) is not None
        and runtime.get("npa_source_revision") == baked_source
        and gpu.get("status") == "available"
        and gpu.get("count") == "1"
        and gpu.get("compute_capability") == "9.0"
        and gpu.get("mig_mode") == "Disabled"
        and "H100" in gpu.get("name", "")
        and memory_mib >= MIN_H100_MEMORY_MIB
        and runtime.get("sequence_parallel_size") == 1
        and runtime.get("color_fix") is False
    )
    if not valid_runtime:
        raise SeedVR2Error("SeedVR2 result runtime identity is invalid")

    model = result["model"]
    expected_files = {
        name: {"bytes": size, "sha256": digest}
        for name, (size, digest) in MODEL_FILES.items()
    }
    input_section = result["input"]
    output_section = result["output"]
    artifacts = result["artifacts"]
    artifact_hashes = result.get("artifact_hashes")
    expected_prefix = restore_request.output_path.rstrip("/") + "/"
    valid_artifacts = (
        result["source"]
        == {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION}
        and model.get("repository") == MODEL_REPOSITORY
        and input_section.get("uri") == restore_request.input_path
        and re.fullmatch(r"[0-9a-f]{64}", str(input_section.get("sha256", "")))
        is not None
        and isinstance(input_section.get("media"), dict)
        and re.fullmatch(r"[0-9a-f]{64}", str(output_section.get("sha256", "")))
        is not None
        and isinstance(output_section.get("media"), dict)
        and isinstance(output_section.get("unique_decoded_frames"), int)
        and output_section["unique_decoded_frames"] > 1
        and output_section.get("derived_sensor_truth") is False
        and artifacts.get("restored_video") == expected_prefix + "restored.mp4"
        and artifacts.get("upstream_log") == expected_prefix + "upstream.log"
        and artifacts.get("result") == request.input_path
        and isinstance(artifact_hashes, dict)
        and artifact_hashes.get("restored_video") == output_section.get("sha256")
        and re.fullmatch(r"[0-9a-f]{64}", str(artifact_hashes.get("upstream_log", "")))
        is not None
        and model.get("files") == expected_files
        and model.get("weights_baked") is False
    )
    if not valid_artifacts:
        raise SeedVR2Error("SeedVR2 result artifact identity is invalid")

    argv = result.get("argv")
    valid_argv = (
        isinstance(argv, list)
        and len(argv) == 16
        and all(isinstance(item, str) for item in argv)
        and argv[0] == "/opt/seedvr2-venv/bin/torchrun"
        and argv[1:3] == ["--standalone", "--nproc-per-node=1"]
        and argv[3].endswith("/projects/inference_seedvr2_3b.py")
        and argv[4] == "--video_path"
        and Path(argv[5]).name == "input"
        and argv[6] == "--output_dir"
        and Path(argv[7]).name == "generated"
        and argv[8:]
        == [
            "--seed",
            str(restore_request.seed),
            "--res_h",
            str(restore_request.output_height),
            "--res_w",
            str(restore_request.output_width),
            "--sp_size",
            "1",
        ]
    )
    if not valid_argv:
        raise SeedVR2Error("SeedVR2 result command identity is invalid")

    source_location = authorize_uri(
        restore_request.input_path, operation="verify SeedVR2 source video"
    )
    if source_location.kind != "s3" or not source_location.key.lower().endswith(".mp4"):
        raise SeedVR2Error("SeedVR2 result source identity is invalid")
    source_media = input_section["media"]
    output_media = output_section["media"]
    try:
        _validate_source_geometry(restore_request, source_media)
        period = 1.0 / float(Fraction(source_media["fps"]))
        valid_media = (
            source_media["codec"] == "h264"
            and output_media["codec"] == "h264"
            and (output_media["height"], output_media["width"])
            == (restore_request.output_height, restore_request.output_width)
            and output_media["frames"] == source_media["frames"]
            and Fraction(output_media["fps"]) == Fraction(source_media["fps"])
            and abs(
                float(output_media["duration_seconds"])
                - float(source_media["duration_seconds"])
            )
            <= period
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise SeedVR2Error("SeedVR2 result media contract is invalid") from exc
    if not valid_media:
        raise SeedVR2Error("SeedVR2 result media contract is invalid")

    probe = input_section.get("probe")
    if not isinstance(probe, dict) or probe.get("uri") != restore_request.probe_path:
        raise SeedVR2Error("SeedVR2 result probe identity is invalid")
    probe_uri = authorize_uri(
        str(probe.get("uri", "")), operation="verify SeedVR2 input probe"
    )
    if probe_uri.kind != "s3" or not probe_uri.key.lower().endswith(".json"):
        raise SeedVR2Error("SeedVR2 result probe identity is invalid")
    probe_path = directory / "probe.json"
    storage.download_file(probe_uri.original, str(probe_path))
    if _sha256(probe_path) != probe.get("sha256"):
        raise SeedVR2Error("SeedVR2 input probe hash disagrees with result.json")
    try:
        probe_document = json.loads(probe_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedVR2Error("SeedVR2 input probe is not valid JSON") from exc
    probe_input = (
        probe_document.get("input") if isinstance(probe_document, dict) else None
    )
    if (
        not isinstance(probe_document, dict)
        or not isinstance(probe_input, dict)
        or probe_document.get("schema") != PROBE_SCHEMA
        or probe_document.get("status") != "ok"
        or probe_document.get("run_id") != request.run_id
        or probe_input.get("uri") != input_section.get("uri")
        or probe_input.get("sha256") != input_section.get("sha256")
        or probe_input.get("media") != input_section.get("media")
    ):
        raise SeedVR2Error("SeedVR2 input probe does not bind result.json")

    log_uri = _authorized_result_uri(
        result,
        "artifacts",
        "upstream_log",
        operation="verify SeedVR2 upstream log",
        suffix=".log",
    )
    log_path = directory / "upstream.log"
    storage.download_file(log_uri, str(log_path))
    if (
        not log_path.is_file()
        or log_path.stat().st_size == 0
        or _sha256(log_path) != artifact_hashes["upstream_log"]
    ):
        raise SeedVR2Error("SeedVR2 upstream log hash disagrees with result.json")


def _validate_candidate_decode(result: dict[str, Any], candidate: Path) -> None:
    hashes = _decoded_frame_hashes(candidate)
    expected_frames = result["output"]["media"]["frames"]
    unique_frames = len(set(hashes))
    if (
        len(hashes) != expected_frames
        or unique_frames != result["output"]["unique_decoded_frames"]
        or (len(hashes) > 1 and unique_frames == 1)
    ):
        raise SeedVR2Error("SeedVR2 candidate decoded-frame identity is invalid")


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
        media = _probe_video(source)
        hashes = _decoded_frame_hashes(source)
        document = {
            "schema": PROBE_SCHEMA,
            "status": "ok",
            "run_id": request.run_id,
            "created_at": _utc_now(),
            "input": {
                "uri": request.input_path,
                "sha256": _sha256(source),
                "media": media,
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
    video_uri = _authorized_result_uri(
        result,
        "artifacts",
        "restored_video",
        operation="verify SeedVR2 restored video",
        suffix=".mp4",
    )
    _validate_result_context(result, request, storage, directory)
    video = directory / "restored.mp4"
    storage.download_file(video_uri, str(video))
    media = _probe_video(video)
    digest = _sha256(video)
    if digest != result.get("output", {}).get("sha256"):
        raise SeedVR2Error("restored video hash disagrees with result.json")
    if media != result.get("output", {}).get("media"):
        raise SeedVR2Error("restored video media metadata disagrees with result.json")
    _validate_candidate_decode(result, video)
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
        "validated_execution": {
            "run_id": result["run_id"],
            "image": result["runtime"]["image"],
            "image_digest": result["runtime"]["image_digest"],
            "npa_source_revision": result["runtime"]["npa_source_revision"],
            "probe_sha256": result["input"]["probe"]["sha256"],
            "upstream_log_sha256": result["artifact_hashes"]["upstream_log"],
        },
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
        "-nostdin",
        "-y",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "mov",
        "-i",
        str(source),
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "mov",
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
    canonical_anchors = (0, 16, 32, 48, 60, 68, 76, 84, 96)
    return sorted(
        {round(anchor * (frame_count - 1) / 99) for anchor in canonical_anchors}
    )


def _contact_sheet(comparison: Path, output: Path, frame_indices: list[int]) -> None:
    expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    columns = 3
    rows = (len(frame_indices) + columns - 1) // columns
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "mov",
        "-i",
        str(comparison),
        "-vf",
        f"select='{expression}',tile={columns}x{rows}:padding=8:margin=8",
        "-frames:v",
        "1",
        str(output),
    ]
    completed = subprocess.run(command, check=False, env=_media_environment())
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
    completed = subprocess.run(command, check=False, env=_media_environment())
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
    source_uri = _authorized_result_uri(
        result,
        "input",
        "uri",
        operation="review SeedVR2 source video",
        suffix=".mp4",
    )
    candidate_uri = _authorized_result_uri(
        result,
        "artifacts",
        "restored_video",
        operation="review SeedVR2 restored video",
        suffix=".mp4",
    )
    _validate_result_context(result, request, storage, directory)
    source = directory / "source.mp4"
    candidate = directory / "restored.mp4"
    storage.download_file(source_uri, str(source))
    storage.download_file(candidate_uri, str(candidate))
    if _sha256(source) != result["input"].get("sha256"):
        raise SeedVR2Error("review source hash disagrees with result.json")
    if _sha256(candidate) != result["output"].get("sha256"):
        raise SeedVR2Error("review candidate hash disagrees with result.json")
    if _probe_video(source) != result["input"].get("media"):
        raise SeedVR2Error("review source media disagrees with result.json")
    if _probe_video(candidate) != result["output"].get("media"):
        raise SeedVR2Error("review candidate media disagrees with result.json")
    _validate_candidate_decode(result, candidate)
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
