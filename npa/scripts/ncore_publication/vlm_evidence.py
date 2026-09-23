#!/usr/bin/env python3
"""Freeze and execute one-shot visual controls for NCore qualification."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import secrets
import stat
import tempfile
import time
from typing import Any
import httpx

from PIL import Image

from npa.clients.token_factory import default_chat_extra
from npa.workbench.nurec.colmap import (
    ColmapConversionRequest,
    extract_colmap_zip,
    find_colmap_root,
    inspect_colmap_source,
)


FREEZE_FORMAT = "npa_ncore_vlm_freeze_v2"
CALIBRATION_FORMAT = "npa_ncore_vlm_calibration_v2"
FINAL_FORMAT = "npa_ncore_vlm_final_v2"
ATTEMPT_FORMAT = "npa_ncore_vlm_attempt_v2"
TRANSPORT_FORMAT = "npa_ncore_vlm_transport_manifest_v2"
FREEZE_REVIEW_FORMAT = "npa_ncore_vlm_freeze_review_v1"
FREEZE_ACCEPTANCE_FORMAT = "npa_ncore_vlm_freeze_acceptance_v1"
MODEL = "openbmb/MiniCPM-V-4_5"
ENDPOINT = "https://api.tokenfactory.nebius.com/v1/chat/completions"
THRESHOLD = 0.8
FRAME_COUNT = 4
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class VlmEvidenceError(RuntimeError):
    """The frozen one-shot visual evidence contract was not satisfied."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_private(path: Path, body: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Any) -> None:
    _write_private(path, _canonical(payload))


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise VlmEvidenceError("required evidence file is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VlmEvidenceError("required evidence file is invalid") from exc
    if not isinstance(payload, dict):
        raise VlmEvidenceError("required evidence is not an object")
    return payload


def _private_root(path: Path, *, existing: bool) -> None:
    if existing:
        if path.is_symlink() or not path.is_dir():
            raise VlmEvidenceError("private evidence directory is missing")
    else:
        path.mkdir(mode=0o700, parents=True)
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise VlmEvidenceError("private evidence directory must have mode 0700")


def _indices(count: int) -> list[int]:
    if count < FRAME_COUNT:
        raise VlmEvidenceError("each visual sequence must contain at least four frames")
    return [0, (count - 1) // 3, (2 * (count - 1)) // 3, count - 1]


def _image_bytes(path: Path) -> tuple[bytes, tuple[int, int]]:
    if path.is_symlink() or not path.is_file():
        raise VlmEvidenceError("selected visual frame is not a regular file")
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            if rgb.width <= 0 or rgb.height <= 0:
                raise ValueError("invalid dimensions")
            from io import BytesIO

            stream = BytesIO()
            rgb.save(stream, format="PNG")
            return stream.getvalue(), rgb.size
    except Exception as exc:
        raise VlmEvidenceError("selected visual frame does not decode") from exc


def _write_case(root: Path, case_id: str, frames: list[bytes]) -> list[dict[str, Any]]:
    directory = root / "controls" / case_id
    directory.mkdir(mode=0o700, parents=True)
    records = []
    for index, body in enumerate(frames):
        path = directory / f"frame-{index:03d}.png"
        _write_private(path, body)
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha_bytes(body),
                "bytes": len(body),
                "order": index,
            }
        )
    return records


def _uniform_controls(
    selected: list[Path],
) -> list[bytes]:
    controls = []
    for path in selected:
        _, size = _image_bytes(path)
        image = Image.new("RGB", size, (127, 127, 127))
        from io import BytesIO

        stream = BytesIO()
        image.save(stream, format="PNG")
        controls.append(stream.getvalue())
    return controls


def _block_corrupt(selected: list[Path]) -> list[bytes]:
    controls = []
    for path in selected:
        try:
            original, _ = _image_bytes(path)
            with Image.open(path) as source:
                image = source.convert("RGB")
                boxes = [
                    (x, y, min(x + 32, image.width), min(y + 32, image.height))
                    for y in range(0, image.height, 32)
                    for x in range(0, image.width, 32)
                ]
                if len(boxes) < 2:
                    transformed = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                else:
                    tiles = [image.crop(box) for box in boxes]
                    transformed = Image.new("RGB", image.size)
                    rotated = tiles[1:] + tiles[:1]
                    for box, tile in zip(boxes, rotated, strict=True):
                        width = box[2] - box[0]
                        height = box[3] - box[1]
                        transformed.paste(tile.resize((width, height)), box[:2])
                from io import BytesIO

                stream = BytesIO()
                transformed.save(stream, format="PNG")
                body = stream.getvalue()
                if body == original:
                    raise VlmEvidenceError(
                        "block control is pixel-identical to its positive source"
                    )
                controls.append(body)
        except Exception as exc:
            if isinstance(exc, VlmEvidenceError):
                raise
            raise VlmEvidenceError("block control generation failed") from exc
    return controls


def _render_defect_control(selected: list[Path]) -> list[bytes]:
    """Add a deterministic local seam/floater defect to rendered views."""
    controls = []
    for path in selected:
        try:
            original, _ = _image_bytes(path)
            with Image.open(path) as source:
                image = source.convert("RGB")
                width = max(4, image.width // 4)
                height = max(4, image.height // 4)
                left = max(0, (image.width - width) // 2)
                top = max(0, (image.height - height) // 2)
                source_left = min(max(0, left + width // 2), image.width - width)
                source_top = min(max(0, top - height // 2), image.height - height)
                patch = image.crop(
                    (
                        source_left,
                        source_top,
                        source_left + width,
                        source_top + height,
                    )
                ).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                image.paste(patch, (left, top))
                border = (255, 0, 255)
                for x in range(left, min(image.width, left + width)):
                    image.putpixel((x, top), border)
                    image.putpixel((x, min(image.height - 1, top + height - 1)), border)
                from io import BytesIO

                stream = BytesIO()
                image.save(stream, format="PNG")
                body = stream.getvalue()
                if body == original:
                    raise VlmEvidenceError(
                        "render defect control is pixel-identical to its source"
                    )
                controls.append(body)
        except Exception as exc:
            if isinstance(exc, VlmEvidenceError):
                raise
            raise VlmEvidenceError("render defect control generation failed") from exc
    return controls


def _render_trajectory(render_dir: Path, camera: str) -> tuple[str, list[Path]]:
    relative = PurePosixPath(str(camera))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise VlmEvidenceError("render camera must be a safe relative directory")
    directory = render_dir.joinpath(*relative.parts)
    if directory.is_symlink() or not directory.is_dir():
        raise VlmEvidenceError("render camera directory is missing")
    frames = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(frames) < FRAME_COUNT:
        raise VlmEvidenceError("render camera trajectory needs at least four frames")
    return relative.as_posix(), frames


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root
    _private_root(root, existing=False)
    if _sha_file(args.source_zip) != args.source_sha256:
        raise VlmEvidenceError("source archive SHA-256 differs")
    with tempfile.TemporaryDirectory(prefix="ncore-vlm-source-") as directory:
        extracted = Path(directory)
        extract_colmap_zip(args.source_zip, extracted)
        request = ColmapConversionRequest(
            input_path="s3://private/source.zip",
            output_path="s3://private/output/",
            expected_archive_sha256=args.source_sha256,
            dataset_root=args.dataset_root,
            colmap_dir=args.colmap_dir,
            images_dir=args.images_dir,
            include_downsampled_images=False,
        )
        source_root = find_colmap_root(
            extracted, args.dataset_root, args.colmap_dir, args.images_dir
        )
        inspected = inspect_colmap_source(source_root, request)
        cameras = [
            (name, value)
            for name, value in sorted(inspected["cameras"].items())
            if "_" not in name
        ]
        if len(cameras) < 2:
            raise VlmEvidenceError("source needs at least two base cameras")
        positives: list[tuple[str, list[Path]]] = []
        for camera_name, camera in cameras[:2]:
            frames = sorted(
                source_root / args.images_dir / frame["name"]
                for frame in camera["frames"]
            )
            positives.append(
                (camera_name, [frames[index] for index in _indices(len(frames))])
            )
        roles: list[tuple[str, bool, list[bytes]]] = []
        for camera_name, selected in positives:
            roles.append(
                (
                    "source-camera:" + camera_name,
                    True,
                    [_image_bytes(path)[0] for path in selected],
                )
            )
        roles.append(("block-rotate-32", False, _block_corrupt(positives[1][1])))
    render_camera, render_frames = _render_trajectory(
        args.render_dir, args.render_camera
    )
    selected_render_frames = [
        render_frames[index] for index in _indices(len(render_frames))
    ]
    roles.append(
        (
            "render-local-seam-floater",
            False,
            _render_defect_control(selected_render_frames),
        )
    )
    case_ids = [f"case-{secrets.token_hex(8)}" for _ in roles]
    case_order = sorted(
        case_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest()
    )
    role_by_case = dict(zip(case_ids, roles, strict=True))
    cases = {}
    labels = {}
    for case_id in case_order:
        role, expected, frames = role_by_case[case_id]
        cases[case_id] = {
            "frames": _write_case(root, case_id, frames),
        }
        labels[case_id] = {"role": role, "expected_label": expected}
    final_records = []
    final_dir = root / "final"
    final_dir.mkdir(mode=0o700)
    for order, source_index in enumerate(_indices(len(render_frames))):
        body, _ = _image_bytes(render_frames[source_index])
        path = final_dir / f"frame-{order:03d}.png"
        _write_private(path, body)
        final_records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha_bytes(body),
                "bytes": len(body),
                "order": order,
                "source_index": source_index,
                "source_sha256": _sha_file(render_frames[source_index]),
            }
        )
    final_frame_manifest_path = root / "final-frame-manifest.json"
    _write_json(
        final_frame_manifest_path,
        {
            "format": "npa_ncore_vlm_final_frames_v1",
            "render_camera_sha256": _sha_bytes(render_camera.encode()),
            "render_inventory_sha256": _sha_bytes(
                _canonical(
                    [
                        {
                            "sha256": _sha_file(path),
                            "bytes": path.stat().st_size,
                        }
                        for path in render_frames
                    ]
                )
            ),
            "frames": final_records,
        },
    )
    label_path = root / "labels.json"
    _write_json(label_path, {"cases": labels})
    manifest = {
        "format": FREEZE_FORMAT,
        "schedule_id": secrets.token_hex(16),
        "source_archive_sha256": args.source_sha256,
        "render_inventory_sha256": _sha_bytes(
            _canonical(
                [
                    {
                        "sha256": _sha_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in render_frames
                ]
            )
        ),
        "model": MODEL,
        "threshold": THRESHOLD,
        "rubric_sha256": _sha_file(args.rubric),
        "calibration_task_sha256": _sha_file(args.calibration_task),
        "final_task_sha256": _sha_file(args.final_task),
        "case_order": case_order,
        "cases": cases,
        "label_commitment_sha256": _sha_file(label_path),
        "final_frame_manifest_sha256": _sha_file(final_frame_manifest_path),
        "final_frames": final_records,
    }
    manifest_path = root / "freeze.json"
    _write_json(manifest_path, manifest)
    return {
        "status": "ok",
        "freeze_sha256": _sha_file(manifest_path),
        "label_commitment_sha256": manifest["label_commitment_sha256"],
        "final_frame_manifest_sha256": manifest["final_frame_manifest_sha256"],
        "cases": len(cases),
        "final_frames": len(final_records),
    }


def _prompt(task: str, rubric: str, frames: int) -> str:
    return "\n".join(
        [
            "Judge only the supplied visual sequence.",
            f"Task/instruction: {task}",
            f"Rubric: {rubric}",
            f"Frames supplied in chronological order: {frames}.",
            "Return only a JSON object with this schema:",
            '{"success": boolean, "score": number between 0 and 1, "rationale": string}',
        ]
    )


def _build_request(*, task: str, rubric: str, frames: list[bytes]) -> tuple[bytes, str]:
    prompt = _prompt(task, rubric, len(frames))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,"
                + base64.b64encode(frame).decode("ascii")
            },
        }
        for frame in frames
    )
    request_payload = {
        "model": MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
        **default_chat_extra(MODEL),
    }
    return _canonical(request_payload), prompt


def _validate_rationale(value: Any) -> str:
    if not isinstance(value, str):
        raise VlmEvidenceError("hosted VLM rationale is not text")
    rationale = value.strip()
    if (
        len(rationale) < 32
        or len(re.findall(r"[A-Za-z]{3,}", rationale)) < 4
        or re.search(
            r"\b(expected|label|metadata|filename|ground.?truth)\b",
            rationale,
            re.IGNORECASE,
        )
        or not re.search(
            r"\b(visible|frame|region|blur|seam|geometry|texture|hole|tearing|scene)\b",
            rationale,
            re.IGNORECASE,
        )
    ):
        raise VlmEvidenceError(
            "hosted VLM rationale does not cite sufficient visible evidence"
        )
    return rationale


def _response_semantics(response_bytes: bytes) -> dict[str, Any]:
    try:
        response_payload = json.loads(response_bytes)
        choices = response_payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise TypeError
        choice = choices[0]
        message = json.loads(choice["message"]["content"])
        served_model = response_payload["model"]
        usage = response_payload["usage"]
    except (
        KeyError,
        IndexError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise VlmEvidenceError("hosted VLM response schema differs") from exc
    if served_model != MODEL or choice.get("finish_reason") != "stop":
        raise VlmEvidenceError("hosted VLM model or finish reason differs")
    score = message.get("score") if isinstance(message, dict) else None
    success = message.get("success") if isinstance(message, dict) else None
    if (
        type(success) is not bool
        or not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or not 0 <= float(score) <= 1
        or not isinstance(usage, dict)
        or not usage
        or success is not (float(score) >= THRESHOLD)
    ):
        raise VlmEvidenceError("hosted VLM structured result differs")
    provider_id = response_payload.get("id")
    return {
        "served_model": served_model,
        "finish_reason": choice["finish_reason"],
        "usage": usage,
        "success": success,
        "score": float(score),
        "rationale": _validate_rationale(message.get("rationale")),
        "provider_request_id": (
            provider_id.strip()
            if isinstance(provider_id, str) and provider_id.strip()
            else "unavailable"
        ),
    }


def _attempt_marker(
    *,
    storage_client: Any,
    prefix: str,
    attempt_id: str,
    purpose: str,
    freeze_sha256: str,
    request_sha256: str,
) -> dict[str, Any]:
    from npa.clients.storage import StoragePreconditionFailed

    uri = prefix.rstrip("/") + f"/attempt-ledger/{attempt_id}.json"
    payload = {
        "format": ATTEMPT_FORMAT,
        "status": "committed",
        "attempt_id": attempt_id,
        "purpose": purpose,
        "freeze_sha256": freeze_sha256,
        "request_sha256": request_sha256,
    }
    body = _canonical(payload)
    try:
        etag = storage_client.put_bytes_conditional(
            body,
            uri,
            if_none_match=True,
            content_type="application/json",
        )
    except StoragePreconditionFailed as exc:
        raise VlmEvidenceError(
            "hosted VLM attempt already exists; retry is prohibited"
        ) from exc
    observed = storage_client.read_bytes_with_etag(uri)
    if observed is None or observed[0] != body or observed[1] != etag:
        raise VlmEvidenceError("external attempt marker read-back differs")
    return {
        "body_sha256": _sha_bytes(body),
        "etag_sha256": _sha_bytes(etag.encode()),
        "uri_sha256": _sha_bytes(uri.encode()),
    }


def _post_hosted_bytes(request_bytes: bytes, api_key: str) -> httpx.Response:
    """Send the frozen request once, without following provider redirects."""
    with httpx.Client(timeout=120, follow_redirects=False) as client:
        response = client.post(
            ENDPOINT,
            content=request_bytes,
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response


def _call_once(
    *,
    root: Path,
    attempt_id: str,
    freeze_sha256: str,
    purpose: str,
    frame_records: list[dict[str, Any]],
    task: str,
    rubric: str,
    api_key: str,
    external_attempt_prefix: str,
    storage_client: Any = None,
) -> dict[str, Any]:
    if _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise VlmEvidenceError("attempt ID is invalid")
    if purpose not in {"calibration", "final"}:
        raise VlmEvidenceError("attempt purpose is invalid")
    if not str(external_attempt_prefix).startswith("s3://"):
        raise VlmEvidenceError("external attempt prefix must be an S3 URI")
    frames = []
    for record in sorted(frame_records, key=lambda item: item["order"]):
        path = root / record["path"]
        body = path.read_bytes()
        if _sha_bytes(body) != record["sha256"]:
            raise VlmEvidenceError("frozen frame hash differs")
        frames.append(body)
    request_bytes, prompt = _build_request(task=task, rubric=rubric, frames=frames)
    if storage_client is None:
        from npa.clients.storage import StorageClient

        storage_client = StorageClient.from_environment()
    marker = _attempt_marker(
        storage_client=storage_client,
        prefix=external_attempt_prefix,
        attempt_id=attempt_id,
        purpose=purpose,
        freeze_sha256=freeze_sha256,
        request_sha256=_sha_bytes(request_bytes),
    )
    ledger_path = root / "attempt-ledger" / f"{attempt_id}.json"
    ledger = {
        "format": ATTEMPT_FORMAT,
        "status": "committed",
        "attempt_id": attempt_id,
        "purpose": purpose,
        "freeze_sha256": freeze_sha256,
        "request_sha256": _sha_bytes(request_bytes),
        "external_marker_body_sha256": marker["body_sha256"],
        "external_marker_etag_sha256": marker["etag_sha256"],
        "external_marker_uri_sha256": marker["uri_sha256"],
    }
    _write_json(ledger_path, ledger)
    attempt_root = root / "transport" / attempt_id
    attempt_root.mkdir(mode=0o700, parents=True)
    _write_json(
        attempt_root / "attempt.json",
        {
            "status": "committed",
            "attempt_id": attempt_id,
            "purpose": purpose,
            "freeze_sha256": freeze_sha256,
            "endpoint_sha256": _sha_bytes(ENDPOINT.encode()),
            "model": MODEL,
            "prompt_sha256": _sha_bytes(prompt.encode()),
            "frame_sha256": [_sha_bytes(frame) for frame in frames],
            "request_sha256": _sha_bytes(request_bytes),
            "external_marker_body_sha256": marker["body_sha256"],
            "external_marker_etag_sha256": marker["etag_sha256"],
            "external_marker_uri_sha256": marker["uri_sha256"],
        },
    )
    _write_private(attempt_root / "request.json", request_bytes)
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    provider_request_id = "unavailable"
    try:
        response = _post_hosted_bytes(request_bytes, api_key)
        status_code = response.status_code
        response_bytes = response.content
    except httpx.HTTPStatusError as exc:
        response_bytes = exc.response.content
        finished_at = datetime.now(timezone.utc).isoformat()
        latency_ms = max(0, round((time.monotonic() - started_clock) * 1000))
        _write_private(attempt_root / "response.json", response_bytes)
        _write_json(
            attempt_root / "outcome.json",
            {
                "status": "response_failed",
                "error_class": "HTTPStatusError",
                "http_status": exc.response.status_code,
                "response_sha256": _sha_bytes(response_bytes),
                "request_started_at": started_at,
                "request_finished_at": finished_at,
                "latency_ms": latency_ms,
                "provider_request_id": "unavailable",
            },
        )
        raise VlmEvidenceError(
            "hosted VLM returned an HTTP error; retry is prohibited"
        ) from exc
    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat()
        latency_ms = max(0, round((time.monotonic() - started_clock) * 1000))
        _write_json(
            attempt_root / "outcome.json",
            {
                "status": "transport_failed",
                "error_class": type(exc).__name__,
                "request_started_at": started_at,
                "request_finished_at": finished_at,
                "latency_ms": latency_ms,
                "provider_request_id": "unavailable",
            },
        )
        raise VlmEvidenceError(
            "hosted VLM transport failed; retry is prohibited"
        ) from exc
    finished_at = datetime.now(timezone.utc).isoformat()
    latency_ms = max(0, round((time.monotonic() - started_clock) * 1000))
    _write_private(attempt_root / "response.json", response_bytes)
    if status_code != 200:
        _write_json(
            attempt_root / "outcome.json",
            {
                "status": "response_failed",
                "http_status": status_code,
                "response_sha256": _sha_bytes(response_bytes),
                "request_started_at": started_at,
                "request_finished_at": finished_at,
                "latency_ms": latency_ms,
                "provider_request_id": provider_request_id,
            },
        )
        raise VlmEvidenceError("hosted VLM returned a non-200 response")
    try:
        semantics = _response_semantics(response_bytes)
    except VlmEvidenceError:
        _write_json(
            attempt_root / "outcome.json",
            {
                "status": "response_failed",
                "error_class": "schema",
                "response_sha256": _sha_bytes(response_bytes),
                "request_started_at": started_at,
                "request_finished_at": finished_at,
                "latency_ms": latency_ms,
                "provider_request_id": provider_request_id,
            },
        )
        raise
    provider_request_id = semantics["provider_request_id"]
    result = {
        "status": "complete",
        "attempt_id": attempt_id,
        "purpose": purpose,
        "freeze_sha256": freeze_sha256,
        "http_status": status_code,
        "requested_model": MODEL,
        "served_model": semantics["served_model"],
        "finish_reason": semantics["finish_reason"],
        "usage": semantics["usage"],
        "request_sha256": _sha_bytes(request_bytes),
        "response_sha256": _sha_bytes(response_bytes),
        "prompt_sha256": _sha_bytes(prompt.encode()),
        "frame_sha256": [_sha_bytes(frame) for frame in frames],
        "success": semantics["success"],
        "score": semantics["score"],
        "rationale": semantics["rationale"],
        "request_started_at": started_at,
        "request_finished_at": finished_at,
        "latency_ms": latency_ms,
        "provider_request_id": provider_request_id,
    }
    _write_json(attempt_root / "outcome.json", result)
    return result


def _verified_freeze(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.evidence_root / "freeze.json"
    if _sha_file(manifest_path) != args.freeze_sha256:
        raise VlmEvidenceError("freeze manifest SHA-256 differs")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("format") != FREEZE_FORMAT
        or manifest.get("model") != MODEL
        or manifest.get("threshold") != THRESHOLD
        or re.fullmatch(r"[0-9a-f]{32}", str(manifest.get("schedule_id") or "")) is None
    ):
        raise VlmEvidenceError("freeze manifest contract differs")
    case_order = manifest.get("case_order")
    cases = manifest.get("cases")
    if (
        not isinstance(case_order, list)
        or len(case_order) != 4
        or len(set(case_order)) != 4
        or not all(isinstance(item, str) for item in case_order)
        or not isinstance(cases, dict)
        or set(cases) != set(case_order)
    ):
        raise VlmEvidenceError("freeze calibration cases differ")
    for case_id in case_order:
        case = cases[case_id]
        if not isinstance(case, dict):
            raise VlmEvidenceError("freeze calibration case differs")
        _verified_frames(args.evidence_root, case.get("frames"))
    final_frames = _verified_frames(args.evidence_root, manifest.get("final_frames"))
    final_manifest_path = args.evidence_root / "final-frame-manifest.json"
    if _sha_file(final_manifest_path) != manifest.get("final_frame_manifest_sha256"):
        raise VlmEvidenceError("final frame manifest hash differs")
    final_manifest = _load_json(final_manifest_path)
    if (
        final_manifest.get("format") != "npa_ncore_vlm_final_frames_v1"
        or final_manifest.get("frames") != final_frames
        or final_manifest.get("render_inventory_sha256")
        != manifest.get("render_inventory_sha256")
    ):
        raise VlmEvidenceError("final frame manifest contract differs")
    return manifest


def accept_freeze(args: argparse.Namespace) -> dict[str, Any]:
    """Bind a separate reviewer's exact freeze verdict before hosted calls."""
    _private_root(args.evidence_root, existing=True)
    manifest = _verified_freeze(args)
    if (
        args.review_path.is_symlink()
        or not args.review_path.is_file()
        or args.review_path.stat().st_uid != os.getuid()
        or args.review_path.stat().st_nlink != 1
        or args.review_path.stat().st_mode & 0o077
    ):
        raise VlmEvidenceError("freeze review receipt is missing or not private")
    review = _load_json(args.review_path)
    prefix_sha256 = _sha_bytes(args.external_attempt_prefix.encode())
    if (
        review.get("format") != FREEZE_REVIEW_FORMAT
        or review.get("verdict") != "ACCEPTED"
        or review.get("freeze_sha256") != args.freeze_sha256
        or review.get("label_commitment_sha256")
        != manifest.get("label_commitment_sha256")
        or review.get("final_frame_manifest_sha256")
        != manifest.get("final_frame_manifest_sha256")
        or review.get("external_attempt_prefix_sha256") != prefix_sha256
        or review.get("controls_opened") != len(manifest["case_order"])
        or review.get("final_frames_opened") != len(manifest["final_frames"])
        or review.get("prompts_reviewed") is not True
        or not isinstance(review.get("reviewer_id"), str)
        or not review["reviewer_id"].strip()
    ):
        raise VlmEvidenceError("independent freeze review contract differs")
    payload = {
        "format": FREEZE_ACCEPTANCE_FORMAT,
        "status": "accepted",
        "freeze_sha256": args.freeze_sha256,
        "label_commitment_sha256": manifest["label_commitment_sha256"],
        "final_frame_manifest_sha256": manifest["final_frame_manifest_sha256"],
        "external_attempt_prefix_sha256": prefix_sha256,
        "review_receipt_sha256": _sha_file(args.review_path),
        "reviewer_id_sha256": _sha_bytes(review["reviewer_id"].encode()),
    }
    _write_private(
        args.evidence_root / "external-attempt-prefix.txt",
        (args.external_attempt_prefix + "\n").encode(),
    )
    path = args.evidence_root / "freeze-acceptance.json"
    _write_json(path, payload)
    return {"status": "ok", "sha256": _sha_file(path)}


def _verified_freeze_acceptance(
    *,
    root: Path,
    manifest: dict[str, Any],
    freeze_sha256: str,
    acceptance_sha256: str,
    external_attempt_prefix: str,
) -> dict[str, Any]:
    path = root / "freeze-acceptance.json"
    if _sha_file(path) != acceptance_sha256:
        raise VlmEvidenceError("freeze acceptance SHA-256 differs")
    acceptance = _load_json(path)
    if (
        acceptance.get("format") != FREEZE_ACCEPTANCE_FORMAT
        or acceptance.get("status") != "accepted"
        or acceptance.get("freeze_sha256") != freeze_sha256
        or acceptance.get("label_commitment_sha256")
        != manifest.get("label_commitment_sha256")
        or acceptance.get("final_frame_manifest_sha256")
        != manifest.get("final_frame_manifest_sha256")
        or acceptance.get("external_attempt_prefix_sha256")
        != _sha_bytes(external_attempt_prefix.encode())
        or re.fullmatch(
            r"[0-9a-f]{64}", str(acceptance.get("review_receipt_sha256") or "")
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(acceptance.get("reviewer_id_sha256") or "")
        )
        is None
    ):
        raise VlmEvidenceError("freeze acceptance contract differs")
    return acceptance


def _verified_frames(root: Path, records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != FRAME_COUNT:
        raise VlmEvidenceError("frozen frame records differ")
    verified = []
    for order, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or record.get("order") != order
            or not isinstance(record.get("path"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256") or "")) is None
            or type(record.get("bytes")) is not int
            or record["bytes"] <= 0
        ):
            raise VlmEvidenceError("frozen frame record differs")
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise VlmEvidenceError("frozen frame path is unsafe")
        path = root.joinpath(*relative.parts)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
            or _sha_file(path) != record["sha256"]
        ):
            raise VlmEvidenceError("frozen frame bytes differ")
        verified.append(record)
    return verified


def _verified_transport(
    root: Path,
    *,
    attempt_id: str,
    freeze_sha256: str,
    purpose: str,
    frame_records: list[dict[str, Any]],
    task: str,
    rubric: str,
) -> dict[str, Any]:
    ledger = _load_json(root / "attempt-ledger" / f"{attempt_id}.json")
    if (
        not {
            "format": ATTEMPT_FORMAT,
            "status": "committed",
            "attempt_id": attempt_id,
            "purpose": purpose,
            "freeze_sha256": freeze_sha256,
        }.items()
        <= ledger.items()
    ):
        raise VlmEvidenceError("VLM attempt ledger differs")
    attempt_root = root / "transport" / attempt_id
    files = {
        path.name
        for path in attempt_root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if files != {"attempt.json", "request.json", "response.json", "outcome.json"}:
        raise VlmEvidenceError("completed VLM transport file set differs")
    attempt = _load_json(attempt_root / "attempt.json")
    outcome = _load_json(attempt_root / "outcome.json")
    frames = []
    for record in sorted(frame_records, key=lambda item: item["order"]):
        body = (root / record["path"]).read_bytes()
        if _sha_bytes(body) != record["sha256"]:
            raise VlmEvidenceError("completed VLM frame binding differs")
        frames.append(body)
    request_bytes, prompt = _build_request(task=task, rubric=rubric, frames=frames)
    marker_body = _canonical(
        {
            "format": ATTEMPT_FORMAT,
            "status": "committed",
            "attempt_id": attempt_id,
            "purpose": purpose,
            "freeze_sha256": freeze_sha256,
            "request_sha256": _sha_bytes(request_bytes),
        }
    )
    response_bytes = (attempt_root / "response.json").read_bytes()
    semantics = _response_semantics(response_bytes)
    semantic_fields = {
        "served_model": semantics["served_model"],
        "finish_reason": semantics["finish_reason"],
        "usage": semantics["usage"],
        "success": semantics["success"],
        "score": semantics["score"],
        "rationale": semantics["rationale"],
        "provider_request_id": semantics["provider_request_id"],
    }
    try:
        started = datetime.fromisoformat(str(outcome.get("request_started_at") or ""))
        finished = datetime.fromisoformat(str(outcome.get("request_finished_at") or ""))
    except ValueError as exc:
        raise VlmEvidenceError("completed VLM transport timing differs") from exc
    if (
        attempt.get("status") != "committed"
        or attempt.get("attempt_id") != attempt_id
        or attempt.get("purpose") != purpose
        or attempt.get("freeze_sha256") != freeze_sha256
        or outcome.get("status") != "complete"
        or outcome.get("attempt_id") != attempt_id
        or outcome.get("purpose") != purpose
        or outcome.get("freeze_sha256") != freeze_sha256
        or outcome.get("requested_model") != MODEL
        or outcome.get("served_model") != MODEL
        or outcome.get("http_status") != 200
        or (attempt_root / "request.json").read_bytes() != request_bytes
        or attempt.get("request_sha256") != _sha_bytes(request_bytes)
        or attempt.get("prompt_sha256") != _sha_bytes(prompt.encode())
        or attempt.get("frame_sha256") != [_sha_bytes(frame) for frame in frames]
        or outcome.get("request_sha256") != attempt.get("request_sha256")
        or outcome.get("response_sha256") != _sha_bytes(response_bytes)
        or outcome.get("prompt_sha256") != attempt.get("prompt_sha256")
        or outcome.get("frame_sha256") != attempt.get("frame_sha256")
        or any(outcome.get(key) != value for key, value in semantic_fields.items())
        or ledger.get("request_sha256") != attempt.get("request_sha256")
        or ledger.get("external_marker_body_sha256") != _sha_bytes(marker_body)
        or attempt.get("external_marker_body_sha256")
        != ledger.get("external_marker_body_sha256")
        or attempt.get("external_marker_etag_sha256")
        != ledger.get("external_marker_etag_sha256")
        or attempt.get("external_marker_uri_sha256")
        != ledger.get("external_marker_uri_sha256")
        or re.fullmatch(
            r"[0-9a-f]{64}", str(ledger.get("external_marker_etag_sha256") or "")
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(ledger.get("external_marker_uri_sha256") or "")
        )
        is None
        or type(outcome.get("latency_ms")) is not int
        or outcome["latency_ms"] < 0
        or started.tzinfo is None
        or finished.tzinfo is None
        or finished < started
    ):
        raise VlmEvidenceError("completed VLM transport binding differs")
    return outcome


def _exact_attempt_set(root: Path, expected: set[str]) -> None:
    for directory_name in ("attempt-ledger", "transport"):
        directory = root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise VlmEvidenceError("VLM attempt evidence directory is missing")
        entries = list(directory.iterdir())
        if any(
            path.is_symlink()
            or (
                not path.is_file()
                if directory_name == "attempt-ledger"
                else not path.is_dir()
            )
            or (directory_name == "attempt-ledger" and path.suffix != ".json")
            for path in entries
        ):
            raise VlmEvidenceError("VLM attempt evidence contains an invalid entry")
        observed = {
            path.stem if directory_name == "attempt-ledger" else path.name
            for path in entries
        }
        if observed != expected:
            raise VlmEvidenceError("VLM attempt schedule differs")


def _verified_calibration(
    root: Path,
    *,
    manifest: dict[str, Any],
    freeze_sha256: str,
    calibration_sha256: str,
    freeze_acceptance_sha256: str,
    task: str,
    rubric: str,
) -> dict[str, Any]:
    path = root / "calibration.json"
    if _sha_file(path) != calibration_sha256:
        raise VlmEvidenceError("calibration SHA-256 differs")
    calibration = _load_json(path)
    case_order = manifest["case_order"]
    labels_path = root / "labels.json"
    if _sha_file(labels_path) != manifest.get("label_commitment_sha256"):
        raise VlmEvidenceError("private label commitment differs")
    labels = _load_json(labels_path).get("cases")
    results = calibration.get("results")
    if (
        calibration.get("format") != CALIBRATION_FORMAT
        or calibration.get("status") != "pass"
        or calibration.get("freeze_sha256") != freeze_sha256
        or calibration.get("freeze_acceptance_sha256") != freeze_acceptance_sha256
        or calibration.get("model") != MODEL
        or calibration.get("served_model") != MODEL
        or calibration.get("threshold") != THRESHOLD
        or calibration.get("total") != len(case_order)
        or calibration.get("attempt_count") != len(case_order)
        or calibration.get("one_shot") is not True
        or not isinstance(labels, dict)
        or not isinstance(results, list)
        or [item.get("case_id") for item in results if isinstance(item, dict)]
        != case_order
    ):
        raise VlmEvidenceError("calibration contract differs")
    _exact_attempt_set(root, set(case_order))
    derived = []
    for result in results:
        case_id = result["case_id"]
        label = labels.get(case_id)
        outcome = _verified_transport(
            root,
            attempt_id=case_id,
            freeze_sha256=freeze_sha256,
            purpose="calibration",
            frame_records=manifest["cases"][case_id]["frames"],
            task=task,
            rubric=rubric,
        )
        if (
            not isinstance(label, dict)
            or type(label.get("expected_label")) is not bool
            or result.get("expected_label") is not label["expected_label"]
            or result.get("predicted_label")
            is not (float(outcome["score"]) >= THRESHOLD)
            or result.get("transport_sha256")
            != _sha_file(root / "transport" / case_id / "outcome.json")
        ):
            raise VlmEvidenceError("calibration result binding differs")
        derived.append(
            (bool(result["expected_label"]), bool(result["predicted_label"]))
        )
    counts = (
        sum(expected and predicted for expected, predicted in derived),
        sum(not expected and not predicted for expected, predicted in derived),
        sum(not expected and predicted for expected, predicted in derived),
        sum(expected and not predicted for expected, predicted in derived),
    )
    if counts != (2, 2, 0, 0) or counts != (
        calibration.get("true_positives"),
        calibration.get("true_negatives"),
        calibration.get("false_positives"),
        calibration.get("false_negatives"),
    ):
        raise VlmEvidenceError("calibration did not pass")
    return calibration


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    _private_root(args.evidence_root, existing=True)
    manifest = _verified_freeze(args)
    _verified_freeze_acceptance(
        root=args.evidence_root,
        manifest=manifest,
        freeze_sha256=args.freeze_sha256,
        acceptance_sha256=args.freeze_acceptance_sha256,
        external_attempt_prefix=args.external_attempt_prefix,
    )
    labels_path = args.evidence_root / "labels.json"
    if _sha_file(labels_path) != manifest.get("label_commitment_sha256"):
        raise VlmEvidenceError("private label commitment differs")
    labels = _load_json(labels_path).get("cases")
    if not isinstance(labels, dict) or set(labels) != set(manifest["case_order"]):
        raise VlmEvidenceError("private label map differs")
    task = args.calibration_task.read_text(encoding="utf-8").strip()
    rubric = args.rubric.read_text(encoding="utf-8").strip()
    if _sha_file(args.calibration_task) != manifest.get(
        "calibration_task_sha256"
    ) or _sha_file(args.rubric) != manifest.get("rubric_sha256"):
        raise VlmEvidenceError("calibration prompt inputs differ")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise VlmEvidenceError("hosted VLM credential is unavailable")
    from npa.clients.storage import StorageClient

    storage_client = StorageClient.from_environment()
    results = []
    for case_id in manifest["case_order"]:
        case = manifest["cases"][case_id]
        result = _call_once(
            root=args.evidence_root,
            attempt_id=case_id,
            freeze_sha256=args.freeze_sha256,
            purpose="calibration",
            frame_records=case["frames"],
            task=task,
            rubric=rubric,
            api_key=api_key,
            external_attempt_prefix=args.external_attempt_prefix,
            storage_client=storage_client,
        )
        label = labels.get(case_id)
        if not isinstance(label, dict) or type(label.get("expected_label")) is not bool:
            raise VlmEvidenceError("private expected label differs")
        expected = label["expected_label"]
        predicted = result["score"] >= THRESHOLD
        results.append(
            {
                "case_id": case_id,
                "expected_label": expected,
                "predicted_label": predicted,
                "transport_sha256": _sha_file(
                    args.evidence_root / "transport" / case_id / "outcome.json"
                ),
            }
        )
    tp = sum(item["expected_label"] and item["predicted_label"] for item in results)
    tn = sum(
        not item["expected_label"] and not item["predicted_label"] for item in results
    )
    fp = sum(not item["expected_label"] and item["predicted_label"] for item in results)
    fn = sum(item["expected_label"] and not item["predicted_label"] for item in results)
    payload = {
        "format": CALIBRATION_FORMAT,
        "status": "pass" if (tp, tn, fp, fn) == (2, 2, 0, 0) else "failed",
        "freeze_sha256": args.freeze_sha256,
        "freeze_acceptance_sha256": args.freeze_acceptance_sha256,
        "model": MODEL,
        "served_model": MODEL,
        "threshold": THRESHOLD,
        "total": len(results),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "attempt_count": len(results),
        "one_shot": True,
        "results": results,
    }
    _exact_attempt_set(args.evidence_root, set(manifest["case_order"]))
    for case_id in manifest["case_order"]:
        _verified_transport(
            args.evidence_root,
            attempt_id=case_id,
            freeze_sha256=args.freeze_sha256,
            purpose="calibration",
            frame_records=manifest["cases"][case_id]["frames"],
            task=task,
            rubric=rubric,
        )
    path = args.evidence_root / "calibration.json"
    _write_json(path, payload)
    return {"status": "ok", "verdict": payload["status"], "sha256": _sha_file(path)}


def final(args: argparse.Namespace) -> dict[str, Any]:
    _private_root(args.evidence_root, existing=True)
    manifest = _verified_freeze(args)
    _verified_freeze_acceptance(
        root=args.evidence_root,
        manifest=manifest,
        freeze_sha256=args.freeze_sha256,
        acceptance_sha256=args.freeze_acceptance_sha256,
        external_attempt_prefix=args.external_attempt_prefix,
    )
    calibration_task = args.calibration_task.read_text(encoding="utf-8").strip()
    rubric = args.rubric.read_text(encoding="utf-8").strip()
    if _sha_file(args.calibration_task) != manifest.get(
        "calibration_task_sha256"
    ) or _sha_file(args.rubric) != manifest.get("rubric_sha256"):
        raise VlmEvidenceError("calibration prompt inputs differ")
    _verified_calibration(
        args.evidence_root,
        manifest=manifest,
        freeze_sha256=args.freeze_sha256,
        calibration_sha256=args.calibration_sha256,
        freeze_acceptance_sha256=args.freeze_acceptance_sha256,
        task=calibration_task,
        rubric=rubric,
    )
    task = args.final_task.read_text(encoding="utf-8").strip()
    if _sha_file(args.final_task) != manifest.get("final_task_sha256") or _sha_file(
        args.rubric
    ) != manifest.get("rubric_sha256"):
        raise VlmEvidenceError("final prompt inputs differ")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise VlmEvidenceError("hosted VLM credential is unavailable")
    from npa.clients.storage import StorageClient

    storage_client = StorageClient.from_environment()
    result = _call_once(
        root=args.evidence_root,
        attempt_id="final-one-shot",
        freeze_sha256=args.freeze_sha256,
        purpose="final",
        frame_records=manifest["final_frames"],
        task=task,
        rubric=rubric,
        api_key=api_key,
        external_attempt_prefix=args.external_attempt_prefix,
        storage_client=storage_client,
    )
    expected_attempts = {*manifest["case_order"], "final-one-shot"}
    _exact_attempt_set(args.evidence_root, expected_attempts)
    _verified_transport(
        args.evidence_root,
        attempt_id="final-one-shot",
        freeze_sha256=args.freeze_sha256,
        purpose="final",
        frame_records=manifest["final_frames"],
        task=task,
        rubric=rubric,
    )
    transport_records = []
    for evidence_dir in ("attempt-ledger", "transport"):
        evidence_root = args.evidence_root / evidence_dir
        for path in sorted(evidence_root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                transport_records.append(
                    {
                        "path": path.relative_to(args.evidence_root).as_posix(),
                        "sha256": _sha_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
    transport_manifest_path = args.evidence_root / "transport-manifest.json"
    _write_json(
        transport_manifest_path,
        {
            "format": TRANSPORT_FORMAT,
            "freeze_sha256": args.freeze_sha256,
            "freeze_acceptance_sha256": args.freeze_acceptance_sha256,
            "external_attempt_prefix_sha256": _sha_bytes(
                args.external_attempt_prefix.encode()
            ),
            "attempt_ids": sorted(expected_attempts),
            "attempts": len(expected_attempts),
            "files": transport_records,
        },
    )
    payload = {
        "format": FINAL_FORMAT,
        "status": "pass" if result["score"] >= THRESHOLD else "failed",
        "freeze_sha256": args.freeze_sha256,
        "freeze_acceptance_sha256": args.freeze_acceptance_sha256,
        "calibration_sha256": args.calibration_sha256,
        "final_frame_manifest_sha256": manifest["final_frame_manifest_sha256"],
        "model": MODEL,
        "served_model": result["served_model"],
        "threshold": THRESHOLD,
        "score": result["score"],
        "rationale": result["rationale"],
        "attempt_count": len(expected_attempts),
        "one_shot": len(expected_attempts) == len(manifest["case_order"]) + 1,
        "transport_sha256": _sha_file(
            args.evidence_root / "transport/final-one-shot/outcome.json"
        ),
        "transport_manifest_sha256": _sha_file(transport_manifest_path),
        "claim": "visual_coherence_only",
    }
    path = args.evidence_root / "final.json"
    _write_json(path, payload)
    return {"status": "ok", "verdict": payload["status"], "sha256": _sha_file(path)}


def verify_complete_evidence(
    root: Path,
    *,
    freeze_sha256: str,
    freeze_acceptance_sha256: str,
    calibration_sha256: str,
    final_sha256: str,
    external_attempt_prefix: str,
    rubric_path: Path,
    calibration_task_path: Path,
    final_task_path: Path,
) -> dict[str, Any]:
    """Rebuild requests and outcomes for one completed calibrated schedule."""
    args = argparse.Namespace(evidence_root=root, freeze_sha256=freeze_sha256)
    _private_root(root, existing=True)
    manifest = _verified_freeze(args)
    _verified_freeze_acceptance(
        root=root,
        manifest=manifest,
        freeze_sha256=freeze_sha256,
        acceptance_sha256=freeze_acceptance_sha256,
        external_attempt_prefix=external_attempt_prefix,
    )
    rubric = rubric_path.read_text(encoding="utf-8").strip()
    calibration_task = calibration_task_path.read_text(encoding="utf-8").strip()
    final_task = final_task_path.read_text(encoding="utf-8").strip()
    if (
        _sha_file(rubric_path) != manifest.get("rubric_sha256")
        or _sha_file(calibration_task_path) != manifest.get("calibration_task_sha256")
        or _sha_file(final_task_path) != manifest.get("final_task_sha256")
    ):
        raise VlmEvidenceError("completed VLM prompt inputs differ")
    calibration = _verified_calibration(
        root,
        manifest=manifest,
        freeze_sha256=freeze_sha256,
        calibration_sha256=calibration_sha256,
        freeze_acceptance_sha256=freeze_acceptance_sha256,
        task=calibration_task,
        rubric=rubric,
    )
    if _sha_file(root / "final.json") != final_sha256:
        raise VlmEvidenceError("final VLM result SHA-256 differs")
    final_result = _load_json(root / "final.json")
    final_outcome = _verified_transport(
        root,
        attempt_id="final-one-shot",
        freeze_sha256=freeze_sha256,
        purpose="final",
        frame_records=manifest["final_frames"],
        task=final_task,
        rubric=rubric,
    )
    expected_attempts = {*manifest["case_order"], "final-one-shot"}
    _exact_attempt_set(root, expected_attempts)
    transport_path = root / "transport-manifest.json"
    transport = _load_json(transport_path)
    records = []
    for directory_name in ("attempt-ledger", "transport"):
        for path in sorted((root / directory_name).rglob("*")):
            if path.is_file() and not path.is_symlink():
                records.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": _sha_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
    if (
        transport.get("format") != TRANSPORT_FORMAT
        or transport.get("freeze_sha256") != freeze_sha256
        or transport.get("freeze_acceptance_sha256") != freeze_acceptance_sha256
        or transport.get("external_attempt_prefix_sha256")
        != _sha_bytes(external_attempt_prefix.encode())
        or transport.get("attempt_ids") != sorted(expected_attempts)
        or transport.get("attempts") != len(expected_attempts)
        or transport.get("files") != records
    ):
        raise VlmEvidenceError("completed VLM transport manifest differs")
    if (
        final_result.get("format") != FINAL_FORMAT
        or final_result.get("status") != "pass"
        or final_result.get("freeze_sha256") != freeze_sha256
        or final_result.get("freeze_acceptance_sha256") != freeze_acceptance_sha256
        or final_result.get("calibration_sha256") != calibration_sha256
        or final_result.get("final_frame_manifest_sha256")
        != manifest["final_frame_manifest_sha256"]
        or final_result.get("model") != MODEL
        or final_result.get("served_model") != MODEL
        or final_result.get("threshold") != THRESHOLD
        or final_result.get("score") != final_outcome["score"]
        or final_result.get("rationale") != final_outcome["rationale"]
        or final_result.get("attempt_count") != len(expected_attempts)
        or final_result.get("one_shot") is not True
        or final_result.get("transport_sha256")
        != _sha_file(root / "transport/final-one-shot/outcome.json")
        or final_result.get("transport_manifest_sha256") != _sha_file(transport_path)
        or final_result.get("claim") != "visual_coherence_only"
    ):
        raise VlmEvidenceError("completed final VLM result differs")
    return {
        "freeze": manifest,
        "calibration": calibration,
        "final": final_result,
        "transport_manifest_sha256": _sha_file(transport_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--source-zip", type=Path, required=True)
    freeze_parser.add_argument("--source-sha256", required=True)
    freeze_parser.add_argument("--dataset-root", default="struktur28")
    freeze_parser.add_argument("--colmap-dir", default="sparse/0")
    freeze_parser.add_argument("--images-dir", default="images")
    freeze_parser.add_argument("--render-dir", type=Path, required=True)
    freeze_parser.add_argument(
        "--render-camera",
        required=True,
        help="One relative camera directory whose ordered frames form the final case.",
    )
    freeze_parser.add_argument("--rubric", type=Path, required=True)
    freeze_parser.add_argument("--calibration-task", type=Path, required=True)
    freeze_parser.add_argument("--final-task", type=Path, required=True)
    freeze_parser.add_argument("--output-root", type=Path, required=True)
    accept_parser = commands.add_parser("accept-freeze")
    accept_parser.add_argument("--evidence-root", type=Path, required=True)
    accept_parser.add_argument("--freeze-sha256", required=True)
    accept_parser.add_argument("--review-path", type=Path, required=True)
    accept_parser.add_argument("--external-attempt-prefix", required=True)
    for name in ("calibrate", "final"):
        command = commands.add_parser(name)
        command.add_argument("--evidence-root", type=Path, required=True)
        command.add_argument("--freeze-sha256", required=True)
        command.add_argument("--freeze-acceptance-sha256", required=True)
        command.add_argument("--external-attempt-prefix", required=True)
        command.add_argument("--rubric", type=Path, required=True)
        command.add_argument("--api-key-env", default="NEBIUS_TOKEN_FACTORY_KEY")
        if name == "calibrate":
            command.add_argument("--calibration-task", type=Path, required=True)
        else:
            command.add_argument("--calibration-task", type=Path, required=True)
            command.add_argument("--final-task", type=Path, required=True)
            command.add_argument("--calibration-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = {
            "freeze": freeze,
            "accept-freeze": accept_freeze,
            "calibrate": calibrate,
            "final": final,
        }[args.command](args)
    except VlmEvidenceError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 1
    except (OSError, ValueError):
        print(
            json.dumps(
                {"status": "failed", "error": "private evidence I/O failed"},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
