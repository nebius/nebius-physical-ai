#!/usr/bin/env python3
"""Freeze and execute one-shot visual controls for NCore qualification."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile
from typing import Any
import urllib.request

from PIL import Image

from npa.clients.token_factory import default_chat_extra
from npa.workbench.nurec.colmap import (
    ColmapConversionRequest,
    extract_colmap_zip,
    find_colmap_root,
    inspect_colmap_source,
)


FREEZE_FORMAT = "npa_ncore_vlm_freeze_v1"
CALIBRATION_FORMAT = "npa_ncore_vlm_calibration_v1"
FINAL_FORMAT = "npa_ncore_vlm_final_v1"
MODEL = "openbmb/MiniCPM-V-4_5"
ENDPOINT = "https://api.tokenfactory.nebius.com/v1/chat/completions"
THRESHOLD = 0.8
FRAME_COUNT = 4
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


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
                controls.append(stream.getvalue())
        except Exception as exc:
            raise VlmEvidenceError("block control generation failed") from exc
    return controls


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
        roles.append(("uniform", False, _uniform_controls(positives[0][1])))
        roles.append(("block-rotate-32", False, _block_corrupt(positives[1][1])))
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
    render_frames = sorted(
        path
        for path in args.render_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
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
            }
        )
    label_path = root / "labels.json"
    _write_json(label_path, {"cases": labels})
    manifest = {
        "format": FREEZE_FORMAT,
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
        "final_frames": final_records,
    }
    manifest_path = root / "freeze.json"
    _write_json(manifest_path, manifest)
    return {
        "status": "ok",
        "freeze_sha256": _sha_file(manifest_path),
        "label_commitment_sha256": manifest["label_commitment_sha256"],
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


def _call_once(
    *,
    root: Path,
    attempt_id: str,
    frame_records: list[dict[str, Any]],
    task: str,
    rubric: str,
    api_key: str,
) -> dict[str, Any]:
    attempt_root = root / "transport" / attempt_id
    attempt_root.mkdir(mode=0o700, parents=True)
    frames = []
    for record in sorted(frame_records, key=lambda item: item["order"]):
        path = root / record["path"]
        body = path.read_bytes()
        if _sha_bytes(body) != record["sha256"]:
            raise VlmEvidenceError("frozen frame hash differs")
        frames.append(body)
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
    request_bytes = _canonical(request_payload)
    _write_json(
        attempt_root / "attempt.json",
        {
            "status": "started",
            "attempt_id": attempt_id,
            "endpoint_sha256": _sha_bytes(ENDPOINT.encode()),
            "model": MODEL,
            "prompt_sha256": _sha_bytes(prompt.encode()),
            "frame_sha256": [_sha_bytes(frame) for frame in frames],
            "request_sha256": _sha_bytes(request_bytes),
        },
    )
    _write_private(attempt_root / "request.json", request_bytes)
    request = urllib.request.Request(
        ENDPOINT,
        data=request_bytes,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status_code = int(response.status)
            response_bytes = response.read()
    except Exception as exc:
        _write_json(
            attempt_root / "outcome.json",
            {"status": "transport_failed", "error_class": type(exc).__name__},
        )
        raise VlmEvidenceError(
            "hosted VLM transport failed; retry is prohibited"
        ) from exc
    _write_private(attempt_root / "response.json", response_bytes)
    if status_code != 200:
        _write_json(
            attempt_root / "outcome.json",
            {
                "status": "response_failed",
                "http_status": status_code,
                "response_sha256": _sha_bytes(response_bytes),
            },
        )
        raise VlmEvidenceError("hosted VLM returned a non-200 response")
    try:
        response_payload = json.loads(response_bytes)
        choice = response_payload["choices"][0]
        message = json.loads(choice["message"]["content"])
        served_model = response_payload["model"]
        usage = response_payload["usage"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        _write_json(
            attempt_root / "outcome.json",
            {
                "status": "response_failed",
                "error_class": "schema",
                "response_sha256": _sha_bytes(response_bytes),
            },
        )
        raise VlmEvidenceError("hosted VLM response schema differs") from exc
    if served_model != MODEL or choice.get("finish_reason") != "stop":
        _write_json(
            attempt_root / "outcome.json",
            {
                "status": "response_failed",
                "error_class": "identity_or_finish",
                "response_sha256": _sha_bytes(response_bytes),
            },
        )
        raise VlmEvidenceError("hosted VLM model or finish reason differs")
    if (
        not isinstance(message, dict)
        or type(message.get("success")) is not bool
        or not isinstance(message.get("score"), (int, float))
        or isinstance(message.get("score"), bool)
        or not 0 <= float(message["score"]) <= 1
        or not isinstance(message.get("rationale"), str)
        or not message["rationale"].strip()
        or not isinstance(usage, dict)
        or message["success"] is not (float(message["score"]) >= THRESHOLD)
    ):
        _write_json(
            attempt_root / "outcome.json",
            {
                "status": "response_failed",
                "error_class": "structured_result",
                "response_sha256": _sha_bytes(response_bytes),
            },
        )
        raise VlmEvidenceError("hosted VLM structured result differs")
    result = {
        "status": "complete",
        "attempt_id": attempt_id,
        "http_status": status_code,
        "requested_model": MODEL,
        "served_model": served_model,
        "finish_reason": choice["finish_reason"],
        "usage": usage,
        "request_sha256": _sha_bytes(request_bytes),
        "response_sha256": _sha_bytes(response_bytes),
        "prompt_sha256": _sha_bytes(prompt.encode()),
        "frame_sha256": [_sha_bytes(frame) for frame in frames],
        "success": message["success"],
        "score": float(message["score"]),
        "rationale": message["rationale"].strip(),
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
    ):
        raise VlmEvidenceError("freeze manifest contract differs")
    return manifest


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    _private_root(args.evidence_root, existing=True)
    manifest = _verified_freeze(args)
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
    results = []
    for case_id in manifest["case_order"]:
        case = manifest["cases"][case_id]
        result = _call_once(
            root=args.evidence_root,
            attempt_id=case_id,
            frame_records=case["frames"],
            task=task,
            rubric=rubric,
            api_key=api_key,
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
        "model": MODEL,
        "served_model": MODEL,
        "threshold": THRESHOLD,
        "total": len(results),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "results": results,
    }
    path = args.evidence_root / "calibration.json"
    _write_json(path, payload)
    return {"status": "ok", "verdict": payload["status"], "sha256": _sha_file(path)}


def final(args: argparse.Namespace) -> dict[str, Any]:
    _private_root(args.evidence_root, existing=True)
    manifest = _verified_freeze(args)
    calibration_path = args.evidence_root / "calibration.json"
    if _sha_file(calibration_path) != args.calibration_sha256:
        raise VlmEvidenceError("calibration SHA-256 differs")
    calibration = _load_json(calibration_path)
    if (
        calibration.get("format") != CALIBRATION_FORMAT
        or calibration.get("status") != "pass"
        or (calibration.get("true_positives"), calibration.get("true_negatives"))
        != (2, 2)
        or (calibration.get("false_positives"), calibration.get("false_negatives"))
        != (0, 0)
    ):
        raise VlmEvidenceError("calibration did not pass")
    task = args.final_task.read_text(encoding="utf-8").strip()
    rubric = args.rubric.read_text(encoding="utf-8").strip()
    if _sha_file(args.final_task) != manifest.get("final_task_sha256") or _sha_file(
        args.rubric
    ) != manifest.get("rubric_sha256"):
        raise VlmEvidenceError("final prompt inputs differ")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise VlmEvidenceError("hosted VLM credential is unavailable")
    result = _call_once(
        root=args.evidence_root,
        attempt_id="final-one-shot",
        frame_records=manifest["final_frames"],
        task=task,
        rubric=rubric,
        api_key=api_key,
    )
    transport_records = []
    transport_root = args.evidence_root / "transport"
    for path in sorted(transport_root.rglob("*")):
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
            "format": "npa_ncore_vlm_transport_manifest_v1",
            "attempts": len(manifest["case_order"]) + 1,
            "files": transport_records,
        },
    )
    payload = {
        "format": FINAL_FORMAT,
        "status": "pass" if result["score"] >= THRESHOLD else "failed",
        "freeze_sha256": args.freeze_sha256,
        "calibration_sha256": args.calibration_sha256,
        "model": MODEL,
        "served_model": result["served_model"],
        "threshold": THRESHOLD,
        "score": result["score"],
        "rationale": result["rationale"],
        "one_shot": True,
        "transport_sha256": _sha_file(
            args.evidence_root / "transport/final-one-shot/outcome.json"
        ),
        "transport_manifest_sha256": _sha_file(transport_manifest_path),
        "claim": "visual_coherence_only",
    }
    path = args.evidence_root / "final.json"
    _write_json(path, payload)
    return {"status": "ok", "verdict": payload["status"], "sha256": _sha_file(path)}


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
    freeze_parser.add_argument("--rubric", type=Path, required=True)
    freeze_parser.add_argument("--calibration-task", type=Path, required=True)
    freeze_parser.add_argument("--final-task", type=Path, required=True)
    freeze_parser.add_argument("--output-root", type=Path, required=True)
    for name in ("calibrate", "final"):
        command = commands.add_parser(name)
        command.add_argument("--evidence-root", type=Path, required=True)
        command.add_argument("--freeze-sha256", required=True)
        command.add_argument("--rubric", type=Path, required=True)
        command.add_argument("--api-key-env", default="NEBIUS_TOKEN_FACTORY_KEY")
        if name == "calibrate":
            command.add_argument("--calibration-task", type=Path, required=True)
        else:
            command.add_argument("--final-task", type=Path, required=True)
            command.add_argument("--calibration-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = {"freeze": freeze, "calibrate": calibrate, "final": final}[
            args.command
        ](args)
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
