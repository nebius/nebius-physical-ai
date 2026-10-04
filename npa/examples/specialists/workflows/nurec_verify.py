"""Independently decode NuRec workflow outputs and bind them to terminal receipts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickletools
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

_STAGES = {"check", "fetch", "reconstruct", "render", "visualize", "finalize"}
_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
_VIDEOS = {".mp4", ".mov", ".mkv", ".webm"}
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document(path: Path) -> dict:
    payload = json.loads(path.read_text())
    _require(isinstance(payload, dict), "expected_object")
    return payload


def _raw_evidence(evidence: Path, payload: dict, prefix: str) -> str:
    digest = payload.get(f"{prefix}_sha256", "")
    _require(bool(_DIGEST.fullmatch(digest)), "missing_raw_evidence_digest")
    source = evidence.parent / payload[f"{prefix}_path"]
    _require(source.is_file() and _digest(source) == digest, "raw_evidence_changed")
    return digest


def _workflow(path: Path) -> dict:
    payload = _document(path)
    _require(payload.get("source") == "workbench-live-status", "wrong_status_source")
    _require(payload.get("terminal") is True, "nonterminal_workflow")
    _require(payload.get("status") == "succeeded", "unsuccessful_workflow")
    _require(bool(payload.get("run_id")), "missing_run_identity")
    observed = datetime.fromisoformat(payload["observed_at"].replace("Z", "+00:00"))
    _require(observed.tzinfo is not None, "status_timestamp_without_timezone")
    stages = payload["stages"]
    _require(isinstance(stages, list) and len(stages) == len(_STAGES), "missing_stages")
    _require({stage["name"] for stage in stages} == _STAGES, "unexpected_stages")
    _require(all(stage["status"] == "succeeded" for stage in stages), "failed_stage")
    _require(
        bool(_DIGEST.fullmatch(payload["submitted_spec_sha256"])), "missing_spec_digest"
    )
    _raw_evidence(path, payload, "raw_status")
    return payload


def _provenance(root: Path, scene: str, variant: str) -> dict:
    manifest = _document(root / "ncore/manifest.json")
    _require(manifest.get("status") == "ok", "failed_fetch")
    _require(manifest.get("scene") == scene, "wrong_scene")
    _require(manifest.get("variant") == variant, "wrong_variant")
    for key, expected in (("observed_scene", scene), ("observed_variant", variant)):
        _require(manifest.get(key) == expected, "unverified_capture_identity")
    _require(isinstance(manifest.get("dataset_id"), str), "missing_dataset")
    _require(manifest.get("shard_count", 0) > 0, "no_capture_shards")
    _require(bool(manifest.get("camera_ids")), "no_capture_cameras")
    return {
        "scene_matches": True,
        "variant_matches": True,
        "camera_count": len(manifest["camera_ids"]),
        "shard_count": manifest["shard_count"],
        "manifest_sha256": _digest(root / "ncore/manifest.json"),
    }


def _package_members(package: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = package.infolist()
    _require(bool(members), "empty_usdz")
    names = [member.filename for member in members]
    _require(len(names) == len(set(names)), "duplicate_usdz_members")
    for member in members:
        name = PurePosixPath(member.filename)
        _require(
            not name.is_absolute() and ".." not in name.parts, "unsafe_usdz_member"
        )
        _require(member.compress_type == zipfile.ZIP_STORED, "compressed_usdz_member")
    _require(package.testzip() is None, "usdz_crc_failure")
    _require(Path(names[0]).suffix in {".usd", ".usda", ".usdc"}, "missing_usd_root")
    return members


def _checkpoint(package: zipfile.ZipFile, members: list) -> dict:
    # NVIDIA's documented NuRec package contains checkpoint.ckpt. Inspect its
    # storage archive without unpickling model objects or importing vendor code.
    matches = [
        member for member in members if Path(member.filename).name == "checkpoint.ckpt"
    ]
    _require(len(matches) == 1 and matches[0].file_size > 0, "missing_checkpoint")
    with package.open(matches[0]) as stream, zipfile.ZipFile(stream) as checkpoint:
        entries = checkpoint.infolist()
        names = [member.filename for member in entries]
        _require(checkpoint.testzip() is None, "checkpoint_crc_failure")
        _checkpoint_metadata(checkpoint, names)
        tensors = [
            entry for entry in entries if re.search(r"/data/\d+$", entry.filename)
        ]
        _require(
            any(entry.file_size > 0 for entry in tensors),
            "no_checkpoint_tensors",
        )
    return {
        "checkpoint_bytes": matches[0].file_size,
        "tensor_storage_count": len(tensors),
        "empty_tensor_storage_count": sum(entry.file_size == 0 for entry in tensors),
        "checkpoint_validation": "archive_structure_only_no_pickle_execution",
    }


def _checkpoint_metadata(checkpoint: zipfile.ZipFile, names: list[str]) -> None:
    metadata = [name for name in names if name.endswith("/data.pkl")]
    _require(len(metadata) == 1, "invalid_checkpoint_metadata")
    operations = list(pickletools.genops(checkpoint.read(metadata[0])))
    _require(
        bool(operations) and operations[-1][0].name == "STOP",
        "truncated_checkpoint_metadata",
    )
    references = [
        argument for _, argument, _ in operations if isinstance(argument, str)
    ]
    _require(
        any("_rebuild_tensor" in value for value in references),
        "no_checkpoint_tensor_references",
    )


def _usdz(root: Path) -> dict:
    from pxr import Usd

    path = root / "reconstruction/last.usdz"
    with zipfile.ZipFile(path) as package:
        members = _package_members(package)
        checkpoint = _checkpoint(package, members)
    stage = Usd.Stage.Open(str(path))
    _require(bool(stage), "unreadable_usd_scene")
    prim_count = sum(1 for _ in stage.Traverse())
    _require(prim_count > 0, "empty_usd_scene")
    return {"member_count": len(members), "prim_count": prim_count, **checkpoint}


def _flatten(payload: dict, prefix: str = "") -> dict:
    result = {}
    for name, value in payload.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            result.update(_flatten(value, key))
        else:
            result[key] = value
    return result


def _metrics(root: Path) -> dict:
    payload = yaml.safe_load((root / "reconstruction/metrics.yaml").read_text())
    _require(isinstance(payload, dict), "invalid_metrics_document")
    flat = _flatten(payload)
    metrics = {}
    for name in ("psnr", "ssim", "lpips"):
        key = f"test/{name}"
        value = flat.get(key, flat.get(f"aggregated_metrics/{key}/value"))
        _require(
            type(value) in (int, float) and math.isfinite(value),
            "missing_or_nonfinite_metrics",
        )
        metrics[key] = float(value)
    _require(metrics["test/psnr"] > 0, "invalid_psnr")
    _require(0 < metrics["test/ssim"] <= 1, "invalid_ssim")
    _require(metrics["test/lpips"] >= 0, "invalid_lpips")
    return metrics


def _media_files(root: Path) -> tuple[list[Path], list[Path]]:
    files = sorted(path for path in (root / "novel_views").rglob("*") if path.is_file())
    return (
        [path for path in files if path.suffix.lower() in _IMAGES],
        [path for path in files if path.suffix.lower() in _VIDEOS],
    )


def _decode_video(path: Path) -> int:
    executable = shutil.which("ffmpeg")
    if not executable:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        executable,
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-progress",
        "pipe:1",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    _require(result.returncode == 0 and not result.stderr.strip(), "corrupt_video")
    frames = re.findall(r"^frame=(\d+)$", result.stdout, re.MULTILINE)
    _require(bool(frames) and int(frames[-1]) > 0, "empty_video")
    _require("progress=end" in result.stdout, "incomplete_video_decode")
    return int(frames[-1])


def _media(root: Path) -> dict:
    from PIL import Image

    images, videos = _media_files(root)
    _require(bool(images) and bool(videos), "missing_novel_images_or_videos")
    for path in images:
        with Image.open(path) as image:
            image.load()
            _require(min(image.size) > 0, "empty_image")
    frames = sum(_decode_video(path) for path in videos)
    return {
        "image_count": len(images),
        "video_count": len(videos),
        "decoded_video_frames": frames,
    }


def _rrd_document(chunks: list, entity: str) -> dict:
    documents = []
    for chunk in chunks:
        if str(chunk.entity_path) != entity:
            continue
        batch = chunk.to_record_batch()
        for row in batch.column("TextDocument:text").to_pylist():
            documents.extend(row or [])
    _require(len(documents) == 1, "missing_or_duplicate_rrd_document")
    match = re.search(r"```json\s*(.*?)\s*```", documents[0], re.DOTALL)
    _require(match is not None, "invalid_rrd_document")
    return json.loads(match.group(1))


def _legacy_expected_rrd_images(root: Path, settings: dict) -> dict:
    from PIL import Image

    groups = {}
    for path in _media_files(root)[0]:
        camera = path.parent.name if path.parent != root / "novel_views" else "frames"
        groups.setdefault(camera, []).append(path)
    result = {}
    for camera, paths in groups.items():
        cap = settings["max_frames_per_entity"]
        selected = (
            paths
            if cap <= 0 or len(paths) <= cap
            else [paths[int(i * len(paths) / cap)] for i in range(cap)]
        )
        for path in selected:
            match = re.search(r"(\d+)\D*$", path.stem)
            identity = (camera, int(match.group(1)) if match else 0)
            _require(identity not in result, "duplicate_source_frame_identity")
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                dimension = settings["max_frame_dim"]
                if dimension > 0 and max(rgb.size) > dimension:
                    rgb.thumbnail((dimension, dimension))
                encoded = io.BytesIO()
                rgb.save(encoded, format="JPEG", quality=settings["jpeg_quality"])
            result[identity] = encoded.getvalue()
    return result


def _legacy_rrd_frames(chunks: list, expected: dict) -> int:
    from PIL import Image

    observed = set()
    for chunk in chunks:
        entity = str(chunk.entity_path)
        if not entity.startswith("/novel_view/"):
            continue
        batch = chunk.to_record_batch()
        _require(
            {"EncodedImage:blob", "frame"} <= set(batch.schema.names),
            "rrd_missing_image_data",
        )
        rows = zip(
            batch.column("frame").to_pylist(),
            batch.column("EncodedImage:blob").to_pylist(),
            strict=True,
        )
        for frame, blobs in rows:
            identity = (entity.removeprefix("/novel_view/"), frame)
            _require(identity not in observed, "duplicate_rrd_image")
            _require(bool(blobs) and len(blobs) == 1, "invalid_rrd_image_row")
            encoded = bytes(blobs[0])
            _require(expected.get(identity) == encoded, "rrd_image_differs_from_run")
            with Image.open(io.BytesIO(encoded)) as image:
                image.load()
            observed.add(identity)
    _require(bool(observed) and observed == set(expected), "incomplete_rrd_images")
    return len(observed)


def _review_image_bytes(path: Path, settings: dict) -> bytes:
    from PIL import Image

    with Image.open(path) as source:
        rgb = source.convert("RGB")
        dimension = settings["max_frame_dim"]
        if dimension > 0 and max(rgb.size) > dimension:
            rgb.thumbnail((dimension, dimension))
        encoded = io.BytesIO()
        rgb.save(encoded, format="JPEG", quality=settings["jpeg_quality"])
    return encoded.getvalue()


def _rrd_source_images(root: Path, settings: dict) -> dict:
    expected = {}
    for directory, prefix in (
        ("novel_views", "/novel_view"),
        ("reconstruction", "/reconstruction"),
    ):
        source = root / directory
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _IMAGES:
                continue
            relative = path.parent.relative_to(source)
            group = "frames" if relative == Path(".") else relative.as_posix()
            match = re.search(r"(\d+)\D*$", path.stem)
            identity = (f"{prefix}/{group}", int(match.group(1)) if match else 0)
            _require(identity not in expected, "duplicate_source_frame_identity")
            expected[identity] = _review_image_bytes(path, settings)
    _require(bool(expected), "missing_rrd_source_images")
    return expected


def _rrd_source_rows(chunks: list, expected: dict) -> set:
    from PIL import Image

    observed = set()
    for chunk in chunks:
        entity = str(chunk.entity_path)
        if not entity.startswith(("/novel_view/", "/reconstruction/")):
            continue
        batch = chunk.to_record_batch()
        _require(
            {"EncodedImage:blob", "frame"} <= set(batch.schema.names),
            "rrd_missing_image_data",
        )
        rows = zip(
            batch.column("frame").to_pylist(),
            batch.column("EncodedImage:blob").to_pylist(),
            strict=True,
        )
        for frame, blobs in rows:
            identity = (entity, frame)
            _require(identity not in observed, "duplicate_rrd_image")
            _require(bool(blobs) and len(blobs) == 1, "invalid_rrd_image_row")
            encoded = bytes(blobs[0])
            _require(expected.get(identity) == encoded, "rrd_image_differs_from_run")
            with Image.open(io.BytesIO(encoded)) as image:
                image.load()
            observed.add(identity)
    return observed


def _rrd_sample_coverage(expected: dict, observed: set, cap: int) -> None:
    groups = {entity for entity, _ in expected}
    _require({entity for entity, _ in observed} == groups, "incomplete_rrd_entities")
    for entity in groups:
        source = {frame for group, frame in expected if group == entity}
        selected = {frame for group, frame in observed if group == entity}
        count = len(source) if cap <= 0 else min(len(source), cap)
        _require(len(selected) == count, "incomplete_rrd_images")
        _require(min(source) in selected, "rrd_missing_first_frame")
        if count > 1:
            _require(max(source) in selected, "rrd_missing_last_frame")
        if cap <= 0 or len(source) <= cap:
            _require(selected == source, "incomplete_rrd_images")


def _rrd_image_checks(root, chunks, settings, settings_source):
    if settings_source == "independent_pinned_producer":
        count = _legacy_rrd_frames(chunks, _legacy_expected_rrd_images(root, settings))
        return {
            "verified_image_rows": count,
            "verified_novel_image_rows": count,
            "verified_reconstruction_image_rows": 0,
            "image_verification_scope": "legacy_novel_views_only",
        }
    expected = _rrd_source_images(root, settings)
    observed = _rrd_source_rows(chunks, expected)
    _rrd_sample_coverage(expected, observed, settings["max_frames_per_entity"])
    novel = sum(entity.startswith("/novel_view/") for entity, _ in observed)
    return {
        "verified_image_rows": len(observed),
        "verified_novel_image_rows": novel,
        "verified_reconstruction_image_rows": len(observed) - novel,
        "image_verification_scope": "novel_views_and_reconstruction",
    }


def _legacy_rrd_review(root: Path, terminal: dict, evidence: Path | None) -> dict:
    _require(evidence is not None, "missing_or_duplicate_rrd_document")
    payload = _document(evidence)
    _require(payload.get("source") == "workbench-pinned-viewer", "wrong_viewer_source")
    _require(payload.get("run_id") == terminal["run_id"], "wrong_viewer_run")
    _require(
        payload.get("submitted_spec_sha256") == terminal["submitted_spec_sha256"],
        "wrong_viewer_spec",
    )
    _require(
        payload.get("rrd_sha256") == _digest(root / "reports/sim2real.rrd"),
        "viewer_recording_changed",
    )
    _raw_evidence(evidence, payload, "raw_producer")
    producer = _document(evidence.parent / payload["raw_producer_path"])
    _require(producer.get("source") == "verified-viewer-producer", "unverified_viewer")
    for key in ("producer_image", "producer_source_sha256", "settings"):
        _require(payload.get(key) == producer.get(key), "viewer_producer_mismatch")
    _require(
        re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", payload.get("producer_image", ""))
        is not None,
        "unpinned_viewer_producer",
    )
    _require(
        _DIGEST.fullmatch(payload.get("producer_source_sha256", "")) is not None,
        "missing_viewer_source_digest",
    )
    return payload["settings"]


def _rrd_review(root, terminal, chunks, evidence):
    embedded = any(
        str(chunk.entity_path) == "/provenance/rrd_review" for chunk in chunks
    )
    if embedded:
        _require(evidence is None, "legacy_evidence_for_current_recording")
        settings = _rrd_document(chunks, "/provenance/rrd_review")
    else:
        settings = _legacy_rrd_review(root, terminal, evidence)
    _require(
        settings.get("schema") == "npa.nurec.rrd-review.v1", "wrong_rrd_review_schema"
    )
    _require(
        all(
            type(settings.get(key)) is int
            for key in ("max_frames_per_entity", "max_frame_dim", "jpeg_quality")
        ),
        "invalid_rrd_settings",
    )
    return settings, "embedded" if embedded else "independent_pinned_producer"


def _rrd(root: Path, terminal: dict, review_evidence: Path | None = None) -> dict:
    from npa.viz.recordings import load_recording

    recording = load_recording(root / "reports/sim2real.rrd")
    _require(
        recording.application_id() == "neural-reconstruction", "wrong_rrd_application"
    )
    _require(recording.recording_id() == terminal["run_id"], "wrong_rrd_run")
    chunks = list(recording.chunks())
    settings, settings_source = _rrd_review(root, terminal, chunks, review_evidence)
    image_checks = _rrd_image_checks(root, chunks, settings, settings_source)
    metrics = yaml.safe_load((root / "reconstruction/metrics.yaml").read_text())
    _require(
        _rrd_document(chunks, "/gaussians/summary") == metrics, "rrd_metrics_mismatch"
    )
    return {
        "chunk_count": len(chunks),
        **image_checks,
        "run_identity_matches": True,
        "review_settings_source": settings_source,
        "review_evidence_sha256": _digest(review_evidence) if review_evidence else None,
    }


def _offset(command: list, flag: str) -> tuple[float, float, float]:
    _require(command.count(flag) == 1, "missing_or_duplicate_offset")
    index = command.index(flag) + 1
    values = command[index : index + 3]
    _require(len(values) == 3, "incomplete_offset")
    return tuple(float(value) for value in values)


def _render_receipt(evidence: Path | None, terminal: dict) -> dict:
    _require(evidence is not None, "missing_render_receipt")
    payload = _document(evidence)
    _require(payload.get("source") == "workbench-render-receipt", "wrong_render_source")
    _require(payload.get("run_id") == terminal["run_id"], "wrong_render_run")
    _require(
        payload.get("submitted_spec_sha256") == terminal["submitted_spec_sha256"],
        "wrong_render_spec",
    )
    _raw_evidence(evidence, payload, "raw_receipt")
    receipt = payload["receipt"]
    raw = _document(evidence.parent / payload["raw_receipt_path"])
    _require(raw == receipt, "render_receipt_differs_from_raw")
    _require(
        receipt.get("status") == "ok" and receipt.get("novel_view") is True,
        "failed_or_training_view_render",
    )
    return payload


def _render_offset(receipt: dict, offsets: tuple) -> tuple:
    command = receipt["command"]
    _require(
        isinstance(command, list) and "render" in command, "invalid_render_command"
    )
    _require(
        "--no-replicate-training-views" in command
        and "--replicate-training-views" not in command,
        "training_view_render",
    )
    observed = _offset(command, "--rig-translation-offset")
    _require(
        all(math.isfinite(value) for value in offsets) and any(offsets),
        "invalid_expected_offset",
    )
    _require(len(offsets) == 3 and observed == tuple(offsets), "wrong_novel_offset")
    rotation = _offset(command, "--rig-rotation-offset")
    _require(rotation == (0.0, 0.0, 0.0), "unexpected_rig_rotation")
    return observed


def _render(root: Path, evidence: Path | None, terminal: dict, offsets: tuple) -> dict:
    payload = _render_receipt(evidence, terminal)
    receipt = payload["receipt"]
    observed = _render_offset(receipt, offsets)
    _require(
        payload["artifact_sha256"] == _digest(root / "reconstruction/last.usdz"),
        "render_model_mismatch",
    )
    images, videos = _media_files(root)
    actual = {str(path.relative_to(root)): _digest(path) for path in images + videos}
    _require(payload["media_sha256"] == actual, "render_media_mismatch")
    _require(receipt.get("frame_count") == len(images), "render_frame_count_mismatch")
    _require(receipt.get("video_count") == len(videos), "render_video_count_mismatch")
    return {
        "offsets_verified": True,
        "translation": list(observed),
        "media_hashes_match": True,
    }


def _final(root: Path, terminal: dict, checks: dict) -> dict:
    report = _document(root / "reports/final.json")
    _require(
        report.get("status") == "ok" and not report.get("errors"), "failed_final_report"
    )
    _require(report.get("run_id") == terminal["run_id"], "wrong_final_run")
    _require(
        report.get("capability") == "neural-reconstruction", "wrong_final_capability"
    )
    _require(
        all(
            report.get(key) is True
            for key in ("has_usdz", "has_rrd", "has_novel_views")
        ),
        "inconsistent_final_flags",
    )
    _require(
        {"usdz", "media", "rrd"} <= checks.keys(), "final_flags_contradict_artifacts"
    )
    return {"flags_match_verified_artifacts": True}


def _inventory(root: Path) -> list[dict]:
    paths = sorted(root.rglob("*"))
    _require(not any(path.is_symlink() for path in paths), "symlink_in_artifacts")
    return [
        {
            "relative_path": str(path.relative_to(root)),
            "sha256": _digest(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
        if path.is_file()
    ]


def _check(result: dict, name: str, operation) -> Any:
    try:
        value = operation()
    except ImportError:
        result["errors"].append({"check": name, "reason": "dependency_unavailable"})
        return None
    except Exception as error:  # noqa: BLE001 - decoder boundaries must return sanitized failure.
        reason = (
            str(error)
            if type(error) is ValueError and re.fullmatch(r"[a-z_]+", str(error))
            else "invalid_or_missing_evidence"
        )
        result["errors"].append({"check": name, "reason": reason})
        return None
    result["checks"][name] = value
    return value


def _artifact_checks(
    root,
    terminal,
    result,
    *,
    scene,
    variant,
    novel_offsets,
    render_evidence,
    legacy_rrd_review_evidence,
):
    operations = (
        ("provenance", lambda: _provenance(root, scene, variant)),
        ("usdz", lambda: _usdz(root)),
        ("metrics", lambda: _metrics(root)),
        ("media", lambda: _media(root)),
        ("rrd", lambda: _rrd(root, terminal, legacy_rrd_review_evidence)),
        ("render", lambda: _render(root, render_evidence, terminal, novel_offsets)),
        ("final", lambda: _final(root, terminal, result["checks"])),
    )
    for name, operation in operations:
        _check(result, name, operation)


def _verification(
    root,
    scene,
    variant,
    novel_offsets,
    terminal_evidence,
    render_evidence,
    legacy_rrd_review_evidence,
):
    root = Path(root)
    result = {"passed": False, "offsets_verified": False, "errors": [], "checks": {}}
    inventory = _check(result, "artifacts", lambda: _inventory(root))
    terminal = _check(result, "workflow", lambda: _workflow(Path(terminal_evidence)))
    if inventory is None or terminal is None:
        result["checks"].pop("workflow", None)
        return result
    result["checks"]["workflow"] = {
        "status": "succeeded",
        "stage_count": len(_STAGES),
        "raw_status_sha256": terminal["raw_status_sha256"],
        "submitted_spec_sha256": terminal["submitted_spec_sha256"],
    }
    _artifact_checks(
        root,
        terminal,
        result,
        scene=scene,
        variant=variant,
        novel_offsets=novel_offsets,
        render_evidence=render_evidence,
        legacy_rrd_review_evidence=legacy_rrd_review_evidence,
    )
    result["offsets_verified"] = (
        result["checks"].get("render", {}).get("offsets_verified", False)
    )
    result["passed"] = not result["errors"]
    return result


def verify_run(
    root: Path,
    *,
    scene: str,
    variant: str,
    novel_offsets: tuple[float, float, float],
    terminal_evidence: Path,
    render_evidence: Path | None = None,
    legacy_rrd_review_evidence: Path | None = None,
) -> dict:
    """Verify downloaded native artifacts against independently collected receipts.

    Args:
        root: Fresh downloaded run tree; no symlinks are accepted.
        scene: Required capture scene from the benchmark protocol.
        variant: Required capture variant from the benchmark protocol.
        novel_offsets: Required rig translation in meters; rotation must be zero.
        terminal_evidence: Normalized live status with raw evidence path and hash.
        render_evidence: Native render result, submitted identity and output hashes.
        legacy_rrd_review_evidence: Optional independent encoding evidence for a
            pinned older viewer without embedded review settings. The collector
            must verify producer origin; local hashes alone do not attest an image.
    Returns:
        Sanitized checks, artifact hashes and a fail-closed overall verdict.
        Checkpoint validation is structural; this does not evaluate model weights.
    Raises:
        None. Unreadable evidence and unavailable decoders produce failed checks.
    """
    return _verification(
        root,
        scene,
        variant,
        novel_offsets,
        terminal_evidence,
        render_evidence,
        legacy_rrd_review_evidence,
    )
