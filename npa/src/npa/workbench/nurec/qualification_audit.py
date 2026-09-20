"""Production audit that binds downloaded NCore/NRE/Rerun qualification bytes."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any
import zipfile

import yaml

from npa.errors import NpaError
from npa.workbench.nurec.evidence import (
    NurecEvidenceError,
    RECONSTRUCTION_RECEIPT_FORMAT,
    RENDER_RECEIPT_FORMAT,
    _decode_image,
    _decode_video,
    _nested_value,
    _sequence_inventory,
    _yaml_mapping,
    validate_runtime_attestation,
)
from npa.workbench.nurec.ncore_audit import AUDIT_FORMAT
from npa.workbench.nurec.nurec import parse_metrics_yaml
from npa.workbench.nurec.qualification_readback import (
    READBACK_FORMAT,
    local_inventory,
)


AUDIT_FORMAT_VERSION = "npa_ncore_qualification_audit_v1"
NATIVE_RECIPE = "configs/experimental/3dgut/3dgut_colmap.yaml"
MIN_PSNR = 15.0
MIN_SSIM = 0.5
MAX_LPIPS = 0.5
USD_RUNTIME_VERSION = (0, 25, 11)


class NcoreQualificationAuditError(NpaError):
    """Downloaded workload evidence failed an objective byte-bound audit."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NcoreQualificationAuditError("required qualification artifact is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NcoreQualificationAuditError(
            "required qualification JSON is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise NcoreQualificationAuditError(
            "required qualification JSON is not an object"
        )
    return value


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _record_matches(path: Path, record: Any) -> bool:
    return (
        isinstance(record, dict)
        and record.get("sha256") == _sha(path)
        and record.get("bytes") == path.stat().st_size
    )


def _nonzero_offset(value: Any) -> bool:
    try:
        values = [float(item.strip()) for item in str(value).split(",")]
    except ValueError:
        return False
    return (
        len(values) == 3
        and all(math.isfinite(item) for item in values)
        and any(abs(item) > 1e-9 for item in values)
    )


def _final_workflow_status(
    path: Path,
    *,
    recording_id: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    status = _json(path)
    stages = status.get("stages")
    if (
        status.get("run_id") != recording_id
        or status.get("status") != "SUCCEEDED"
        or not isinstance(stages, dict)
    ):
        raise NcoreQualificationAuditError("final workflow status differs")
    by_state = {
        str(value.get("workflow_state") or ""): value
        for value in stages.values()
        if isinstance(value, dict)
    }
    if set(by_state) != {"reconstruct", "render", "visualize", "finalize"} or any(
        stage.get("state") != "SUCCEEDED" for stage in by_state.values()
    ):
        raise NcoreQualificationAuditError(
            "final workflow did not complete the qualification graph"
        )
    run_sha = hashlib.sha256(recording_id.encode()).hexdigest()
    if runtime.get("workflow_run_id_sha256") != run_sha:
        raise NcoreQualificationAuditError("runtime workflow run binding differs")
    selected: list[dict[str, Any]] = []
    for stage_name in ("reconstruct", "render"):
        stage = by_state[stage_name]
        job_id = str(stage.get("managed_job_id") or "")
        job_name = str(stage.get("job_name") or "")
        attempts = stage.get("managed_job_attempts")
        matches = (
            [
                attempt
                for attempt in attempts
                if isinstance(attempt, dict)
                and str(attempt.get("job_id") or "") == job_id
                and attempt.get("job_name") == job_name
                and attempt.get("state") == "SUCCEEDED"
            ]
            if isinstance(attempts, list)
            else []
        )
        observed = runtime["stages"][stage_name]
        if (
            stage.get("job_attribution") == "ambiguous"
            or len(matches) != 1
            or observed.get("managed_job_id_sha256")
            != hashlib.sha256(job_id.encode()).hexdigest()
            or observed.get("managed_job_name_sha256")
            != hashlib.sha256(job_name.encode()).hexdigest()
        ):
            raise NcoreQualificationAuditError(
                "runtime attestation is not the final successful workflow attempt"
            )
        selected.append(
            {
                "stage": stage_name,
                "managed_job_id_sha256": observed["managed_job_id_sha256"],
                "managed_job_name_sha256": observed["managed_job_name_sha256"],
            }
        )
    return {
        "sha256": _sha(path),
        "status": "SUCCEEDED",
        "run_id_sha256": run_sha,
        "selected_stage_jobs_sha256": _canonical_sha(selected),
    }


def _complete_readback(root: Path, receipt_path: Path) -> dict[str, Any]:
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.stat().st_uid != os.getuid()
        or receipt_path.stat().st_nlink != 1
        or receipt_path.stat().st_mode & 0o077
    ):
        raise NcoreQualificationAuditError(
            "complete readback receipt is missing or not private"
        )
    receipt = _json(receipt_path)
    local = local_inventory(root)
    if (
        receipt.get("format") != READBACK_FORMAT
        or receipt.get("status") != "pass"
        or receipt.get("stable_listing") is not True
        or receipt.get("object_count") != len(local)
        or receipt.get("local_inventory") != local
        or receipt.get("local_inventory_sha256") != _canonical_sha(local)
        or not isinstance(receipt.get("s3_inventory"), list)
        or receipt.get("object_count") != len(receipt["s3_inventory"])
        or receipt.get("s3_inventory_sha256") != _canonical_sha(receipt["s3_inventory"])
        or {item["path"] for item in local}
        != {
            str(item.get("path") or "")
            for item in receipt["s3_inventory"]
            if isinstance(item, dict)
        }
    ):
        raise NcoreQualificationAuditError(
            "complete qualification readback binding differs"
        )
    return {
        "receipt_sha256": _sha(receipt_path),
        "object_count": len(local),
        "local_inventory_sha256": _canonical_sha(local),
        "s3_inventory_sha256": receipt["s3_inventory_sha256"],
    }


def _usdz(path: Path) -> dict[str, Any]:
    try:
        from pxr import Usd
    except ImportError as exc:
        raise NcoreQualificationAuditError(
            "USD runtime is required for qualification audit"
        ) from exc
    try:
        if tuple(Usd.GetVersion()) != USD_RUNTIME_VERSION:
            raise ValueError("USD runtime version differs")
        with zipfile.ZipFile(path) as package:
            members = package.infolist()
            if (
                not members
                or package.testzip() is not None
                or Path(members[0].filename).suffix.lower()
                not in {".usd", ".usda", ".usdc"}
                or any(
                    member.compress_type != zipfile.ZIP_STORED or member.file_size <= 0
                    for member in members
                )
            ):
                raise ValueError("USDZ package contract differs")
        stage = Usd.Stage.Open(str(path))
        if not stage or not any(stage.Traverse()):
            raise ValueError("USDZ stage has no scene prims")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise NcoreQualificationAuditError("trained USDZ did not reopen") from exc
    return {
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        "reopened": True,
        "usd_runtime_version": ".".join(map(str, USD_RUNTIME_VERSION[1:])),
    }


def _rrd_document(chunks: list[Any], entity: str) -> dict[str, Any]:
    texts: list[str] = []
    for chunk in chunks:
        if str(chunk.entity_path) != entity:
            continue
        batch = chunk.to_record_batch()
        if "TextDocument:text" in batch.schema.names:
            texts.extend(
                row[0] for row in batch.column("TextDocument:text").to_pylist() if row
            )
    if len(texts) != 1:
        raise NcoreQualificationAuditError("RRD provenance document is not unique")
    match = re.search(r"```json\s*\n(.*?)\n```", texts[0], re.DOTALL)
    if match is None:
        raise NcoreQualificationAuditError("RRD provenance document is invalid")
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise NcoreQualificationAuditError("RRD provenance JSON is invalid") from exc
    if not isinstance(value, dict):
        raise NcoreQualificationAuditError("RRD provenance JSON differs")
    return value


def _rrd(root: Path, recording_id: str) -> dict[str, Any]:
    from PIL import Image
    from rerun.recording import load_recording
    from npa.workflows.data_factory_viz import _frame_index, _grouped_images, _subsample

    path = root / "reports/sim2real.rrd"
    verified = subprocess.run(
        [str(Path(os.sys.executable).with_name("rerun")), "rrd", "verify", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verified.returncode != 0:
        raise NcoreQualificationAuditError("Rerun rejected the recording")
    recording = load_recording(path)
    if (
        recording.application_id() != "neural-reconstruction"
        or recording.recording_id() != recording_id
    ):
        raise NcoreQualificationAuditError("Rerun recording identity differs")
    chunks = list(recording.chunks())
    for entity, relative, required in (
        ("source", "source/attribution.json", {"revision", "sha256", "license"}),
        (
            "conversion",
            "ncore/sequence/conversion.json",
            {"engine", "counts", "source", "converter"},
        ),
        (
            "rig",
            "ncore/sequence/npa-rig.json",
            {"reference_camera", "pose_count", "poses_component_group"},
        ),
    ):
        source_path = root / relative
        source = _json(source_path)
        decoded = _rrd_document(chunks, f"/provenance/{entity}")
        if not required <= decoded.keys() or decoded.get("artifact_sha256") != _sha(
            source_path
        ):
            raise NcoreQualificationAuditError("RRD lineage binding differs")
        for key, value in decoded.items():
            if key == "artifact_sha256":
                continue
            if isinstance(value, dict):
                if not isinstance(source.get(key), dict) or any(
                    source[key].get(name) != item for name, item in value.items()
                ):
                    raise NcoreQualificationAuditError("RRD lineage values differ")
            elif source.get(key) != value:
                raise NcoreQualificationAuditError("RRD lineage values differ")
    settings = _rrd_document(chunks, "/provenance/rrd_review")
    if settings.get("schema") != "npa.nurec.rrd-review.v1" or any(
        type(settings.get(name)) is not int
        for name in ("max_frames_per_entity", "max_frame_dim", "jpeg_quality")
    ):
        raise NcoreQualificationAuditError("RRD review settings differ")
    grouped = _grouped_images(root / "novel_views")
    selected = {
        (camera, _frame_index(source.stem))
        for camera, paths in grouped.items()
        for source in _subsample(paths, settings["max_frames_per_entity"])
    }
    expected: dict[tuple[str, int], bytes] = {}
    for camera, paths in grouped.items():
        for source in paths:
            identity = (camera, _frame_index(source.stem))
            if identity in expected:
                raise NcoreQualificationAuditError(
                    "duplicate rendered camera/frame identity"
                )
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                maximum = settings["max_frame_dim"]
                if maximum > 0 and max(rgb.size) > maximum:
                    rgb.thumbnail((maximum, maximum))
                stream = io.BytesIO()
                rgb.save(stream, format="JPEG", quality=settings["jpeg_quality"])
                expected[identity] = stream.getvalue()
    observed: set[tuple[str, int]] = set()
    decoded_rows = 0
    decoded_frames = 0
    for chunk in chunks:
        batch = chunk.to_record_batch()
        decoded_rows += batch.num_rows
        entity = str(chunk.entity_path)
        if not entity.startswith("/novel_view/"):
            continue
        if (
            "EncodedImage:blob" not in batch.schema.names
            or "frame" not in batch.schema.names
        ):
            raise NcoreQualificationAuditError("RRD frame columns differ")
        camera = entity.removeprefix("/novel_view/")
        for row, index in zip(
            batch.column("EncodedImage:blob").to_pylist(),
            batch.column("frame").to_pylist(),
            strict=True,
        ):
            identity = (camera, index)
            if (
                identity in observed
                or not row
                or len(row) != 1
                or bytes(row[0]) != expected.get(identity)
            ):
                raise NcoreQualificationAuditError("RRD frame bytes differ")
            with Image.open(io.BytesIO(bytes(row[0]))) as image:
                image.load()
                if min(image.size) <= 0:
                    raise NcoreQualificationAuditError("RRD frame is empty")
            observed.add(identity)
            decoded_frames += 1
    if observed != selected:
        raise NcoreQualificationAuditError("RRD frame selection differs")
    metrics = yaml.safe_load((root / "reconstruction/metrics.yaml").read_text())
    if _rrd_document(chunks, "/gaussians/summary") != metrics:
        raise NcoreQualificationAuditError("RRD metrics differ")
    return {
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        "decoded_rows": decoded_rows,
        "decoded_frames": decoded_frames,
        "lineage_verified": True,
        "frame_artifacts_verified": True,
    }


def audit_qualification(
    root: Path,
    *,
    recording_id: str,
    expected_image: str,
    expected_source_sha256: str,
    readback_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate downloaded workload bytes and emit one immutable objective receipt."""
    root = root.absolute()
    readback = _complete_readback(root, readback_receipt_path)
    conversion_path = root / "ncore/sequence/conversion.json"
    audit_path = root / "evidence/ncore-conversion-audit.json"
    runtime_path = root / "evidence/nre-runtime.json"
    workflow_status_path = root / "evidence/workflow-status.json"
    reconstruction_path = root / "reconstruction/reconstruction.json"
    render_path = root / "novel_views/nre-render.json"
    conversion = _json(conversion_path)
    audit = _json(audit_path)
    runtime = _json(runtime_path)
    reconstruction = _json(reconstruction_path)
    render = _json(render_path)
    if (
        conversion.get("status") != "ok"
        or conversion.get("source", {}).get("archive_sha256") != expected_source_sha256
        or audit.get("format") != AUDIT_FORMAT
        or audit.get("status") != "pass"
        or audit.get("source", {}).get("archive_sha256") != expected_source_sha256
        or audit.get("conversion", {}).get("report_sha256") != _sha(conversion_path)
    ):
        raise NcoreQualificationAuditError("conversion or independent audit differs")
    validate_runtime_attestation(
        runtime,
        expected_image=expected_image,
        required_stages=("reconstruct", "render"),
    )
    workflow_status = _final_workflow_status(
        workflow_status_path,
        recording_id=recording_id,
        runtime=runtime,
    )
    if (
        reconstruction.get("format") != RECONSTRUCTION_RECEIPT_FORMAT
        or reconstruction.get("status") != "pass"
        or reconstruction.get("nre_image") != expected_image
        or reconstruction.get("input", {}).get("conversion_report_sha256")
        != _sha(conversion_path)
        or render.get("format") != RENDER_RECEIPT_FORMAT
        or render.get("status") != "pass"
        or render.get("nre_image") != expected_image
    ):
        raise NcoreQualificationAuditError("native NRE receipts differ")
    sequence_path = root / "ncore/sequence/sequence.json"
    try:
        sequence, sequence_sha256 = _sequence_inventory(sequence_path)
    except NurecEvidenceError as exc:
        raise NcoreQualificationAuditError(
            "final NCore sequence did not reopen"
        ) from exc
    reconstruction_input = reconstruction.get("input")
    report_members = conversion.get("members")
    if (
        not isinstance(reconstruction_input, dict)
        or reconstruction_input.get("sequence_members") != sequence
        or reconstruction_input.get("sequence_inventory_sha256") != sequence_sha256
        or not isinstance(report_members, list)
        or audit.get("conversion", {}).get("inventory_sha256")
        != _canonical_sha(report_members)
    ):
        raise NcoreQualificationAuditError(
            "final NCore sequence differs from the audited conversion"
        )
    sequence_records = {item["path"]: item for item in sequence}
    if any(
        not isinstance(member, dict)
        or sequence_records.get(str(member.get("path") or "")) != member
        for member in report_members
    ):
        raise NcoreQualificationAuditError(
            "converted member bytes differ from the producer inventory"
        )
    audit_readback = audit.get("s3_readback")
    audit_objects = (
        audit_readback.get("objects") if isinstance(audit_readback, dict) else None
    )
    expected_conversion_objects = {
        "conversion.json",
        ".npa-colmap-claim.json",
        *(str(member.get("path") or "") for member in report_members),
    }
    if (
        not isinstance(audit_objects, list)
        or not all(isinstance(item, dict) for item in audit_objects)
        or audit_readback.get("stable_listing") is not True
        or audit_readback.get("object_count") != len(audit_objects)
        or audit_readback.get("objects_sha256") != _canonical_sha(audit_objects)
        or {
            str(item.get("path") or "")
            for item in audit_objects
            if isinstance(item, dict)
        }
        != expected_conversion_objects
    ):
        raise NcoreQualificationAuditError(
            "conversion audit does not bind the complete object inventory"
        )
    for item in audit_objects:
        relative = str(item.get("path") or "")
        candidate = root / "ncore/sequence" / relative
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or item.get("bytes") != candidate.stat().st_size
            or not isinstance(item.get("etag"), str)
            or not item["etag"]
        ):
            raise NcoreQualificationAuditError(
                "conversion readback inventory differs from final bytes"
            )
    parsed_path = root / "reconstruction/parsed.yaml"
    metrics_path = root / "reconstruction/metrics.yaml"
    usdz_path = root / "reconstruction/last.usdz"
    usdz = _usdz(usdz_path)
    recipe = reconstruction.get("recipe")
    try:
        parsed = _yaml_mapping(parsed_path)
    except NurecEvidenceError as exc:
        raise NcoreQualificationAuditError("native NRE recipe is invalid") from exc
    resolved_epochs = _nested_value(parsed, ("trainer", "max_epochs"))
    resolved_samples = _nested_value(
        parsed,
        ("dataset", "samples_per_epoch"),
        ("data", "samples_per_epoch"),
    )
    if (
        reconstruction.get("outputs", {}).get("usdz", {}).get("sha256")
        != usdz["sha256"]
        or not _record_matches(
            parsed_path, reconstruction.get("outputs", {}).get("parsed_config")
        )
        or not _record_matches(
            metrics_path, reconstruction.get("outputs", {}).get("metrics")
        )
        or render.get("input_usdz", {}).get("sha256") != usdz["sha256"]
        or not isinstance(recipe, dict)
        or recipe.get("name") != NATIVE_RECIPE
        or recipe.get("mode") != "trainval"
        or recipe.get("max_epochs_argument") != 0
        or recipe.get("resolved_epochs") != 1
        or recipe.get("resolved_samples_per_epoch") != 30000
        or resolved_epochs != 1
        or resolved_samples != 30000
    ):
        raise NcoreQualificationAuditError(
            "native recipe or trained output identity differs"
        )
    metrics = parse_metrics_yaml(metrics_path)
    selected_metrics = {
        name: metrics.get(name) for name in ("test/psnr", "test/ssim", "test/lpips")
    }
    if (
        not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in selected_metrics.values()
        )
        or float(selected_metrics["test/psnr"]) < MIN_PSNR
        or float(selected_metrics["test/ssim"]) < MIN_SSIM
        or float(selected_metrics["test/ssim"]) > 1
        or float(selected_metrics["test/lpips"]) < 0
        or float(selected_metrics["test/lpips"]) > MAX_LPIPS
        or reconstruction.get("observed_metrics") != selected_metrics
    ):
        raise NcoreQualificationAuditError("NRE objective metrics differ")
    image_paths = sorted(
        path
        for path in (root / "novel_views").rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    video_paths = sorted(
        path for path in (root / "novel_views").rglob("*.mp4") if path.is_file()
    )
    images = [_decode_image(path) for path in image_paths]
    videos = [_decode_video(path) for path in video_paths]
    for record, path in zip(images, image_paths, strict=True):
        record["path"] = path.relative_to(root / "novel_views").as_posix()
    for record, path in zip(videos, video_paths, strict=True):
        record["path"] = path.relative_to(root / "novel_views").as_posix()
    inventory = sorted(
        [*images, *videos], key=lambda item: (item["path"], item["sha256"])
    )
    output = render.get("output")
    invocation = render.get("invocation")
    if (
        not isinstance(invocation, dict)
        or invocation.get("render_exit_code") != 0
        or invocation.get("novel_view") is not True
        or not (
            _nonzero_offset(invocation.get("rig_translation_offset"))
            or _nonzero_offset(invocation.get("rig_rotation_offset"))
        )
        or not isinstance(output, dict)
        or len(images) != output.get("frame_count")
        or len(videos) != output.get("video_count")
        or not videos
        or sum(item["decoded_frames"] for item in videos)
        != output.get("decoded_video_frames")
        or output.get("all_videos_decoded") is not True
        or output.get("finite_pixels") is not True
        or output.get("nonuniform_frames") is not True
        or output.get("inventory") != inventory
        or output.get("inventory_sha256") != _canonical_sha(inventory)
    ):
        raise NcoreQualificationAuditError("rendered media receipt differs")
    final_path = root / "reports/final.json"
    final = _json(final_path)
    if final != {"has_usdz": True, "has_novel_views": True, "has_rrd": True}:
        raise NcoreQualificationAuditError("terminal qualification report differs")
    rrd = _rrd(root, recording_id)
    receipt = {
        "format": AUDIT_FORMAT_VERSION,
        "status": "pass",
        "source_archive_sha256": expected_source_sha256,
        "complete_readback": readback,
        "conversion_report_sha256": _sha(conversion_path),
        "conversion_audit_sha256": _sha(audit_path),
        "runtime_attestation_sha256": _sha(runtime_path),
        "workflow_status": workflow_status,
        "reconstruction_receipt_sha256": _sha(reconstruction_path),
        "render_receipt_sha256": _sha(render_path),
        "nre_image": expected_image,
        "observed_nre_digest": expected_image.split("@", 1)[-1],
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "gpu_count": 1,
        "usdz": usdz,
        "recipe": {
            "name": NATIVE_RECIPE,
            "mode": "trainval",
            "max_epochs_argument": 0,
            "resolved_epochs": 1,
            "resolved_samples_per_epoch": 30000,
        },
        "sequence_inventory_sha256": sequence_sha256,
        "objective_thresholds": {
            "minimum_psnr": MIN_PSNR,
            "minimum_ssim": MIN_SSIM,
            "maximum_lpips": MAX_LPIPS,
        },
        "metrics": selected_metrics,
        "render": {
            "frame_count": len(images),
            "video_count": len(videos),
            "decoded_video_frames": sum(item["decoded_frames"] for item in videos),
            "inventory_sha256": output.get("inventory_sha256"),
            "finite_pixels": True,
            "novel_view": True,
            "rig_translation_offset": invocation["rig_translation_offset"],
            "rig_rotation_offset": invocation["rig_rotation_offset"],
        },
        "final_report_sha256": _sha(final_path),
        "rrd": rrd,
    }
    _write_private(output_path, receipt)
    return receipt
