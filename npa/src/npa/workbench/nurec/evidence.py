"""Private, byte-bound receipts for real NRE reconstruction and render runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from npa.errors import NpaError


RECONSTRUCTION_RECEIPT_FORMAT = "npa_nurec_reconstruction_receipt_v1"
RENDER_RECEIPT_FORMAT = "npa_nurec_render_receipt_v1"


class NurecEvidenceError(NpaError):
    """A run completed but its objective receipt could not be produced."""


def validate_runtime_attestation(
    payload: Any,
    *,
    expected_image: str,
    required_stages: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate a sanitized control-plane image-ID and GPU observation."""
    expected_digest = _requested_digest(expected_image)
    if not expected_digest:
        raise NurecEvidenceError("expected NRE image must use an exact digest")
    if not isinstance(payload, dict):
        raise NurecEvidenceError("runtime attestation has an invalid schema")
    legacy_fields = {
        "format",
        "status",
        "source",
        "requested_image",
        "observed_image_digest",
        "gpu_names",
        "gpu_count",
        "resource_identity_sha256",
    }
    if payload.get("format") == "npa_nurec_runtime_attestation_v1":
        if required_stages or set(payload) != legacy_fields:
            raise NurecEvidenceError("runtime attestation has an invalid schema")
        names = payload.get("gpu_names")
        identity = payload.get("resource_identity_sha256")
        if (
            payload.get("status") != "pass"
            or payload.get("source") != "kubernetes_pod_status"
            or payload.get("requested_image") != expected_image
            or payload.get("observed_image_digest") != expected_digest
            or names != ["NVIDIA RTX PRO 6000 Blackwell Server Edition"]
            or payload.get("gpu_count") != 1
            or not isinstance(identity, str)
            or re.fullmatch(r"[0-9a-f]{64}", identity) is None
        ):
            raise NurecEvidenceError("runtime image or GPU attestation differs")
        return payload
    from npa.workbench.nurec.runtime_attestation import (
        BUNDLE_FORMAT,
        GPU_NAME,
        STAGE_FORMAT,
    )

    if set(payload) != {
        "format",
        "status",
        "source",
        "requested_image",
        "observed_image_digest",
        "gpu_names",
        "gpu_count",
        "stages",
    }:
        raise NurecEvidenceError("runtime attestation has an invalid schema")
    stages = payload.get("stages")
    required = tuple(required_stages) or ("reconstruct", "render")
    if (
        payload.get("format") != BUNDLE_FORMAT
        or payload.get("status") != "pass"
        or payload.get("source") != "kubernetes_control_plane"
        or payload.get("requested_image") != expected_image
        or payload.get("observed_image_digest") != expected_digest
        or payload.get("gpu_names") != [GPU_NAME]
        or payload.get("gpu_count") != 1
        or not isinstance(stages, dict)
        or set(stages) != set(required)
    ):
        raise NurecEvidenceError("runtime image or GPU attestation differs")
    stage_fields = {
        "format",
        "status",
        "source",
        "stage",
        "requested_image",
        "observed_image_digest",
        "gpu_names",
        "gpu_count",
        "resource_identity_sha256",
        "control_plane_record_sha256",
        "container_state",
        "receipt_sha256",
    }
    for stage_name in required:
        stage = stages.get(stage_name)
        if not isinstance(stage, dict) or set(stage) != stage_fields:
            raise NurecEvidenceError("runtime stage attestation has an invalid schema")
        if (
            stage.get("format") != STAGE_FORMAT
            or stage.get("status") != "pass"
            or stage.get("source") != "kubernetes_control_plane"
            or stage.get("stage") != stage_name
            or stage.get("requested_image") != expected_image
            or stage.get("observed_image_digest") != expected_digest
            or stage.get("gpu_names") != [GPU_NAME]
            or stage.get("gpu_count") != 1
            or stage.get("container_state") not in {"running", "terminated_zero"}
        ):
            raise NurecEvidenceError("runtime stage image or GPU differs")
        for field in (
            "resource_identity_sha256",
            "control_plane_record_sha256",
            "receipt_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(stage.get(field) or "")) is None:
                raise NurecEvidenceError("runtime stage identity hash differs")
    return payload


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise NurecEvidenceError(f"{label} is not a regular file")
    return path


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    source = _regular_file(path, path.name)
    name = source.name if root is None else source.relative_to(root).as_posix()
    return {
        "path": name,
        "bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def _sequence_inventory(ncore_json: Path) -> tuple[list[dict[str, Any]], str]:
    from npa.workbench.nurec.colmap import NcoreConversionError, sequence_members

    root = ncore_json.parent
    try:
        members = [ncore_json, *sequence_members(ncore_json)]
    except (NcoreConversionError, OSError, ValueError) as exc:
        raise NurecEvidenceError("NCore sequence inventory is invalid") from exc
    for name in ("conversion.json", "npa-rig.json", ".npa-colmap-claim.json"):
        candidate = root / name
        if candidate.exists() or candidate.is_symlink():
            members.append(candidate)
    records = sorted(
        (_file_record(path, root=root) for path in members),
        key=lambda item: item["path"],
    )
    names = [item["path"] for item in records]
    if len(names) != len(set(names)):
        raise NurecEvidenceError("NCore sequence inventory contains duplicate members")
    return records, _canonical_sha(records)


def _yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml

        payload = yaml.safe_load(_regular_file(path, "parsed config").read_text())
    except Exception as exc:  # noqa: BLE001 - receipt must fail closed
        raise NurecEvidenceError("parsed NRE config is unreadable") from exc
    if not isinstance(payload, dict):
        raise NurecEvidenceError("parsed NRE config is not an object")
    return payload


def _nested_value(payload: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    values = []
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            values.append(current)
    if not values:
        return None
    first = values[0]
    if any(value != first for value in values[1:]):
        raise NurecEvidenceError("parsed NRE config has conflicting recipe values")
    return first


def _positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _requested_digest(image: str) -> str:
    match = re.search(r"@(sha256:[0-9a-f]{64})$", image)
    return match.group(1) if match else ""


def _gpu_record(names: Sequence[str]) -> dict[str, Any]:
    cleaned = [str(name).strip() for name in names if str(name).strip()]
    return {
        "count": len(cleaned),
        "names": cleaned,
        "all_rt_core_models": bool(cleaned)
        and all("L40S" in name or "RTX PRO 6000" in name.upper() for name in cleaned),
    }


def write_reconstruction_receipt(
    *,
    receipt_path: Path,
    ncore_json: Path,
    nre_image: str,
    config_name: str,
    mode: str,
    max_epochs_argument: int,
    command: Sequence[str],
    train_exit_code: int,
    gpu_names: Sequence[str] = (),
    parsed_config_path: Path | None = None,
    metrics_path: Path | None = None,
    usdz_path: Path | None = None,
    metrics: Mapping[str, float] | None = None,
    error: str = "",
) -> dict[str, Any]:
    """Write a private receipt from actual local NCore and NRE output bytes."""
    sequence, sequence_sha = _sequence_inventory(
        _regular_file(ncore_json, "NCore meta-file")
    )
    conversion = next(
        (item for item in sequence if item["path"] == "conversion.json"), None
    )
    parsed_record = (
        _file_record(parsed_config_path)
        if parsed_config_path is not None and parsed_config_path.is_file()
        else None
    )
    metrics_record = (
        _file_record(metrics_path)
        if metrics_path is not None and metrics_path.is_file()
        else None
    )
    usdz_record = (
        _file_record(usdz_path)
        if usdz_path is not None and usdz_path.is_file()
        else None
    )
    recipe: dict[str, Any] = {
        "name": config_name,
        "mode": mode,
        "max_epochs_argument": max_epochs_argument,
        "resolved_epochs": None,
        "resolved_samples_per_epoch": None,
    }
    if parsed_config_path is not None and parsed_config_path.is_file():
        parsed = _yaml_mapping(parsed_config_path)
        recipe["resolved_epochs"] = _positive_int(
            _nested_value(parsed, ("trainer", "max_epochs"))
        )
        recipe["resolved_samples_per_epoch"] = _positive_int(
            _nested_value(
                parsed,
                ("dataset", "samples_per_epoch"),
                ("data", "samples_per_epoch"),
            )
        )
    observed_metrics = {
        str(name): float(value)
        for name, value in (metrics or {}).items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
    required_metrics = {
        name: observed_metrics.get(name)
        for name in ("test/psnr", "test/ssim", "test/lpips")
    }
    output_complete = all(
        record is not None and record["bytes"] > 0
        for record in (parsed_record, metrics_record, usdz_record)
    )
    metrics_complete = all(value is not None for value in required_metrics.values())
    requested_digest = _requested_digest(nre_image)
    gpu = _gpu_record(gpu_names)
    receipt = {
        "format": RECONSTRUCTION_RECEIPT_FORMAT,
        "status": (
            "pass"
            if train_exit_code == 0
            and output_complete
            and metrics_complete
            and bool(requested_digest)
            and gpu["count"] == 1
            and gpu["all_rt_core_models"]
            else "failed"
        ),
        "engine": "nvidia-nre-3dgut",
        "nre_image": nre_image,
        "requested_nre_digest": requested_digest,
        "gpu": gpu,
        "invocation": {
            "train_exit_code": train_exit_code,
            "command": list(command),
            "command_sha256": _canonical_sha(list(command)),
        },
        "input": {
            "ncore_meta": ncore_json.name,
            "sequence_inventory_sha256": sequence_sha,
            "sequence_members": sequence,
            "conversion_report_sha256": (
                conversion["sha256"] if conversion is not None else ""
            ),
        },
        "recipe": recipe,
        "outputs": {
            "parsed_config": parsed_record,
            "metrics": metrics_record,
            "usdz": usdz_record,
        },
        "observed_metrics": required_metrics,
        "error": str(error),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def _decode_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageStat

        with Image.open(_regular_file(path, "rendered frame")) as image:
            image.load()
            converted = image.convert("RGB")
            extrema = converted.getextrema()
            finite_pixels = all(
                math.isfinite(float(value))
                for channel in ImageStat.Stat(converted).extrema
                for value in channel
            )
            nonuniform = any(low != high for low, high in extrema)
            return {
                **_file_record(path),
                "width": converted.width,
                "height": converted.height,
                "finite_pixels": finite_pixels,
                "nonuniform": nonuniform,
            }
    except Exception as exc:  # noqa: BLE001 - receipt must fail closed
        raise NurecEvidenceError(
            "rendered frame is not independently decodable"
        ) from exc


def _decode_video(path: Path) -> dict[str, Any]:
    reader = None
    try:
        import imageio_ffmpeg

        source = _regular_file(path, "rendered video")
        reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="rgb24")
        metadata = next(reader)
        size = metadata.get("size")
        if (
            not isinstance(size, tuple)
            or len(size) != 2
            or type(size[0]) is not int
            or type(size[1]) is not int
            or size[0] <= 0
            or size[1] <= 0
        ):
            raise ValueError("video dimensions are invalid")
        expected_bytes = size[0] * size[1] * 3
        frames = 0
        for frame in reader:
            if len(frame) != expected_bytes:
                raise ValueError("decoded frame size differs")
            frames += 1
        if frames <= 0:
            raise ValueError("video has no decoded frames")
        reader.close()
        reader = None
        return {
            **_file_record(path),
            "width": size[0],
            "height": size[1],
            "decoded_frames": frames,
        }
    except Exception as exc:  # noqa: BLE001 - receipt must fail closed
        raise NurecEvidenceError(
            "rendered video is not independently decodable"
        ) from exc
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass


def write_render_receipt(
    *,
    receipt_path: Path,
    artifact_path: Path,
    output_dir: Path,
    nre_image: str,
    command: Sequence[str],
    render_exit_code: int,
    novel_view: bool,
    rig_translation_offset: str,
    rig_rotation_offset: str,
    gpu_names: Sequence[str] = (),
    error: str = "",
) -> dict[str, Any]:
    """Decode and hash every rendered frame after a real NRE render call."""
    from npa.workbench.nurec.nurec import IMAGE_SUFFIXES

    artifact = _file_record(_regular_file(artifact_path, "trained USDZ"))
    frames = [
        _decode_image(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    videos = [
        _decode_video(path)
        for path in sorted(output_dir.rglob("*.mp4"))
        if path.is_file()
    ]
    for record, path in zip(
        frames,
        (
            path
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        strict=True,
    ):
        record["path"] = path.relative_to(output_dir).as_posix()
    for record, path in zip(
        videos,
        (path for path in sorted(output_dir.rglob("*.mp4")) if path.is_file()),
        strict=True,
    ):
        record["path"] = path.relative_to(output_dir).as_posix()
    inventory = sorted(
        [*frames, *videos], key=lambda item: (item["path"], item["sha256"])
    )
    complete = bool(frames) and all(
        item["finite_pixels"] and item["nonuniform"] for item in frames
    )
    requested_digest = _requested_digest(nre_image)
    gpu = _gpu_record(gpu_names)
    receipt = {
        "format": RENDER_RECEIPT_FORMAT,
        "status": (
            "pass"
            if render_exit_code == 0
            and novel_view
            and complete
            and bool(requested_digest)
            and gpu["count"] == 1
            and gpu["all_rt_core_models"]
            else "failed"
        ),
        "engine": "nvidia-nre-render",
        "nre_image": nre_image,
        "requested_nre_digest": requested_digest,
        "gpu": gpu,
        "invocation": {
            "render_exit_code": render_exit_code,
            "command": list(command),
            "command_sha256": _canonical_sha(list(command)),
            "novel_view": novel_view,
            "rig_translation_offset": rig_translation_offset,
            "rig_rotation_offset": rig_rotation_offset,
        },
        "input_usdz": artifact,
        "output": {
            "frame_count": len(frames),
            "video_count": len(videos),
            "bytes": sum(item["bytes"] for item in inventory),
            "inventory_sha256": _canonical_sha(inventory),
            "inventory": inventory,
            "all_frames_decoded": bool(frames),
            "all_videos_decoded": all(item["decoded_frames"] > 0 for item in videos),
            "decoded_video_frames": sum(item["decoded_frames"] for item in videos),
            "finite_pixels": bool(frames)
            and all(item["finite_pixels"] for item in frames),
            "nonuniform_frames": bool(frames)
            and all(item["nonuniform"] for item in frames),
        },
        "error": str(error),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt
