"""Independent post-S3 audit for a complete COLMAP-to-NCore V4 conversion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.errors import NpaError
from npa.workbench.ncore_staging import (
    DEFAULT_COLMAP_CACHE_DIR,
    DEFAULT_COLMAP_SCRATCH_DIR,
    private_staging_directory,
)
from npa.workbench.nurec.colmap import (
    CONVERSION_REPORT,
    NCORE_REVISION,
    PUBLICATION_CLAIM,
    ColmapConversionRequest,
    NcoreConversionError,
    _hash_file,
    _inventory,
    _regular_files,
    extract_colmap_zip,
    find_colmap_root,
    inspect_colmap_source,
    sequence_members,
    validate_ncore_sequence,
    verify_conversion_inventory,
)

AUDIT_FORMAT = "npa_ncore_colmap_conversion_audit_v1"


class NcoreAuditError(NpaError):
    """A post-publication conversion audit failed closed."""


class ColmapAuditRequest(BaseModel):
    """Immutable S3 source, converted prefix, and separate audit destination."""

    model_config = ConfigDict(extra="forbid")

    input_path: str
    conversion_path: str
    output_path: str
    expected_archive_sha256: str
    cache_dir: Path = DEFAULT_COLMAP_CACHE_DIR
    scratch_dir: Path = DEFAULT_COLMAP_SCRATCH_DIR
    dataset_root: str = "."
    colmap_dir: str = "sparse/0"
    images_dir: str = "images"
    masks_dir: str = ""
    rig_mode: Literal["derive", "preserve"] = "derive"
    reference_camera: str = ""
    include_downsampled_images: bool = True

    @field_validator("input_path", "conversion_path")
    @classmethod
    def read_handoff(cls, value: str) -> str:
        return validate_read_path(value, tool="nurec audit-colmap", allow_hf=False)

    @field_validator("output_path")
    @classmethod
    def write_handoff(cls, value: str) -> str:
        return validate_write_path(value, tool="nurec audit-colmap", required=True)

    @field_validator("expected_archive_sha256")
    @classmethod
    def archive_digest(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("expected_archive_sha256 must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def distinct_destinations(self) -> "ColmapAuditRequest":
        conversion = self.conversion_path.rstrip("/") + "/"
        output = self.output_path.rstrip("/") + "/"
        if output.startswith(conversion) or conversion.startswith(output):
            raise ValueError("audit output and conversion prefix must be separate")
        if not urlparse(self.input_path).path.lower().endswith(".zip"):
            raise ValueError("independent audit requires the original source ZIP")
        return self


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _object_snapshot(client: Any, uri: str) -> list[dict[str, Any]]:
    parsed = urlparse(uri.rstrip("/") + "/")
    prefix = parsed.path.lstrip("/")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in client.s3.get_paginator("list_objects_v2").paginate(
        Bucket=parsed.netloc, Prefix=prefix
    ):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if not isinstance(key, str) or not key.startswith(prefix):
                raise NcoreAuditError("conversion object listing is invalid")
            relative = key[len(prefix) :]
            if (
                not relative
                or "/" in relative
                or "\\" in relative
                or relative in seen
                or type(item.get("Size")) is not int
                or item["Size"] < 0
            ):
                raise NcoreAuditError("conversion object listing is unsafe")
            etag = item.get("ETag")
            if not isinstance(etag, str) or not etag.strip('"'):
                raise NcoreAuditError("conversion object identity is unavailable")
            seen.add(relative)
            rows.append(
                {
                    "path": relative,
                    "bytes": item["Size"],
                    "etag": etag.strip('"'),
                }
            )
    if not rows:
        raise NcoreAuditError("conversion prefix is empty")
    return sorted(rows, key=lambda row: row["path"])


def _download_snapshot(
    client: Any, conversion_uri: str, snapshot: list[dict[str, Any]], root: Path
) -> None:
    base = conversion_uri.rstrip("/")
    for item in snapshot:
        destination = root / item["path"]
        client.download_file(f"{base}/{item['path']}", str(destination))
        if destination.stat().st_size != item["bytes"]:
            raise NcoreAuditError("downloaded conversion object size differs")


def _frame_inventory(source: dict[str, Any]) -> tuple[dict[str, int], str]:
    rows = []
    counts = {}
    for camera_id in sorted(source["cameras"]):
        camera = source["cameras"][camera_id]
        frames = camera["frames"]
        counts[camera_id] = len(frames)
        calibration = {
            name: value for name, value in camera.items() if name != "frames"
        }
        for index, frame in enumerate(frames):
            rows.append(
                {
                    "camera": camera_id,
                    "index": index,
                    "name": frame["name"],
                    "encoded_sha256": frame["encoded_sha256"],
                    "pose_sha256": _canonical_sha(frame["pose"]),
                    "calibration_sha256": _canonical_sha(calibration),
                }
            )
    return counts, _canonical_sha(rows)


def _require_report_matches_source(
    report: dict[str, Any],
    request: ColmapAuditRequest,
    source: dict[str, Any],
    source_members: list[dict[str, Any]],
) -> None:
    expected_options = {
        "dataset_root": request.dataset_root,
        "colmap_dir": request.colmap_dir,
        "images_dir": request.images_dir,
        "masks_dir": request.masks_dir,
        "rig_mode": request.rig_mode,
        "reference_camera": request.reference_camera,
        "include_downsampled_images": request.include_downsampled_images,
    }
    source_record = report.get("source")
    if (
        not isinstance(source_record, dict)
        or report.get("status") != "ok"
        or report.get("engine") != "nvidia-ncore-colmap"
        or report.get("converter", {}).get("revision") != NCORE_REVISION
        or source_record.get("archive_sha256") != request.expected_archive_sha256
        or source_record.get("input_uri_sha256")
        != hashlib.sha256(request.input_path.encode()).hexdigest()
        or source_record.get("members") != source_members
        or source_record.get("counts") != source["counts"]
        or source_record.get("origin_points_filtered")
        != source["origin_points_filtered"]
        or report.get("options") != expected_options
        or report.get("poses_component_group")
        != ("npa_rig" if request.rig_mode == "derive" else "default")
    ):
        raise NcoreAuditError(
            "conversion report differs from independently read source"
        )


def audit_colmap_conversion(
    request: ColmapAuditRequest, *, storage_client: Any = None
) -> dict[str, Any]:
    """Download, enumerate, and independently reopen a published V4 generation."""
    from npa.clients.storage import StorageClient, StoragePreconditionFailed

    phase = "initialization"
    try:
        client = storage_client or StorageClient.from_environment()
        with (
            private_staging_directory(
                request.cache_dir, prefix="ncore-audit-source-"
            ) as source_cache,
            private_staging_directory(
                request.scratch_dir, prefix="ncore-audit-output-"
            ) as audit_scratch,
        ):
            phase = "source readback"
            source_archive = Path(source_cache) / "source.zip"
            client.download_file(request.input_path, str(source_archive))
            if _hash_file(source_archive) != request.expected_archive_sha256:
                raise NcoreAuditError("source archive SHA-256 differs")
            source_tree = Path(source_cache) / "dataset"
            extract_colmap_zip(source_archive, source_tree)
            conversion_options = ColmapConversionRequest(
                input_path=request.input_path,
                output_path=request.conversion_path,
                cache_dir=request.cache_dir,
                scratch_dir=request.scratch_dir,
                dataset_root=request.dataset_root,
                colmap_dir=request.colmap_dir,
                images_dir=request.images_dir,
                masks_dir=request.masks_dir,
                rig_mode=request.rig_mode,
                reference_camera=request.reference_camera,
                include_downsampled_images=request.include_downsampled_images,
            )
            source_root = find_colmap_root(
                source_tree,
                request.dataset_root,
                request.colmap_dir,
                request.images_dir,
            )
            source_members = _inventory(source_root, _regular_files(source_root))
            source = inspect_colmap_source(source_root, conversion_options)

            phase = "conversion readback"
            before = _object_snapshot(client, request.conversion_path)
            conversion_root = Path(audit_scratch) / "conversion"
            conversion_root.mkdir()
            _download_snapshot(client, request.conversion_path, before, conversion_root)
            after = _object_snapshot(client, request.conversion_path)
            if before != after:
                raise NcoreAuditError("conversion object set changed during audit")
            verify_conversion_inventory(conversion_root)

            phase = "independent V4 decode"
            report_path = conversion_root / CONVERSION_REPORT
            report = json.loads(report_path.read_text())
            if not isinstance(report, dict):
                raise NcoreAuditError("conversion report is not an object")
            _require_report_matches_source(report, request, source, source_members)
            meta = conversion_root / report["ncore_meta"]
            counts = validate_ncore_sequence(
                meta,
                source,
                rig_mode=request.rig_mode,
                reference_camera=request.reference_camera,
            )
            if counts != report.get("counts"):
                raise NcoreAuditError("producer and independent V4 counts differ")
            expected_objects = {
                PUBLICATION_CLAIM,
                CONVERSION_REPORT,
                meta.name,
                *(member.name for member in sequence_members(meta)),
            }
            if request.rig_mode == "derive":
                expected_objects.add("npa-rig.json")
            if {item["path"] for item in after} != expected_objects:
                raise NcoreAuditError("conversion prefix contains unexpected objects")
            camera_counts, frame_inventory_sha = _frame_inventory(source)
            raw_points = counts["points"] + source["origin_points_filtered"]

            audit = {
                "format": AUDIT_FORMAT,
                "status": "pass",
                "engine": "independent-ncore-v4-readback",
                "converter_revision": NCORE_REVISION,
                "source": {
                    "archive_sha256": request.expected_archive_sha256,
                    "inventory_sha256": _canonical_sha(source_members),
                    "counts": {
                        "images": source["counts"]["images"],
                        "cameras": source["counts"]["cameras"],
                        "poses": source["counts"]["poses"],
                        "points": raw_points,
                    },
                    "camera_frame_counts": camera_counts,
                    "camera_frame_inventory_sha256": frame_inventory_sha,
                },
                "conversion": {
                    "report_sha256": _hash_file(report_path),
                    "inventory_sha256": _canonical_sha(report["members"]),
                    "counts": counts,
                    "origin_points_filtered": source["origin_points_filtered"],
                    "poses_component_group": report["poses_component_group"],
                    "all_members_reopened": True,
                    "member_hashes_verified": True,
                    "calibration_verified": True,
                    "poses_verified": True,
                    "finite_geometry": True,
                },
                "s3_readback": {
                    "stable_listing": True,
                    "object_count": len(after),
                    "objects_sha256": _canonical_sha(after),
                    "objects": after,
                },
            }
            raw = (
                json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode()
            phase = "audit publication"
            try:
                client.put_bytes_conditional(
                    raw,
                    request.output_path,
                    if_none_match=True,
                    content_type="application/json",
                )
            except StoragePreconditionFailed as exc:
                raise NcoreAuditError(
                    "audit output exists; use a fresh destination"
                ) from exc
            return {
                "status": "ok",
                "format": AUDIT_FORMAT,
                "audit_uri": request.output_path,
                "audit_sha256": hashlib.sha256(raw).hexdigest(),
                "source_counts": audit["source"]["counts"],
                "converted_counts": counts,
                "objects": len(after),
            }
    except NcoreAuditError:
        raise
    except (NcoreConversionError, OSError, ValueError, TypeError, KeyError) as exc:
        raise NcoreAuditError(f"NCore conversion audit failed during {phase}") from exc
