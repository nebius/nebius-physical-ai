"""Require complete capture and decoded visualization evidence for NCore release."""

from __future__ import annotations

import re
from typing import Any


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise RuntimeError(f"NCore acceptance requires valid {field}")


def _record(parent: dict[str, Any], name: str) -> dict[str, Any]:
    value = parent.get(name)
    _require(isinstance(value, dict), name)
    return value


def _hash(parent: dict[str, Any], name: str) -> str:
    value = parent.get(name)
    _require(
        isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value)), name
    )
    return value


def _count(parent: dict[str, Any], name: str, minimum: int = 1) -> int:
    value = parent.get(name)
    _require(type(value) is int and value >= minimum, name)
    return value


def _camera_counts(parent: dict[str, Any], name: str, minimum: int = 1) -> dict:
    value = _record(parent, name)
    _require(bool(value), name)
    for camera in value:
        _require(isinstance(camera, str) and bool(camera.strip()), name)
        _count(value, camera, minimum)
    return value


def _frame_coverage(conversion: dict, training: dict) -> None:
    source = _camera_counts(conversion, "camera_frame_counts")
    _require(len(source) == conversion["source_counts"]["cameras"], "source cameras")
    _require(
        sum(source.values()) == conversion["source_counts"]["images"], "source frames"
    )
    loaded = _camera_counts(training, "loaded_camera_frame_counts")
    train = _camera_counts(training, "eligible_training_camera_frame_counts")
    val = _camera_counts(training, "validation_camera_frame_counts", 0)
    covered = _camera_counts(training, "covered_camera_frame_counts")
    _require(loaded == covered == source, "complete loaded and covered source frames")
    _require(train.keys() == val.keys() == source.keys(), "all source training cameras")
    for camera, count in source.items():
        _require(
            max(train[camera], val[camera]) <= count <= train[camera] + val[camera],
            "training/validation frame accounting",
        )
    source_hash = _hash(conversion, "camera_frame_inventory_sha256")
    for field in ("source_frame_inventory_sha256", "covered_frame_inventory_sha256"):
        _require(_hash(training, field) == source_hash, field)
    _require(training.get("independent_frame_readback") is True, "frame readback")


def _recipe_evidence(training: dict) -> None:
    # These are acceptance comparisons against the pinned 26.04 recipe. They
    # never set or truncate a caller's runtime training configuration.
    for field in (
        "inventory_report_sha256",
        "parsed_config_sha256",
        "native_recipe_sha256",
        "datasource_summary_sha256",
        "split_report_sha256",
    ):
        _hash(training, field)
    _require(
        training.get("native_recipe") == "configs/experimental/3dgut/3dgut_colmap.yaml",
        "native COLMAP recipe",
    )
    for field, native in (("epochs", 1), ("samples_per_epoch", 30000)):
        _require(_count(training, f"native_{field}") == native, f"native {field}")
        _require(_count(training, f"resolved_{field}") == native, f"resolved {field}")
    _require(training.get("native_recipe_completed") is True, "completed native recipe")


def _visualization_evidence(conversion: dict, proof: dict) -> None:
    recording = _record(proof, "rrd")
    for field in ("report_sha256", "sha256"):
        _hash(recording, field)
    for field in ("bytes", "decoded_rows", "decoded_frames"):
        _count(recording, field)
    for field in ("conversion_report_sha256", "source_inventory_sha256"):
        source_field = "report_sha256" if field == "conversion_report_sha256" else field
        _require(_hash(recording, field) == conversion[source_field], field)
    for field in ("lineage_verified", "frame_artifacts_verified"):
        _require(recording.get(field) is True, field)


def validate_full_input_proof(conversion: dict, proof: dict) -> None:
    """Check source-bound frame, recipe and RRD accounting in retained evidence.

    Frame counts describe eligible training inputs and validation splits, not a
    fabricated history of the native random sampler. Hashes identify independently
    inspected artifacts; this offline check does not itself execute a workload.

    Args:
        conversion: Independently validated full-capture conversion evidence.
        proof: RTX reconstruction/render evidence for that converted capture.
    Returns:
        None when all required identities and counts are consistent.
    Raises:
        RuntimeError: Required evidence is absent, partial or inconsistent.
    """
    training = _record(proof, "training_input")
    _frame_coverage(conversion, training)
    _recipe_evidence(training)
    _visualization_evidence(conversion, proof)
