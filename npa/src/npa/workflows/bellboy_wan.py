"""Bellboy-shaped episode validation and honest Wan evaluation boundaries.

The public Wan 2.2 baseline generates video; it does not predict robot actions.
These helpers therefore validate real-robot episode references and make the
held-out boundary machine-readable without manufacturing action metrics.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from npa.workbench.dataset.storage import read_json_uri, write_json_uri

EPISODE_MANIFEST_SCHEMA = "npa.bellboy.episode_manifest.v1"
EPISODE_VALIDATION_SCHEMA = "npa.bellboy.episode_validation.v1"
EVALUATION_BOUNDARY_SCHEMA = "npa.bellboy.wan_evaluation_boundary.v1"
DATASET_VALIDATION_SCHEMA = "npa.dataset.validation_report.v1"
WAN_ARTIFACT_SCHEMA = "npa.workbench.byof.wan2_2_ti2v_5b.v1"
WAN_VIDEO_CAPABILITIES = {
    "wan2.2_ti2v_5b_text_to_video",
    "wan2.2_ti2v_5b_image_to_video",
}
WAN_VALIDATION_CAPABILITY = "wan2.2_decoded_mp4_validation"
VALID_OUTCOMES = {"success", "failure", "partial", "aborted"}
VALID_SPLITS = {"train", "validation", "heldout"}


class BellboyManifestError(ValueError):
    """Raised when an episode manifest violates the public interchange contract."""


def _is_s3_object_uri(value: object) -> bool:
    parsed = urlparse(str(value or ""))
    return parsed.scheme == "s3" and bool(parsed.netloc and parsed.path.lstrip("/"))


def _require_s3_uri(value: object, field: str, errors: list[str]) -> None:
    if not _is_s3_object_uri(value):
        errors.append(f"{field} must be an s3:// object URI")


def _manifest_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_payload(
    payload: dict[str, Any], *, required_split: str = ""
) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("schema") != EPISODE_MANIFEST_SCHEMA:
        errors.append(f"schema must equal {EPISODE_MANIFEST_SCHEMA}")
    for field in ("dataset_id", "version"):
        if not str(payload.get(field) or "").strip():
            errors.append(f"{field} must be non-empty")

    camera = payload.get("camera")
    if not isinstance(camera, dict):
        errors.append("camera must be an object")
    else:
        if camera.get("modality") != "rgb":
            errors.append("camera.modality must equal rgb")
        if camera.get("mount") != "gripper":
            errors.append("camera.mount must equal gripper")
        if not str(camera.get("projection") or "").strip():
            errors.append("camera.projection must describe the wide-angle projection")

    action_schema = payload.get("action_schema")
    if not isinstance(action_schema, dict):
        errors.append("action_schema must be an object")
    else:
        _require_s3_uri(action_schema.get("uri"), "action_schema.uri", errors)
        if not str(action_schema.get("version") or "").strip():
            errors.append("action_schema.version must be non-empty")

    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("episodes must be a non-empty array")
        episodes = []

    ids: set[str] = set()
    task_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    recovery_count = 0
    for index, raw in enumerate(episodes):
        prefix = f"episodes[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{prefix} must be an object")
            continue
        episode_id = str(raw.get("episode_id") or "").strip()
        if not episode_id:
            errors.append(f"{prefix}.episode_id must be non-empty")
        elif episode_id in ids:
            errors.append(f"{prefix}.episode_id duplicates {episode_id!r}")
        ids.add(episode_id)

        split = str(raw.get("split") or "").strip()
        if split not in VALID_SPLITS:
            errors.append(f"{prefix}.split must be one of {sorted(VALID_SPLITS)}")
        if required_split and split != required_split:
            errors.append(f"{prefix}.split must equal {required_split}")
        split_counts[split] += 1

        task = str(raw.get("task") or "").strip()
        if not task:
            errors.append(f"{prefix}.task must be non-empty")
        task_counts[task] += 1
        outcome = str(raw.get("outcome") or "").strip()
        if outcome not in VALID_OUTCOMES:
            errors.append(f"{prefix}.outcome must be one of {sorted(VALID_OUTCOMES)}")
        outcome_counts[outcome] += 1

        observation = raw.get("observation")
        if not isinstance(observation, dict):
            errors.append(f"{prefix}.observation must be an object")
        else:
            _require_s3_uri(
                observation.get("gripper_rgb_uri"),
                f"{prefix}.observation.gripper_rgb_uri",
                errors,
            )
            _require_s3_uri(
                observation.get("timestamps_uri"),
                f"{prefix}.observation.timestamps_uri",
                errors,
            )

        actions = raw.get("actions")
        if not isinstance(actions, dict):
            errors.append(f"{prefix}.actions must be an object")
        else:
            _require_s3_uri(actions.get("uri"), f"{prefix}.actions.uri", errors)
            _require_s3_uri(
                actions.get("timestamps_uri"),
                f"{prefix}.actions.timestamps_uri",
                errors,
            )

        joint_state = raw.get("joint_state")
        if not isinstance(joint_state, dict):
            errors.append(f"{prefix}.joint_state must be an object")
        else:
            _require_s3_uri(joint_state.get("uri"), f"{prefix}.joint_state.uri", errors)
            _require_s3_uri(
                joint_state.get("timestamps_uri"),
                f"{prefix}.joint_state.timestamps_uri",
                errors,
            )

        timing = raw.get("timing")
        if not isinstance(timing, dict):
            errors.append(f"{prefix}.timing must be an object")
        else:
            if not str(timing.get("clock") or "").strip():
                errors.append(f"{prefix}.timing.clock must be non-empty")
            for field in ("start_ns", "end_ns"):
                if not isinstance(timing.get(field), int):
                    errors.append(f"{prefix}.timing.{field} must be an integer")
            if isinstance(timing.get("start_ns"), int) and isinstance(
                timing.get("end_ns"), int
            ):
                if timing["end_ns"] <= timing["start_ns"]:
                    errors.append(
                        f"{prefix}.timing.end_ns must be greater than start_ns"
                    )

        recovery = raw.get("recovery")
        if recovery is not None:
            if not isinstance(recovery, dict):
                errors.append(f"{prefix}.recovery must be an object when present")
            else:
                recovery_count += 1
                if (
                    not isinstance(recovery.get("attempt"), int)
                    or recovery["attempt"] < 1
                ):
                    errors.append(f"{prefix}.recovery.attempt must be an integer >= 1")
                elif recovery["attempt"] > 1 and not str(
                    recovery.get("parent_episode_id") or ""
                ).strip():
                    errors.append(
                        f"{prefix}.recovery.parent_episode_id is required after attempt 1"
                    )
                if not str(recovery.get("correction") or "").strip():
                    errors.append(f"{prefix}.recovery.correction must be non-empty")

    # The canonical dataset records let existing Dataset-of-Record validation and
    # indexing consume the same manifest. Bellboy-specific alignment remains in
    # episodes[] above so the generic schema does not pretend to understand actions.
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        errors.append(
            "records must be a non-empty npa.dataset.manifest.v1-compatible array"
        )
        records = []
    else:
        record_ids: set[str] = set()
        for index, record in enumerate(records):
            prefix = f"records[{index}]"
            if not isinstance(record, dict):
                errors.append(f"{prefix} must be an object")
                continue
            record_id = str(record.get("record_id") or "").strip()
            if not record_id or record_id in record_ids:
                errors.append(f"{prefix}.record_id must be non-empty and unique")
            record_ids.add(record_id)
            if record.get("modality") != "gripper_rgb":
                errors.append(f"{prefix}.modality must equal gripper_rgb")
            _require_s3_uri(record.get("uri"), f"{prefix}.uri", errors)

    quality_stats = payload.get("quality_stats")
    if not isinstance(quality_stats, dict):
        errors.append(
            "quality_stats must be an npa.dataset.manifest.v1-compatible object"
        )
    else:
        if quality_stats.get("record_count") != len(records):
            errors.append("quality_stats.record_count must equal len(records)")
        completeness = quality_stats.get("mean_completeness")
        if not isinstance(completeness, (int, float)) or not 0 <= completeness <= 1:
            errors.append("quality_stats.mean_completeness must be between 0 and 1")
        corrupt_count = quality_stats.get("corrupt_count")
        if not isinstance(corrupt_count, int) or not 0 <= corrupt_count <= len(records):
            errors.append(
                "quality_stats.corrupt_count must be between 0 and len(records)"
            )

    if errors:
        raise BellboyManifestError("; ".join(errors))

    return {
        "episode_count": len(episodes),
        "record_count": len(records),
        "task_counts": dict(sorted(task_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "recovery_episode_count": recovery_count,
        "alignment_level": "S3 reference + shared clock contract; array-level alignment is customer-validated at data access time",
    }


def validate_episode_manifest(
    manifest_uri: str,
    output_uri: str,
    dataset_validation_uri: str,
    required_split: str = "",
) -> dict[str, Any]:
    """Validate a versioned real-robot episode manifest and publish evidence."""

    payload = read_json_uri(manifest_uri)
    if not isinstance(payload, dict):
        raise BellboyManifestError("episode manifest must be a JSON object")
    if required_split and required_split not in VALID_SPLITS:
        raise BellboyManifestError(
            f"required_split must be one of {sorted(VALID_SPLITS)}"
        )
    dataset_validation = read_json_uri(dataset_validation_uri)
    if not isinstance(dataset_validation, dict):
        raise BellboyManifestError("dataset validation report must be a JSON object")
    if dataset_validation.get("schema") != DATASET_VALIDATION_SCHEMA:
        raise BellboyManifestError(
            f"dataset validation schema must equal {DATASET_VALIDATION_SCHEMA}"
        )
    if dataset_validation.get("source_manifest_uri") != manifest_uri:
        raise BellboyManifestError(
            "dataset validation report belongs to a different manifest"
        )
    if dataset_validation.get("passed") is not True:
        raise BellboyManifestError("dataset-of-record validation did not pass")
    stats = _validate_payload(payload, required_split=required_split)
    report = {
        "schema": EPISODE_VALIDATION_SCHEMA,
        "status": "validated",
        "source_manifest_uri": manifest_uri,
        "source_schema": payload["schema"],
        "dataset_id": payload["dataset_id"],
        "version": payload["version"],
        "manifest_sha256": _manifest_digest(payload),
        "dataset_validation_uri": dataset_validation_uri,
        "required_split": required_split,
        **stats,
    }
    write_json_uri(output_uri, report)
    print(json.dumps(report, sort_keys=True))
    return report


def evaluate_heldout_boundary(
    heldout_manifest_uri: str,
    episode_validation_uri: str,
    wan_artifact_uri: str,
    output_uri: str,
) -> dict[str, Any]:
    """Verify the public Wan artifact and bind it to held-out robot evidence.

    This is deliberately not a task-success or action-prediction score. It proves
    the generated artifact is real and that a disjoint held-out real-robot set is
    present, then records the private capabilities still required for comparison.
    """

    heldout = read_json_uri(heldout_manifest_uri)
    validation = read_json_uri(episode_validation_uri)
    wan_artifact = read_json_uri(wan_artifact_uri)
    if (
        not isinstance(heldout, dict)
        or not isinstance(validation, dict)
        or not isinstance(wan_artifact, dict)
    ):
        raise BellboyManifestError("evaluation inputs must be JSON objects")
    heldout_stats = _validate_payload(heldout, required_split="heldout")
    if (
        validation.get("schema") != EPISODE_VALIDATION_SCHEMA
        or validation.get("status") != "validated"
    ):
        raise BellboyManifestError(
            "episode_validation_uri is not a successful Bellboy validation report"
        )
    if validation.get("source_manifest_uri") == heldout_manifest_uri:
        raise BellboyManifestError(
            "training validation and heldout evaluation must use different manifests"
        )
    if wan_artifact.get("schema") != WAN_ARTIFACT_SCHEMA:
        raise BellboyManifestError(
            f"Wan artifact schema must equal {WAN_ARTIFACT_SCHEMA}"
        )
    exercised = set(wan_artifact.get("capabilities_exercised") or [])
    video_capabilities = exercised.intersection(WAN_VIDEO_CAPABILITIES)
    if len(video_capabilities) != 1 or WAN_VALIDATION_CAPABILITY not in exercised:
        raise BellboyManifestError(
            "Wan artifact does not prove one stock generation mode plus decoded MP4 validation"
        )
    video_capability = next(iter(video_capabilities))
    if wan_artifact.get("capability") != video_capability:
        raise BellboyManifestError(
            "Wan artifact capability disagrees with exercised capabilities"
        )
    if wan_artifact.get("deferred"):
        raise BellboyManifestError(
            "Wan baseline artifact contains unresolved hard-gate deferrals"
        )
    if int(wan_artifact.get("output_size_bytes") or 0) < 4096:
        raise BellboyManifestError(
            "Wan video evidence reports an implausibly small artifact"
        )
    if not str(wan_artifact.get("output_filename") or "").lower().endswith(".mp4"):
        raise BellboyManifestError("Wan video evidence does not name an MP4")
    observed = wan_artifact.get("observed")
    if not isinstance(observed, dict):
        raise BellboyManifestError("Wan artifact has no decoded observation")
    if any(
        float(observed.get(field) or 0) <= 0
        for field in ("width", "height", "frame_count", "fps")
    ):
        raise BellboyManifestError(
            "Wan decoded observation has invalid dimensions, frames, or fps"
        )
    if (
        float(observed.get("max_spatial_std") or 0) < 1.0
        or int(observed.get("pixel_range") or 0) < 4
        or float(observed.get("mean_temporal_abs_delta") or 0) <= 0.001
    ):
        raise BellboyManifestError("Wan decoded observation is blank or uniform")

    report = {
        "schema": EVALUATION_BOUNDARY_SCHEMA,
        "status": "boundary_verified",
        "baseline": {
            "kind": "official Wan 2.2 TI2V-5B video generation",
            "capability": video_capability,
            "artifact_uri": wan_artifact_uri,
            "output_filename": wan_artifact.get("output_filename"),
            "output_size_bytes": wan_artifact.get("output_size_bytes"),
            "decoded_observation": observed,
        },
        "real_robot_boundary": {
            "heldout_manifest_uri": heldout_manifest_uri,
            "heldout_manifest_sha256": _manifest_digest(heldout),
            **heldout_stats,
            "synthetic_or_generated_video_replaces_heldout_evaluation": False,
        },
        "checks_executed": [
            "versioned episode and S3-reference validation",
            "heldout split enforcement",
            "stock Wan capability evidence verification",
            "decoded non-uniform MP4 evidence verification",
        ],
        "release_gate": {
            "satisfied": False,
            "reason": "customer action model and held-out real-task metrics are not supplied",
        },
        "deferred": [
            {
                "capability": "bellboy_action_conditioned_training",
                "requires": [
                    "private repo/ref",
                    "training entrypoint",
                    "checkpoint URI",
                    "exact action schema",
                    "authorized data access",
                ],
            },
            {
                "capability": "bellboy_action_prediction_inference_and_evaluation",
                "requires": [
                    "private inference driver",
                    "action_prediction artifact schema",
                    "heldout task metric implementation",
                ],
            },
        ],
    }
    write_json_uri(output_uri, report)
    print(json.dumps(report, sort_keys=True))
    return report
