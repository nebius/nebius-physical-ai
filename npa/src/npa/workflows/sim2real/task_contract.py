"""Normalized task/data contract for the canonical Sim2Real manipulation loop.

The contract is deliberately simulator-facing.  It binds a source dataset to the
exact Isaac task, embodiment, action/observation spaces, randomized quantities,
and success metric that consume it.  A dataset name is not treated as provenance:
real-required runs also need a concrete source URI containing task-aligned media.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


LIFT_TASK_ID = "Isaac-Lift-Cube-Franka-v0"
LIFT_DATASET_ID = "npa/isaac-lift-cube-franka-seed-v1"
PUSHT_DATASET_ID = "lerobot/pusht"
TASK_CONTRACT_SCHEMA = "npa.sim2real.task_contract.v1"
SEED_DATASET_SCHEMA = "npa.sim2real.task_seed_dataset.v1"
STRICT_SUCCESS_DISTANCE_M = 0.05


class TaskContractError(ValueError):
    """Raised when task, dataset, or runtime semantics do not agree."""


def canonical_digest(payload: dict[str, Any]) -> str:
    """Return a stable sha256 over a JSON object, excluding a prior digest."""

    normalized = dict(payload)
    normalized.pop("task_contract_digest", None)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_task_family(dataset_id: str) -> str:
    """Classify a known source without pretending unrelated data is compatible."""

    value = str(dataset_id or "").strip().lower()
    if value == LIFT_DATASET_ID.lower() or (
        "lift" in value and "franka" in value and "cube" in value
    ):
        return "franka_lift_cube"
    if value == PUSHT_DATASET_ID or "pusht" in value:
        return "pusht_planar"
    return "unknown"


def task_family(task_id: str) -> str:
    value = str(task_id or "").strip().lower()
    if value == LIFT_TASK_ID.lower() or (
        "lift" in value and "cube" in value and "franka" in value
    ):
        return "franka_lift_cube"
    if "pusht" in value:
        return "pusht_planar"
    return "unknown"


def validate_task_dataset(
    *, task_id: str, dataset_id: str, dataset_uri: str, real_required: bool
) -> None:
    """Fail closed when a real run would learn from semantically unrelated data."""

    tf = task_family(task_id)
    df = dataset_task_family(dataset_id)
    if tf == "unknown":
        raise TaskContractError(
            f"task {task_id!r} has no registered task/data contract; provide an "
            "explicit compatible task preset before running"
        )
    if df != tf:
        raise TaskContractError(
            f"dataset {dataset_id!r} ({df}) is incompatible with task "
            f"{task_id!r} ({tf}); PushT data must never silently seed Franka lift PPO"
        )
    if real_required and not str(dataset_uri or "").strip().startswith("s3://"):
        raise TaskContractError(
            "real-required Sim2Real needs a concrete s3:// task-aligned seed "
            "dataset URI; a dataset label alone is not provenance"
        )


def build_task_contract(
    *,
    task_id: str,
    dataset_id: str,
    dataset_uri: str,
    robot_source: str = "",
    robot_preset: str = "",
) -> dict[str, Any]:
    """Build the normalized stock-Franka lift contract consumed by all phases."""

    validate_task_dataset(
        task_id=task_id,
        dataset_id=dataset_id,
        dataset_uri=dataset_uri,
        real_required=False,
    )
    if task_family(task_id) != "franka_lift_cube":
        raise TaskContractError(
            f"task {task_id!r} and dataset {dataset_id!r} agree semantically, but "
            "the canonical Sim2Real engine has no normalized simulator-facing "
            "contract preset for that task; add an explicit preset instead of "
            "reusing the Franka lift contract"
        )
    runtime_robot = str(robot_preset or robot_source or "franka")
    stock_franka = runtime_robot.lower() in {"franka", "stock_franka"} and str(
        robot_source or "stock_franka"
    ).lower() in {"", "stock_franka"}
    contract: dict[str, Any] = {
        "schema": TASK_CONTRACT_SCHEMA,
        "task_id": task_id,
        "task_family": task_family(task_id),
        "train_eval_parity_id": "isaac-lift-cube-franka-state-v1",
        "dataset": {
            "id": dataset_id,
            "uri": dataset_uri,
            "provenance": "Isaac-generated Franka lift-cube RGB trajectories",
            "synthetic": True,
            "relabeled_from_another_task": False,
        },
        "embodiment": {
            "base_task_robot": "Franka Emika Panda",
            "runtime_robot": "Franka Emika Panda" if stock_franka else runtime_robot,
            "robot_source": robot_source or "stock_franka",
            "robot_preset": robot_preset or "franka",
            "end_effector": "parallel_jaw_gripper",
            "stage_2_runtime_assertion_required": not stock_franka,
        },
        "object": {
            "kind": "rigid_cube",
            "asset": "Isaac/Props/Blocks/MultiColorCube/multi_color_cube_instanceable.usd",
            "scale": [1.0, 1.0, 1.0],
            "nominal_edge_m": 0.05,
        },
        "goal_distribution": {
            "frame": "robot_base",
            "x_m": [0.42, 0.58],
            "y_m": [-0.20, 0.20],
            "z_m": [0.20, 0.42],
        },
        "object_initial_distribution": {
            "frame": "default_object_pose_offset",
            "x_m": [-0.08, 0.08],
            "y_m": [-0.20, 0.20],
            "z_m": [0.0, 0.0],
        },
        "action_contract": {
            "kind": "joint_position_delta_plus_binary_gripper",
            "dimensions": 8 if stock_franka else None,
            "dimensions_source": (
                "stock_franka_contract"
                if stock_franka
                else "stage_2_consumed_robot_spec_and_runtime_checkpoint_shape"
            ),
        },
        "observation_contract": {
            "policy": "state",
            "terms": [
                "joint_pos",
                "joint_vel",
                "object_position",
                "target_object_position",
                "last_action",
            ],
            "cosmos_pixels_directly_train_policy": False,
        },
        "physics_ranges": {
            "object_friction": [0.45, 1.25],
            "object_mass_scale": [0.85, 1.15],
        },
        "scene": {
            "id": "isaac-lift-franka-stock-table-v1",
            "lighting_intensity": 3000.0,
            "lighting_is_global_fixed": True,
            "camera_profile": "primary-side-overhead-v1",
        },
        "cameras": ["primary", "side", "overhead"],
        "success": {
            "authoritative_metric": "final_object_goal_distance_m",
            "strict_distance_m": STRICT_SUCCESS_DISTANCE_M,
            "diagnostic_distances_m": [0.10, 0.15, 0.20],
            "placement_stability_required": True,
        },
        "augmentation_coupling": {
            "consumer": "scenario_source_lineage_and_cosmos_reason_context",
            "state_policy_pixel_use": "lineage_only_not_policy_observation",
            "required_scenario_field": "source_augmentation",
        },
    }
    contract["task_contract_digest"] = canonical_digest(contract)
    return contract


def assert_contract_digest(contract: dict[str, Any]) -> str:
    expected = str(contract.get("task_contract_digest") or "")
    actual = canonical_digest(contract)
    if not expected or expected != actual:
        raise TaskContractError(
            f"task contract digest mismatch: expected={expected or 'missing'} actual={actual}"
        )
    return actual


def validate_seed_dataset_manifest(
    manifest: dict[str, Any],
    *,
    contract: dict[str, Any],
    trigger_objects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate real Isaac trajectory provenance without trusting a dataset label."""

    if manifest.get("schema") != SEED_DATASET_SCHEMA:
        raise TaskContractError(
            "task seed dataset manifest schema is missing or unsupported"
        )
    expected = {
        "task_id": contract["task_id"],
        "dataset_id": contract["dataset"]["id"],
        "task_contract_digest": contract["task_contract_digest"],
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise TaskContractError(f"task seed dataset manifest mismatch: {mismatches}")
    if manifest.get("source_backend") != "isaac":
        raise TaskContractError("task seed dataset must prove an Isaac source backend")
    if manifest.get("relabeled_from_another_task") is not False:
        raise TaskContractError(
            "task seed dataset must explicitly disavow cross-task relabeling"
        )
    for count_key in ("trajectory_count", "action_count", "camera_observation_count"):
        if int(manifest.get(count_key) or 0) <= 0:
            raise TaskContractError(f"task seed dataset has no {count_key}")
    sample_uri = str(manifest.get("sample_rollout_manifest_uri") or "")
    if not sample_uri.startswith("s3://"):
        raise TaskContractError("task seed dataset lacks an S3 sample rollout manifest")
    proof = {
        "schema": manifest["schema"],
        **expected,
        "source_backend": "isaac",
        "source_run_id": str(manifest.get("source_run_id") or ""),
        "trajectory_count": int(manifest["trajectory_count"]),
        "action_count": int(manifest["action_count"]),
        "camera_observation_count": int(manifest["camera_observation_count"]),
        "sample_rollout_manifest_uri": sample_uri,
        "manifest_digest": canonical_digest(manifest),
    }
    if manifest.get("preset") == "public-franka-lift":
        _validate_public_franka_lift_seed(
            manifest,
            trigger_objects=trigger_objects,
        )
        proof.update(
            {
                "preset": "public-franka-lift",
                "source_provenance": manifest["source_provenance"],
                "source_contract": manifest["source_contract"],
                "compatibility_boundary": manifest["compatibility_boundary"],
            }
        )
    return proof


def _validate_public_franka_lift_seed(
    manifest: dict[str, Any], *, trigger_objects: list[dict[str, Any]] | None
) -> None:
    """Keep the public source and canonical PPO contracts visibly separate."""

    from npa.orchestration.npa_workflow.presets import (
        PUBLIC_FRANKA_LIFT_CANONICAL_TASK_ID,
        PUBLIC_FRANKA_LIFT_DATASET_ID,
        PUBLIC_FRANKA_LIFT_DATASET_REPOSITORY,
        PUBLIC_FRANKA_LIFT_DATASET_REVISION,
        PUBLIC_FRANKA_LIFT_SOURCE_TASK_ID,
    )

    provenance = manifest.get("source_provenance")
    source = manifest.get("source_contract")
    boundary = manifest.get("compatibility_boundary")
    if not all(isinstance(value, dict) for value in (provenance, source, boundary)):
        raise TaskContractError("public Franka seed lacks explicit provenance/contract boundary")
    if (
        manifest.get("dataset_id") != PUBLIC_FRANKA_LIFT_DATASET_ID
        or manifest.get("task_id") != PUBLIC_FRANKA_LIFT_CANONICAL_TASK_ID
        or provenance.get("repository") != PUBLIC_FRANKA_LIFT_DATASET_REPOSITORY
        or provenance.get("revision") != PUBLIC_FRANKA_LIFT_DATASET_REVISION
        or provenance.get("declared_license") != "apache-2.0"
        or provenance.get("anonymous_public") is not True
    ):
        raise TaskContractError("public Franka seed provenance does not match its immutable pin")
    required_source_paths = {
        "final_dataset/meta/info.json",
        "final_dataset/meta/tasks.jsonl",
        "final_dataset/meta/episodes.jsonl",
        "final_dataset/data/chunk-000/episode_000000.parquet",
        "final_dataset/videos/chunk-000/image/episode_000000.mp4",
        "final_dataset/videos/chunk-000/wrist_image/episode_000000.mp4",
    }
    if set(provenance.get("source_paths") or []) != required_source_paths:
        raise TaskContractError("public Franka seed source path provenance is incomplete")
    if source != {
        "task_id": PUBLIC_FRANKA_LIFT_SOURCE_TASK_ID,
        "robot": "franka",
        "action": {"dimensions": 7, "representation": "IK-relative"},
        "cameras": ["image", "wrist_image"],
    }:
        raise TaskContractError("public Franka seed source action/camera contract changed")
    if (
        boundary.get("source_actions_reused_as_canonical_ppo_actions") is not False
        or (boundary.get("canonical_action") or {}).get("dimensions") != 8
        or boundary.get("canonical_cameras") != ["primary", "side", "overhead"]
        or boundary.get("strict_success_distance_m") != STRICT_SUCCESS_DISTANCE_M
        or boundary.get("placement_stability_required") is not True
    ):
        raise TaskContractError("public Franka seed weakens the canonical policy/evaluation boundary")
    records = provenance.get("objects")
    if not isinstance(records, list) or not records:
        raise TaskContractError("public Franka seed lacks staged object hashes")
    for record in records:
        if (
            not isinstance(record, dict)
            or not str(record.get("uri") or "").startswith("s3://")
            or int(record.get("bytes") or 0) <= 0
            or len(str(record.get("sha256") or "")) != 64
            or any(
                char not in "0123456789abcdef"
                for char in str(record.get("sha256") or "")
            )
        ):
            raise TaskContractError("public Franka seed has malformed staged object evidence")
    frame_records = [record for record in records if "/frames/camera-" in record["uri"]]
    if len(frame_records) < 4 or len(frame_records) != int(manifest["camera_observation_count"]):
        raise TaskContractError("public Franka seed must contain at least four decoded camera frames")
    record_uris = {record["uri"] for record in records}
    if sample_uri := str(manifest.get("sample_rollout_manifest_uri") or ""):
        if sample_uri not in record_uris:
            raise TaskContractError("public Franka seed sample rollout was not staged")
    if not any(uri.endswith("/evidence/actions.json") for uri in record_uris):
        raise TaskContractError("public Franka seed action evidence was not staged")
    if trigger_objects is not None:
        listed = {
            f"s3://{item.get('Bucket')}/{item.get('Key')}": int(item.get("Size") or 0)
            for item in trigger_objects
            if item.get("Bucket") and item.get("Key")
        }
        missing_or_changed = sorted(
            record["uri"]
            for record in records
            if listed.get(record["uri"]) != int(record["bytes"])
        )
        if missing_or_changed:
            raise TaskContractError(
                f"public Franka staged objects are missing or changed: {missing_or_changed}"
            )
