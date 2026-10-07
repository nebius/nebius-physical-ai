"""Fail-closed contracts for the pi0.5 released-surface pick-and-place variant.

This module is deliberately import-light.  Isaac and OpenPI runtimes both use
these transforms, and unit tests exercise the exact boundary without requiring
either large runtime.  It does not redefine the canonical PPO task.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "npa.sim2real.pi05.task_contract.v1"
DATASET_SCHEMA = "npa.sim2real.pi05.dense_dataset.v1"
EPISODE_SCHEMA = "npa.sim2real.pi05.episode.v1"
TASK_ID = "Isaac-Surface-PickPlace-Franka-OpenPI-v0"
JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
# Franka limits from the stock articulation.  Keep a small boundary margin so
# a predicted target cannot be accepted and then clipped invisibly by Isaac.
JOINT_LIMITS_RAD = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
MAX_GRIPPER_WIDTH_M = 0.08
CONTROL_HZ = 15.0
ACTION_HORIZON = 15
EXECUTION_PREFIX = 8


class Pi05ContractError(ValueError):
    """Raised before incompatible task, observation, or action bytes are used."""


def _sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _task_contract() -> dict[str, Any]:
    return {
        "comparison_scope": {
            "canonical_ppo": "historical_diagnostic_only",
            "reason": "canonical goal z is airborne; this task ends released on a support surface",
            "old_strict_5cm_diagnostic_preserved": True,
        },
        "task": {
            "instruction": "Pick up the object and place it inside the target area on the table.",
            "support_surface": "table",
            "target_xy_m": {"x": [0.42, 0.58], "y": [0.10, 0.25]},
            "target_z": "derived from measured table top plus half object height",
            "object_initial_xy_m": {"x": [0.42, 0.55], "y": [-0.25, -0.08]},
            "object_motion": "physics_only_no_teleport_or_direct_object_state_write",
        },
        "success": {
            "authoritative_metric": "released_supported_surface_placement_v1",
            "required_event_order": [
                "contact",
                "grasp",
                "lift",
                "transport",
                "release",
                "support_contact",
                "stable",
            ],
            "target_xy_distance_m_lt": 0.05,
            "object_speed_m_s_lt": 0.03,
            "stable_steps_gte": 3,
            "lift_delta_m_gte": 0.05,
            "gripper_released_width_m_gte": 0.06,
            "hand_object_retreat_distance_m_gte": 0.10,
            "old_goal_distance_5cm": "diagnostic_only_not_cross_task_comparison",
        },
    }


def _observation_contract() -> dict[str, Any]:
    return {
        "observation": {
            "exterior_image_1_left": {
                "dtype": "uint8",
                "shape": [224, 224, 3],
                "mount": "world",
            },
            "wrist_image_left": {
                "dtype": "uint8",
                "shape": [224, 224, 3],
                "mount": "panda_hand",
                "mount_translation_m": [0.045, 0.0, 0.035],
                "mount_quaternion_wxyz": [0.0, 0.0, 1.0, 0.0],
                "moving_with_end_effector": True,
            },
            "joint_position": {
                "dtype": "float32",
                "shape": [7],
                "units": "rad",
                "order": list(JOINT_NAMES),
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": [1],
                "units": "normalized",
                "open": 0.0,
                "closed": 1.0,
            },
            "prompt": "non_empty_utf8",
            "alignment": (
                "one causal pre-command exterior+wrist capture and state per control step; "
                "the label is the absolute controller target computed from that observation, "
                "with observation, command-application, and resulting-state simulator times"
            ),
        },
    }


def _action_contract() -> dict[str, Any]:
    return {
        "action": {
            "kind": "absolute_joint_position_plus_normalized_gripper",
            "shape": [ACTION_HORIZON, 8],
            "joint_order": list(JOINT_NAMES),
            "joint_units": "rad",
            "joint_limits_rad": [list(v) for v in JOINT_LIMITS_RAD],
            "gripper": {
                "open": 0.0,
                "closed": 1.0,
                "prediction_range": [0.0, 1.0],
                "close_threshold": ">0.5",
                "physical_width_m": [MAX_GRIPPER_WIDTH_M, 0.0],
                "isaac_finger_targets_m": {"open": [0.04, 0.04], "closed": [0.0, 0.0]},
            },
            "control_hz": CONTROL_HZ,
            "hold_seconds": 1.0 / CONTROL_HZ,
            "receding_horizon": {
                "model_horizon": ACTION_HORIZON,
                "execute_prefix": EXECUTION_PREFIX,
            },
            "isaac_bridge": "(absolute_target_rad-default_joint_position_rad)/action_scale_rad",
            "clipping": "forbidden",
        },
    }


def _data_contract() -> dict[str, Any]:
    return {
        "dataset": {
            "schema": DATASET_SCHEMA,
            "successful_physics_demonstrations_only_for_training": True,
            "failures_retained_as_evidence": True,
            "normalization": "train_split_only",
            "split_keys": ["object_identity", "scene_configuration_digest"],
            "public_seed": "rejected: lift-only 7D relative actions and insufficient task/camera contract",
        },
        "upstream": {
            "openpi_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
            "config": "pi05_droid_jointpos_polaris",
            "checkpoint": "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris",
            "right_wrist": "adapter-created zeros with false mask; not a dataset input",
        },
    }


def build_contract() -> dict[str, Any]:
    """Return the exact variant contract. Args: None. Returns: Contract. Raises: None."""

    contract: dict[str, Any] = {"schema": SCHEMA, "task_id": TASK_ID}
    for section in (
        _task_contract(),
        _observation_contract(),
        _action_contract(),
        _data_contract(),
    ):
        contract.update(section)
    contract["contract_sha256"] = _sha(contract)
    return contract


def assert_contract(contract: Mapping[str, Any]) -> str:
    """Verify final bytes. Args: contract. Returns: SHA-256. Raises: Pi05ContractError."""
    if contract.get("schema") != SCHEMA:
        raise Pi05ContractError("unsupported pi0.5 task contract schema")
    given = str(contract.get("contract_sha256") or "")
    body = dict(contract)
    body.pop("contract_sha256", None)
    actual = _sha(body)
    if not given or given != actual:
        raise Pi05ContractError("pi0.5 task contract digest mismatch")
    return actual


def _finite_vector(value: Sequence[float], size: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or len(value) != size:
        raise Pi05ContractError(f"{name} must have exactly {size} values")
    result = [float(v) for v in value]
    if not all(math.isfinite(v) for v in result):
        raise Pi05ContractError(f"{name} contains non-finite values")
    return result


def droid_gripper_to_width(value: float) -> float:
    """Convert DROID grip. Args: value. Returns: Width. Raises: Pi05ContractError."""

    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise Pi05ContractError("DROID gripper target must be finite and in [0, 1]")
    return (1.0 - normalized) * MAX_GRIPPER_WIDTH_M


def width_to_droid_gripper(width_m: float) -> float:
    """Convert finger width. Args: width_m. Returns: DROID grip. Raises: Pi05ContractError."""
    width = float(width_m)
    if not math.isfinite(width) or not 0.0 <= width <= MAX_GRIPPER_WIDTH_M:
        raise Pi05ContractError("physical gripper width must be in [0, 0.08] m")
    return 1.0 - width / MAX_GRIPPER_WIDTH_M


def absolute_to_isaac_action(
    target: Sequence[float], *, defaults: Sequence[float], scales: Sequence[float]
) -> list[float]:
    """Map Polaris to Isaac. Args: target/defaults/scales. Returns: Action. Raises: Pi05ContractError."""

    action = _finite_vector(target, 8, "Polaris action")
    default = _finite_vector(defaults, 7, "Isaac default joint positions")
    scale = _finite_vector(scales, 7, "Isaac joint action scales")
    if any(v == 0.0 for v in scale):
        raise Pi05ContractError("Isaac joint action scales must be non-zero")
    for index, (value, limits) in enumerate(
        zip(action[:7], JOINT_LIMITS_RAD, strict=True)
    ):
        if not limits[0] <= value <= limits[1]:
            raise Pi05ContractError(
                f"absolute target for {JOINT_NAMES[index]} is outside its physical limits"
            )
    arm = [
        (value - offset) / gain
        for value, offset, gain in zip(action[:7], default, scale, strict=True)
    ]
    if any(abs(value) > 1.0 for value in arm):
        raise Pi05ContractError(
            "absolute target is outside the configured Isaac action range; clipping is forbidden"
        )
    if not 0.0 <= action[7] <= 1.0:
        raise Pi05ContractError("DROID gripper prediction must be in [0, 1]")
    # The pinned upstream robot example explicitly binarizes policy output at
    # >0.5. Stock Isaac uses negative=close and positive=open.
    return [*arm, -1.0 if action[7] > 0.5 else 1.0]


def verify_controller_roundtrip(
    target: Sequence[float],
    *,
    defaults: Sequence[float],
    scales: Sequence[float],
    controller_target: Sequence[float],
    finger_controller_target_m: Sequence[float],
) -> dict[str, Any]:
    """Prove target roundtrip. Args: requested and observed targets. Returns: Proof. Raises: Pi05ContractError."""

    requested = _finite_vector(target, 8, "Polaris action")
    rendered = absolute_to_isaac_action(requested, defaults=defaults, scales=scales)
    observed = _finite_vector(controller_target, 7, "Isaac controller target")
    observed_fingers = _finite_vector(
        finger_controller_target_m, 2, "Isaac finger controller target"
    )
    maximum_error, reconstruction_error = _joint_roundtrip_errors(
        requested, rendered, observed, defaults, scales
    )
    if maximum_error > 1.0e-5 or reconstruction_error > 1.0e-9:
        raise Pi05ContractError(
            "Isaac controller target does not roundtrip to the Polaris target"
        )
    expected_fingers = [0.0, 0.0] if requested[7] > 0.5 else [0.04, 0.04]
    finger_error = max(
        abs(a - b) for a, b in zip(expected_fingers, observed_fingers, strict=True)
    )
    if finger_error > 1.0e-5:
        raise Pi05ContractError(
            "Isaac two-finger controller target does not match the DROID threshold mapping"
        )
    return _roundtrip_evidence(
        requested,
        rendered,
        expected_fingers,
        maximum_error,
        reconstruction_error,
        finger_error,
    )


def _joint_roundtrip_errors(
    requested: list[float],
    rendered: list[float],
    observed: list[float],
    defaults: Sequence[float],
    scales: Sequence[float],
) -> tuple[float, float]:
    reconstructed = [
        rendered[i] * float(scales[i]) + float(defaults[i]) for i in range(7)
    ]
    target_error = max(abs(a - b) for a, b in zip(requested[:7], observed, strict=True))
    bridge_error = max(
        abs(a - b) for a, b in zip(requested[:7], reconstructed, strict=True)
    )
    return target_error, bridge_error


def _roundtrip_evidence(
    requested: list[float],
    rendered: list[float],
    fingers: list[float],
    target_error: float,
    bridge_error: float,
    finger_error: float,
) -> dict[str, Any]:
    return {
        "joint_order": list(JOINT_NAMES),
        "max_controller_target_error_rad": target_error,
        "max_bridge_roundtrip_error_rad": bridge_error,
        "gripper_prediction_normalized": requested[7],
        "gripper_close_threshold": ">0.5",
        "thresholded_gripper_target": 1.0 if requested[7] > 0.5 else 0.0,
        "finger_controller_target_m": fingers,
        "max_finger_controller_target_error_m": finger_error,
        "expected_observed_gripper_width_m": 0.0
        if requested[7] > 0.5
        else MAX_GRIPPER_WIDTH_M,
        "isaac_action": rendered,
    }


def _validate_timing(row: Mapping[str, Any], previous: float) -> float:
    timestamp = float(row.get("timestamp_s", math.nan))
    if not math.isfinite(timestamp) or timestamp <= previous:
        raise Pi05ContractError("camera/state/action timestamps must strictly increase")
    values = [
        float(row.get(key, math.nan))
        for key in (
            "exterior_timestamp_s",
            "wrist_timestamp_s",
            "command_application_timestamp_s",
            "resulting_state_timestamp_s",
        )
    ]
    exterior, wrist, command, resulting = values
    invalid = not all(math.isfinite(value) for value in values)
    invalid |= (
        max(abs(exterior - timestamp), abs(wrist - timestamp)) > 1.0 / 60.0 + 1.0e-6
    )
    invalid |= command < timestamp
    invalid |= abs((resulting - command) - 1.0 / CONTROL_HZ) > 1.0e-6
    if invalid:
        raise Pi05ContractError(
            "every action/state must retain causal simulator and camera timing"
        )
    return timestamp


def _validate_dense_row(row: Mapping[str, Any], index: int, previous: float) -> float:
    if int(row.get("step", -1)) != index:
        raise Pi05ContractError("episode steps must be contiguous and ordered")
    timestamp = _validate_timing(row, previous)
    _finite_vector(row.get("joint_position", []), 7, "joint_position")
    action = _finite_vector(row.get("action", []), 8, "action")
    for joint_index, (value, limits) in enumerate(
        zip(action[:7], JOINT_LIMITS_RAD, strict=True)
    ):
        if not limits[0] <= value <= limits[1]:
            raise Pi05ContractError(
                f"action target for {JOINT_NAMES[joint_index]} is outside its physical limits"
            )
    if not 0.0 <= action[7] <= 1.0:
        raise Pi05ContractError("DROID gripper action must be in [0, 1]")
    for key in ("exterior_image_1_left", "wrist_image_left"):
        image = row.get(key)
        valid = isinstance(image, Mapping) and image.get("shape") == [224, 224, 3]
        valid = valid and image.get("dtype") == "uint8" and bool(image.get("path"))
        if not valid:
            raise Pi05ContractError(
                f"{key} is missing a dense 224x224 uint8 observation"
            )
    return timestamp


def _validate_success_evidence(episode: Mapping[str, Any]) -> None:
    evidence = episode.get("success_evidence")
    if not isinstance(evidence, Mapping):
        raise Pi05ContractError("episode lacks measured success evidence")
    ordered = list(evidence.get("ordered_events") or [])
    required = build_contract()["success"]["required_event_order"]
    positions = [ordered.index(name) if name in ordered else -1 for name in required]
    if positions != sorted(positions) or any(value < 0 for value in positions):
        raise Pi05ContractError(
            "episode does not prove ordered grasp/lift/release/stability events"
        )
    if not bool(evidence.get("released_supported_surface_placement")):
        raise Pi05ContractError(
            "training episode is not a successful released surface placement"
        )
    if int(evidence.get("stable_steps") or 0) < 3:
        raise Pi05ContractError(
            "released placement was not stable for three physics steps"
        )


def _validate_episode_header(episode: Mapping[str, Any]) -> None:
    if episode.get("schema") != EPISODE_SCHEMA:
        raise Pi05ContractError("unsupported dense episode schema")
    if (
        episode.get("source_backend") != "isaac"
        or episode.get("object_motion") != "physics_only"
    ):
        raise Pi05ContractError(
            "demonstrations must come from Isaac physics without object state writes"
        )
    if (
        episode.get("action_semantics")
        != "absolute_joint_position_plus_normalized_gripper"
    ):
        raise Pi05ContractError(
            "7D, relative, delta, or unspecified action targets are incompatible"
        )
    _finite_vector(episode.get("target_position_m", []), 3, "target_position_m")
    _finite_vector(
        episode.get("initial_object_position_m", []),
        3,
        "initial_object_position_m",
    )
    width = float(episode.get("object_width_m", math.nan))
    if not math.isfinite(width) or not 0.01 <= width <= MAX_GRIPPER_WIDTH_M:
        raise Pi05ContractError("object width is missing or incompatible with Franka")
    if not isinstance(episode.get("scene_seed"), int):
        raise Pi05ContractError("scene randomization seed provenance is missing")


def validate_dense_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an episode. Args: episode. Returns: Split identity. Raises: Pi05ContractError."""

    _validate_episode_header(episode)
    rows = episode.get("steps")
    if not isinstance(rows, list) or len(rows) < ACTION_HORIZON:
        raise Pi05ContractError(
            "episode has too few dense steps for one pi0.5 action horizon"
        )
    previous = -math.inf
    for index, row in enumerate(rows):
        previous = _validate_dense_row(row, index, previous)
    _validate_success_evidence(episode)
    return {
        "episode_id": str(episode.get("episode_id") or ""),
        "object_identity": str(episode.get("object_identity") or ""),
        "scene_configuration_digest": str(
            episode.get("scene_configuration_digest") or ""
        ),
        "step_count": len(rows),
    }


def validate_split_manifest(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Validate split isolation. Args: manifest. Returns: Counts. Raises: Pi05ContractError."""

    if manifest.get("schema") != DATASET_SCHEMA:
        raise Pi05ContractError("unsupported pi0.5 dataset schema")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {
        "train",
        "validation",
        "gold",
    }:
        raise Pi05ContractError(
            "dataset must contain exact train/validation/gold splits"
        )
    counts = _validate_split_rows(splits)
    normalization = manifest.get("normalization")
    if (
        not isinstance(normalization, Mapping)
        or normalization.get("source_split") != "train"
    ):
        raise Pi05ContractError(
            "normalization statistics must be computed from train only"
        )
    if not str(normalization.get("sha256") or ""):
        raise Pi05ContractError("normalization provenance digest is missing")
    return counts


def _validate_split_rows(splits: Mapping[str, Any]) -> dict[str, int]:
    seen_objects: set[str] = set()
    seen_scenes: set[str] = set()
    counts = {}
    for name in ("train", "validation", "gold"):
        rows = splits[name]
        if not isinstance(rows, list) or not rows:
            raise Pi05ContractError(f"{name} split is empty")
        objects = {str(row.get("object_identity") or "") for row in rows}
        scenes = {str(row.get("scene_configuration_digest") or "") for row in rows}
        if (
            "" in objects
            or "" in scenes
            or objects & seen_objects
            or scenes & seen_scenes
        ):
            raise Pi05ContractError(
                "object identities and scene configurations must be disjoint by split"
            )
        seen_objects |= objects
        seen_scenes |= scenes
        counts[name] = len(rows)
    return counts
