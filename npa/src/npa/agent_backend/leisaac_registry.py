"""Authoritative registry for browser-teleoperable LeIsaac environments.

Keep this module dependency-free: the agent bootstrap and the LeIsaac container
ship the same source file so the CLI, runtime, manifest validation, and UI all
use identical task metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

REGISTRY_SCHEMA = "npa.leisaac.task-registry.v1"
CONFIGURATION_SCHEMA = "npa.leisaac.configuration.v1"
DEFAULT_TASK = "LeIsaac-SO101-LiftCube-v0"
DEFAULT_ENVIRONMENT_ID = "operator-0"
TELEOP_DEVICE = "keyboard"
DEFAULT_ROBOT = "so101_follower"
DEFAULT_SCENE = "table_with_cube"
DEFAULT_DEVICE = "browser_keyboard_so101"

_ENVIRONMENT_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,61}[A-Za-z0-9])?$")

RUNTIME_ASSETS: tuple[dict[str, str], ...] = (
    {
        "id": "so101_follower",
        "url": "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd",
        "sha256": "64a877c3b82cdc4a48ab8a1f321a2dd3ef7c55d4b10bce222b58c530d978ae58",
        "destination": "robots/so101_follower.usd",
        "archive": "false",
    },
    {
        "id": "kitchen_with_orange",
        "url": "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip",
        "sha256": "d314c54b63a17e91402bfaddf26e21ff614adf2430fa092b78897f15b8adea34",
        "destination": "scenes/kitchen_with_orange/scene.usd",
        "archive": "true",
    },
    {
        "id": "table_with_cube",
        "url": "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.2/table_with_cube.zip",
        "sha256": "917c66a724019d235cc9f442a30ae72e5663b44ef4ed8d4d5324e549e11952b7",
        "destination": "scenes/table_with_cube/scene.usd",
        "archive": "true",
    },
)

BUILTIN_ROBOTS: tuple[dict[str, str], ...] = (
    {
        "id": DEFAULT_ROBOT,
        "display_name": "Built-in SO-101 follower robot",
        "asset_id": "so101_follower",
        "runtime_reference": "runtime-fetched SO-101 follower USD",
    },
)

BUILTIN_SCENES: tuple[dict[str, str], ...] = (
    {
        "id": "kitchen_with_orange",
        "display_name": "Built-in kitchen counter and orange scene",
        "asset_id": "kitchen_with_orange",
        "runtime_reference": "runtime-fetched kitchen_with_orange USD",
    },
    {
        "id": DEFAULT_SCENE,
        "display_name": "Built-in table and lift-cube scene",
        "asset_id": "table_with_cube",
        "runtime_reference": "runtime-fetched table_with_cube USD",
    },
)

BUILTIN_DEVICES: tuple[dict[str, str], ...] = (
    {
        "id": DEFAULT_DEVICE,
        "display_name": "Browser keyboard SO-101 teleoperator (default test device)",
        "driver": TELEOP_DEVICE,
        "runtime_reference": "upstream SO101Keyboard",
    },
)

# This is intentionally smaller than upstream's complete Gym registry. These
# are the single-arm SO101 tasks at the pinned source commit that accept the
# exact eight-dimensional SO101Keyboard action path used by the browser relay.
SUPPORTED_TASKS: tuple[dict[str, Any], ...] = (
    {
        "task": "LeIsaac-SO101-PickOrange-v0",
        "display_name": "SO101 Pick Orange",
        "description": "Pick an orange from the kitchen counter and place it on the plate.",
        "robot": "SO101",
        "robot_id": DEFAULT_ROBOT,
        "scene_id": "kitchen_with_orange",
        "device_id": DEFAULT_DEVICE,
        "teleop_device": TELEOP_DEVICE,
        "action_dimension": 8,
        "state_joint_names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        "asset_ids": ["so101_follower", "kitchen_with_orange"],
    },
    {
        "task": "LeIsaac-SO101-LiftCube-v0",
        "display_name": "SO101 Lift Cube",
        "description": "Grasp and lift the red cube from the table.",
        "robot": "SO101",
        "robot_id": DEFAULT_ROBOT,
        "scene_id": DEFAULT_SCENE,
        "device_id": DEFAULT_DEVICE,
        "teleop_device": TELEOP_DEVICE,
        "action_dimension": 8,
        "state_joint_names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        "asset_ids": ["so101_follower", "table_with_cube"],
    },
)


def registry_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": REGISTRY_SCHEMA,
        "source": {
            "version": "0.4.0",
            "commit": "1651c321e9b0c1bb54233211fc7b3cd70d8373d5",
        },
        "default_task": DEFAULT_TASK,
        "environment_model": "named-sequential",
        "max_parallel_environments": 1,
        "tasks": [dict(item) for item in SUPPORTED_TASKS],
        "runtime_assets": [dict(item) for item in RUNTIME_ASSETS],
        "builtins": {
            "robots": [dict(item) for item in BUILTIN_ROBOTS],
            "scenes": [dict(item) for item in BUILTIN_SCENES],
            "devices": [dict(item) for item in BUILTIN_DEVICES],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return payload


REGISTRY_FINGERPRINT = registry_payload()["fingerprint"]


def task_metadata(task: str) -> dict[str, Any]:
    value = str(task or "").strip()
    for item in SUPPORTED_TASKS:
        if item["task"] == value:
            return dict(item)
    allowed = ", ".join(item["task"] for item in SUPPORTED_TASKS)
    raise ValueError(f"unsupported LeIsaac task {value!r}; choose one of: {allowed}")


def _builtin(items: tuple[dict[str, str], ...], identifier: str) -> dict[str, str]:
    for item in items:
        if item["id"] == identifier:
            return dict(item)
    raise ValueError(f"LeIsaac registry has no built-in {identifier!r}")


def resolve_configuration(
    task: str = DEFAULT_TASK,
    selected_bundles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve real built-ins plus any validated uploaded-bundle overrides."""

    metadata = task_metadata(task)
    configuration: dict[str, Any] = {
        "schema": CONFIGURATION_SCHEMA,
        "robot": {
            **_builtin(BUILTIN_ROBOTS, str(metadata["robot_id"])),
            "source": "built-in-runtime",
        },
        "scene": {
            **_builtin(BUILTIN_SCENES, str(metadata["scene_id"])),
            "source": "built-in-runtime",
        },
        "device": {
            **_builtin(BUILTIN_DEVICES, str(metadata["device_id"])),
            "source": "built-in-runtime",
        },
        "task": {
            "id": str(metadata["task"]),
            "display_name": str(metadata["display_name"]),
            "source": "built-in-registry",
        },
    }
    selection = selected_bundles if isinstance(selected_bundles, dict) else {}
    for kind in ("robot", "scene", "device"):
        item = selection.get(kind)
        if not isinstance(item, dict):
            continue
        digest = str(item.get("bundle_sha256") or "")
        name = str(item.get("name") or "")
        entrypoint = str(item.get("entrypoint") or "")
        if not digest or not name or not entrypoint:
            continue
        configuration[kind] = {
            "id": name,
            "display_name": name,
            "source": "uploaded-bundle",
            "bundle_sha256": digest,
            "entrypoint": entrypoint,
        }
    configuration["custom_bundle_count"] = sum(
        configuration[kind].get("source") == "uploaded-bundle"
        for kind in ("robot", "scene", "device")
    )
    return configuration


def validate_task(task: str) -> str:
    return str(task_metadata(task)["task"])


def validate_environment_id(environment_id: str) -> str:
    value = str(environment_id or "").strip()
    if not _ENVIRONMENT_ID.fullmatch(value):
        raise ValueError(
            "environment id must start with a letter or number and contain only "
            "letters, numbers, '.', '_' and '-', end with a letter or number, "
            "and contain at most 63 characters"
        )
    return value


def validate_environment_index(environment_index: int) -> int:
    value = int(environment_index)
    if value < 0 or value > 2**31 - 1:
        raise ValueError("environment index must be between 0 and 2147483647")
    return value


def validate_seed(seed: int) -> int:
    value = int(seed)
    if value < 0 or value > 2**32 - 1:
        raise ValueError("seed must be between 0 and 4294967295")
    return value


def validate_num_envs(num_envs: int) -> int:
    value = int(num_envs)
    if value != 1:
        raise ValueError(
            "browser teleoperation supports exactly one active environment per "
            "session; use distinct environment IDs across sequential launches"
        )
    return value
