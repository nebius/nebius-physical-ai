"""Preparation and evidence adapters for model-neutral Sim2Real policies.

The input is one checksum-sealed profile directory.  Preparing a run validates
the robot, calibrated sensors, rigid object geometry, unchanged placement event
contract, and native policy adapter before a GPU job can be submitted.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from npa.workflows.lerobot_transfer_data import materialize, tree_hashes
from npa.workflows.sim2real.interchange_backends import (
    backend_contract,
    openpi_pi05_policy_profile,
    smolvla_policy_profile,
)
from npa.workflows.sim2real.interchange_contract import (
    InterchangeContractError,
    SurfacePlacementTask,
    Transform,
    assert_policy_compatible,
    assert_surface_placement_compatible,
    rigid_object_profile_from_asset_manifest,
    robot_sensor_profile_from_resolved,
)
from npa.workflows.sim2real.robot_contract import assert_robot_contract
from npa.workflows.sim2real.workflow_io import (
    POLICY_INTERCHANGE_COMPONENT_SCHEMA,
    publish_component_record,
    write_json,
)

PROFILE_SCHEMA = "npa.sim2real.policy_interchange.profile.v1"
PREPARATION_SCHEMA = "npa.sim2real.policy_interchange.preparation.v1"
_PROFILE_FILES = {
    "profile.json",
    "robot-contract.json",
    "sensor-config.json",
    "object-manifest.json",
    "object-config.json",
    "task.json",
    "policy-adapter.json",
}


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InterchangeContractError(f"invalid profile JSON {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise InterchangeContractError(f"profile file {path.name} must be an object")
    return value


def _transform(value: object, *, name: str) -> Transform:
    if not isinstance(value, Mapping):
        raise InterchangeContractError(f"{name} transform is required")
    return Transform(
        tuple(value.get("translation_m") or ()),  # type: ignore[arg-type]
        tuple(value.get("quaternion_xyzw") or ()),  # type: ignore[arg-type]
    )


def _task(value: Mapping[str, Any]) -> SurfacePlacementTask:
    if value.get("schema") != "npa.sim2real.surface_placement_task.v1":
        raise InterchangeContractError("unsupported task profile schema")
    if value.get("privileged_object_state") not in (None, False):
        raise InterchangeContractError("privileged object state is forbidden")
    return SurfacePlacementTask(
        identity=str(value.get("identity") or ""),
        target_world_xyz_m=tuple(value.get("target_world_xyz_m") or ()),  # type: ignore[arg-type]
        approach_clearance_m=float(value.get("approach_clearance_m", "nan")),
        lift_delta_m=float(value.get("lift_delta_m", "nan")),
        retreat_distance_m=float(value.get("retreat_distance_m", "nan")),
        success_events=tuple(value.get("success_events") or ()),
    )


def build_preparation(profile_root: Path, *, run_id: str) -> dict[str, Any]:
    """Validate a materialized sealed profile and return its launch authority."""

    checksums = _object(profile_root / "checksums.json")
    if not checksums or checksums != tree_hashes(profile_root):
        raise InterchangeContractError("interchange profile checksum mismatch")
    missing = sorted(name for name in _PROFILE_FILES if not (profile_root / name).is_file())
    if missing:
        raise InterchangeContractError(
            "sealed interchange profile is incomplete: " + ", ".join(missing)
        )
    profile = _object(profile_root / "profile.json")
    if profile.get("schema") != PROFILE_SCHEMA:
        raise InterchangeContractError("unsupported interchange profile schema")
    backend_name = str(profile.get("backend") or "")
    backend = backend_contract(backend_name)

    robot_contract = _object(profile_root / "robot-contract.json")
    robot_identity = assert_robot_contract(robot_contract)
    robot = robot_sensor_profile_from_resolved(
        robot_contract, sensor_config=_object(profile_root / "sensor-config.json")
    )
    if robot.identity != robot_identity:
        raise InterchangeContractError("robot sensor profile identity is stale")

    object_config = _object(profile_root / "object-config.json")
    prop = rigid_object_profile_from_asset_manifest(
        _object(profile_root / "object-manifest.json"),
        target=str(object_config.get("target") or ""),
        source_root_uri=str(object_config.get("source_root_uri") or ""),
        world_from_root=_transform(
            object_config.get("world_from_root"), name="world_from_root"
        ),
        root_from_reference=_transform(
            object_config.get("root_from_reference"), name="root_from_reference"
        ),
        root_from_grasp=_transform(
            object_config.get("root_from_grasp"), name="root_from_grasp"
        ),
        grasp_extents_m=tuple(object_config.get("grasp_extents_m") or ()),
        grasp_axis=str(object_config.get("grasp_axis") or ""),  # type: ignore[arg-type]
        scale_xyz=tuple(object_config.get("scale_xyz") or ()),
    )
    task = _task(_object(profile_root / "task.json"))
    geometry_receipt = assert_surface_placement_compatible(robot, prop, task)

    adapter = _object(profile_root / "policy-adapter.json")
    topologies = tuple(str(value) for value in profile.get("hardware_topologies") or ())
    if backend_name == "lerobot_smolvla":
        policy = smolvla_policy_profile(
            adapter, robot=robot, task=task, hardware_topologies=topologies
        )
    elif backend_name == "openpi_pi05":
        policy = openpi_pi05_policy_profile(
            robot=robot,
            task=task,
            policy_identity=str(adapter.get("policy_identity") or ""),
            checkpoint_sha256=str(adapter.get("checkpoint_sha256") or ""),
            normalization_identity=str(adapter.get("normalization_identity") or ""),
            hardware_topologies=topologies,
        )
    else:  # backend_contract already rejects this; retain exhaustive dispatch.
        raise InterchangeContractError(f"unsupported backend {backend_name!r}")
    policy_receipt = assert_policy_compatible(
        robot,
        task,
        policy,
        cartesian_controller_available=bool(
            profile.get("cartesian_controller_qualified", False)
        ),
    )
    return {
        "schema": PREPARATION_SCHEMA,
        "run_id": run_id,
        "backend": backend.to_dict(),
        "robot": robot.to_dict(),
        "object": prop.to_dict(),
        "task": {
            "identity": task.identity,
            **task.teacher_frames(prop),
        },
        "geometry_compatibility": geometry_receipt,
        "policy_compatibility": policy_receipt,
        "capabilities": policy.capabilities.to_dict(),
        "profile_checksums": checksums,
        "launch_authorized": True,
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="npa-sim2real-interchange-") as directory:
        root = materialize(args.profile_uri, Path(directory) / "profile")
        result = build_preparation(root, run_id=args.run_id)
        write_json(args.output_uri, result, directory=Path(directory) / "output")
    publish_component_record(
        component_schema=POLICY_INTERCHANGE_COMPONENT_SCHEMA,
        root_uri=args.component_root_uri,
        stage=1,
        name="policy_interchange_prepare",
        tier="WORKS",
        evidence="Validated sealed robot, sensor, object, task, and policy contracts.",
        artifacts={
            "preparation": args.output_uri,
            "backend": result["backend"]["name"],
            "robot_identity": result["robot"]["identity"],
            "object_identity": result["object"]["identity"],
            "task_identity": result["task"]["identity"],
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--profile-uri", required=True)
    prepare.add_argument("--output-uri", required=True)
    prepare.add_argument("--component-root-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = {"prepare": _prepare}[args.command](args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
