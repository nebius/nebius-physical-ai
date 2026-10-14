"""Preparation and evidence adapters for model-neutral Sim2Real policies.

The input is one checksum-sealed profile directory.  Preparing a run validates
the robot, calibrated sensors, rigid object geometry, unchanged placement event
contract, and native policy adapter before a GPU job can be submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from npa.workbench.lerobot.policy_container import prepare_smolvla_base_checkpoint
from npa.workflows.lerobot_dataset import materialize_lerobot_dataset
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
    read_json,
    storage,
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
_HF_SOURCE_RE = re.compile(r"^hf://(?P<repo>[^@\s]+)@(?P<revision>[0-9a-f]{40})$")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InterchangeContractError(
            f"invalid profile JSON {path.name}: {exc}"
        ) from exc
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


def _validated_robot_asset_binding(
    robot_contract: Mapping[str, Any], robot: Any
) -> dict[str, Any]:
    """Retain the exact canonical RobotSpec and reject incomplete GPU inputs."""

    robot_spec = robot_contract.get("robot_spec")
    isaac_asset = robot_contract.get("isaac_asset")
    if not isinstance(robot_spec, Mapping) or not isinstance(isaac_asset, Mapping):
        raise InterchangeContractError(
            "non-stock policy interchange requires the normalized RobotSpec and "
            "resolved Isaac asset contract"
        )
    expected_names = [*robot.arm_joint_names, *robot.gripper.joint_names]
    if list(robot_spec.get("joint_names") or ()) != expected_names:
        raise InterchangeContractError(
            "RobotSpec joint order differs from the policy robot profile"
        )
    for field in ("home_qpos", "kp", "kv", "force_lower", "force_upper"):
        values = robot_spec.get(field)
        if not isinstance(values, list) or len(values) != len(expected_names):
            raise InterchangeContractError(
                f"RobotSpec {field} must cover every ordered physical joint"
            )
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise InterchangeContractError(
                f"RobotSpec {field} must contain numeric values"
            ) from exc
        if not all(math.isfinite(value) for value in numeric):
            raise InterchangeContractError(
                f"RobotSpec {field} must contain finite values"
            )
    if (
        robot_spec.get("embodiment_digest") != robot.identity
        or robot_spec.get("source_tree_sha256") != robot.bundle_sha256
        or robot_spec.get("source_sha256") != robot.source_sha256
        or isaac_asset.get("resolved_usd_uri") != robot_spec.get("resolved_usd_uri")
        or not robot_spec.get("resolved_manifest_uri")
    ):
        raise InterchangeContractError(
            "RobotSpec, sensor profile, and resolved Isaac asset identities disagree"
        )
    return {
        "contract_digest": robot_contract.get("contract_digest"),
        "robot_spec": dict(robot_spec),
        "isaac_asset": dict(isaac_asset),
    }


def build_preparation(profile_root: Path, *, run_id: str) -> dict[str, Any]:
    """Validate a materialized sealed profile and return its launch authority."""

    checksums = _object(profile_root / "checksums.json")
    if not checksums or checksums != tree_hashes(profile_root):
        raise InterchangeContractError("interchange profile checksum mismatch")
    missing = sorted(
        name for name in _PROFILE_FILES if not (profile_root / name).is_file()
    )
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
    robot_asset = _validated_robot_asset_binding(robot_contract, robot)
    return {
        "schema": PREPARATION_SCHEMA,
        "run_id": run_id,
        "backend": backend.to_dict(),
        "robot": robot.to_dict(),
        "robot_asset": robot_asset,
        "policy": policy.to_dict(),
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


def training_adapter(preparation: Mapping[str, Any]) -> dict[str, Any]:
    """Project a validated preparation into a native LeRobot adapter spec.

    Checkpoint weight and processor hashes are deliberately filled by the
    native binder after loading the selected base or trained checkpoint.
    """

    policy = preparation.get("policy")
    robot = preparation.get("robot")
    if not isinstance(policy, Mapping) or not isinstance(robot, Mapping):
        raise InterchangeContractError("preparation lacks robot/policy profiles")
    if policy.get("backend") != "lerobot_smolvla":
        raise InterchangeContractError(
            "native LeRobot adapter export requires the SmolVLA backend"
        )
    cameras = robot.get("cameras")
    if not isinstance(cameras, list):
        raise InterchangeContractError("prepared camera declarations are required")
    feature_by_role = {
        "exterior": "observation.images.workspace",
        "wrist": "observation.images.wrist",
    }
    adapter_cameras = []
    for camera in cameras:
        if not isinstance(camera, Mapping) or camera.get("role") not in feature_by_role:
            raise InterchangeContractError("unsupported prepared camera role")
        adapter_cameras.append(
            {
                "key": camera.get("key"),
                "feature_key": feature_by_role[str(camera["role"])],
                "layout": camera.get("layout"),
                "image_size_wh": camera.get("image_size_wh"),
                "maximum_state_skew_s": camera.get("maximum_state_skew_s"),
            }
        )
    return {
        "schema": "npa.sim2real.lerobot_policy_adapter.v1",
        "backend": "lerobot_smolvla",
        "policy_identity": policy.get("identity"),
        "robot_identity": policy.get("robot_identity"),
        "task_identity": policy.get("task_identity"),
        "normalization_identity": "",
        "measured_state_names": policy.get("measured_state_names"),
        "measured_state_units": policy.get("measured_state_units"),
        "state_feature_key": "observation.state",
        "action_feature_key": "action",
        "cameras": adapter_cameras,
        "action": policy.get("action"),
        "cartesian_controller_qualified": False,
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="npa-sim2real-interchange-") as directory:
        root = materialize(args.profile_uri, Path(directory) / "profile")
        result = build_preparation(root, run_id=args.run_id)
        write_json(args.output_uri, result, directory=Path(directory) / "output")
        if args.adapter_output_uri:
            write_json(
                args.adapter_output_uri,
                training_adapter(result),
                directory=Path(directory) / "adapter",
            )
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
            **(
                {"policy_adapter": args.adapter_output_uri}
                if args.adapter_output_uri
                else {}
            ),
        },
    )
    return result


def _external_seam(args: argparse.Namespace) -> dict[str, Any]:
    result = {
        "schema": "npa.sim2real.policy_interchange.external_seam.v1",
        "run_id": args.run_id,
        "status": "operator_seam_not_executed",
        "physical_robot_deployed": False,
        "required_before_robot_claim": [
            "selected embodiment and calibrated sensor profile",
            "controller and gripper safety qualification",
            "workspace and force safety review",
            "supervised physical task trials",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="npa-policy-external-") as directory:
        write_json(args.output_uri, result, directory=Path(directory))
    publish_component_record(
        component_schema=POLICY_INTERCHANGE_COMPONENT_SCHEMA,
        root_uri=args.component_root_uri,
        stage=12,
        name="policy_interchange_physical_robot_seam",
        tier="SEAM",
        evidence="Physical robot deployment remains an explicit operator seam.",
        artifacts={"seam": args.output_uri},
    )
    return result


def _artifact_preflight(args: argparse.Namespace) -> dict[str, Any]:
    match = _HF_SOURCE_RE.fullmatch(args.checkpoint_uri)
    if not match:
        raise InterchangeContractError(
            "base SmolVLA checkpoint must be hf://repo@40-hex-revision"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise InterchangeContractError("huggingface_hub is required") from exc
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    with tempfile.TemporaryDirectory(prefix="npa-smolvla-preflight-") as directory:
        work = Path(directory)
        base = work / "base"
        snapshot_download(
            repo_id=match.group("repo"),
            revision=match.group("revision"),
            token=os.environ.get("HF_TOKEN") or None,
            local_dir=base,
        )
        missing = [name for name in required if not (base / name).is_file()]
        if missing:
            raise InterchangeContractError(
                "exact SmolVLA snapshot lacks required files: " + ", ".join(missing)
            )
        files = [
            {
                "path": path.relative_to(base).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
            for path in sorted(base.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        ]
        dataset = materialize_lerobot_dataset(
            args.dataset_uri,
            work / "dataset",
            repo_id=args.dataset_repo_id,
        )
        adapter = read_json(args.adapter_uri, directory=work / "adapter")
        prepared = work / "prepared"
        prepared_receipt = prepare_smolvla_base_checkpoint(
            base,
            dataset_root=dataset,
            dataset_repo_id=args.dataset_repo_id,
            adapter=adapter,
            output_dir=prepared,
        )
        (prepared / "prepared-base-receipt.json").write_text(
            json.dumps(prepared_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (prepared / "checksums.json").write_text(
            json.dumps(tree_hashes(prepared), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        storage().upload_directory(
            str(prepared), args.prepared_checkpoint_uri, require_empty=True
        )
        materialize(args.prepared_checkpoint_uri, work / "prepared-readback")
        result = {
            "schema": "npa.sim2real.smolvla_artifact_preflight.v1",
            "checkpoint_uri": args.checkpoint_uri,
            "repository": match.group("repo"),
            "revision": match.group("revision"),
            "anonymous_access": not bool(os.environ.get("HF_TOKEN")),
            "files": files,
            "complete_snapshot_bytes_verified": True,
            "complete_required_bytes_verified": True,
            "license": "Apache-2.0",
            "prepared_checkpoint_uri": args.prepared_checkpoint_uri,
            "prepared_base": prepared_receipt,
        }
        write_json(args.output_uri, result, directory=work / "result")
    publish_component_record(
        component_schema=POLICY_INTERCHANGE_COMPONENT_SCHEMA,
        root_uri=args.component_root_uri,
        stage=4,
        name="smolvla_exact_artifact_preflight",
        tier="WORKS",
        evidence="Downloaded and hashed exact-revision base weights and processors.",
        artifacts={
            "preflight": args.output_uri,
            "revision": result["revision"],
            "prepared_checkpoint": args.prepared_checkpoint_uri,
        },
    )
    return result


def _copy_artifact(source: str, destination: str, work: Path) -> dict[str, Any]:
    target = work / Path(source).name
    target.parent.mkdir(parents=True, exist_ok=True)
    storage().download_file(source, str(target))
    payload = target.read_bytes()
    if not payload:
        raise InterchangeContractError(f"review artifact is empty: {source}")
    storage().upload_file(str(target), destination)
    return {
        "uri": destination,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _fresh_rate(report: Mapping[str, Any]) -> float:
    ablation = report.get("wrist_ablation")
    if not isinstance(ablation, Mapping):
        raise InterchangeContractError("closed-loop report lacks wrist interventions")
    modes = ablation.get("per_mode")
    if not isinstance(modes, Mapping) or not isinstance(modes.get("fresh"), Mapping):
        raise InterchangeContractError("closed-loop report lacks fresh wrist trials")
    return float(modes["fresh"].get("success_rate"))


def _finalize(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="npa-policy-finalize-") as directory:
        work = Path(directory)
        baseline = read_json(args.baseline_uri, directory=work / "baseline")
        adapted = read_json(args.adapted_uri, directory=work / "adapted")
        training = read_json(args.training_uri, directory=work / "training")
        reload = read_json(args.reload_uri, directory=work / "reload")
        external = read_json(args.external_seam_uri, directory=work / "external")
        expected = "npa.sim2real.policy_interchange.closed_loop_eval.v1"
        if baseline.get("schema") != expected or adapted.get("schema") != expected:
            raise InterchangeContractError(
                "closed-loop interchange reports are invalid"
            )
        before = baseline.get("checkpoint_provenance") or {}
        after = adapted.get("checkpoint_provenance") or {}
        same_protocol = bool(
            baseline.get("split") == adapted.get("split") == "gold"
            and (baseline.get("protocol") or {}).get("sha256")
            == (adapted.get("protocol") or {}).get("sha256")
            and baseline.get("episode_count") == adapted.get("episode_count")
            and before.get("source_commit") == after.get("source_commit")
            and before.get("image_digest") == after.get("image_digest")
        )
        if not same_protocol:
            raise InterchangeContractError(
                "baseline and adapted trials do not share an identical protocol/runtime"
            )
        if before.get("checkpoint_sha256") == after.get("checkpoint_sha256"):
            raise InterchangeContractError("adapted checkpoint is not distinct")
        if (
            reload.get("schema") != "npa.sim2real.smolvla_reload_validation.v1"
            or reload.get("status") != "native_checkpoint_reloaded"
            or (reload.get("interchange") or {}).get("checkpoint_sha256")
            != after.get("checkpoint_sha256")
            or training.get("selected_checkpoint_uri") != args.checkpoint_uri
        ):
            raise InterchangeContractError(
                "training, native reload and adapted service checkpoint differ"
            )
        before_rate, after_rate = _fresh_rate(baseline), _fresh_rate(adapted)
        copied = {
            "mp4": _copy_artifact(
                args.adapted_artifact_root_uri.rstrip("/") + "/rollouts.mp4",
                args.final_mp4_uri,
                work / "media",
            ),
            "rrd": _copy_artifact(
                args.adapted_artifact_root_uri.rstrip("/") + "/rollouts.rrd",
                args.final_rrd_uri,
                work / "media",
            ),
        }
        result = {
            "schema": "npa.sim2real.policy_interchange.final_report.v1",
            "run_id": args.run_id,
            "backend": "lerobot_smolvla",
            "task": "released_supported_surface_placement_v1",
            "comparison_protocol_identical": True,
            "baseline_fresh_success_rate": before_rate,
            "adapted_fresh_success_rate": after_rate,
            "measured_improvement": after_rate - before_rate,
            "improvement_claimed": after_rate > before_rate,
            "sampler_noise_coupling": "unavailable_for_native_smolvla",
            "cross_model_loss_ranking": False,
            "checkpoint": {
                "selected_uri": args.checkpoint_uri,
                "sha256": after.get("checkpoint_sha256"),
                "native_reload_verified": True,
            },
            "review_artifacts": copied,
            "physical_robot": external,
        }
        write_json(args.output_uri, result, directory=work / "output")
    publish_component_record(
        component_schema=POLICY_INTERCHANGE_COMPONENT_SCHEMA,
        root_uri=args.component_root_uri,
        stage=14,
        name="policy_interchange_finalize",
        tier="WORKS",
        evidence=(
            "Bound same-task repeated trials, selected native checkpoint and factual media."
        ),
        artifacts={
            "report": args.output_uri,
            "mp4": args.final_mp4_uri,
            "rrd": args.final_rrd_uri,
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
    prepare.add_argument("--adapter-output-uri", default="")
    external = commands.add_parser("external-seam")
    external.add_argument("--run-id", required=True)
    external.add_argument("--output-uri", required=True)
    external.add_argument("--component-root-uri", required=True)
    artifact = commands.add_parser("artifact-preflight")
    artifact.add_argument("--checkpoint-uri", required=True)
    artifact.add_argument("--dataset-uri", required=True)
    artifact.add_argument("--dataset-repo-id", required=True)
    artifact.add_argument("--adapter-uri", required=True)
    artifact.add_argument("--prepared-checkpoint-uri", required=True)
    artifact.add_argument("--output-uri", required=True)
    artifact.add_argument("--component-root-uri", required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--run-id", required=True)
    for name in (
        "baseline-uri",
        "adapted-uri",
        "training-uri",
        "reload-uri",
        "checkpoint-uri",
        "external-seam-uri",
        "adapted-artifact-root-uri",
        "final-mp4-uri",
        "final-rrd-uri",
        "output-uri",
        "component-root-uri",
    ):
        final.add_argument(f"--{name}", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = {
        "prepare": _prepare,
        "external-seam": _external_seam,
        "artifact-preflight": _artifact_preflight,
        "finalize": _finalize,
    }[args.command](args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
