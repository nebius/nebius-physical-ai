"""Two explicit policy backends for the model-neutral Sim2Real seam.

This is deliberately a closed catalog, not a dynamic plugin registry.  Each
backend advertises only operations implemented by its native runtime.  A new
backend must add its own export/train/reload/serve adapter and qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from npa.workflows.sim2real.interchange_contract import (
    InterchangeContractError,
    PolicyAction,
    PolicyCapabilities,
    PolicyProfile,
    RobotSensorProfile,
    SurfacePlacementTask,
)


BackendName = Literal["openpi_pi05", "lerobot_smolvla"]


@dataclass(frozen=True)
class BackendAdapterContract:
    """Native runtime and stage capabilities for one bounded backend."""

    name: BackendName
    runtime_family: str
    pinned_version: str
    observation_adapter: str
    transport: str
    dataset_format: str
    training_adapter: str
    reload_adapter: str
    serving_adapter: str
    deterministic_noise_protocol: str | None
    repeated_paired_trials_required: bool
    language_conditioned: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runtime_family": self.runtime_family,
            "pinned_version": self.pinned_version,
            "observation_adapter": self.observation_adapter,
            "transport": self.transport,
            "dataset_format": self.dataset_format,
            "training_adapter": self.training_adapter,
            "reload_adapter": self.reload_adapter,
            "serving_adapter": self.serving_adapter,
            "deterministic_noise_protocol": self.deterministic_noise_protocol,
            "repeated_paired_trials_required": self.repeated_paired_trials_required,
            "language_conditioned": self.language_conditioned,
        }


BACKENDS: dict[BackendName, BackendAdapterContract] = {
    "openpi_pi05": BackendAdapterContract(
        name="openpi_pi05",
        runtime_family="Physical Intelligence OpenPI pi0.5",
        pinned_version="repository-pinned-openpi-commit",
        observation_adapter="droid_aliases_and_exact_7d_joint_state",
        transport="authenticated_binary_msgpack_websocket",
        dataset_format="openpi_npz_with_train_only_normalization",
        training_adapter="upstream_jax_lora",
        reload_adapter="openpi_checkpoint_reload_and_all_heldout_loss",
        serving_adapter="private_openpi_clusterip",
        deterministic_noise_protocol="npa.openpi.pi05-explicit-noise.v1",
        repeated_paired_trials_required=False,
        language_conditioned=True,
    ),
    "lerobot_smolvla": BackendAdapterContract(
        name="lerobot_smolvla",
        runtime_family="Hugging Face LeRobot SmolVLA",
        pinned_version="0.6.0-d6-extras-20260912",
        observation_adapter="native_smolvla_preprocessor",
        transport="private_json_http_with_hash_bound_rgb8",
        dataset_format="lerobot_dataset_v3",
        training_adapter="native_lerobot_train_smolvla",
        reload_adapter="smolvla_from_pretrained_and_checkpoint_local_processors",
        serving_adapter="private_lerobot_fastapi",
        deterministic_noise_protocol=None,
        repeated_paired_trials_required=True,
        language_conditioned=True,
    ),
}


def backend_contract(name: str) -> BackendAdapterContract:
    """Resolve one supported backend and reject undeclared fallbacks."""

    try:
        return BACKENDS[name]  # type: ignore[index]
    except KeyError as exc:
        raise InterchangeContractError(
            f"unsupported policy backend {name!r}; choose one of {sorted(BACKENDS)}"
        ) from exc


def _manifest_action(action: Mapping[str, Any]) -> PolicyAction:
    """Parse the declared SmolVLA action contract."""

    return PolicyAction(
        representation=str(action.get("representation") or ""),
        names=tuple(action.get("names") or ()),
        units=tuple(action.get("units") or ()),
        frame=str(action.get("frame") or "joint"),
        gripper_semantics=str(action.get("gripper_semantics") or ""),
        horizon=int(action.get("horizon") or 0),
        control_hz=float(action.get("control_hz") or 0.0),
        execute_prefix=int(action.get("execute_prefix") or 0),
    )


def _smolvla_capabilities(hardware_topologies: tuple[str, ...]) -> PolicyCapabilities:
    """Declare only capabilities implemented by the pinned SmolVLA runtime."""

    return PolicyCapabilities(
        inference=True,
        native_training=True,
        checkpoint_reload=True,
        model_loss=True,
        action_metrics=True,
        deterministic_noise_coupling=False,
        repeated_paired_trials=True,
        hardware_topologies=hardware_topologies,
    )


def smolvla_policy_profile(
    manifest: Mapping[str, Any],
    *,
    robot: RobotSensorProfile,
    task: SurfacePlacementTask,
    hardware_topologies: tuple[str, ...],
) -> PolicyProfile:
    """Project a checkpoint-bound SmolVLA adapter manifest into the neutral API."""

    if (
        manifest.get("schema") != "npa.sim2real.lerobot_policy_adapter.v1"
        or manifest.get("backend") != "lerobot_smolvla"
    ):
        raise InterchangeContractError("invalid SmolVLA adapter manifest")
    action = manifest.get("action")
    cameras = manifest.get("cameras")
    if not isinstance(action, Mapping) or not isinstance(cameras, list):
        raise InterchangeContractError("SmolVLA manifest lacks action/camera adapters")
    profile = PolicyProfile(
        backend="lerobot_smolvla",
        identity=str(manifest.get("policy_identity") or ""),
        checkpoint_sha256=str(manifest.get("checkpoint_sha256") or ""),
        robot_identity=str(manifest.get("robot_identity") or ""),
        task_identity=str(manifest.get("task_identity") or ""),
        camera_keys=tuple(str(row.get("key") or "") for row in cameras),
        measured_state_names=tuple(manifest.get("measured_state_names") or ()),
        measured_state_units=tuple(manifest.get("measured_state_units") or ()),
        action=_manifest_action(action),
        capabilities=_smolvla_capabilities(hardware_topologies),
        normalization_identity=str(manifest.get("normalization_identity") or ""),
    )
    if (
        profile.robot_identity != robot.identity
        or profile.task_identity != task.identity
    ):
        raise InterchangeContractError(
            "SmolVLA manifest robot/task binding does not match the run"
        )
    return profile


def _require_pi05_franka(robot: RobotSensorProfile) -> None:
    """Reject kinematics outside the retained DROID/Franka profile."""

    if len(robot.arm_joint_names) != 7 or len(robot.gripper.joint_names) != 2:
        raise InterchangeContractError(
            "pi0.5 DROID profile requires seven arm joints and two Franka fingers"
        )
    if abs(robot.gripper.maximum_width_m - 0.08) > 1.0e-9:
        raise InterchangeContractError(
            "pi0.5 DROID profile requires the calibrated 80 mm Franka aperture"
        )


def _pi05_capabilities(hardware_topologies: tuple[str, ...]) -> PolicyCapabilities:
    """Declare the retained OpenPI training and replay capabilities."""

    return PolicyCapabilities(
        inference=True,
        native_training=True,
        checkpoint_reload=True,
        model_loss=True,
        action_metrics=True,
        deterministic_noise_coupling=True,
        repeated_paired_trials=False,
        hardware_topologies=hardware_topologies,
    )


def openpi_pi05_policy_profile(
    *,
    robot: RobotSensorProfile,
    task: SurfacePlacementTask,
    policy_identity: str,
    checkpoint_sha256: str,
    normalization_identity: str,
    hardware_topologies: tuple[str, ...],
) -> PolicyProfile:
    """Build the retained exact pi0.5 profile without pretending other kinematics."""

    _require_pi05_franka(robot)
    return PolicyProfile(
        backend="openpi_pi05",
        identity=policy_identity,
        checkpoint_sha256=checkpoint_sha256,
        robot_identity=robot.identity,
        task_identity=task.identity,
        camera_keys=tuple(camera.key for camera in robot.cameras),
        measured_state_names=robot.measured_state_names,
        measured_state_units=robot.measured_state_units,
        action=PolicyAction(
            representation="absolute_joint_position",
            names=robot.action_names,
            units=("rad",) * 7 + ("normalized_closed",),
            frame="joint",
            gripper_semantics="normalized_closed",
            horizon=15,
            control_hz=15.0,
            execute_prefix=1,
        ),
        capabilities=_pi05_capabilities(hardware_topologies),
        normalization_identity=normalization_identity,
    )
