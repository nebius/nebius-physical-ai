"""Isaac controller and camera bridge for typed Sim2Real policy interchange.

This module deliberately contains no Franka names or fixed action dimensions.
It translates one validated :class:`RobotSensorProfile` into the normalized
joint-position and binary parallel-jaw action used by the supported Isaac Lab
manipulation environments.  Native stages must still prove that the live
articulation and action-manager names match the sealed profile before applying
the returned action.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from npa.workflows.sim2real.interchange_contract import (
    InterchangeContractError,
    RigidObjectProfile,
    RobotSensorProfile,
    SurfacePlacementTask,
    Transform,
)


@dataclass(frozen=True)
class IsaacActionLayout:
    """Locations and physical parameters of two native Isaac action terms."""

    total_dimension: int
    arm_start: int
    gripper_start: int
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    default_arm_positions_rad: tuple[float, ...]
    arm_scales_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        dof = len(self.arm_joint_names)
        if dof < 1 or len(set(self.arm_joint_names)) != dof:
            raise InterchangeContractError("Isaac arm joint names must be unique")
        if not self.gripper_joint_names or len(set(self.gripper_joint_names)) != len(
            self.gripper_joint_names
        ):
            raise InterchangeContractError("Isaac gripper joint names must be unique")
        if (
            len(self.default_arm_positions_rad) != dof
            or len(self.arm_scales_rad) != dof
        ):
            raise InterchangeContractError(
                "Isaac defaults/scales must match the ordered arm joints"
            )
        if not all(math.isfinite(value) for value in self.default_arm_positions_rad):
            raise InterchangeContractError("Isaac arm defaults must be finite")
        if not all(
            math.isfinite(value) and value != 0.0 for value in self.arm_scales_rad
        ):
            raise InterchangeContractError(
                "Isaac arm scales must be finite and nonzero"
            )
        if (
            self.total_dimension < dof + 1
            or self.arm_start < 0
            or self.gripper_start < 0
            or self.arm_start + dof > self.total_dimension
            or self.gripper_start >= self.total_dimension
            or self.gripper_start in range(self.arm_start, self.arm_start + dof)
        ):
            raise InterchangeContractError(
                "Isaac action term offsets overlap or escape"
            )


@dataclass(frozen=True)
class TeacherWaypoint:
    """One geometry-derived Cartesian target for native teacher execution."""

    phase: str
    world_from_ee: Transform
    gripper_closed: bool


def surface_teacher_waypoints(
    prop: RigidObjectProfile, task: SurfacePlacementTask
) -> tuple[TeacherWaypoint, ...]:
    """Derive a released-surface plan from declared object frames.

    The object root is allowed to be offset from both the stable reference and
    the gripper-compatible grasp frame. The target names the desired reference
    point, so the destination root is solved explicitly instead of assuming a
    centered cube or guessing from rendered bounds.
    """

    grasp = compose_transform(prop.world_from_root, prop.root_from_grasp)
    reference = compose_transform(prop.world_from_root, prop.root_from_reference)
    destination_reference = Transform(
        task.target_world_xyz_m, reference.quaternion_xyzw
    )
    destination_root = compose_transform(
        destination_reference, inverse_transform(prop.root_from_reference)
    )
    destination_grasp = compose_transform(destination_root, prop.root_from_grasp)

    def raised(value: Transform, amount: float) -> Transform:
        x, y, z = value.translation_m
        return Transform((x, y, z + amount), value.quaternion_xyzw)

    lifted = max(task.approach_clearance_m, task.lift_delta_m)
    return (
        TeacherWaypoint("approach", raised(grasp, task.approach_clearance_m), False),
        TeacherWaypoint("contact", grasp, False),
        TeacherWaypoint("grasp", grasp, True),
        TeacherWaypoint("lift", raised(grasp, lifted), True),
        TeacherWaypoint("transport", raised(destination_grasp, lifted), True),
        TeacherWaypoint("lower", destination_grasp, True),
        TeacherWaypoint("release", destination_grasp, False),
        TeacherWaypoint(
            "retreat", raised(destination_grasp, task.retreat_distance_m), False
        ),
    )


def compose_transform(left: Transform, right: Transform) -> Transform:
    """Compose two validated XYZW rigid transforms."""

    rotated = _rotate_xyzw(left.quaternion_xyzw, right.translation_m)
    return Transform(
        tuple(left.translation_m[index] + rotated[index] for index in range(3)),
        _multiply_xyzw(left.quaternion_xyzw, right.quaternion_xyzw),
    )


def inverse_transform(value: Transform) -> Transform:
    """Invert one validated XYZW rigid transform."""

    x, y, z, w = value.quaternion_xyzw
    inverse_rotation = (-x, -y, -z, w)
    rotated = _rotate_xyzw(
        inverse_rotation, tuple(-item for item in value.translation_m)
    )
    return Transform(rotated, inverse_rotation)


def action_layout_from_live_isaac(
    native: Any, profile: RobotSensorProfile
) -> IsaacActionLayout:
    """Read the initialized action-manager layout without assuming an arm DOF."""

    manager = native.action_manager
    names = list(manager.active_terms)
    dimensions = [int(value) for value in manager.action_term_dim]
    if names.count("arm_action") != 1 or names.count("gripper_action") != 1:
        raise InterchangeContractError(
            "live Isaac task must expose one named arm and gripper action"
        )
    arm_index = names.index("arm_action")
    gripper_index = names.index("gripper_action")
    dof = len(profile.arm_joint_names)
    if dimensions[arm_index] != dof or dimensions[gripper_index] != 1:
        raise InterchangeContractError(
            "live Isaac action dimensions differ from the sealed robot profile"
        )
    arm = manager.get_term("arm_action")
    gripper = manager.get_term("gripper_action")
    _require_unclipped_arm_action(arm)
    arm_names = tuple(arm.IO_descriptor.joint_names)
    gripper_names = tuple(gripper.IO_descriptor.joint_names)
    return IsaacActionLayout(
        total_dimension=sum(dimensions),
        arm_start=sum(dimensions[:arm_index]),
        gripper_start=sum(dimensions[:gripper_index]),
        arm_joint_names=arm_names,
        gripper_joint_names=gripper_names,
        default_arm_positions_rad=_parameter_row(
            getattr(arm, "_offset", None), dof, "arm action offsets"
        ),
        arm_scales_rad=_parameter_row(
            getattr(arm, "_scale", None), dof, "arm action scales", scalar=True
        ),
    )


def _require_unclipped_arm_action(arm: Any) -> None:
    """Reject native clipping that would invalidate absolute target evidence."""

    configured = getattr(getattr(arm, "cfg", None), "clip", None)
    runtime = getattr(arm, "_clip", None)
    if configured is not None or runtime is not None:
        raise InterchangeContractError(
            "absolute joint interchange requires an unclipped affine Isaac action"
        )


def _controller_targets(
    robot: Any, profile: RobotSensorProfile
) -> tuple[list[float], list[float]]:
    """Read ordered physical targets from the live articulation controller."""

    arm_indices = [robot.joint_names.index(name) for name in profile.arm_joint_names]
    gripper_indices = [
        robot.joint_names.index(name) for name in profile.gripper.joint_names
    ]
    controller = robot.data.joint_pos_target
    if hasattr(controller, "torch"):
        controller = controller.torch
    arm = controller[0, arm_indices].detach().cpu().tolist()
    gripper = controller[0, gripper_indices].detach().cpu().tolist()
    return arm, gripper


def _require_applied_targets(
    target: Sequence[float],
    mapping: dict[str, Any],
    arm: list[float],
    gripper: list[float],
) -> None:
    """Prove the controller consumed both physical target components."""

    if not _close(arm, [float(value) for value in target[:-1]], tolerance=1.0e-5):
        raise InterchangeContractError(
            "live Isaac arm controller target differs from the typed policy action"
        )
    expected = list(mapping["physical_gripper_joint_targets"].values())
    if not _close(gripper, expected, tolerance=1.0e-5):
        raise InterchangeContractError(
            "live Isaac gripper target differs from the declared physical mapping"
        )


def _applied_control_receipt(
    *,
    binding: dict[str, Any],
    mapping: dict[str, Any],
    before: float,
    after: float,
    arm: list[float],
    gripper: list[float],
    terminated: Any,
    truncated: Any,
) -> dict[str, Any]:
    """Build the causal native-controller receipt after verification."""

    return {
        "schema": "npa.sim2real.isaac_applied_control.v1",
        "binding": binding,
        "mapping": mapping,
        "command_application_time_s": before,
        "resulting_state_time_s": after,
        "controller_arm_target_rad": arm,
        "controller_gripper_joint_targets": gripper,
        "terminated": bool(terminated[0]),
        "truncated": bool(truncated[0]),
    }


def apply_absolute_joint_target(
    env: Any,
    target: Sequence[float],
    *,
    profile: RobotSensorProfile,
) -> dict[str, Any]:
    """Apply one typed target and prove the live controller consumed it."""

    import torch

    native = env.unwrapped
    robot = native.scene["robot"]
    layout = action_layout_from_live_isaac(native, profile)
    binding = assert_live_robot_matches_profile(
        robot_joint_names=robot.joint_names,
        robot_body_names=robot.body_names,
        layout=layout,
        profile=profile,
    )
    rendered, mapping = render_absolute_joint_action(
        target, profile=profile, layout=layout
    )
    before, after, terminated, truncated = _step_absolute_action(
        env,
        native,
        torch.as_tensor([rendered], dtype=torch.float32, device=native.device),
    )
    arm_targets, gripper_targets = _controller_targets(robot, profile)
    _require_applied_targets(target, mapping, arm_targets, gripper_targets)
    return _applied_control_receipt(
        binding=binding,
        mapping=mapping,
        before=before,
        after=after,
        arm=arm_targets,
        gripper=gripper_targets,
        terminated=terminated,
        truncated=truncated,
    )


def _step_absolute_action(
    env: Any, native: Any, action: Any
) -> tuple[float, float, Any, Any]:
    """Apply one native action and verify the measured controller cadence."""

    before = _simulator_time(native)
    _, _, terminated, truncated, _ = env.step(action)
    after = _simulator_time(native)
    if abs((after - before) - float(native.step_dt)) > 1.0e-6:
        raise InterchangeContractError(
            "native controller cadence differs from the measured simulator step"
        )
    return before, after, terminated, truncated


def assert_live_robot_matches_profile(
    *,
    robot_joint_names: Sequence[str],
    robot_body_names: Sequence[str],
    layout: IsaacActionLayout,
    profile: RobotSensorProfile,
) -> dict[str, Any]:
    """Fail before control when a live articulation differs from sealed morphology."""

    joints = tuple(robot_joint_names)
    bodies = tuple(robot_body_names)
    missing_joints, missing_bodies = _missing_live_names(joints, bodies, profile)
    if missing_joints or missing_bodies:
        raise InterchangeContractError(
            "live Isaac articulation lacks sealed joints or links: "
            f"joints={missing_joints}, links={missing_bodies}"
        )
    if layout.arm_joint_names != profile.arm_joint_names:
        raise InterchangeContractError(
            "live Isaac arm action order differs from the sealed policy order"
        )
    if layout.gripper_joint_names != profile.gripper.joint_names:
        raise InterchangeContractError(
            "live Isaac gripper action order differs from the sealed physical mapping"
        )
    return {
        "schema": "npa.sim2real.isaac_robot_binding.v1",
        "robot_identity": profile.identity,
        "arm_joint_names": list(layout.arm_joint_names),
        "gripper_joint_names": list(layout.gripper_joint_names),
        "base_link": profile.base_link,
        "ee_link": profile.ee_link,
        "arm_dof": len(profile.arm_joint_names),
        "action_dimension": layout.total_dimension,
        "exact_order_verified": True,
    }


def _missing_live_names(
    joints: tuple[str, ...], bodies: tuple[str, ...], profile: RobotSensorProfile
) -> tuple[list[str], list[str]]:
    """Return sealed articulation names absent from the initialized task."""

    missing_joints = [
        name
        for name in (*profile.arm_joint_names, *profile.gripper.joint_names)
        if name not in joints
    ]
    missing_bodies = [
        name
        for name in (profile.base_link, profile.ee_link, *profile.gripper.finger_links)
        if name not in bodies
    ]
    return missing_joints, missing_bodies


def _normalized_arm_target(
    values: tuple[float, ...], profile: RobotSensorProfile, layout: IsaacActionLayout
) -> list[float]:
    """Validate physical joint limits and return the native affine action."""

    normalized: list[float] = []
    rows = zip(
        values[:-1],
        profile.joint_limits_rad,
        layout.default_arm_positions_rad,
        layout.arm_scales_rad,
        strict=True,
    )
    for index, (value, limits, default, scale) in enumerate(rows):
        lower, upper = limits
        if value < lower - 1.0e-9 or value > upper + 1.0e-9:
            raise InterchangeContractError(
                f"policy joint {profile.arm_joint_names[index]!r} is outside physical limits"
            )
        normalized.append((value - default) / scale)
    return normalized


def _binary_gripper_target(
    score: float, profile: RobotSensorProfile
) -> tuple[bool, float, tuple[float, ...]]:
    """Map the declared semantic scalar to one binary native command."""

    low, high = sorted((profile.gripper.model_open, profile.gripper.model_closed))
    if score < low - 1.0e-9 or score > high + 1.0e-9:
        raise InterchangeContractError(
            "policy gripper scalar is outside declared endpoints"
        )
    midpoint = (profile.gripper.model_open + profile.gripper.model_closed) / 2.0
    increases = profile.gripper.model_closed > profile.gripper.model_open
    close = score >= midpoint if increases else score <= midpoint
    binary = -1.0 if close else 1.0
    endpoint = profile.gripper.model_closed if close else profile.gripper.model_open
    return close, binary, profile.gripper.model_to_joint_positions(endpoint)


def _action_mapping_receipt(
    *,
    profile: RobotSensorProfile,
    values: tuple[float, ...],
    normalized: list[float],
    score: float,
    close: bool,
    binary: float,
    physical: tuple[float, ...],
    rendered: list[float],
) -> dict[str, Any]:
    """Record the semantic-to-native action translation."""

    return {
        "schema": "npa.sim2real.isaac_action_mapping.v1",
        "robot_identity": profile.identity,
        "representation": "absolute_joint_position",
        "policy_action_names": list(profile.action_names),
        "policy_target": list(values),
        "normalized_arm_action": normalized,
        "model_gripper_score": score,
        "binary_gripper_command": "close" if close else "open",
        "native_gripper_action": binary,
        "physical_gripper_joint_targets": dict(
            zip(profile.gripper.joint_names, physical, strict=True)
        ),
        "native_action": rendered,
    }


def render_absolute_joint_action(
    target: Sequence[float],
    *,
    profile: RobotSensorProfile,
    layout: IsaacActionLayout,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Translate an exact named policy target to one native Isaac action.

    The native parallel-jaw action is binary: positive opens and negative
    closes.  A continuous model scalar is therefore thresholded at the exact
    midpoint of its declared open/closed endpoints, and both the scalar and
    resulting physical joint endpoints are retained in the receipt.
    """

    values = _validated_policy_target(target, profile, layout)
    normalized_arm = _normalized_arm_target(values, profile, layout)
    rendered = [0.0] * layout.total_dimension
    rendered[layout.arm_start : layout.arm_start + len(normalized_arm)] = normalized_arm
    score = values[-1]
    close, binary_native, physical = _binary_gripper_target(score, profile)
    rendered[layout.gripper_start] = binary_native
    receipt = _action_mapping_receipt(
        profile=profile,
        values=values,
        normalized=normalized_arm,
        score=score,
        close=close,
        binary=binary_native,
        physical=physical,
        rendered=rendered,
    )
    return tuple(rendered), receipt


def _validated_policy_target(
    target: Sequence[float], profile: RobotSensorProfile, layout: IsaacActionLayout
) -> tuple[float, ...]:
    """Validate one exact ordered policy target before native translation."""

    expected = len(profile.arm_joint_names) + 1
    if len(target) != expected:
        raise InterchangeContractError(
            f"policy action must contain exactly {expected} values for this robot"
        )
    values = tuple(float(value) for value in target)
    if not all(math.isfinite(value) for value in values):
        raise InterchangeContractError("policy action must contain finite values")
    if layout.arm_joint_names != profile.arm_joint_names:
        raise InterchangeContractError("Isaac arm action order does not match policy")
    if layout.gripper_joint_names != profile.gripper.joint_names:
        raise InterchangeContractError("Isaac gripper mapping does not match policy")
    return values


def camera_world_pose(
    *,
    link_position_m: Sequence[float],
    link_quaternion_xyzw: Sequence[float],
    mount_from_camera: Transform,
) -> Transform:
    """Compose a declared camera mount with a measured live link pose."""

    position = _finite_tuple(link_position_m, 3, "link position")
    quaternion = _unit_quaternion(link_quaternion_xyzw, "link quaternion")
    mount_position = mount_from_camera.translation_m
    rotated = _rotate_xyzw(quaternion, mount_position)
    return Transform(
        tuple(position[index] + rotated[index] for index in range(3)),
        _multiply_xyzw(quaternion, mount_from_camera.quaternion_xyzw),
    )


def measured_policy_state(native: Any, profile: RobotSensorProfile) -> dict[str, float]:
    """Read ordered arm state plus one semantic gripper value from physical joints."""

    robot = native.scene["robot"]
    positions = robot.data.joint_pos
    if hasattr(positions, "torch"):
        positions = positions.torch
    arm_indices = [robot.joint_names.index(name) for name in profile.arm_joint_names]
    gripper_indices = [
        robot.joint_names.index(name) for name in profile.gripper.joint_names
    ]
    arm = positions[0, arm_indices].detach().cpu().tolist()
    fingers = positions[0, gripper_indices].detach().cpu().tolist()
    width = profile.gripper.joint_positions_to_width(fingers)
    values = [*arm, profile.gripper.width_to_model(width)]
    if len(values) != len(profile.measured_state_names):
        raise InterchangeContractError(
            "live measured state dimension differs from the sealed profile"
        )
    return {
        name: float(value)
        for name, value in zip(profile.measured_state_names, values, strict=True)
    }


def live_camera_world_pose(
    native: Any, profile: RobotSensorProfile, key: str
) -> Transform:
    """Resolve a configured camera against the current articulation state."""

    matches = [camera for camera in profile.cameras if camera.key == key]
    if len(matches) != 1:
        raise InterchangeContractError("camera key must resolve exactly once")
    camera = matches[0]
    if camera.mount_link == "world":
        return camera.mount_from_camera
    robot = native.scene["robot"]
    if camera.mount_link not in robot.body_names:
        raise InterchangeContractError(
            "live articulation lacks the declared camera mount link"
        )
    index = robot.body_names.index(camera.mount_link)
    positions = robot.data.body_link_pos_w
    quaternions = robot.data.body_link_quat_w
    if hasattr(positions, "torch"):
        positions = positions.torch
    if hasattr(quaternions, "torch"):
        quaternions = quaternions.torch
    return camera_world_pose(
        link_position_m=positions[0, index].detach().cpu().tolist(),
        link_quaternion_xyzw=quaternions[0, index].detach().cpu().tolist(),
        mount_from_camera=camera.mount_from_camera,
    )


def _finite_tuple(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if len(values) != size:
        raise InterchangeContractError(f"{name} must contain {size} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise InterchangeContractError(f"{name} must be finite")
    return result


def _parameter_row(
    value: Any, size: int, name: str, *, scalar: bool = False
) -> tuple[float, ...]:
    if value is None:
        raise InterchangeContractError(f"live Isaac term does not expose {name}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, (int, float)):
        values = [float(value)] * size
    elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        values = value[0]
    elif isinstance(value, list):
        values = value
    else:
        raise InterchangeContractError(f"unsupported live Isaac {name} shape")
    if scalar and len(values) == 1:
        values = values * size
    if len(values) != size or not all(
        math.isfinite(float(item)) and (not scalar or float(item) != 0.0)
        for item in values
    ):
        raise InterchangeContractError(f"live Isaac {name} differ from arm DOF")
    return tuple(float(item) for item in values)


def _simulator_time(native: Any) -> float:
    return float(native.common_step_counter) * float(native.step_dt)


def _close(
    actual: Sequence[float], expected: Sequence[float], *, tolerance: float
) -> bool:
    return len(actual) == len(expected) and all(
        math.isfinite(float(left)) and abs(float(left) - float(right)) <= tolerance
        for left, right in zip(actual, expected, strict=True)
    )


def _unit_quaternion(values: Sequence[float], name: str) -> tuple[float, ...]:
    quaternion = _finite_tuple(values, 4, name)
    norm = math.sqrt(sum(value * value for value in quaternion))
    if abs(norm - 1.0) > 1.0e-6:
        raise InterchangeContractError(f"{name} must be unit length")
    return quaternion


def _multiply_xyzw(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _rotate_xyzw(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, ...]:
    x, y, z, w = quaternion
    vx, vy, vz = vector
    # q * v * q^-1, expanded to avoid a runtime NumPy dependency.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
