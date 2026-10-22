"""Typed embodiment, sensor, object, task, and policy interchange contracts.

The canonical Sim2Real graph is intentionally model-neutral.  This module keeps
that property at the bytes exchanged between simulation and a learned policy:
robot morphology, calibrated sensors, rigid-object geometry, and policy actions
are validated before a GPU job or controller can consume them.  Model-specific
aliases and transports belong in adapters, not in this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence


SCHEMA = "npa.sim2real.policy_interchange.v1"
CAMERA_ROLES = frozenset({"exterior", "wrist"})
IMAGE_LAYOUTS = frozenset({"HWC", "CHW"})
ACTION_REPRESENTATIONS = frozenset(
    {"absolute_joint_position", "delta_joint_position", "ee_pose"}
)
GRIPPER_SEMANTICS = frozenset(
    {"width_m", "normalized_open", "normalized_closed", "binary_open"}
)
RIGID_OBJECT_KINDS = frozenset({"rigid"})
GRASP_AXES = frozenset({"x", "y", "z"})


class InterchangeContractError(ValueError):
    """Raised before incompatible observations, actions, or assets are launched."""


def _finite(values: Sequence[float], *, size: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != size:
        raise InterchangeContractError(f"{name} must contain exactly {size} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise InterchangeContractError(f"{name} must contain only finite values")
    return result


def _names(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if not result or any(not value for value in result):
        raise InterchangeContractError(f"{name} must contain non-empty names")
    if len(result) != len(set(result)):
        raise InterchangeContractError(f"{name} must contain unique ordered names")
    return result


def _sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _lowercase_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise InterchangeContractError(f"{name} must be lowercase SHA-256")


def _validate_gripper_endpoints(mapping: "GripperMapping") -> None:
    if mapping.model_semantics not in GRIPPER_SEMANTICS:
        raise InterchangeContractError("unsupported gripper model semantics")
    if not all(math.isfinite(v) for v in (mapping.model_open, mapping.model_closed)):
        raise InterchangeContractError("model gripper endpoints must be finite")
    if mapping.model_open == mapping.model_closed:
        raise InterchangeContractError("model gripper endpoints must differ")
    expected = {
        "normalized_open": (1.0, 0.0),
        "normalized_closed": (0.0, 1.0),
        "binary_open": (1.0, 0.0),
    }.get(mapping.model_semantics)
    endpoints = (mapping.model_open, mapping.model_closed)
    if expected is not None and any(
        abs(actual - required) > 1.0e-9
        for actual, required in zip(endpoints, expected, strict=True)
    ):
        raise InterchangeContractError(
            f"{mapping.model_semantics} gripper endpoints must be "
            f"open={expected[0]:g}, closed={expected[1]:g}"
        )


def _validate_joint_limits(limits: Sequence[Sequence[float]], *, size: int) -> None:
    if len(limits) != size:
        raise InterchangeContractError(
            "one physical joint limit is required per arm joint"
        )
    for lower, upper in limits:
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise InterchangeContractError(
                "arm joint limits must be finite and ordered"
            )


def _validate_camera_set(cameras: Sequence["CameraView"]) -> None:
    roles = [camera.role for camera in cameras]
    if roles.count("exterior") < 1 or roles.count("wrist") < 1:
        raise InterchangeContractError("one exterior and one wrist camera are required")
    keys = [camera.key for camera in cameras]
    if len(keys) != len(set(keys)):
        raise InterchangeContractError("camera keys must be unique")


def _validate_rigid_material(profile: "RigidObjectProfile") -> None:
    declarations = (
        profile.collision_declared,
        profile.inertia_declared,
        profile.material_declared,
    )
    if not all(declarations):
        raise InterchangeContractError(
            "rigid props require explicit collision, inertia, and material data"
        )
    values = (profile.static_friction, profile.dynamic_friction, profile.restitution)
    if not all(math.isfinite(value) for value in values):
        raise InterchangeContractError("object material values are invalid")
    if not 0.0 <= profile.dynamic_friction <= profile.static_friction:
        raise InterchangeContractError("object material values are invalid")
    if not 0.0 <= profile.restitution <= 1.0:
        raise InterchangeContractError("object material values are invalid")


@dataclass(frozen=True)
class Transform:
    """One explicit parent-to-child rigid transform in XYZW convention."""

    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        translation = _finite(self.translation_m, size=3, name="translation_m")
        quaternion = _finite(self.quaternion_xyzw, size=4, name="quaternion_xyzw")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if abs(norm - 1.0) > 1.0e-6:
            raise InterchangeContractError("quaternion_xyzw must be unit length")
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "quaternion_xyzw", quaternion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation_m": list(self.translation_m),
            "quaternion_xyzw": list(self.quaternion_xyzw),
        }


IDENTITY_TRANSFORM = Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


@dataclass(frozen=True)
class CameraView:
    """Calibrated causal camera view presented to a policy."""

    key: str
    role: Literal["exterior", "wrist"]
    mount_link: str
    mount_from_camera: Transform
    intrinsics: tuple[float, float, float, float]
    image_size: tuple[int, int]
    layout: Literal["HWC", "CHW"]
    mask_key: str
    capture_timestamp_key: str
    state_timestamp_key: str
    maximum_state_skew_s: float

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.mount_link.strip():
            raise InterchangeContractError("camera key and mount_link are required")
        if self.role not in CAMERA_ROLES:
            raise InterchangeContractError(f"unsupported camera role {self.role!r}")
        if self.role == "wrist" and self.mount_link == "world":
            raise InterchangeContractError("a wrist camera must name its moving link")
        _finite(self.intrinsics, size=4, name=f"{self.key} intrinsics")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise InterchangeContractError("camera image dimensions must be positive")
        if self.layout not in IMAGE_LAYOUTS:
            raise InterchangeContractError(f"unsupported image layout {self.layout!r}")
        for value, label in (
            (self.mask_key, "mask_key"),
            (self.capture_timestamp_key, "capture_timestamp_key"),
            (self.state_timestamp_key, "state_timestamp_key"),
        ):
            if not value.strip():
                raise InterchangeContractError(f"camera {label} is required")
        skew = float(self.maximum_state_skew_s)
        if not math.isfinite(skew) or skew < 0.0:
            raise InterchangeContractError("maximum camera/state skew must be finite")

    def validate_timestamps(self, *, capture_s: float, state_s: float) -> None:
        """Reject a future or stale image relative to its measured state."""

        capture = float(capture_s)
        state = float(state_s)
        if not math.isfinite(capture) or not math.isfinite(state):
            raise InterchangeContractError("camera and state timestamps must be finite")
        if capture > state + 1.0e-9:
            raise InterchangeContractError(
                "camera capture cannot follow measured state"
            )
        if state - capture > self.maximum_state_skew_s + 1.0e-9:
            raise InterchangeContractError(
                f"camera {self.key!r} is stale relative to measured state"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "mount_link": self.mount_link,
            "mount_from_camera": self.mount_from_camera.to_dict(),
            "intrinsics_fx_fy_cx_cy": list(self.intrinsics),
            "image_size_wh": list(self.image_size),
            "layout": self.layout,
            "mask_key": self.mask_key,
            "capture_timestamp_key": self.capture_timestamp_key,
            "state_timestamp_key": self.state_timestamp_key,
            "maximum_state_skew_s": self.maximum_state_skew_s,
        }


@dataclass(frozen=True)
class GripperMapping:
    """Measured parallel-jaw aperture and model-facing scalar semantics."""

    joint_names: tuple[str, ...]
    joint_units: tuple[str, ...]
    finger_links: tuple[str, ...]
    open_joint_positions: tuple[float, ...]
    closed_joint_positions: tuple[float, ...]
    maximum_width_m: float
    model_semantics: Literal[
        "width_m", "normalized_open", "normalized_closed", "binary_open"
    ]
    model_open: float
    model_closed: float

    def __post_init__(self) -> None:
        joints = _names(self.joint_names, name="gripper joint_names")
        fingers = _names(self.finger_links, name="gripper finger_links")
        if len(fingers) != 2:
            raise InterchangeContractError(
                "surface placement currently supports exactly two physical "
                "parallel-jaw fingers"
            )
        object.__setattr__(self, "joint_names", joints)
        object.__setattr__(self, "finger_links", fingers)
        if len(self.joint_units) != len(joints) or any(
            not str(unit).strip() for unit in self.joint_units
        ):
            raise InterchangeContractError(
                "one physical unit is required per gripper joint"
            )
        _finite(self.open_joint_positions, size=len(joints), name="gripper open")
        _finite(self.closed_joint_positions, size=len(joints), name="gripper closed")
        width = float(self.maximum_width_m)
        if not math.isfinite(width) or width <= 0.0:
            raise InterchangeContractError("maximum gripper width must be positive")
        _validate_gripper_endpoints(self)

    def width_to_model(self, width_m: float) -> float:
        width = float(width_m)
        if not math.isfinite(width) or not 0.0 <= width <= self.maximum_width_m:
            raise InterchangeContractError("measured gripper width is out of range")
        ratio = width / self.maximum_width_m
        return self.model_closed + ratio * (self.model_open - self.model_closed)

    def model_to_joint_positions(self, value: float) -> tuple[float, ...]:
        scalar = float(value)
        low, high = sorted((self.model_open, self.model_closed))
        if not math.isfinite(scalar) or not low <= scalar <= high:
            raise InterchangeContractError("model gripper action is out of range")
        ratio = (scalar - self.model_closed) / (self.model_open - self.model_closed)
        return tuple(
            closed + ratio * (opened - closed)
            for opened, closed in zip(
                self.open_joint_positions,
                self.closed_joint_positions,
                strict=True,
            )
        )

    def model_value_is_closed(self, value: float) -> bool:
        """Classify aperture state without assuming either scalar direction."""

        scalar = float(value)
        low, high = sorted((self.model_open, self.model_closed))
        if not math.isfinite(scalar) or not low <= scalar <= high:
            raise InterchangeContractError("model gripper state is out of range")
        return abs(scalar - self.model_closed) < abs(scalar - self.model_open)

    def joint_positions_to_width(self, values: Sequence[float]) -> float:
        """Recover aperture from each measured parallel-jaw joint contribution.

        Command targets follow the declared coupled endpoint trajectory, but
        measured fingers can legitimately differ under dynamics and contact.
        Each joint therefore contributes its own normalized physical travel to
        the measured aperture.  Endpoint validation remains per joint so an
        invalid measurement cannot be hidden by averaging it with another.
        """

        positions = _finite(
            values, size=len(self.joint_names), name="measured gripper positions"
        )
        ratios: list[float] = []
        for value, opened, closed in zip(
            positions,
            self.open_joint_positions,
            self.closed_joint_positions,
            strict=True,
        ):
            span = opened - closed
            if abs(span) < 1.0e-12:
                raise InterchangeContractError(
                    "gripper joint endpoints do not define an aperture mapping"
                )
            ratios.append((value - closed) / span)
        if any(not 0.0 <= ratio <= 1.0 for ratio in ratios):
            raise InterchangeContractError(
                "measured gripper positions are outside declared endpoints"
            )
        return (sum(ratios) / len(ratios)) * self.maximum_width_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "joint_units": list(self.joint_units),
            "finger_links": list(self.finger_links),
            "open_joint_positions": list(self.open_joint_positions),
            "closed_joint_positions": list(self.closed_joint_positions),
            "maximum_width_m": self.maximum_width_m,
            "model_semantics": self.model_semantics,
            "model_open": self.model_open,
            "model_closed": self.model_closed,
        }


@dataclass(frozen=True)
class RobotSensorProfile:
    """Resolved fixed-base arm morphology and calibrated policy sensors."""

    identity: str
    bundle_sha256: str
    source_sha256: str
    base_link: str
    ee_link: str
    arm_joint_names: tuple[str, ...]
    measured_state_names: tuple[str, ...]
    measured_state_units: tuple[str, ...]
    joint_limits_rad: tuple[tuple[float, float], ...]
    home_positions_rad: tuple[float, ...]
    controller_representation: Literal[
        "absolute_joint_position", "delta_joint_position", "ee_pose"
    ]
    gripper: GripperMapping
    cameras: tuple[CameraView, ...]
    fixed_base: bool = True

    def __post_init__(self) -> None:
        if (
            not self.identity.strip()
            or not self.base_link.strip()
            or not self.ee_link.strip()
        ):
            raise InterchangeContractError(
                "robot identity, base_link and ee_link are required"
            )
        _lowercase_sha256(self.bundle_sha256, name="robot bundle digest")
        _lowercase_sha256(self.source_sha256, name="robot source digest")
        arm = _names(self.arm_joint_names, name="arm_joint_names")
        measured = _names(self.measured_state_names, name="measured_state_names")
        if measured[: len(arm)] != arm or len(measured) != len(arm) + 1:
            raise InterchangeContractError(
                "measured_state_names must equal the ordered arm joints followed "
                "by one semantic gripper value"
            )
        expected_units = ("rad",) * len(arm) + (self.gripper.model_semantics,)
        if self.measured_state_units != expected_units:
            raise InterchangeContractError(
                "measured_state_units must declare radians for ordered arm joints "
                "and the gripper model semantics for the final value"
            )
        _validate_joint_limits(self.joint_limits_rad, size=len(arm))
        _finite(self.home_positions_rad, size=len(arm), name="home_positions_rad")
        if self.controller_representation not in ACTION_REPRESENTATIONS:
            raise InterchangeContractError("unsupported controller representation")
        if not self.fixed_base:
            raise InterchangeContractError(
                "only fixed-base arms are currently supported"
            )
        _validate_camera_set(self.cameras)

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.arm_joint_names + ("gripper",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "bundle_sha256": self.bundle_sha256,
            "source_sha256": self.source_sha256,
            "fixed_base": self.fixed_base,
            "base_link": self.base_link,
            "ee_link": self.ee_link,
            "arm_joint_names": list(self.arm_joint_names),
            "measured_state_names": list(self.measured_state_names),
            "measured_state_units": list(self.measured_state_units),
            "joint_limits_rad": [list(limit) for limit in self.joint_limits_rad],
            "home_positions_rad": list(self.home_positions_rad),
            "controller_representation": self.controller_representation,
            "gripper": self.gripper.to_dict(),
            "cameras": [camera.to_dict() for camera in self.cameras],
        }


def _gripper_mapping_from_dict(value: Mapping[str, Any]) -> GripperMapping:
    return GripperMapping(
        joint_names=tuple(value.get("joint_names") or ()),
        joint_units=tuple(value.get("joint_units") or ()),
        finger_links=tuple(value.get("finger_links") or ()),
        open_joint_positions=tuple(value.get("open_joint_positions") or ()),
        closed_joint_positions=tuple(value.get("closed_joint_positions") or ()),
        maximum_width_m=float(value.get("maximum_width_m", math.nan)),
        model_semantics=str(value.get("model_semantics") or ""),
        model_open=float(value.get("model_open", math.nan)),
        model_closed=float(value.get("model_closed", math.nan)),
    )


def _camera_view_from_dict(value: Mapping[str, Any]) -> CameraView:
    transform = value.get("mount_from_camera") or {}
    return CameraView(
        key=str(value.get("key") or ""),
        role=str(value.get("role") or ""),  # type: ignore[arg-type]
        mount_link=str(value.get("mount_link") or ""),
        mount_from_camera=Transform(
            tuple(transform.get("translation_m") or ()),
            tuple(transform.get("quaternion_xyzw") or ()),
        ),
        intrinsics=tuple(value.get("intrinsics_fx_fy_cx_cy") or ()),
        image_size=tuple(value.get("image_size_wh") or ()),
        layout=str(value.get("layout") or ""),  # type: ignore[arg-type]
        mask_key=str(value.get("mask_key") or ""),
        capture_timestamp_key=str(value.get("capture_timestamp_key") or ""),
        state_timestamp_key=str(value.get("state_timestamp_key") or ""),
        maximum_state_skew_s=float(value.get("maximum_state_skew_s", math.nan)),
    )


def robot_sensor_profile_from_dict(value: Mapping[str, Any]) -> RobotSensorProfile:
    """Rebuild a validated robot/sensor profile from durable preparation bytes."""

    gripper = value.get("gripper")
    cameras = value.get("cameras")
    if not isinstance(gripper, Mapping) or not isinstance(cameras, list):
        raise InterchangeContractError("robot preparation lacks gripper/camera data")
    return RobotSensorProfile(
        identity=str(value.get("identity") or ""),
        bundle_sha256=str(value.get("bundle_sha256") or ""),
        source_sha256=str(value.get("source_sha256") or ""),
        fixed_base=bool(value.get("fixed_base", False)),
        base_link=str(value.get("base_link") or ""),
        ee_link=str(value.get("ee_link") or ""),
        arm_joint_names=tuple(value.get("arm_joint_names") or ()),
        measured_state_names=tuple(
            value.get("measured_state_names") or value.get("measured_joint_names") or ()
        ),
        measured_state_units=tuple(value.get("measured_state_units") or ()),
        joint_limits_rad=tuple(
            tuple(float(item) for item in row)
            for row in value.get("joint_limits_rad") or ()
        ),
        home_positions_rad=tuple(value.get("home_positions_rad") or ()),
        controller_representation=str(value.get("controller_representation") or ""),
        gripper=_gripper_mapping_from_dict(gripper),
        cameras=tuple(
            _camera_view_from_dict(camera)
            for camera in cameras
            if isinstance(camera, Mapping)
        ),
    )


def _resolved_embodiment(
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    embodiment = contract.get("embodiment")
    if not isinstance(embodiment, Mapping):
        raise InterchangeContractError("resolved robot contract lacks embodiment")
    source = embodiment.get("source")
    if not isinstance(source, Mapping):
        raise InterchangeContractError(
            "resolved robot contract lacks source provenance"
        )
    return embodiment, source


def _camera_views_from_config(
    sensor_config: Mapping[str, Any],
) -> tuple[CameraView, ...]:
    cameras_config = sensor_config.get("cameras")
    if not isinstance(cameras_config, Sequence) or isinstance(
        cameras_config, (str, bytes)
    ):
        raise InterchangeContractError("explicit calibrated cameras are required")
    cameras = []
    for raw in cameras_config:
        if not isinstance(raw, Mapping):
            raise InterchangeContractError("camera configuration must be an object")
        transform = raw.get("mount_from_camera")
        if not isinstance(transform, Mapping):
            raise InterchangeContractError("camera mount transform is required")
        cameras.append(_camera_view_from_dict({**raw, "mount_from_camera": transform}))
    return tuple(cameras)


def _gripper_from_resolved(
    embodiment: Mapping[str, Any], sensor_config: Mapping[str, Any]
) -> GripperMapping:
    config = sensor_config.get("gripper")
    commands = embodiment.get("gripper_commands")
    if not isinstance(config, Mapping):
        raise InterchangeContractError("explicit gripper width mapping is required")
    if not isinstance(commands, Mapping):
        raise InterchangeContractError("resolved robot contract lacks gripper commands")
    names = tuple(embodiment.get("gripper_joint_names") or ())
    open_positions = config.get("open_joint_positions")
    closed_positions = config.get("closed_joint_positions")
    if open_positions is None:
        open_positions = [commands.get("open")] * len(names)
    if closed_positions is None:
        closed_positions = [commands.get("close")] * len(names)
    return GripperMapping(
        joint_names=names,
        joint_units=tuple(config.get("joint_units") or ()),
        finger_links=tuple(embodiment.get("finger_links") or ()),
        open_joint_positions=tuple(open_positions),
        closed_joint_positions=tuple(closed_positions),
        maximum_width_m=float(config.get("maximum_width_m", math.nan)),
        model_semantics=str(config.get("model_semantics") or ""),  # type: ignore[arg-type]
        model_open=float(config.get("model_open", math.nan)),
        model_closed=float(config.get("model_closed", math.nan)),
    )


def robot_sensor_profile_from_resolved(
    contract: Mapping[str, Any], *, sensor_config: Mapping[str, Any]
) -> RobotSensorProfile:
    """Build a profile without inheriting unknown embodiment calibration."""

    embodiment, source = _resolved_embodiment(contract)
    arm_names = tuple(embodiment.get("arm_joint_names") or ())
    limits = sensor_config.get("joint_limits_rad")
    if not isinstance(limits, Sequence) or isinstance(limits, (str, bytes)):
        raise InterchangeContractError(
            "generalized policy profile requires actual ordered joint_limits_rad"
        )
    limit_values = tuple(_finite(limit, size=2, name="joint limit") for limit in limits)
    home = tuple(embodiment.get("home_qpos") or ())[: len(arm_names)]
    measured_state = sensor_config.get("measured_state")
    if not isinstance(measured_state, Mapping):
        raise InterchangeContractError(
            "explicit measured_state names and units are required"
        )
    measured_state_names = tuple(measured_state.get("names") or ())
    measured_state_units = tuple(measured_state.get("units") or ())
    return RobotSensorProfile(
        identity=str(contract.get("embodiment_digest") or ""),
        bundle_sha256=str(source.get("source_tree_sha256") or ""),
        source_sha256=str(source.get("source_sha256") or ""),
        base_link=str(embodiment.get("base_link") or ""),
        ee_link=str(embodiment.get("ee_link") or ""),
        arm_joint_names=arm_names,
        measured_state_names=measured_state_names,
        measured_state_units=measured_state_units,
        joint_limits_rad=limit_values,
        home_positions_rad=home,
        controller_representation=str(
            sensor_config.get("controller_representation") or ""
        ),  # type: ignore[arg-type]
        gripper=_gripper_from_resolved(embodiment, sensor_config),
        cameras=_camera_views_from_config(sensor_config),
        fixed_base=bool(sensor_config.get("fixed_base", True)),
    )


@dataclass(frozen=True)
class RigidObjectProfile:
    """Immutable rigid-prop geometry with explicit root and grasp frames."""

    identity: str
    source_uri: str
    source_sha256: str
    bundle_sha256: str
    kind: str
    units_m_per_unit: float
    scale_xyz: tuple[float, float, float]
    world_from_root: Transform
    root_from_reference: Transform
    root_from_grasp: Transform
    grasp_extents_m: tuple[float, float, float]
    collision_declared: bool
    inertia_declared: bool
    material_declared: bool
    mass_kg: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    grasp_axis: Literal["x", "y", "z"] = "y"

    def __post_init__(self) -> None:
        if self.kind not in RIGID_OBJECT_KINDS:
            raise InterchangeContractError(
                "only rigid, non-articulated, non-deformable props are supported"
            )
        if not self.identity.strip() or not self.source_uri.strip():
            raise InterchangeContractError(
                "object identity and source_uri are required"
            )
        _lowercase_sha256(self.source_sha256, name="object source digest")
        _lowercase_sha256(self.bundle_sha256, name="object bundle digest")
        units = float(self.units_m_per_unit)
        if not math.isfinite(units) or units <= 0.0:
            raise InterchangeContractError("object units must be positive")
        scale = _finite(self.scale_xyz, size=3, name="object scale_xyz")
        extents = _finite(self.grasp_extents_m, size=3, name="grasp_extents_m")
        if any(value <= 0.0 for value in (*scale, *extents)):
            raise InterchangeContractError(
                "object scale and grasp extents must be positive"
            )
        _validate_rigid_material(self)
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise InterchangeContractError("object mass_kg must be positive")
        if self.grasp_axis not in GRASP_AXES:
            raise InterchangeContractError("object grasp_axis must be x, y, or z")

    @property
    def required_gripper_width_m(self) -> float:
        """Return the declared object extent along the closing-axis normal."""

        return self.grasp_extents_m[("x", "y", "z").index(self.grasp_axis)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "source_uri": self.source_uri,
            "source_sha256": self.source_sha256,
            "bundle_sha256": self.bundle_sha256,
            "kind": self.kind,
            "units_m_per_unit": self.units_m_per_unit,
            "scale_xyz": list(self.scale_xyz),
            "world_from_root": self.world_from_root.to_dict(),
            "root_from_reference": self.root_from_reference.to_dict(),
            "root_from_grasp": self.root_from_grasp.to_dict(),
            "grasp_extents_m": list(self.grasp_extents_m),
            "collision_declared": self.collision_declared,
            "inertia_declared": self.inertia_declared,
            "material_declared": self.material_declared,
            "mass_kg": self.mass_kg,
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "restitution": self.restitution,
            "grasp_axis": self.grasp_axis,
            "required_gripper_width_m": self.required_gripper_width_m,
        }


def rigid_object_profile_from_dict(value: Mapping[str, Any]) -> RigidObjectProfile:
    """Rebuild a validated rigid-object profile from durable preparation bytes."""

    def transform(name: str) -> Transform:
        raw = value.get(name)
        if not isinstance(raw, Mapping):
            raise InterchangeContractError(f"object preparation lacks {name}")
        return Transform(
            tuple(raw.get("translation_m") or ()),  # type: ignore[arg-type]
            tuple(raw.get("quaternion_xyzw") or ()),  # type: ignore[arg-type]
        )

    return RigidObjectProfile(
        identity=str(value.get("identity") or ""),
        source_uri=str(value.get("source_uri") or ""),
        source_sha256=str(value.get("source_sha256") or ""),
        bundle_sha256=str(value.get("bundle_sha256") or ""),
        kind=str(value.get("kind") or ""),
        units_m_per_unit=float(value.get("units_m_per_unit", math.nan)),
        scale_xyz=tuple(value.get("scale_xyz") or ()),  # type: ignore[arg-type]
        world_from_root=transform("world_from_root"),
        root_from_reference=transform("root_from_reference"),
        root_from_grasp=transform("root_from_grasp"),
        grasp_extents_m=tuple(value.get("grasp_extents_m") or ()),  # type: ignore[arg-type]
        collision_declared=bool(value.get("collision_declared", False)),
        inertia_declared=bool(value.get("inertia_declared", False)),
        material_declared=bool(value.get("material_declared", False)),
        mass_kg=float(value.get("mass_kg", math.nan)),
        static_friction=float(value.get("static_friction", math.nan)),
        dynamic_friction=float(value.get("dynamic_friction", math.nan)),
        restitution=float(value.get("restitution", math.nan)),
        grasp_axis=str(value.get("grasp_axis") or ""),  # type: ignore[arg-type]
    )


def _sealed_rigid_asset(
    manifest: Mapping[str, Any], target: str
) -> tuple[str, dict[str, str]]:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise InterchangeContractError("object asset manifest lacks sealed files")
    selected = f"assets/{target}.usda"
    source_sha256 = str(files.get(selected) or "")
    if len(source_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in source_sha256
    ):
        raise InterchangeContractError(
            f"object target {target!r} is not an exact sealed USD asset"
        )
    if manifest.get("license") != "Apache-2.0":
        raise InterchangeContractError(
            "object asset manifest must declare its redistributable license"
        )
    units = str(manifest.get("units") or "").strip().lower()
    if units not in {"metres", "meters"}:
        raise InterchangeContractError("object asset units must be metres")
    return source_sha256, {str(key): str(value) for key, value in sorted(files.items())}


def _rigid_physics(manifest: Mapping[str, Any]) -> tuple[float, Mapping[str, Any]]:
    physics = manifest.get("physics")
    if not isinstance(physics, Mapping):
        raise InterchangeContractError(
            "object asset manifest requires explicit physics metadata"
        )
    if physics.get("collision") != "declared":
        raise InterchangeContractError("object collision metadata is not declared")
    mass = physics.get("mass_kg")
    if not isinstance(mass, (int, float)) or not math.isfinite(mass) or mass <= 0:
        raise InterchangeContractError("object inertial mass metadata is invalid")
    material = physics.get("material")
    if not isinstance(material, Mapping):
        raise InterchangeContractError("object physics material metadata is required")
    for key in ("static_friction", "dynamic_friction", "restitution"):
        value = material.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise InterchangeContractError(
                f"object physics material {key} must be finite"
            )
    if not (
        0 <= float(material["dynamic_friction"]) <= float(material["static_friction"])
        and 0 <= float(material["restitution"]) <= 1
    ):
        raise InterchangeContractError("object physics material values are invalid")
    return float(mass), material


def _object_source_uri(source_root_uri: str, target: str) -> str:
    root = source_root_uri.rstrip("/")
    if not root:
        raise InterchangeContractError("object source root URI is required")
    return f"{root}/assets/{target}.usda"


def rigid_object_profile_from_asset_manifest(
    manifest: Mapping[str, Any],
    *,
    target: str,
    source_root_uri: str,
    world_from_root: Transform,
    root_from_reference: Transform,
    root_from_grasp: Transform,
    grasp_extents_m: Sequence[float],
    grasp_axis: Literal["x", "y", "z"] = "y",
    scale_xyz: Sequence[float] = (1.0, 1.0, 1.0),
) -> RigidObjectProfile:
    """Bind a sealed rigid asset to explicit task, reference, and grasp frames."""

    source_sha256, sealed = _sealed_rigid_asset(manifest, target)
    mass, material = _rigid_physics(manifest)
    profile = RigidObjectProfile(
        identity=f"{manifest.get('schema')}:{target}:{source_sha256}",
        source_uri=_object_source_uri(source_root_uri, target),
        source_sha256=source_sha256,
        bundle_sha256=_sha({"files": sealed}),
        kind="rigid",
        units_m_per_unit=1.0,
        scale_xyz=tuple(scale_xyz),  # type: ignore[arg-type]
        world_from_root=world_from_root,
        root_from_reference=root_from_reference,
        root_from_grasp=root_from_grasp,
        grasp_extents_m=tuple(grasp_extents_m),  # type: ignore[arg-type]
        collision_declared=True,
        inertia_declared=True,
        material_declared=True,
        mass_kg=float(mass),
        static_friction=float(material["static_friction"]),
        dynamic_friction=float(material["dynamic_friction"]),
        restitution=float(material["restitution"]),
        grasp_axis=grasp_axis,
    )
    return profile


@dataclass(frozen=True)
class SurfacePlacementTask:
    """Geometry-aware teacher and unchanged measured placement event protocol."""

    identity: str
    target_world_xyz_m: tuple[float, float, float]
    approach_clearance_m: float
    lift_delta_m: float
    retreat_distance_m: float
    success_events: tuple[str, ...] = (
        "contact",
        "grasp",
        "lift",
        "transport",
        "release",
        "support_contact",
        "stable",
    )

    def __post_init__(self) -> None:
        if not self.identity.strip():
            raise InterchangeContractError("task identity is required")
        _finite(self.target_world_xyz_m, size=3, name="target_world_xyz_m")
        values = (
            self.approach_clearance_m,
            self.lift_delta_m,
            self.retreat_distance_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise InterchangeContractError(
                "task clearances must be positive and finite"
            )
        if self.success_events != (
            "contact",
            "grasp",
            "lift",
            "transport",
            "release",
            "support_contact",
            "stable",
        ):
            raise InterchangeContractError(
                "surface-placement success events are immutable"
            )

    def teacher_frames(self, prop: RigidObjectProfile) -> dict[str, Any]:
        """Return geometry-aware grasp and target frames without assuming root centering."""

        return {
            "object_root_from_grasp": prop.root_from_grasp.to_dict(),
            "object_root_from_reference": prop.root_from_reference.to_dict(),
            "approach_clearance_m": self.approach_clearance_m,
            "lift_delta_m": self.lift_delta_m,
            "target_world_xyz_m": list(self.target_world_xyz_m),
            "release_at": "declared object reference frame over target",
            "retreat_distance_m": self.retreat_distance_m,
            "success_events": list(self.success_events),
        }


def surface_placement_task_from_dict(value: Mapping[str, Any]) -> SurfacePlacementTask:
    """Rebuild the immutable success protocol from a preparation receipt."""

    return SurfacePlacementTask(
        identity=str(value.get("identity") or ""),
        target_world_xyz_m=tuple(value.get("target_world_xyz_m") or ()),  # type: ignore[arg-type]
        approach_clearance_m=float(value.get("approach_clearance_m", math.nan)),
        lift_delta_m=float(value.get("lift_delta_m", math.nan)),
        retreat_distance_m=float(value.get("retreat_distance_m", math.nan)),
        success_events=tuple(value.get("success_events") or ()),
    )


def assert_surface_placement_compatible(
    robot: RobotSensorProfile,
    prop: RigidObjectProfile,
    task: SurfacePlacementTask,
) -> dict[str, Any]:
    """Reject geometry the declared parallel jaws cannot grasp before launch."""

    required = prop.required_gripper_width_m
    available = robot.gripper.maximum_width_m
    if required > available + 1.0e-9:
        raise InterchangeContractError(
            "object grasp width exceeds the declared gripper aperture"
        )
    receipt = {
        "schema": "npa.sim2real.surface_placement_compatibility.v1",
        "robot_identity": robot.identity,
        "object_identity": prop.identity,
        "task_identity": task.identity,
        "grasp_axis": prop.grasp_axis,
        "required_gripper_width_m": required,
        "available_gripper_width_m": available,
        "teacher_frames": task.teacher_frames(prop),
    }
    return {**receipt, "compatibility_sha256": _sha(receipt)}


@dataclass(frozen=True)
class PolicyAction:
    """Declared model output and controller mapping contract."""

    representation: str
    names: tuple[str, ...]
    units: tuple[str, ...]
    frame: str
    gripper_semantics: str
    horizon: int
    control_hz: float
    execute_prefix: int

    def __post_init__(self) -> None:
        if self.representation not in ACTION_REPRESENTATIONS:
            raise InterchangeContractError("unsupported policy action representation")
        names = _names(self.names, name="policy action names")
        if len(self.units) != len(names) or any(
            not str(unit).strip() for unit in self.units
        ):
            raise InterchangeContractError(
                "one action unit is required per named action"
            )
        if self.gripper_semantics not in GRIPPER_SEMANTICS:
            raise InterchangeContractError("unsupported policy gripper semantics")
        if self.horizon <= 0 or not 0 < self.execute_prefix <= self.horizon:
            raise InterchangeContractError("invalid action horizon or execute_prefix")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise InterchangeContractError("policy control_hz must be positive")
        if self.representation == "ee_pose" and not self.frame.strip():
            raise InterchangeContractError(
                "Cartesian actions require an explicit frame"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "names": list(self.names),
            "units": list(self.units),
            "frame": self.frame,
            "gripper_semantics": self.gripper_semantics,
            "horizon": self.horizon,
            "control_hz": self.control_hz,
            "execute_prefix": self.execute_prefix,
        }


@dataclass(frozen=True)
class PolicyCapabilities:
    """Backend-specific evidence capabilities; unavailable is not a zero metric."""

    inference: bool
    native_training: bool
    checkpoint_reload: bool
    model_loss: bool
    action_metrics: bool
    deterministic_noise_coupling: bool
    repeated_paired_trials: bool
    hardware_topologies: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.inference:
            raise InterchangeContractError(
                "a selectable policy profile must support inference"
            )
        if not self.deterministic_noise_coupling and not self.repeated_paired_trials:
            raise InterchangeContractError(
                "a backend without deterministic noise coupling requires repeated paired trials"
            )
        if not self.hardware_topologies:
            raise InterchangeContractError(
                "qualified hardware topology must be explicit"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return explicit support rather than fabricating unavailable metrics."""

        return {
            "inference": self.inference,
            "native_training": self.native_training,
            "checkpoint_reload": self.checkpoint_reload,
            "model_loss": self.model_loss,
            "action_metrics": self.action_metrics,
            "deterministic_noise_coupling": self.deterministic_noise_coupling,
            "repeated_paired_trials": self.repeated_paired_trials,
            "hardware_topologies": list(self.hardware_topologies),
        }


@dataclass(frozen=True)
class PolicyProfile:
    """Model identity, observation keys, output semantics, and real capabilities."""

    backend: Literal["openpi_pi05", "lerobot_smolvla"]
    identity: str
    checkpoint_sha256: str
    robot_identity: str
    task_identity: str
    camera_keys: tuple[str, ...]
    measured_state_names: tuple[str, ...]
    measured_state_units: tuple[str, ...]
    action: PolicyAction
    capabilities: PolicyCapabilities
    normalization_identity: str

    def __post_init__(self) -> None:
        if self.backend not in {"openpi_pi05", "lerobot_smolvla"}:
            raise InterchangeContractError("unsupported catalog policy backend")
        if any(
            not value.strip()
            for value in (
                self.identity,
                self.robot_identity,
                self.task_identity,
                self.normalization_identity,
            )
        ):
            raise InterchangeContractError("policy identities must be non-empty")
        if len(self.checkpoint_sha256) != 64:
            raise InterchangeContractError("checkpoint identity must be SHA-256")
        _names(self.camera_keys, name="policy camera_keys")
        names = _names(self.measured_state_names, name="policy measured_state_names")
        if len(self.measured_state_units) != len(names) or any(
            not str(unit).strip() for unit in self.measured_state_units
        ):
            raise InterchangeContractError(
                "one measured-state unit is required per policy input"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "identity": self.identity,
            "checkpoint_sha256": self.checkpoint_sha256,
            "robot_identity": self.robot_identity,
            "task_identity": self.task_identity,
            "camera_keys": list(self.camera_keys),
            "measured_state_names": list(self.measured_state_names),
            "measured_state_units": list(self.measured_state_units),
            "action": self.action.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "normalization_identity": self.normalization_identity,
        }


def _policy_action_from_dict(value: Mapping[str, Any]) -> PolicyAction:
    return PolicyAction(
        representation=str(value.get("representation") or ""),
        names=tuple(value.get("names") or ()),
        units=tuple(value.get("units") or ()),
        frame=str(value.get("frame") or ""),
        gripper_semantics=str(value.get("gripper_semantics") or ""),
        horizon=int(value.get("horizon") or 0),
        control_hz=float(value.get("control_hz") or 0.0),
        execute_prefix=int(value.get("execute_prefix") or 0),
    )


def _policy_capabilities_from_dict(value: Mapping[str, Any]) -> PolicyCapabilities:
    return PolicyCapabilities(
        inference=bool(value.get("inference")),
        native_training=bool(value.get("native_training")),
        checkpoint_reload=bool(value.get("checkpoint_reload")),
        model_loss=bool(value.get("model_loss")),
        action_metrics=bool(value.get("action_metrics")),
        deterministic_noise_coupling=bool(value.get("deterministic_noise_coupling")),
        repeated_paired_trials=bool(value.get("repeated_paired_trials")),
        hardware_topologies=tuple(value.get("hardware_topologies") or ()),
    )


def policy_profile_from_dict(value: Mapping[str, Any]) -> PolicyProfile:
    """Rebuild a validated policy profile without backend-specific defaults."""

    action = value.get("action")
    capabilities = value.get("capabilities")
    if not isinstance(action, Mapping) or not isinstance(capabilities, Mapping):
        raise InterchangeContractError("policy preparation is incomplete")
    return PolicyProfile(
        backend=str(value.get("backend") or ""),
        identity=str(value.get("identity") or ""),
        checkpoint_sha256=str(value.get("checkpoint_sha256") or ""),
        robot_identity=str(value.get("robot_identity") or ""),
        task_identity=str(value.get("task_identity") or ""),
        camera_keys=tuple(value.get("camera_keys") or ()),
        measured_state_names=tuple(value.get("measured_state_names") or ()),
        measured_state_units=tuple(value.get("measured_state_units") or ()),
        action=_policy_action_from_dict(action),
        capabilities=_policy_capabilities_from_dict(capabilities),
        normalization_identity=str(value.get("normalization_identity") or ""),
    )


@dataclass(frozen=True)
class TimestampedCameraObservation:
    """One exact RGB observation bound to calibration and robot state time."""

    key: str
    role: Literal["exterior", "wrist"]
    calibration_sha256: str
    image_size: tuple[int, int]
    layout: Literal["HWC", "CHW"]
    mask: bool
    capture_timestamp_s: float
    state_timestamp_s: float
    rgb8: bytes
    rgb_sha256: str

    def __post_init__(self) -> None:
        if not self.key.strip() or self.role not in CAMERA_ROLES:
            raise InterchangeContractError("camera observation key/role is invalid")
        if len(self.calibration_sha256) != 64:
            raise InterchangeContractError("camera calibration identity is invalid")
        width, height = self.image_size
        if width <= 0 or height <= 0 or self.layout not in IMAGE_LAYOUTS:
            raise InterchangeContractError("camera observation geometry is invalid")
        if not self.mask:
            raise InterchangeContractError("camera observation is masked unavailable")
        if not isinstance(self.rgb8, bytes) or len(self.rgb8) != width * height * 3:
            raise InterchangeContractError(
                "camera RGB bytes do not match declared image dimensions"
            )
        actual = hashlib.sha256(self.rgb8).hexdigest()
        if self.rgb_sha256 != actual:
            raise InterchangeContractError("camera RGB hash does not match bytes")

    def to_wire_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "calibration_sha256": self.calibration_sha256,
            "image_size_wh": list(self.image_size),
            "layout": self.layout,
            "mask": self.mask,
            "capture_timestamp_s": self.capture_timestamp_s,
            "state_timestamp_s": self.state_timestamp_s,
            "image_rgb8_base64": base64.b64encode(self.rgb8).decode("ascii"),
            "image_sha256": self.rgb_sha256,
        }


@dataclass(frozen=True)
class PolicyObservation:
    """Model-neutral observation; object truth is intentionally absent."""

    input_identity: str
    robot_identity: str
    task_identity: str
    policy_identity: str
    prompt: str
    views: tuple[TimestampedCameraObservation, ...]
    measured_state_names: tuple[str, ...]
    measured_state_units: tuple[str, ...]
    measured_state: tuple[float, ...]

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.input_identity,
                self.robot_identity,
                self.task_identity,
                self.policy_identity,
                self.prompt,
            )
        ):
            raise InterchangeContractError(
                "observation identities and prompt must be non-empty"
            )
        names = _names(self.measured_state_names, name="measured state names")
        if len(self.measured_state_units) != len(names) or any(
            not str(unit).strip() for unit in self.measured_state_units
        ):
            raise InterchangeContractError(
                "one measured-state unit is required per named value"
            )
        _finite(self.measured_state, size=len(names), name="measured state")
        if not self.views:
            raise InterchangeContractError("at least one camera view is required")
        if len({view.key for view in self.views}) != len(self.views):
            raise InterchangeContractError("camera observation keys must be unique")

    def to_wire_dict(self) -> dict[str, Any]:
        return {
            "schema": "npa.sim2real.policy_observation.v1",
            "input_identity": self.input_identity,
            "robot_identity": self.robot_identity,
            "task_identity": self.task_identity,
            "policy_identity": self.policy_identity,
            "prompt": self.prompt,
            "views": [view.to_wire_dict() for view in self.views],
            "measured_state_names": list(self.measured_state_names),
            "measured_state_units": list(self.measured_state_units),
            "measured_state": list(self.measured_state),
            "privileged_object_state": False,
        }


def _camera_observation_from_wire(
    raw: Mapping[str, Any],
) -> TimestampedCameraObservation:
    capture = float(raw.get("capture_timestamp_s", math.nan))
    state = float(raw.get("state_timestamp_s", math.nan))
    if not math.isfinite(capture) or not math.isfinite(state) or capture > state:
        raise InterchangeContractError("camera timestamps are invalid")
    try:
        rgb8 = base64.b64decode(str(raw.get("image_rgb8_base64") or ""), validate=True)
    except ValueError as exc:
        raise InterchangeContractError("camera RGB payload is not base64") from exc
    return TimestampedCameraObservation(
        key=str(raw.get("key") or ""),
        role=str(raw.get("role") or ""),  # type: ignore[arg-type]
        calibration_sha256=str(raw.get("calibration_sha256") or ""),
        image_size=tuple(raw.get("image_size_wh") or ()),  # type: ignore[arg-type]
        layout=str(raw.get("layout") or ""),  # type: ignore[arg-type]
        mask=raw.get("mask") is True,
        capture_timestamp_s=capture,
        state_timestamp_s=state,
        rgb8=rgb8,
        rgb_sha256=str(raw.get("image_sha256") or ""),
    )


def policy_observation_from_wire(payload: Mapping[str, Any]) -> PolicyObservation:
    """Decode the exact public wire schema into the typed observation contract."""

    if payload.get("schema") != "npa.sim2real.policy_observation.v1":
        raise InterchangeContractError("unsupported policy observation schema")
    privileged = payload.get("privileged_object_state")
    if privileged is not None and privileged is not False:
        raise InterchangeContractError("privileged object state is forbidden")
    raw_views = payload.get("views")
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, (str, bytes)):
        raise InterchangeContractError("camera views must be an ordered list")
    if not all(isinstance(raw, Mapping) for raw in raw_views):
        raise InterchangeContractError("camera view must be an object")
    views = tuple(_camera_observation_from_wire(raw) for raw in raw_views)
    return PolicyObservation(
        input_identity=str(payload.get("input_identity") or ""),
        robot_identity=str(payload.get("robot_identity") or ""),
        task_identity=str(payload.get("task_identity") or ""),
        policy_identity=str(payload.get("policy_identity") or ""),
        prompt=str(payload.get("prompt") or ""),
        views=views,
        measured_state_names=tuple(payload.get("measured_state_names") or ()),
        measured_state_units=tuple(payload.get("measured_state_units") or ()),
        measured_state=tuple(payload.get("measured_state") or ()),
    )


@dataclass(frozen=True)
class PolicyOutput:
    """Finite action chunk with the complete declared controller semantics."""

    observation_input_identity: str
    policy_identity: str
    checkpoint_sha256: str
    action: PolicyAction
    action_chunk: tuple[tuple[float, ...], ...]
    generated_timestamp_s: float

    def __post_init__(self) -> None:
        if (
            not self.observation_input_identity.strip()
            or not self.policy_identity.strip()
        ):
            raise InterchangeContractError(
                "policy output observation and policy identities are required"
            )
        if len(self.checkpoint_sha256) != 64:
            raise InterchangeContractError(
                "policy output checkpoint identity must be SHA-256"
            )
        rows = tuple(
            _finite(row, size=len(self.action.names), name="policy action row")
            for row in self.action_chunk
        )
        if not self.action.execute_prefix <= len(rows) <= self.action.horizon:
            raise InterchangeContractError(
                "policy output rows must cover execute_prefix without exceeding horizon"
            )
        if not math.isfinite(self.generated_timestamp_s):
            raise InterchangeContractError(
                "policy output generation timestamp must be finite"
            )
        object.__setattr__(self, "action_chunk", rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "npa.sim2real.policy_output.v1",
            "observation_input_identity": self.observation_input_identity,
            "policy_identity": self.policy_identity,
            "checkpoint_sha256": self.checkpoint_sha256,
            "action": self.action.to_dict(),
            "action_chunk": [list(row) for row in self.action_chunk],
            "generated_timestamp_s": self.generated_timestamp_s,
        }


def policy_output_from_wire(
    payload: Mapping[str, Any], *, expected_action: PolicyAction
) -> PolicyOutput:
    """Decode a policy response without accepting changed controller semantics."""

    if payload.get("schema") != "npa.sim2real.policy_output.v1":
        raise InterchangeContractError("unsupported policy output schema")
    raw_action = payload.get("action")
    if not isinstance(raw_action, Mapping):
        raise InterchangeContractError("policy output action contract is required")
    action = PolicyAction(
        representation=str(raw_action.get("representation") or ""),  # type: ignore[arg-type]
        names=tuple(raw_action.get("names") or ()),
        units=tuple(raw_action.get("units") or ()),
        frame=str(raw_action.get("frame") or ""),
        gripper_semantics=str(raw_action.get("gripper_semantics") or ""),  # type: ignore[arg-type]
        horizon=int(raw_action.get("horizon") or 0),
        control_hz=float(raw_action.get("control_hz") or 0.0),
        execute_prefix=int(raw_action.get("execute_prefix") or 0),
    )
    if action != expected_action:
        raise InterchangeContractError(
            "policy response changed the declared action/controller contract"
        )
    raw_chunk = payload.get("action_chunk")
    if not isinstance(raw_chunk, Sequence) or isinstance(raw_chunk, (str, bytes)):
        raise InterchangeContractError("policy output action_chunk must be a list")
    return PolicyOutput(
        observation_input_identity=str(payload.get("observation_input_identity") or ""),
        policy_identity=str(payload.get("policy_identity") or ""),
        checkpoint_sha256=str(payload.get("checkpoint_sha256") or ""),
        action=action,
        action_chunk=tuple(tuple(row) for row in raw_chunk),
        generated_timestamp_s=float(payload.get("generated_timestamp_s", math.nan)),
    )


def _assert_policy_inputs_compatible(
    robot: RobotSensorProfile, task: SurfacePlacementTask, policy: PolicyProfile
) -> tuple[str, ...]:
    if policy.robot_identity != robot.identity or policy.task_identity != task.identity:
        raise InterchangeContractError(
            "policy robot/task identities do not match the run"
        )
    robot_cameras = tuple(camera.key for camera in robot.cameras)
    if policy.camera_keys != robot_cameras:
        raise InterchangeContractError(
            "policy camera keys/order do not match calibrated views"
        )
    if policy.measured_state_names != robot.measured_state_names:
        raise InterchangeContractError(
            "policy measured state names/order do not match the embodiment"
        )
    if policy.measured_state_units != robot.measured_state_units:
        raise InterchangeContractError(
            "policy measured state units do not match the embodiment"
        )
    return robot_cameras


def _assert_policy_action_compatible(
    robot: RobotSensorProfile,
    policy: PolicyProfile,
    *,
    cartesian_controller_available: bool,
) -> None:
    if policy.action.names != robot.action_names:
        raise InterchangeContractError(
            "policy action names/order do not match the embodiment; padding is forbidden"
        )
    expected_units = ("rad",) * len(robot.arm_joint_names) + (
        robot.gripper.model_semantics,
    )
    if policy.action.units != expected_units:
        raise InterchangeContractError(
            "policy action units do not match controller units"
        )
    if policy.action.gripper_semantics != robot.gripper.model_semantics:
        raise InterchangeContractError(
            "policy gripper semantics do not match the robot"
        )
    if policy.action.representation == "ee_pose" and not cartesian_controller_available:
        raise InterchangeContractError(
            "Cartesian policy requires a qualified IK/controller mapping"
        )
    if policy.action.representation != robot.controller_representation:
        raise InterchangeContractError(
            "policy action representation does not match the physical controller"
        )


def assert_policy_compatible(
    robot: RobotSensorProfile,
    task: SurfacePlacementTask,
    policy: PolicyProfile,
    *,
    cartesian_controller_available: bool = False,
) -> dict[str, Any]:
    """Validate exact kinematics, cameras, units, cadence, and controller mapping."""

    robot_cameras = _assert_policy_inputs_compatible(robot, task, policy)
    _assert_policy_action_compatible(
        robot,
        policy,
        cartesian_controller_available=cartesian_controller_available,
    )
    receipt = {
        "schema": SCHEMA,
        "robot_identity": robot.identity,
        "task_identity": task.identity,
        "policy_identity": policy.identity,
        "backend": policy.backend,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "arm_dof": len(robot.arm_joint_names),
        "ee_link": robot.ee_link,
        "camera_keys": list(robot_cameras),
        "action": policy.action.to_dict(),
        "normalization_identity": policy.normalization_identity,
        "capabilities": policy.capabilities.to_dict(),
    }
    return {**receipt, "compatibility_sha256": _sha(receipt)}


def _validated_camera_observation(
    camera: CameraView, payload: Mapping[str, Any]
) -> TimestampedCameraObservation:
    if payload.get("layout") != camera.layout:
        raise InterchangeContractError(f"camera {camera.key!r} layout does not match")
    if tuple(payload.get("image_size_wh") or ()) != camera.image_size:
        raise InterchangeContractError(
            f"camera {camera.key!r} image size does not match"
        )
    if payload.get("mask") is not True:
        raise InterchangeContractError(f"camera {camera.key!r} is masked unavailable")
    capture = float(payload.get("capture_timestamp_s", math.nan))
    state = float(payload.get("state_timestamp_s", math.nan))
    camera.validate_timestamps(capture_s=capture, state_s=state)
    rgb8 = payload.get("rgb8")
    if not isinstance(rgb8, bytes):
        raise InterchangeContractError(f"camera {camera.key!r} RGB bytes are required")
    return TimestampedCameraObservation(
        key=camera.key,
        role=camera.role,
        calibration_sha256=_sha(camera.to_dict()),
        image_size=camera.image_size,
        layout=camera.layout,
        mask=True,
        capture_timestamp_s=capture,
        state_timestamp_s=state,
        rgb8=rgb8,
        rgb_sha256=str(payload.get("image_sha256") or ""),
    )


def _validate_observation_identity(
    *, policy: PolicyProfile, prompt: str, input_identity: str, task_identity: str
) -> None:
    if not prompt.strip() or not input_identity.strip():
        raise InterchangeContractError("prompt and input identity are required")
    if task_identity != policy.task_identity:
        raise InterchangeContractError(
            "observation task identity does not match policy"
        )


def _ordered_measured_state(
    policy: PolicyProfile, measured_state: Mapping[str, float]
) -> tuple[float, ...]:
    if tuple(measured_state) != policy.measured_state_names:
        raise InterchangeContractError("measured state names/order do not match policy")
    return _finite(
        tuple(measured_state.values()),
        size=len(policy.measured_state_names),
        name="measured state",
    )


def _validated_camera_views(
    robot: RobotSensorProfile,
    policy: PolicyProfile,
    camera_payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[TimestampedCameraObservation, ...]:
    unexpected = set(camera_payloads) - set(policy.camera_keys)
    missing = set(policy.camera_keys) - set(camera_payloads)
    if unexpected or missing:
        raise InterchangeContractError(
            f"camera payload mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    by_key = {camera.key: camera for camera in robot.cameras}
    return tuple(
        _validated_camera_observation(by_key[key], camera_payloads[key])
        for key in policy.camera_keys
    )


def validate_observation(
    *,
    robot: RobotSensorProfile,
    policy: PolicyProfile,
    camera_payloads: Mapping[str, Mapping[str, Any]],
    measured_state: Mapping[str, float],
    prompt: str,
    input_identity: str,
    task_identity: str,
) -> PolicyObservation:
    """Validate one causal model observation without privileged object state."""

    _validate_observation_identity(
        policy=policy,
        prompt=prompt,
        input_identity=input_identity,
        task_identity=task_identity,
    )
    state_values = _ordered_measured_state(policy, measured_state)
    views = _validated_camera_views(robot, policy, camera_payloads)
    return PolicyObservation(
        input_identity=input_identity,
        robot_identity=robot.identity,
        task_identity=task_identity,
        policy_identity=policy.identity,
        prompt=prompt,
        views=views,
        measured_state_names=policy.measured_state_names,
        measured_state_units=robot.measured_state_units,
        measured_state=state_values,
    )
