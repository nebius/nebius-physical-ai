"""Classify manipulation stalls and validate separately owned recovery actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Protocol, Sequence


class RecoveryKind(StrEnum):
    """Name one observable recovery condition.

    Args: None.
    Returns: None.
    Raises: None.
    """

    NONE = "none"
    COMMAND_BLOCKED = "command_blocked"
    MOVING_WITHOUT_TASK_PROGRESS = "moving_without_task_progress"
    POOR_VISIBILITY = "poor_visibility"


@dataclass(frozen=True)
class RecoveryThresholds:
    """Hold trace-calibrated thresholds for one recovery experiment.

    Args: None.
    Returns: None.
    Raises: ValueError if a threshold is non-positive.
    """

    window_decisions: int
    commanded_motion_min: float
    measured_motion_min: float
    task_progress_min_m: float
    target_pixels_min: int
    finger_pixels_min: int

    def __post_init__(self) -> None:
        values = (
            self.window_decisions,
            self.commanded_motion_min,
            self.measured_motion_min,
            self.task_progress_min_m,
            self.target_pixels_min,
            self.finger_pixels_min,
        )
        if any(value <= 0 for value in values):
            raise ValueError("recovery thresholds must be positive")


@dataclass(frozen=True)
class RecoverySample:
    """Describe one policy decision using measured state and visibility.

    Args: None.
    Returns: None.
    Raises: None.
    """

    commanded_motion: float
    measured_motion: float
    task_progress_m: float
    workspace_target_pixels: int
    wrist_target_pixels: int
    wrist_left_finger_pixels: int
    wrist_right_finger_pixels: int
    workspace_target_clipped: bool = False
    wrist_target_clipped: bool = False
    wrist_left_finger_clipped: bool = False
    wrist_right_finger_clipped: bool = False
    controller_blocked: bool = False
    collision: bool = False
    joint_limit: bool = False


@dataclass(frozen=True)
class RecoveryClassification:
    """Bind a stall class to its measured decision window.

    Args: None.
    Returns: None.
    Raises: None.
    """

    kind: RecoveryKind
    window_decisions: int
    evidence: tuple[str, ...]


def classify_recovery(
    samples: Sequence[RecoverySample], thresholds: RecoveryThresholds
) -> RecoveryClassification:
    """Classify blockage, task stagnation, or inadequate policy visibility.

    Args:
        samples: Ordered decision samples from the current episode.
        thresholds: Values calibrated from retained expert and stall traces.
    Returns:
        The current recovery classification and its evidence labels.
    Raises:
        ValueError: If a numeric sample is negative or non-finite.
    """

    window = tuple(samples[-thresholds.window_decisions :])
    _validate_samples(window)
    if len(window) < thresholds.window_decisions:
        return RecoveryClassification(RecoveryKind.NONE, len(window), ("warming",))
    if any(
        row.controller_blocked or row.collision or row.joint_limit for row in window
    ):
        return RecoveryClassification(
            RecoveryKind.COMMAND_BLOCKED, len(window), ("controller_or_limit",)
        )
    commanded = sum(row.commanded_motion for row in window)
    measured = sum(row.measured_motion for row in window)
    if (
        commanded >= thresholds.commanded_motion_min
        and measured < thresholds.measured_motion_min
    ):
        return RecoveryClassification(
            RecoveryKind.COMMAND_BLOCKED, len(window), ("command_measurement_delta",)
        )
    if _poor_visibility(window, thresholds):
        return RecoveryClassification(
            RecoveryKind.POOR_VISIBILITY, len(window), ("workspace_wrist_geometry",)
        )
    progress = sum(row.task_progress_m for row in window)
    if (
        measured >= thresholds.measured_motion_min
        and progress < thresholds.task_progress_min_m
    ):
        return RecoveryClassification(
            RecoveryKind.MOVING_WITHOUT_TASK_PROGRESS,
            len(window),
            ("motion_without_object_progress",),
        )
    return RecoveryClassification(RecoveryKind.NONE, len(window), ("progressing",))


def _validate_samples(samples: Sequence[RecoverySample]) -> None:
    for sample in samples:
        numeric = (
            sample.commanded_motion,
            sample.measured_motion,
            sample.task_progress_m,
            sample.workspace_target_pixels,
            sample.wrist_target_pixels,
            sample.wrist_left_finger_pixels,
            sample.wrist_right_finger_pixels,
        )
        if any(not math.isfinite(float(value)) or value < 0 for value in numeric):
            raise ValueError("recovery evidence must be finite and nonnegative")


def _poor_visibility(
    samples: Sequence[RecoverySample], thresholds: RecoveryThresholds
) -> bool:
    return all(
        row.workspace_target_pixels < thresholds.target_pixels_min
        or row.wrist_target_pixels < thresholds.target_pixels_min
        or min(row.wrist_left_finger_pixels, row.wrist_right_finger_pixels)
        < thresholds.finger_pixels_min
        or row.workspace_target_clipped
        or row.wrist_target_clipped
        or row.wrist_left_finger_clipped
        or row.wrist_right_finger_clipped
        for row in samples
    )


@dataclass(frozen=True)
class RecoveryRequest:
    """Provide robot, camera, and grasp constraints to a recovery planner.

    Args: None.
    Returns: None.
    Raises: None.
    """

    classification: RecoveryClassification
    robot_identity: str
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]
    joint_limits: tuple[tuple[float, float], ...]
    camera_roles: tuple[str, ...]
    camera_extrinsics_sha256: str
    wrist_camera_world_pose: tuple[float, ...]
    collision_model_identity: str
    grasp_active: bool


@dataclass(frozen=True)
class RecoveryAction:
    """Declare a planner-owned action independently from the VLA output.

    Args: None.
    Returns: None.
    Raises: None.
    """

    owner: str
    representation: str
    joint_names: tuple[str, ...]
    values: tuple[float, ...]
    units: str
    collision_checked: bool
    preserves_grasp: bool


class ViewRecoveryPlanner(Protocol):
    """Plan a constrained view-seeking action for a declared robot profile.

    Args: None.
    Returns: None.
    Raises: None.
    """

    def plan(self, request: RecoveryRequest) -> RecoveryAction:
        """Return one explicitly planner-owned recovery action.

        Args:
            request: Current observation-derived stall and robot constraints.
        Returns:
            A collision-checked action carrying distinct planner ownership.
        Raises:
            RuntimeError: If the planner cannot find a safe view-seeking action.
        """


def validate_recovery_action(request: RecoveryRequest, action: RecoveryAction) -> None:
    """Reject unsafe or falsely attributed view-recovery actions.

    Args:
        request: Current robot, camera, collision, and grasp constraints.
        action: Candidate action emitted by the declared recovery planner.
    Returns:
        None.
    Raises:
        ValueError: If ownership, kinematics, limits, or grasp safety fail.
    """

    _validate_recovery_request(request)
    if action.owner != "recovery_planner" or not action.collision_checked:
        raise ValueError("recovery action lacks planner ownership or collision check")
    if action.representation != "absolute_joint_position" or action.units != "rad":
        raise ValueError("recovery action representation is unsupported")
    if action.joint_names != request.joint_names:
        raise ValueError("recovery action joint order differs from the robot profile")
    if len(action.values) != len(request.joint_limits):
        raise ValueError("recovery action width differs from the robot profile")
    for value, limits in zip(action.values, request.joint_limits, strict=True):
        if not math.isfinite(value) or not limits[0] <= value <= limits[1]:
            raise ValueError("recovery action violates a physical joint limit")
    if request.grasp_active and not action.preserves_grasp:
        raise ValueError("view recovery may not release an active grasp")


def _validate_recovery_request(request: RecoveryRequest) -> None:
    if request.classification.kind is RecoveryKind.NONE:
        raise ValueError("recovery action requires an observed stall classification")
    width = len(request.joint_names)
    if width == 0 or len(request.joint_positions) != width:
        raise ValueError("recovery request joint state differs from the robot profile")
    if len(request.joint_limits) != width or len(set(request.joint_names)) != width:
        raise ValueError("recovery request joint limits or ordering are invalid")
    if request.camera_roles != ("workspace", "wrist"):
        raise ValueError("recovery request must bind workspace and wrist roles")
    if len(request.camera_extrinsics_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in request.camera_extrinsics_sha256
    ):
        raise ValueError("recovery request camera calibration identity is invalid")
    if len(request.wrist_camera_world_pose) != 7 or not all(
        math.isfinite(value) for value in request.wrist_camera_world_pose
    ):
        raise ValueError("recovery request wrist camera pose is invalid")
    if not request.robot_identity or not request.collision_model_identity:
        raise ValueError("recovery request lacks robot or collision identity")
