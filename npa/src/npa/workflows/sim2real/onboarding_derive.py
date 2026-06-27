"""Auto-derive a robot-aware Lift/reach/place config from an onboarding spec (B2).

The stock ``Isaac-Lift-Cube-Franka-v0`` task hardcodes Franka-tuned numbers:
the action scale, the reward thresholds (lift height, goal distance), and the
object / goal placement ranges are all sized to the Franka Panda. A different
arm can be *retargeted* onto the task (``isaac_byo_robot_task``) and will RUN,
but the learning signal is mis-scaled, so it does not LEARN (observed on Kinova
Jaco2: flat reward 0.71->0.70). This module derives a robot-aware config so a
non-Franka arm gets a correctly-scaled task.

Everything here is **pure math** — no ``torch`` / ``isaaclab`` / ``pxr`` import.
On-cluster the BYO wrapper can introspect the swapped USD's joint limits and link
lengths and pass them in as ``arm_joint_ranges`` / ``arm_link_lengths`` /
``finger_joint_ranges``; off-GPU (and in unit tests) the derivation falls back to
preset reach values and conservative heuristics. Either way the output is a
``DerivedTaskConfig`` the B3 task variant applies.

Calibration contract: the Franka reference (its joint ranges + 0.855 m reach)
reproduces the *stock* numbers exactly, so a Franka-shaped input derives the
stock action scale / placement / thresholds — i.e. the derivation is a no-op for
Franka and B3 keeps the Franka path byte-for-byte. A different arm gets values
scaled away from the stock by its own range / reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from npa.workflows.sim2real import onboarding_spec as ob

# --------------------------------------------------------------------------- #
# Franka reference (the calibration point)
# --------------------------------------------------------------------------- #
# Franka Emika Panda arm joint limits (radians) — the 7 arm joints.
FRANKA_ARM_JOINT_RANGES: tuple[tuple[float, float], ...] = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
FRANKA_MEAN_ARM_RANGE = sum(hi - lo for lo, hi in FRANKA_ARM_JOINT_RANGES) / len(
    FRANKA_ARM_JOINT_RANGES
)  # ~4.782 rad
# Franka maximum reach (m); used to scale workspace-relative placement.
FRANKA_REACH_M = 0.855
# Stock JointPositionAction scale for the Franka Lift task.
STOCK_ACTION_SCALE = 0.5
# Stock Franka Lift placement ranges (m), in the robot base frame. These mirror
# the manager-based Lift cfg: the object spawns on the table near the base, the
# goal command samples a box in front of and above the base.
STOCK_OBJECT_INIT_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.1, 0.1),
    "y": (-0.25, 0.25),
    "z": (0.0, 0.0),
}
STOCK_GOAL_RANGE: dict[str, tuple[float, float]] = {
    "x": (0.4, 0.6),
    "y": (-0.25, 0.25),
    "z": (0.25, 0.5),
}
# Stock reward/eval thresholds.
STOCK_MINIMAL_HEIGHT_M = 0.04  # lifting_object term minimal_height
STOCK_SUCCESS_DISTANCE_M = 0.02  # goal-tracking success tolerance

# Approximate maximum reach (m) per known arm class, used when link lengths are
# not introspected. Conservative published nominal reaches.
PRESET_REACH_M: dict[str, float] = {
    "franka": FRANKA_REACH_M,
    "franka_panda": FRANKA_REACH_M,
    "stock_franka": FRANKA_REACH_M,
    "ur5e": 0.85,
    "ur10e": 1.30,
    "flexiv": 0.93,
    "flexiv_rizon": 0.93,
    "rizon": 0.93,
    "kinova": 0.985,
    "kinova_j2n7s300": 0.985,
    "j2n7s300": 0.985,
}

# Bounds keep a derived value sane even with garbage / missing measurements.
ACTION_SCALE_MIN, ACTION_SCALE_MAX = 0.1, STOCK_ACTION_SCALE
REACH_MIN, REACH_MAX = 0.3, 2.0
PLACEMENT_SCALE_MIN, PLACEMENT_SCALE_MAX = 0.5, 1.8


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class DerivedTaskConfig:
    """The robot-aware Lift/reach/place config the B3 variant applies.

    ``source`` records, per field, whether the value came from an explicit spec
    value (``explicit``), a measured USD range (``measured``), a named preset
    (``preset``), or a heuristic (``heuristic``) — so the onboarding CLI can show
    the customer what was derived vs. supplied.
    """

    skill: str = ob.SKILL_LIFT
    action_scale: float = STOCK_ACTION_SCALE
    workspace_reach_m: float = FRANKA_REACH_M
    object_init_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(STOCK_OBJECT_INIT_RANGE)
    )
    goal_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(STOCK_GOAL_RANGE)
    )
    goal_pos: tuple[float, ...] = ()  # explicit fixed goal, if the customer gave one
    minimal_height_m: float = STOCK_MINIMAL_HEIGHT_M
    success_distance_m: float = STOCK_SUCCESS_DISTANCE_M
    gripper_open: float = 0.04
    gripper_close: float = 0.0
    init_joint_pos: tuple[float, ...] = ()
    source: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "action_scale": self.action_scale,
            "workspace_reach_m": self.workspace_reach_m,
            "object_init_range": {k: list(v) for k, v in self.object_init_range.items()},
            "goal_range": {k: list(v) for k, v in self.goal_range.items()},
            "goal_pos": list(self.goal_pos),
            "minimal_height_m": self.minimal_height_m,
            "success_distance_m": self.success_distance_m,
            "gripper_open": self.gripper_open,
            "gripper_close": self.gripper_close,
            "init_joint_pos": list(self.init_joint_pos),
            "source": dict(self.source),
        }


# --------------------------------------------------------------------------- #
# Individual derivations (each pure, each independently testable)
# --------------------------------------------------------------------------- #
def derive_workspace_reach(
    *,
    arm_link_lengths: list[float] | None = None,
    preset: str = "",
    robot_name: str = "",
) -> tuple[float, str]:
    """Approximate the arm's maximum reach (m) and how it was obtained.

    Order: summed introspected link lengths -> preset/name lookup -> Franka
    default. Clamped to a physically plausible band.
    """

    if arm_link_lengths:
        total = sum(abs(float(x)) for x in arm_link_lengths)
        if total > 0:
            return round(_clamp(total, REACH_MIN, REACH_MAX), 4), "measured"
    for key in (preset, robot_name):
        k = str(key or "").strip().lower()
        if k in PRESET_REACH_M:
            return PRESET_REACH_M[k], "preset"
    return FRANKA_REACH_M, "heuristic"


def derive_action_scale(
    *, arm_joint_ranges: list[tuple[float, float]] | None = None
) -> tuple[float, str]:
    """Derive the JointPositionAction scale from the arm's joint ranges.

    The action is a per-step joint-position offset (radians). The stock 0.5 is
    sized to the Franka's mean joint range; a smaller arm needs a smaller offset
    so a unit action stays a comparable fraction of its travel. We never exceed
    the stock 0.5 (it is already aggressive), only reduce for tighter arms.
    A Franka-shaped range reproduces 0.5 exactly.
    """

    if not arm_joint_ranges:
        return STOCK_ACTION_SCALE, "heuristic"
    widths = [abs(float(hi) - float(lo)) for lo, hi in arm_joint_ranges if hi != lo]
    if not widths:
        return STOCK_ACTION_SCALE, "heuristic"
    mean_range = sum(widths) / len(widths)
    scale = STOCK_ACTION_SCALE * (mean_range / FRANKA_MEAN_ARM_RANGE)
    return round(_clamp(scale, ACTION_SCALE_MIN, ACTION_SCALE_MAX), 4), "measured"


def derive_gripper_targets(
    *,
    explicit_open: float | None = None,
    explicit_close: float | None = None,
    finger_joint_ranges: list[tuple[float, float]] | None = None,
) -> tuple[float, float, str]:
    """Derive (open, close) finger joint targets and how they were obtained.

    Explicit spec values win. Otherwise, from measured finger joint ranges we use
    the convention that a revolute finger *opens* at its lower bound and *closes*
    (curls) at its upper bound. Falls back to the Franka parallel-jaw defaults
    (open 0.04, close 0.0).
    """

    if explicit_open is not None and explicit_close is not None:
        return float(explicit_open), float(explicit_close), "explicit"
    if finger_joint_ranges:
        los = [float(lo) for lo, _ in finger_joint_ranges]
        his = [float(hi) for _, hi in finger_joint_ranges]
        if los and his:
            return (
                round(sum(los) / len(los), 4),
                round(sum(his) / len(his), 4),
                "measured",
            )
    # Respect a single explicit value if only one was given.
    open_v = 0.04 if explicit_open is None else float(explicit_open)
    close_v = 0.0 if explicit_close is None else float(explicit_close)
    return open_v, close_v, "heuristic"


def derive_placement(
    reach_m: float,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], str]:
    """Scale the stock object-init / goal ranges by the arm's reach vs Franka.

    A Franka reach reproduces the stock ranges (scale 1.0); a longer arm pushes
    the goal box outward, a shorter arm pulls it in, so the object is reachable
    and the goal is within the workspace.
    """

    scale = _clamp(float(reach_m) / FRANKA_REACH_M, PLACEMENT_SCALE_MIN, PLACEMENT_SCALE_MAX)

    def _scaled(rng: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        return {axis: (round(lo * scale, 4), round(hi * scale, 4)) for axis, (lo, hi) in rng.items()}

    src = "measured" if scale != 1.0 else "heuristic"
    return _scaled(STOCK_OBJECT_INIT_RANGE), _scaled(STOCK_GOAL_RANGE), src


def derive_init_joint_pos(
    spec_robot: ob.RobotInput,
    *,
    measured_home: list[float] | None = None,
) -> tuple[tuple[float, ...], str]:
    """Resolve the init/home joint pose.

    Explicit ``home_qpos`` wins; otherwise a measured/introspected home; otherwise
    an empty tuple (the variant then resets joints to zero, which the retarget
    layer already handles).
    """

    if spec_robot.home_qpos and "home_qpos" not in spec_robot.auto_fields:
        return tuple(float(x) for x in spec_robot.home_qpos), "explicit"
    if measured_home:
        return tuple(float(x) for x in measured_home), "measured"
    return (), "heuristic"


# --------------------------------------------------------------------------- #
# Top-level: derive the full task config from an onboarding spec
# --------------------------------------------------------------------------- #
def derive_task_config(
    spec: ob.OnboardingSpec,
    *,
    arm_joint_ranges: list[tuple[float, float]] | None = None,
    finger_joint_ranges: list[tuple[float, float]] | None = None,
    arm_link_lengths: list[float] | None = None,
    measured_home: list[float] | None = None,
) -> DerivedTaskConfig:
    """Derive the robot-aware Lift/reach/place config for an onboarding spec.

    Pure: explicit spec values win, then measured USD introspection (passed in by
    the on-cluster wrapper), then preset/heuristic. The result feeds the B3 task
    variant. For a stock-Franka spec the derivation reproduces the stock numbers,
    so applying it is a no-op (B3 still gates the apply on non-Franka).
    """

    robot = spec.robot
    task = spec.task
    cfg = DerivedTaskConfig(skill=task.skill)
    src = cfg.source

    reach, reach_src = derive_workspace_reach(
        arm_link_lengths=arm_link_lengths, preset=robot.preset, robot_name=robot.name
    )
    cfg.workspace_reach_m = reach
    src["workspace_reach_m"] = reach_src

    action_scale, action_src = derive_action_scale(arm_joint_ranges=arm_joint_ranges)
    cfg.action_scale = action_scale
    src["action_scale"] = action_src

    obj_range, goal_range, place_src = derive_placement(reach)
    cfg.object_init_range = obj_range
    cfg.goal_range = goal_range
    src["object_init_range"] = place_src
    src["goal_range"] = place_src

    # Explicit fixed goal from the customer overrides the sampled range.
    if task.goal_pos and not task.goal_pos_auto:
        cfg.goal_pos = tuple(task.goal_pos)
        src["goal_pos"] = "explicit"

    # Reward / eval thresholds: explicit task values win, else stock.
    cfg.minimal_height_m = float(task.lift_height_m or STOCK_MINIMAL_HEIGHT_M)
    src["minimal_height_m"] = (
        "explicit" if task.lift_height_m != ob.DEFAULT_LIFT_HEIGHT_M else "heuristic"
    )
    cfg.success_distance_m = float(task.success_distance_m or STOCK_SUCCESS_DISTANCE_M)
    src["success_distance_m"] = (
        "explicit" if task.success_distance_m != ob.DEFAULT_SUCCESS_DISTANCE_M else "heuristic"
    )

    g_open, g_close, g_src = derive_gripper_targets(
        explicit_open=robot.gripper_open if "gripper_open" not in robot.auto_fields else None,
        explicit_close=robot.gripper_close if "gripper_close" not in robot.auto_fields else None,
        finger_joint_ranges=finger_joint_ranges,
    )
    cfg.gripper_open = g_open
    cfg.gripper_close = g_close
    src["gripper"] = g_src

    init_pos, init_src = derive_init_joint_pos(robot, measured_home=measured_home)
    cfg.init_joint_pos = init_pos
    src["init_joint_pos"] = init_src

    return cfg
