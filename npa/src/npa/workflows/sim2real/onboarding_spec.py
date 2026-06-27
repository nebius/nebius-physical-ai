"""Declarative customer-facing robot+task onboarding spec (``npa.sim2real.onboarding.v1``).

A customer onboards their own robot into the sim2real Lift/reach/place workflow
with a single YAML file describing two things:

* ``robot``: the embodiment — a ``usd_path`` (or ``urdf_path``) plus *either*
  explicit morphology (``ee_link`` / ``joint_names`` / ``gripper_joint_names`` /
  ``home_qpos`` / gains) *or* the literal ``auto`` for any field we can derive
  from the asset / a sane heuristic (see ``onboarding_derive``).
* ``task``: the skill — ``lift`` / ``reach`` / ``place`` — plus the manipulated
  object USD, goal, success distance, and the eval success threshold.

This module is **pure schema + validation only** (no ``torch`` / ``isaaclab`` /
``genesis`` imports), so it is unit-testable off-GPU. It reuses
``npa.genesis.robot_assets`` for robot-source inference and the eventual
``RobotSpec`` projection; the auto-derivation of ``auto`` fields and the
robot-aware task config live in ``onboarding_derive`` (B2) and
``isaac_byo_robot_task`` (B3). The split mirrors the rest of sim2real: the spec
layer validates intent, the derive layer turns it into a concrete config, and
the task layer applies it inside Isaac.

The contract:

* A field set to the string ``auto`` (case-insensitive) is recorded in
  ``RobotInput.auto_fields`` and left for the derive layer to fill — validation
  does not require it to be present/consistent yet.
* A field given explicitly is validated now (lengths, ranges, types) so a typo
  fails fast, before any GPU job is submitted.
* ``stock_franka`` reproduces today's Franka embodiment with no asset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from npa.genesis import robot_assets

ONBOARDING_SCHEMA = "npa.sim2real.onboarding.v1"

# Literal sentinel a customer writes to defer a field to auto-derivation (B2).
AUTO = "auto"

# Skills the onboarding flow supports. Each maps to the stock Lift task family
# with a skill-specific reward/goal emphasis applied robot-aware in B3.
SKILL_LIFT = "lift"
SKILL_REACH = "reach"
SKILL_PLACE = "place"
SKILLS = (SKILL_LIFT, SKILL_REACH, SKILL_PLACE)

# Defaults that reproduce the proven Franka Lift contract when a task omits them.
# The lift-height threshold matches the stock ``lifting_object`` term's
# ``minimal_height`` (0.04 m); the success distance matches the goal-tracking
# tolerance used by the held-out eval.
DEFAULT_LIFT_HEIGHT_M = 0.04
DEFAULT_SUCCESS_DISTANCE_M = 0.02
DEFAULT_SUCCESS_THRESHOLD = 0.5


class OnboardingSpecError(ValueError):
    """Raised when an onboarding spec is malformed."""


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO


def _require(doc: dict[str, Any], key: str, label: str) -> Any:
    if key not in doc or doc[key] in (None, ""):
        raise OnboardingSpecError(f"{label} requires a non-empty '{key}'")
    return doc[key]


def _opt_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise OnboardingSpecError(f"{label} must be a number, got {value!r}") from exc


def _opt_pos_float(value: Any, label: str) -> float:
    out = _opt_float(value, label)
    if out <= 0:
        raise OnboardingSpecError(f"{label} must be > 0, got {out}")
    return out


def _str_list_or_auto(value: Any, label: str, auto_fields: set[str], field_name: str) -> tuple[str, ...]:
    """Parse a list-of-strings field that may be the ``auto`` sentinel."""

    if _is_auto(value):
        auto_fields.add(field_name)
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise OnboardingSpecError(f"{label} must be a list of names or '{AUTO}', got {value!r}")
    return tuple(str(v) for v in value)


def _float_list_or_auto(value: Any, label: str, auto_fields: set[str], field_name: str) -> tuple[float, ...]:
    if _is_auto(value):
        auto_fields.add(field_name)
        return ()
    if not isinstance(value, (list, tuple)):
        raise OnboardingSpecError(f"{label} must be a list of numbers or '{AUTO}', got {value!r}")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise OnboardingSpecError(f"{label} must be numeric: {value!r}") from exc


def _str_or_auto(value: Any, label: str, auto_fields: set[str], field_name: str) -> str:
    if _is_auto(value):
        auto_fields.add(field_name)
        return ""
    return str(value).strip()


@dataclass
class RobotInput:
    """The customer's robot block — minimal morphology + ``auto`` placeholders.

    Fields left as ``auto`` in the YAML are recorded in ``auto_fields`` and stay
    empty/zero here; the derive layer (B2) fills them from the asset / heuristics.
    Explicit fields are validated at parse time.
    """

    name: str = "robot"
    robot_source: str = robot_assets.ROBOT_SOURCE_STOCK_FRANKA
    robot_uri: str = ""  # the usd_path / urdf_path / s3 uri (the asset)
    preset: str = ""  # optional named preset to seed defaults (ur5e/flexiv/...)
    ee_link: str = ""
    base_link: str = ""
    joint_names: tuple[str, ...] = ()
    gripper_joint_names: tuple[str, ...] = ()
    finger_links: tuple[str, ...] = ()
    n_arm_joints: int = 0
    n_gripper_joints: int = 0
    home_qpos: tuple[float, ...] = ()
    kp: tuple[float, ...] = ()
    kv: tuple[float, ...] = ()
    force_lower: tuple[float, ...] = ()
    force_upper: tuple[float, ...] = ()
    gripper_open: float | None = None
    gripper_close: float | None = None
    auto_fields: set[str] = field(default_factory=set)

    @property
    def is_stock_franka(self) -> bool:
        return self.robot_source == robot_assets.ROBOT_SOURCE_STOCK_FRANKA

    def is_auto(self, name: str) -> bool:
        return name in self.auto_fields


@dataclass
class TaskSpec:
    """The customer's task block — the skill, object, goal, and success gate."""

    skill: str = SKILL_LIFT
    object_usd: str = ""  # empty -> trainer default (rigid-ready MultiColorCube)
    object_scale: str = ""  # e.g. "(0.8, 0.8, 0.8)"; empty -> task default
    # Goal position [x, y, z] in the robot base frame, or ``auto`` to derive from
    # the arm workspace (B2). Empty -> task default goal-command range.
    goal_pos: tuple[float, ...] = ()
    goal_pos_auto: bool = False
    lift_height_m: float = DEFAULT_LIFT_HEIGHT_M
    success_distance_m: float = DEFAULT_SUCCESS_DISTANCE_M
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
    num_envs: int = 0  # 0 -> trainer default
    iterations: int = 0  # 0 -> trainer default

    @property
    def needs_gripper(self) -> bool:
        """Whether the skill structurally requires a grasp (drives compat check)."""

        return self.skill in (SKILL_LIFT, SKILL_PLACE)


@dataclass
class OnboardingSpec:
    """A parsed robot+task onboarding spec."""

    schema: str = ONBOARDING_SCHEMA
    robot: RobotInput = field(default_factory=RobotInput)
    task: TaskSpec = field(default_factory=TaskSpec)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_robot(doc: dict[str, Any]) -> RobotInput:
    if not isinstance(doc, dict):
        raise OnboardingSpecError(f"'robot' must be a mapping, got {type(doc).__name__}")

    auto_fields: set[str] = set()
    ri = RobotInput(auto_fields=auto_fields)
    ri.name = str(doc.get("name") or "robot").strip() or "robot"
    ri.preset = str(doc.get("preset") or "").strip().lower()

    # The asset: usd_path / urdf_path / robot_uri (aliases), or a stock/preset robot.
    uri = str(
        doc.get("usd_path")
        or doc.get("urdf_path")
        or doc.get("robot_uri")
        or ""
    ).strip()
    ri.robot_uri = uri

    explicit_source = doc.get("robot_source")
    if explicit_source:
        source = str(explicit_source).strip().lower()
        if source not in robot_assets.ROBOT_SOURCES:
            raise OnboardingSpecError(
                f"robot_source must be one of {robot_assets.ROBOT_SOURCES}, got {source!r}"
            )
        ri.robot_source = source
    elif uri:
        inferred = robot_assets._infer_robot_source_from_uri(uri)
        if inferred is None:
            raise OnboardingSpecError(
                f"could not infer robot_source from asset {uri!r}; the robot must be an "
                f"articulated description {robot_assets.ARTICULATED_SUFFIXES} "
                "(or set robot_source explicitly)"
            )
        ri.robot_source = inferred
    elif ri.preset:
        # A bare preset (no asset) inherits the preset's source (e.g. byo_urdf).
        ri.robot_source = robot_assets.robot_spec_from_preset(ri.preset).robot_source
    else:
        ri.robot_source = robot_assets.ROBOT_SOURCE_STOCK_FRANKA

    # A real BYO source needs an asset to stage.
    if ri.robot_source in robot_assets.ROBOT_SOURCES_BYO and not uri:
        raise OnboardingSpecError(
            f"robot_source={ri.robot_source} requires a 'usd_path'/'urdf_path' asset"
        )

    ri.ee_link = _str_or_auto(doc.get("ee_link", ""), "robot.ee_link", auto_fields, "ee_link")
    ri.base_link = _str_or_auto(doc.get("base_link", ""), "robot.base_link", auto_fields, "base_link")
    ri.joint_names = _str_list_or_auto(
        doc.get("joint_names", ()), "robot.joint_names", auto_fields, "joint_names"
    )
    ri.gripper_joint_names = _str_list_or_auto(
        doc.get("gripper_joint_names", ()), "robot.gripper_joint_names", auto_fields, "gripper_joint_names"
    )
    ri.finger_links = _str_list_or_auto(
        doc.get("finger_links", ()), "robot.finger_links", auto_fields, "finger_links"
    )

    for fld in ("home_qpos", "kp", "kv", "force_lower", "force_upper"):
        if fld in doc:
            setattr(ri, fld, _float_list_or_auto(doc[fld], f"robot.{fld}", auto_fields, fld))

    if "n_arm_joints" in doc and not _is_auto(doc["n_arm_joints"]):
        ri.n_arm_joints = int(doc["n_arm_joints"])
    elif _is_auto(doc.get("n_arm_joints")):
        auto_fields.add("n_arm_joints")
    if "n_gripper_joints" in doc and not _is_auto(doc["n_gripper_joints"]):
        ri.n_gripper_joints = int(doc["n_gripper_joints"])
    elif _is_auto(doc.get("n_gripper_joints")):
        auto_fields.add("n_gripper_joints")

    for fld in ("gripper_open", "gripper_close"):
        if fld in doc and doc[fld] is not None:
            if _is_auto(doc[fld]):
                auto_fields.add(fld)
            else:
                setattr(ri, fld, _opt_float(doc[fld], f"robot.{fld}"))

    _validate_robot(ri)
    return ri


def _validate_robot(ri: RobotInput) -> None:
    """Validate the explicit (non-auto) parts of a RobotInput."""

    if not ri.name:
        raise OnboardingSpecError("robot.name must be non-empty")

    # Counts: when both arm + (gripper or zero) given, basic sanity.
    if "n_arm_joints" not in ri.auto_fields and ri.n_arm_joints < 0:
        raise OnboardingSpecError("robot.n_arm_joints must be >= 0")
    if "n_gripper_joints" not in ri.auto_fields and ri.n_gripper_joints < 0:
        raise OnboardingSpecError("robot.n_gripper_joints must be >= 0")

    # Explicit per-DoF vectors must agree in length with each other (the dof_count
    # consistency vs n_arm+n_gripper is enforced once derived into a RobotSpec).
    vectors = {
        name: getattr(ri, name)
        for name in ("home_qpos", "kp", "kv", "force_lower", "force_upper")
        if getattr(ri, name) and name not in ri.auto_fields
    }
    lengths = {len(v) for v in vectors.values()}
    if len(lengths) > 1:
        raise OnboardingSpecError(
            "explicit per-DoF vectors must share a length; got "
            + ", ".join(f"{k}={len(v)}" for k, v in vectors.items())
        )

    # A BYO robot doing a grasping task needs a gripper declared (or auto). We do
    # not enforce here (the skill is in the task block); compat is checked at the
    # spec level in validate_onboarding_spec once both blocks are parsed.


def _parse_task(doc: dict[str, Any]) -> TaskSpec:
    if not isinstance(doc, dict):
        raise OnboardingSpecError(f"'task' must be a mapping, got {type(doc).__name__}")

    ts = TaskSpec()
    skill = str(_require(doc, "skill", "task")).strip().lower()
    if skill not in SKILLS:
        raise OnboardingSpecError(f"task.skill must be one of {SKILLS}, got {skill!r}")
    ts.skill = skill

    ts.object_usd = str(doc.get("object_usd") or "").strip()
    ts.object_scale = str(doc.get("object_scale") or "").strip()

    goal = doc.get("goal_pos", doc.get("goal"))
    if _is_auto(goal):
        ts.goal_pos_auto = True
    elif goal not in (None, ""):
        if not isinstance(goal, (list, tuple)) or len(goal) != 3:
            raise OnboardingSpecError(
                f"task.goal_pos must be [x, y, z] or '{AUTO}', got {goal!r}"
            )
        try:
            ts.goal_pos = tuple(float(v) for v in goal)
        except (TypeError, ValueError) as exc:
            raise OnboardingSpecError(f"task.goal_pos must be numeric: {goal!r}") from exc

    if "lift_height_m" in doc:
        ts.lift_height_m = _opt_pos_float(doc["lift_height_m"], "task.lift_height_m")
    if "success_distance_m" in doc:
        ts.success_distance_m = _opt_pos_float(doc["success_distance_m"], "task.success_distance_m")
    if "success_threshold" in doc:
        ts.success_threshold = _opt_float(doc["success_threshold"], "task.success_threshold")
        if not 0.0 <= ts.success_threshold <= 1.0:
            raise OnboardingSpecError(
                f"task.success_threshold must be in [0, 1], got {ts.success_threshold}"
            )
    if "num_envs" in doc and doc["num_envs"] not in (None, ""):
        ts.num_envs = int(doc["num_envs"])
        if ts.num_envs < 0:
            raise OnboardingSpecError("task.num_envs must be >= 0")
    if "iterations" in doc and doc["iterations"] not in (None, ""):
        ts.iterations = int(doc["iterations"])
        if ts.iterations < 0:
            raise OnboardingSpecError("task.iterations must be >= 0")

    return ts


def parse_onboarding_spec(doc: dict[str, Any]) -> OnboardingSpec:
    """Parse + validate an onboarding spec document into an ``OnboardingSpec``."""

    if not isinstance(doc, dict):
        raise OnboardingSpecError(f"onboarding spec must be a mapping, got {type(doc).__name__}")

    schema = str(doc.get("schema") or ONBOARDING_SCHEMA).strip()
    if schema != ONBOARDING_SCHEMA:
        raise OnboardingSpecError(
            f"unsupported onboarding schema {schema!r}; expected {ONBOARDING_SCHEMA!r}"
        )

    if "robot" not in doc:
        raise OnboardingSpecError("onboarding spec requires a 'robot' block")
    if "task" not in doc:
        raise OnboardingSpecError("onboarding spec requires a 'task' block")

    spec = OnboardingSpec(
        schema=schema,
        robot=_parse_robot(doc["robot"]),
        task=_parse_task(doc["task"]),
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: OnboardingSpec) -> None:
    """Cross-block validation once robot + task are parsed."""

    robot, task = spec.robot, spec.task
    # A grasping skill on a real BYO robot needs a gripper declared, unless the
    # gripper morphology is left to auto-derivation. Catch the obvious mismatch
    # (explicit zero gripper joints + a grasping skill) before any GPU job.
    if (
        task.needs_gripper
        and not robot.is_stock_franka
        and "n_gripper_joints" not in robot.auto_fields
        and robot.n_gripper_joints == 0
        and not robot.gripper_joint_names
    ):
        raise OnboardingSpecError(
            f"task.skill={task.skill!r} requires a gripper, but robot {robot.name!r} "
            "declares no gripper (n_gripper_joints=0, no gripper_joint_names). Set "
            f"'auto' to derive it from the asset, or declare the gripper joints. A "
            "gripperless arm cannot lift/place."
        )


def load_onboarding_spec(path: str | Path) -> OnboardingSpec:
    """Load + parse an onboarding spec from a YAML (or JSON) file."""

    import yaml  # local import keeps module import light

    text = Path(path).read_text(encoding="utf-8")
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OnboardingSpecError(f"could not parse onboarding YAML {path}: {exc}") from exc
    return parse_onboarding_spec(doc)
