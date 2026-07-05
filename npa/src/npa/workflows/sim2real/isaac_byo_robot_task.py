"""Route a customer robot_spec into a registered Isaac-Lab Lift task variant.

The BYO Isaac trainer/eval run the stock ``Isaac-Lift-Cube-Franka-v0`` task, so a
customer ``robot_spec`` (USD/URDF + joints + ee_link + gains) never reaches RL
training — it is only assets-stage provenance plus an in-process held-out-eval USD
swap. This module closes that gap for training: it registers a task *variant* that
subclasses the Franka Lift env cfg and swaps in the customer's robot articulation
(spawn USD, init joint positions, actuator gains), registered POST-AppLauncher-
boot. The trainer ships this module into the Isaac container and runs the post-boot
wrapper against ``NPA_BYO_ROBOT_TASK_ID``.

This mirrors ``isaac_physics_task.py`` exactly: the pure helpers (``robot_spec_
from_env``, ``robot_articulation_overrides``) are unit-tested off-GPU; ``register``
touches Isaac-Lab internals and is exercised on-cluster (it imports gymnasium /
isaaclab, unavailable off-GPU).

Opt-in: with no robot spec env var set, ``robot_spec_from_env`` returns ``None``
and ``register`` is a no-op, so the loop stays on the stock task — the proven path
is untouched. A ``stock_franka`` spec produces no overrides, so the variant
degenerates to the stock task (the mechanism is proven without changing the
policy). Only a real BYO robot (with a resolved USD) actually swaps the
articulation.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

STOCK_TASK_ID = "Isaac-Lift-Cube-Franka-v0"
STOCK_ROBOT_SOURCE = "stock_franka"
# JSON blob carrying the (already S3-staged) customer robot fields, set by the
# trainer/eval wiring; mirrors how NPA_GEN_FRICTION/NPA_GEN_MASS_SCALE carry the
# generated physics for the physics variant.
ROBOT_SPEC_ENV = "NPA_BYO_ROBOT_SPEC_JSON"
# JSON blob carrying the B2-derived robot-aware task config (action scale, object
# + goal placement, reward thresholds, gripper targets). Set by the onboarding
# CLI / trainer so the Lift variant is scaled to the robot instead of using the
# Franka-tuned stock numbers. Unset -> the variant keeps the stock task config
# (only the articulation + link/joint names are swapped).
TASK_CONFIG_ENV = "NPA_BYO_TASK_CONFIG_JSON"

# Bounds keep a garbage gain from producing a numerically unstable drive. The
# defaults sit well inside Franka's own range (kp up to 4500, kv up to 450).
STIFFNESS_MIN, STIFFNESS_MAX = 1.0, 100000.0
DAMPING_MIN, DAMPING_MAX = 0.1, 10000.0
EFFORT_MIN, EFFORT_MAX = 1.0, 10000.0

# Robot-agnostic gripper drive floors. The stock swap gives every joint one
# arm-averaged actuator group, so a manipulator's fingers inherit arm-shaped gains
# rather than a drive tuned for a light-object grasp. When a spec declares a
# gripper we give the finger joints their OWN actuator group with a modest
# stiffness/effort FLOOR (or the spec's own gripper_kp/kv/force when higher).
# Driven off ``gripper_joint_names`` (+ any per-gripper gain the spec supplies) —
# never off a specific robot's joint names.
#
# Values are sized for a small REVOLUTE finger holding a light tabletop object: a
# position-controlled finger's holding torque ≈ stiffness × (close_target −
# contact_angle); with a ~0.3 rad contact error a stiffness of ~20 N·m/rad yields
# a few N·m of firm-but-bounded clamp, capped by the effort floor. An earlier
# on-cluster trial with a much higher floor (400/200) DESTABILIZED the arm policy
# (the finger group double-actuates joints the catch-all group also drives, and an
# over-stiff finger drive injects large contact forces) — reaching collapsed and
# mean reward went negative. These modest floors keep the per-robot gripper drive
# sane and non-destabilizing. Franka/preset specs resolve no BYO gripper group
# (see ``robot_articulation_overrides``), so their path is unchanged.
GRIPPER_STIFFNESS_FLOOR = 20.0
GRIPPER_DAMPING_FLOOR = 4.0
GRIPPER_EFFORT_FLOOR = 10.0

# Stock ``Isaac-Lift-Cube-Franka-v0`` link/joint names hardcoded in the task cfg's
# ee_frame FrameTransformer, arm/gripper action terms, and the object_pose command
# body. The retarget mapping resolves to EXACTLY these for a stock-Franka spec (or
# any field a BYO spec omits), so the Franka path is byte-for-byte unchanged; a
# non-Franka spec resolves them to its own link/joint names instead.
FRANKA_BASE_LINK = "panda_link0"  # ee_frame source frame (articulation root)
FRANKA_EE_LINK = "panda_hand"  # ee_frame target frame + object_pose command body
EE_FRAME_NAME = "end_effector"  # stock target-frame name; kept stable for rewards
FRANKA_ARM_JOINT_EXPR = ["panda_joint.*"]  # JointPositionAction joint_names
FRANKA_GRIPPER_JOINT_EXPR = ["panda_finger.*"]  # BinaryJointPositionAction joints
FRANKA_GRIPPER_OPEN = {"panda_finger_.*": 0.04}
FRANKA_GRIPPER_CLOSE = {"panda_finger_.*": 0.0}

# Task kinds whose stock reward/action contract structurally requires a gripper.
GRIPPER_TASK_KINDS = ("lift", "manipulation", "stack", "pick", "place")


def _task_id(name: str) -> str:
    """Build a safe gym id from the robot name: ``NPA-Lift-Cube-<Name>-v0``."""

    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(name or "robot")).strip("-") or "robot"
    return f"NPA-Lift-Cube-{slug}-v0"


def _num_list(value: Any) -> list[float]:
    """Coerce a JSON value to a list of floats; drop anything unparseable."""

    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        try:
            x = float(item)
        except (TypeError, ValueError):
            continue
        if x == x:  # not NaN
            out.append(x)
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def robot_spec_from_env(env: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Parse the customer robot spec from ``NPA_BYO_ROBOT_SPEC_JSON``.

    Returns ``None`` when the env var is unset/empty/invalid OR when the spec is
    the stock Franka — in every such case the caller falls back to the stock task
    (BYO-robot routing disabled or no swap to perform).
    """

    env = os.environ if env is None else env
    raw = (env.get(ROBOT_SPEC_ENV) or "").strip()
    if not raw:
        return None
    try:
        spec = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(spec, dict):
        return None
    if str(spec.get("robot_source") or STOCK_ROBOT_SOURCE) == STOCK_ROBOT_SOURCE:
        return None
    return spec


def robot_articulation_overrides(spec: dict[str, Any] | None) -> dict[str, Any]:
    """Map a robot_spec dict to Lift-env ``scene.robot`` ArticulationCfg overrides.

    Returns the structured overrides applied post-boot by ``register``:
    ``usd_path`` (spawn), ``init_joint_pos`` (per-joint home), ``stiffness`` /
    ``damping`` / ``effort_limit`` (a single coarse actuator group), ``ee_link``.

    Returns ``{}`` for a stock-Franka spec or one without a resolved USD: there is
    nothing to swap, so the variant degenerates to the stock task. Per-joint
    actuator fidelity and reward retuning are deliberate non-goals (see
    ``docs/architecture/byo-robot-task-registration.md``).
    """

    if not isinstance(spec, dict):
        return {}
    if str(spec.get("robot_source") or STOCK_ROBOT_SOURCE) == STOCK_ROBOT_SOURCE:
        return {}
    usd_path = str(spec.get("usd_path") or spec.get("local_path") or "").strip()
    if not usd_path:
        return {}

    overrides: dict[str, Any] = {"usd_path": usd_path}

    ee_link = str(spec.get("ee_link") or "").strip()
    if ee_link:
        overrides["ee_link"] = ee_link

    joint_names = [str(n) for n in (spec.get("joint_names") or []) if str(n)]
    home_qpos = _num_list(spec.get("home_qpos"))
    if joint_names and home_qpos and len(joint_names) == len(home_qpos):
        overrides["init_joint_pos"] = dict(zip(joint_names, home_qpos))
    else:
        # Names/qpos absent or mismatched: cannot map per-joint reliably. Reset to
        # zero so the swapped arm starts from a defined pose rather than inheriting
        # the Franka's named init (whose joints won't exist on a different arm).
        overrides["init_joint_pos"] = {".*": 0.0}

    kp = _num_list(spec.get("kp"))
    kv = _num_list(spec.get("kv"))
    if kp:
        overrides["stiffness"] = round(_clamp(sum(kp) / len(kp), STIFFNESS_MIN, STIFFNESS_MAX), 6)
    if kv:
        overrides["damping"] = round(_clamp(sum(kv) / len(kv), DAMPING_MIN, DAMPING_MAX), 6)
    forces = [abs(f) for f in _num_list(spec.get("force_upper")) + _num_list(spec.get("force_lower"))]
    if forces:
        overrides["effort_limit"] = round(_clamp(max(forces), EFFORT_MIN, EFFORT_MAX), 6)

    # Dedicated gripper drive: when the spec declares finger joints, give them their
    # OWN actuator group with a stiffness/effort FLOOR so the fingers can clamp AND
    # hold the object rather than inheriting the (too-soft-for-gripping) arm-averaged
    # group. Robot-agnostic — the joint names come from the spec, the gains from any
    # ``gripper_kp``/``gripper_kv``/``gripper_force`` the spec supplies, else the
    # floors. Absent finger joints (bare arm / stock Franka) -> no group, path
    # unchanged.
    gripper_joints = [str(n) for n in (spec.get("gripper_joint_names") or []) if str(n)]
    if gripper_joints and _spec_has_gripper(spec):
        g_kp = _num_list(spec.get("gripper_kp"))
        g_kv = _num_list(spec.get("gripper_kv"))
        g_force = [abs(f) for f in _num_list(spec.get("gripper_force"))]
        g_stiff = max(sum(g_kp) / len(g_kp), GRIPPER_STIFFNESS_FLOOR) if g_kp else GRIPPER_STIFFNESS_FLOOR
        g_damp = max(sum(g_kv) / len(g_kv), GRIPPER_DAMPING_FLOOR) if g_kv else GRIPPER_DAMPING_FLOOR
        g_eff = max(max(g_force), GRIPPER_EFFORT_FLOOR) if g_force else GRIPPER_EFFORT_FLOOR
        overrides["gripper_actuator"] = {
            "joint_names": list(gripper_joints),
            "stiffness": round(_clamp(g_stiff, STIFFNESS_MIN, STIFFNESS_MAX), 6),
            "damping": round(_clamp(g_damp, DAMPING_MIN, DAMPING_MAX), 6),
            "effort_limit": round(_clamp(g_eff, EFFORT_MIN, EFFORT_MAX), 6),
        }

    return overrides


def _spec_has_gripper(spec: dict[str, Any]) -> bool:
    """Whether the spec declares an actuated gripper (finger joints + links).

    Mirrors ``RobotSpec.has_gripper``: a positive gripper-joint count AND at least
    one finger link. An explicit ``has_gripper`` flag, when present, wins.
    """

    if "has_gripper" in spec:
        return bool(spec.get("has_gripper"))
    n_gripper = 0
    try:
        n_gripper = int(spec.get("n_gripper_joints") or 0)
    except (TypeError, ValueError):
        n_gripper = 0
    finger_links = [str(f) for f in (spec.get("finger_links") or []) if str(f)]
    return n_gripper > 0 and bool(finger_links)


def task_retarget_overrides(spec: dict[str, Any] | None) -> dict[str, Any]:
    """Map a robot_spec to the Lift task's link/joint *renames* (the seam's gap).

    Swapping the articulation USD alone is not enough: the stock task cfg hardcodes
    Franka link/joint names in three places that break on a different arm —

      * ``scene.ee_frame`` FrameTransformer source (``panda_link0``) + target
        (``panda_hand``) prim paths,
      * ``actions.arm_action`` joint names (``panda_joint.*``) and
        ``actions.gripper_action`` joint names / open-close command expressions
        (``panda_finger.*``),
      * ``commands.object_pose`` resolution body (``panda_hand``).

    This pure function returns the rename set ``register`` applies post-boot. Every
    field a BYO spec omits resolves to the Franka stock value, so a stock-Franka (or
    field-less) spec yields the panda_* names verbatim and the Franka path is
    unchanged. Returns ``{}`` when there is nothing to swap (no/stock/USD-less spec),
    matching ``robot_articulation_overrides``.

    ``gripper`` is ``None`` when the spec declares no gripper (e.g. a bare UR/Flexiv
    arm): the caller then surfaces a task/robot incompatibility (see
    ``task_robot_compatibility``) rather than wiring a gripper action onto an arm
    that has none.
    """

    if not isinstance(spec, dict):
        return {}
    if str(spec.get("robot_source") or STOCK_ROBOT_SOURCE) == STOCK_ROBOT_SOURCE:
        return {}
    usd_path = str(spec.get("usd_path") or spec.get("local_path") or "").strip()
    if not usd_path:
        return {}

    base_link = str(spec.get("base_link") or "").strip() or FRANKA_BASE_LINK
    ee_link = str(spec.get("ee_link") or "").strip() or FRANKA_EE_LINK

    joint_names = [str(n) for n in (spec.get("joint_names") or []) if str(n)]
    if joint_names:
        try:
            n_arm = int(spec.get("n_arm_joints") or 0)
        except (TypeError, ValueError):
            n_arm = 0
        arm_joint_names = joint_names[:n_arm] if n_arm > 0 else list(joint_names)
    else:
        arm_joint_names = list(FRANKA_ARM_JOINT_EXPR)

    overrides: dict[str, Any] = {
        "ee_frame_source": base_link,
        "ee_frame_target": ee_link,
        "ee_frame_name": EE_FRAME_NAME,
        "arm_joint_names": arm_joint_names,
        "command_body_name": ee_link,
    }

    if _spec_has_gripper(spec):
        # A declared gripper. Without explicit gripper joint names we fall back to
        # the Franka finger pattern (correct only for a Franka-class hand); a
        # non-Franka gripper must declare ``gripper_joint_names`` to retarget the
        # finger joints — recorded as a remaining requirement.
        gripper_joints = [str(n) for n in (spec.get("gripper_joint_names") or []) if str(n)]
        try:
            open_pos = float(spec.get("gripper_open", FRANKA_GRIPPER_OPEN["panda_finger_.*"]))
        except (TypeError, ValueError):
            open_pos = FRANKA_GRIPPER_OPEN["panda_finger_.*"]
        try:
            close_pos = float(spec.get("gripper_close", FRANKA_GRIPPER_CLOSE["panda_finger_.*"]))
        except (TypeError, ValueError):
            close_pos = FRANKA_GRIPPER_CLOSE["panda_finger_.*"]
        if gripper_joints:
            expr = "|".join(re.escape(j) for j in gripper_joints)
            overrides["gripper"] = {
                "joint_names": list(gripper_joints),
                "open": {expr: open_pos},
                "close": {expr: close_pos},
            }
        else:
            overrides["gripper"] = {
                "joint_names": list(FRANKA_GRIPPER_JOINT_EXPR),
                "open": {"panda_finger_.*": open_pos},
                "close": {"panda_finger_.*": close_pos},
            }
    else:
        overrides["gripper"] = None

    return overrides


def task_robot_compatibility(
    spec: dict[str, Any] | None, task_kind: str = "lift"
) -> dict[str, Any]:
    """Whether ``spec``'s embodiment can physically perform ``task_kind``.

    Honest gate, not retargeting: a cube-lift (and every manipulation task built on
    the stock Lift reward) needs an actuated end-effector to grasp the object. A
    gripperless arm (a bare UR10/UR5e/Flexiv) cannot lift no matter how the links
    are renamed, so we report ``task_robot_compatible=False`` with the reason and
    the customer requirement rather than training a policy that can never succeed.

    A stock-Franka / ``None`` spec is the proven, gripper-bearing path → compatible.
    """

    kind = str(task_kind or "lift").strip().lower()
    needs_gripper = any(k in kind for k in GRIPPER_TASK_KINDS)

    if not isinstance(spec, dict) or str(
        spec.get("robot_source") or STOCK_ROBOT_SOURCE
    ) == STOCK_ROBOT_SOURCE:
        return {
            "task_robot_compatible": True,
            "task_kind": kind,
            "has_gripper": True,
            "reason": "stock Franka (proven gripper-bearing embodiment)",
            "requirements": [],
        }

    has_gripper = _spec_has_gripper(spec)
    if needs_gripper and not has_gripper:
        return {
            "task_robot_compatible": False,
            "task_kind": kind,
            "has_gripper": False,
            "reason": (
                f"task '{kind}' requires grasping but robot_spec '"
                f"{spec.get('name') or 'robot'}' declares no gripper "
                "(n_gripper_joints=0 / no finger_links). A gripperless arm "
                "cannot lift a cube regardless of link/joint retargeting."
            ),
            "requirements": [
                "a parallel-jaw (or equivalent) gripper actuated in sim",
                "gripper finger joint names (gripper_joint_names) to retarget the "
                "BinaryJointPositionAction term",
                "gripper_open / gripper_close command targets",
                "an action space matching arm DoF + gripper (stock Lift = arm + "
                "1 binary gripper)",
            ],
        }

    return {
        "task_robot_compatible": True,
        "task_kind": kind,
        "has_gripper": has_gripper,
        "reason": (
            "embodiment has the actuators the task requires"
            if not needs_gripper or has_gripper
            else "task does not require a gripper"
        ),
        "requirements": [],
    }


def task_config_from_env(env: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Parse the B2-derived robot-aware task config from ``NPA_BYO_TASK_CONFIG_JSON``.

    Returns ``None`` when unset/empty/invalid, in which case the variant keeps the
    stock Lift task numbers (articulation swap only).
    """

    env = os.environ if env is None else env
    raw = (env.get(TASK_CONFIG_ENV) or "").strip()
    if not raw:
        return None
    try:
        cfg = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return cfg if isinstance(cfg, dict) else None


def _range_dict(value: Any) -> dict[str, tuple[float, float]]:
    """Coerce a {axis: [lo, hi]} mapping to {axis: (lo, hi)} floats; drop bad axes."""

    out: dict[str, tuple[float, float]] = {}
    if not isinstance(value, dict):
        return out
    for axis in ("x", "y", "z"):
        pair = value.get(axis)
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            try:
                out[axis] = (float(pair[0]), float(pair[1]))
            except (TypeError, ValueError):
                continue
    return out


def _scale_triple(value: Any) -> tuple[float, float, float] | None:
    """Coerce an object-scale value to a positive ``(x, y, z)`` tuple, or ``None``.

    Accepts a scalar (uniform scale) or a 3-element list/tuple. A small-aperture
    gripper (e.g. a parallel jaw) cannot force-close on the stock ~5 cm Lift cube,
    so the manipuland must be shrunk to fit; this normalizes whatever the derived
    task config carries into the tuple the object spawn cfg expects. Non-positive
    or malformed values return ``None`` (term dropped) rather than corrupting the
    scene — mirrors the other ``task_config_overrides`` fields.
    """

    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        return None
    if isinstance(value, (int, float)):
        s = float(value)
        return (s, s, s) if s > 0 else None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            triple = tuple(float(v) for v in value)
        except (TypeError, ValueError):
            return None
        return triple if all(v > 0 for v in triple) else None
    return None


def task_config_overrides(task_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Map a B2-derived task-config dict to the Lift-env mutations ``register`` applies.

    Pure + unit-tested off-GPU (the GPU-side ``_apply_task_config`` consumes this).
    Returns ``{}`` for ``None`` / non-dict so the variant keeps the stock numbers.
    Includes only the fields the derived config actually provides, so a partial
    config touches only what it specifies.
    """

    if not isinstance(task_cfg, dict):
        return {}
    out: dict[str, Any] = {}

    action_scale = task_cfg.get("action_scale")
    if action_scale is not None:
        try:
            out["action_scale"] = float(action_scale)
        except (TypeError, ValueError):
            pass

    # Object scale: shrink the Lift manipuland to the gripper's aperture. Only
    # carried when the config explicitly sets it, so the Franka path (never sets
    # it) and any large-gripper robot keep the stock object. See _scale_triple.
    scale = _scale_triple(task_cfg.get("object_scale"))
    if scale is not None:
        out["object_scale"] = scale

    obj_range = _range_dict(task_cfg.get("object_init_range"))
    if obj_range:
        out["object_init_range"] = obj_range
    goal_range = _range_dict(task_cfg.get("goal_range"))
    if goal_range:
        out["goal_range"] = goal_range

    goal_pos = _num_list(task_cfg.get("goal_pos"))
    if len(goal_pos) == 3:
        out["goal_pos"] = tuple(goal_pos)

    for key in ("minimal_height_m", "success_distance_m", "gripper_open", "gripper_close"):
        val = task_cfg.get(key)
        if val is not None:
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                continue

    # Dense, robot-agnostic lift-progress shaping (B5 structural fix): the stock
    # lifting_object reward is a STEP at minimal_height, so partial lifts give zero
    # gradient — a non-Franka arm reaches the object but the grasp->lift never
    # bootstraps. dense_lift_weight (>0) adds a continuous reward for raising the
    # object above its spawn height, breaking the flat region. Stock-Franka derived
    # config never sets it, so the Franka path is unchanged.
    dlw = task_cfg.get("dense_lift_weight")
    if dlw is not None:
        try:
            w = float(dlw)
            if w > 0:
                out["dense_lift_weight"] = w
                std = task_cfg.get("dense_lift_std")
                out["dense_lift_std"] = float(std) if std is not None else 0.05
        except (TypeError, ValueError):
            pass

    # Grasp-shaping reward weight (>0): reward closing the gripper near the object,
    # breaking the grasp<->lift chicken-and-egg. Same gating/Franka-safety as above.
    gsw = task_cfg.get("grasp_shaping_weight")
    if gsw is not None:
        try:
            w = float(gsw)
            if w > 0:
                out["grasp_shaping_weight"] = w
                std = task_cfg.get("grasp_shaping_std")
                out["grasp_shaping_std"] = float(std) if std is not None else 0.06
        except (TypeError, ValueError):
            pass

    # Grasp-hold reward weight (>0): reward a MAINTAINED grasp while the object is
    # lifted (closedness x height) so the gripper learns to keep holding rather than
    # bat the object and reopen. Complements grasp_shaping (close-near precursor) and
    # dense_lift (height only). Same gating/Franka-safety: stock-derived config never
    # sets it, so the Franka path is unchanged.
    ghw = task_cfg.get("grasp_hold_weight")
    if ghw is not None:
        try:
            w = float(ghw)
            if w > 0:
                out["grasp_hold_weight"] = w
                std = task_cfg.get("grasp_hold_std")
                out["grasp_hold_std"] = float(std) if std is not None else 0.05
        except (TypeError, ValueError):
            pass

    return out


def object_lift_progress(env, std: float = 0.05, object_name: str = "object"):
    """Continuous, robot-agnostic reward for raising the object above its spawn height.

    The stock Isaac-Lift ``lifting_object`` reward is a STEP (height > minimal_height
    -> 1), and ``object_goal_tracking`` is gated on the same threshold, so before the
    first clean grasp+lift the whole lift phase has ZERO gradient. A non-Franka arm
    learns to reach the object (reaching_object climbs) but the 3-finger grasp->lift
    is never discovered, so lifting_object stays flat. This term rewards ANY upward
    object motion (``tanh`` of the height gained over spawn), giving PPO a gradient
    to bootstrap the grasp. Depends only on the object, not the embodiment.

    Defined at module level so it ships in ``module_source`` and runs in-container.
    Imports torch lazily (GPU-only). Returns a per-env tensor.
    """

    import torch  # noqa: WPS433 (GPU-only; lazy so the module imports off-GPU)

    obj = env.scene[object_name]
    z = obj.data.root_pos_w[:, 2]
    # default_root_state is env-local; add env origin z for a world-frame baseline.
    z0 = obj.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    gained = torch.clamp(z - z0, min=0.0)
    return torch.tanh(gained / float(std))


_GRASP_WARNED = {"done": False}


def grasp_shaping(
    env,
    std: float = 0.06,
    object_name: str = "object",
    ee_frame_name: str = "ee_frame",
    gripper_joint_names: tuple[str, ...] | list[str] | None = None,
    gripper_open: float = 0.0,
    gripper_close: float = 1.0,
):
    """Reward closing the gripper while the ee is at the object (grasp precursor).

    The deepest BYO-lift wall: the gripper-close action has NO reward gradient
    until a clean grasp+lift, but the lift needs the close — a chicken-and-egg that
    even a dense lift reward can't break, because the object never moves during
    random exploration (dense_lift_progress stays 0 while reaching_object climbs).
    This term gives a gradient to CLOSE the gripper when near the object: reward =
    near(ee,object) * gripper_closedness. Robot-agnostic — the finger joints +
    open/close targets come from the retarget/spec. Once the policy closes on the
    object, the dense lift reward takes over.

    Fully defensive: any per-step error returns zeros (logged once) so a cfg/API
    shape mismatch contributes 0 instead of crashing the run.
    """

    import torch  # noqa: WPS433 (GPU-only)

    try:
        obj = env.scene[object_name]
        ee = env.scene[ee_frame_name]
        ee_w = ee.data.target_pos_w[..., 0, :]
        obj_w = obj.data.root_pos_w
        dist = torch.norm(obj_w - ee_w, dim=1)
        near = 1.0 - torch.tanh(dist / float(std))
        robot = env.scene["robot"]
        if gripper_joint_names:
            idx, _ = robot.find_joints(list(gripper_joint_names))
            qpos = robot.data.joint_pos[:, idx].mean(dim=1)
            denom = float(gripper_close) - float(gripper_open)
            denom = denom if abs(denom) > 1e-6 else 1.0
            closed = ((qpos - float(gripper_open)) / denom).clamp(0.0, 1.0)
        else:
            closed = torch.ones_like(near)
        return near * closed
    except Exception as exc:  # noqa: BLE001 (never crash training on a reward call)
        if not _GRASP_WARNED["done"]:
            print("ROBOT_GRASP_SHAPING_ERR", repr(exc), flush=True)
            _GRASP_WARNED["done"] = True
        import torch as _t  # noqa: WPS433

        return _t.zeros(env.num_envs, device=env.device)


_GRASP_HOLD_WARNED = {"done": False}


def grasp_lift_hold(
    env,
    std: float = 0.05,
    object_name: str = "object",
    ee_frame_name: str = "ee_frame",
    gripper_joint_names: tuple[str, ...] | list[str] | None = None,
    gripper_open: float = 0.0,
    gripper_close: float = 1.0,
):
    """Reward raising a CLOSED gripper that is at the object — a bootstrap-capable
    grasp+lift signal.

    The deepest BYO-lift wall (5 prior escalation rounds + on-cluster confirmation):
    a non-Franka arm learns to REACH and to CLOSE near the object (``grasp_shaping``
    saturates), but the object's height stays EXACTLY 0 across 1000 iters — the arm
    never even attempts to raise itself while gripping. Any reward gated on *object*
    height (``object_lift_progress``, or a naive closed×object_height term) is dead
    flat there: it cannot bootstrap the lift because the object never moves, the
    same chicken-and-egg that defeated the dense object-height reward.

    This term instead multiplies three quantities the policy can influence from the
    reached+closed state IMMEDIATELY: ``near(ee, object) × closedness ×
    tanh(ee_height_above_table / std)``. Raising the (closed) end-effector is
    directly controllable, so there is a live gradient the instant the arm is
    closed at the object — that is the missing "attempt the lift" signal. Crucially
    the ``near`` factor keeps it honest: lifting the hand away from a NOT-grasped
    object drops ``near`` toward 0, so sustained reward requires the object to
    travel UP WITH the hand — i.e. a grasp that actually holds. Robot-agnostic:
    closedness from the spec's ``gripper_joint_names`` + open/close targets, heights
    from the object/ee frames only. Once a real hold forms and the object rises, the
    dense lift term + stock ``lifting_object`` / ``object_goal_tracking`` take over.

    Fully defensive: any per-step error returns zeros (logged once) so a cfg/API
    shape mismatch contributes 0 instead of crashing the run.
    """

    import torch  # noqa: WPS433 (GPU-only)

    try:
        obj = env.scene[object_name]
        ee = env.scene[ee_frame_name]
        ee_w = ee.data.target_pos_w[..., 0, :]
        obj_w = obj.data.root_pos_w
        dist = torch.norm(obj_w - ee_w, dim=1)
        near = 1.0 - torch.tanh(dist / float(std))
        # ee height gained above the object's spawn (table) height — controllable
        # from the reached pose, so this bootstraps where object-height cannot.
        table_z = obj.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
        ee_gain = torch.clamp(ee_w[:, 2] - table_z, min=0.0)
        height = torch.tanh(ee_gain / float(std))
        robot = env.scene["robot"]
        if gripper_joint_names:
            idx, _ = robot.find_joints(list(gripper_joint_names))
            qpos = robot.data.joint_pos[:, idx].mean(dim=1)
            denom = float(gripper_close) - float(gripper_open)
            denom = denom if abs(denom) > 1e-6 else 1.0
            closed = ((qpos - float(gripper_open)) / denom).clamp(0.0, 1.0)
        else:
            closed = torch.ones_like(near)
        return near * closed * height
    except Exception as exc:  # noqa: BLE001 (never crash training on a reward call)
        if not _GRASP_HOLD_WARNED["done"]:
            print("ROBOT_GRASP_HOLD_ERR", repr(exc), flush=True)
            _GRASP_HOLD_WARNED["done"] = True
        import torch as _t  # noqa: WPS433

        return _t.zeros(env.num_envs, device=env.device)


def register(spec: dict[str, Any] | None = None, task_cfg: dict[str, Any] | None = None) -> str | None:
    """Register the BYO-robot Lift variant in the gym registry; return its id.

    No-op (returns ``None``) when there are no articulation overrides (no spec, or
    a stock-Franka spec, or a spec without a resolved USD). Imports Isaac-Lab
    lazily so this module stays importable off-GPU for unit tests.
    """

    spec = spec if spec is not None else robot_spec_from_env()
    overrides = robot_articulation_overrides(spec)
    if not overrides:
        return None

    retarget = task_retarget_overrides(spec)
    task_over = task_config_overrides(task_cfg if task_cfg is not None else task_config_from_env())
    compat = task_robot_compatibility(spec, task_kind="lift")
    print("ROBOT_TASKCFG_PLAN", json.dumps(task_over, default=list), flush=True)
    # Honest, loud signal: a gripperless arm cannot lift no matter the renames. We
    # still register (so the swap/retarget mechanism is exercised), but the trainer
    # surfaces this in the report and the operator is not misled into expecting
    # success. See task_robot_compatibility() for the customer requirements.
    print("ROBOT_COMPAT", json.dumps(compat), flush=True)
    if not compat.get("task_robot_compatible", True):
        print("ROBOT_TASK_INCOMPATIBLE", compat.get("reason"), flush=True)

    import gymnasium as gym  # noqa: WPS433 (lazy: GPU-only dep)
    import isaaclab.sim as sim_utils  # noqa: WPS433
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka import (  # noqa: WPS433
        joint_pos_env_cfg as franka_lift,
    )

    usd_path = overrides["usd_path"]
    init_joint_pos = overrides.get("init_joint_pos")
    stiffness = overrides.get("stiffness")
    damping = overrides.get("damping")
    effort_limit = overrides.get("effort_limit")
    gripper_actuator = overrides.get("gripper_actuator")
    task_id = _task_id((spec or {}).get("name") or "robot")

    def _swap_prim_tail(prim_path: str, link: str) -> str:
        """Replace the trailing link segment of a ``.../Robot/<link>`` prim path,
        preserving the ``{ENV_REGEX_NS}/Robot`` namespace the task assigned."""

        base = str(prim_path or "").rsplit("/", 1)[0] or "{ENV_REGEX_NS}/Robot"
        return f"{base}/{link}"

    def _apply_retarget(env_cfg: Any) -> None:
        """Retarget the Franka-hardcoded link/joint names onto the swapped robot.

        Defensive: each term is guarded so a cfg-shape change on a newer Isaac-Lab
        skips that term (recorded via ROBOT_RETARGET) instead of crashing the run.
        """

        applied: list[str] = []
        skipped: list[str] = []

        # (a) ee_frame FrameTransformer: source + target prim paths, target name.
        try:
            ee_frame = getattr(env_cfg.scene, "ee_frame", None)
            if ee_frame is not None and hasattr(ee_frame, "prim_path"):
                ee_frame.prim_path = _swap_prim_tail(
                    ee_frame.prim_path, retarget["ee_frame_source"]
                )
                targets = getattr(ee_frame, "target_frames", None) or []
                for tf in targets:
                    if hasattr(tf, "prim_path"):
                        tf.prim_path = _swap_prim_tail(
                            tf.prim_path, retarget["ee_frame_target"]
                        )
                    if hasattr(tf, "name"):
                        tf.name = retarget["ee_frame_name"]
                applied.append("ee_frame")
            else:
                skipped.append("ee_frame")
        except Exception as exc:  # noqa: BLE001 (record, do not crash the run)
            print("ROBOT_RETARGET_ERR ee_frame", repr(exc), flush=True)
            skipped.append("ee_frame")

        # (b) action terms: arm joint names + (optional) gripper joints/commands.
        try:
            actions = getattr(env_cfg, "actions", None)
            arm_action = getattr(actions, "arm_action", None)
            if arm_action is not None and hasattr(arm_action, "joint_names"):
                arm_action.joint_names = list(retarget["arm_joint_names"])
                applied.append("arm_action.joint_names")
            elif arm_action is not None and hasattr(arm_action, "body_name"):
                # IK-style action variant: the controlled body is the ee link.
                arm_action.body_name = retarget["ee_frame_target"]
                applied.append("arm_action.body_name")
            else:
                skipped.append("arm_action")

            gripper_action = getattr(actions, "gripper_action", None)
            gripper = retarget.get("gripper")
            if gripper_action is not None and gripper is not None:
                if hasattr(gripper_action, "joint_names"):
                    gripper_action.joint_names = list(gripper["joint_names"])
                if hasattr(gripper_action, "open_command_expr"):
                    gripper_action.open_command_expr = dict(gripper["open"])
                if hasattr(gripper_action, "close_command_expr"):
                    gripper_action.close_command_expr = dict(gripper["close"])
                applied.append("gripper_action")
            elif gripper_action is not None and gripper is None:
                # No gripper on this robot: leave the stock term in place but flag it
                # (the env build may still fail; compatibility already reported it).
                skipped.append("gripper_action(no-gripper)")
        except Exception as exc:  # noqa: BLE001
            print("ROBOT_RETARGET_ERR actions", repr(exc), flush=True)
            skipped.append("actions")

        # (c) command term: object_pose resolves against the ee body.
        try:
            commands = getattr(env_cfg, "commands", None)
            object_pose = getattr(commands, "object_pose", None)
            if object_pose is not None and hasattr(object_pose, "body_name"):
                object_pose.body_name = retarget["command_body_name"]
                applied.append("commands.object_pose.body_name")
            else:
                skipped.append("commands.object_pose")
        except Exception as exc:  # noqa: BLE001
            print("ROBOT_RETARGET_ERR commands", repr(exc), flush=True)
            skipped.append("commands.object_pose")

        print("ROBOT_RETARGET applied=%s skipped=%s" % (applied, skipped), flush=True)

    def _apply_task_config(env_cfg: Any) -> None:
        """Apply the B2-derived robot-aware task config onto the Lift env cfg.

        Drives the action scale, object-init + goal placement ranges, and the
        lift/goal reward thresholds from ``task_over`` (derived from the robot's
        workspace) instead of the Franka-tuned stock numbers — the change that
        lets a non-Franka arm actually LEARN. No-op when ``task_over`` is empty
        (stock numbers kept). Each term is guarded so a cfg-shape change on a
        newer Isaac-Lab skips that term (recorded) rather than crashing the run.
        """

        if not task_over:
            print("ROBOT_TASKCFG applied=[] skipped=[] (no derived config)", flush=True)
            return

        applied: list[str] = []
        skipped: list[str] = []

        # (a) action scale: per-step joint-position offset magnitude.
        if "action_scale" in task_over:
            try:
                arm_action = getattr(getattr(env_cfg, "actions", None), "arm_action", None)
                if arm_action is not None and hasattr(arm_action, "scale"):
                    arm_action.scale = float(task_over["action_scale"])
                    applied.append("arm_action.scale")
                else:
                    skipped.append("arm_action.scale")
            except Exception as exc:  # noqa: BLE001
                print("ROBOT_TASKCFG_ERR action_scale", repr(exc), flush=True)
                skipped.append("arm_action.scale")

        # (b) object init placement: the reset_object_position event pose_range.
        if "object_init_range" in task_over:
            try:
                events = getattr(env_cfg, "events", None)
                rop = getattr(events, "reset_object_position", None)
                params = getattr(rop, "params", None)
                if isinstance(params, dict) and isinstance(params.get("pose_range"), dict):
                    params["pose_range"].update(
                        {k: tuple(v) for k, v in task_over["object_init_range"].items()}
                    )
                    applied.append("events.reset_object_position.pose_range")
                else:
                    skipped.append("events.reset_object_position.pose_range")
            except Exception as exc:  # noqa: BLE001
                print("ROBOT_TASKCFG_ERR object_init_range", repr(exc), flush=True)
                skipped.append("events.reset_object_position.pose_range")

        # (c) goal placement: object_pose command sampling ranges.
        if "goal_range" in task_over:
            try:
                commands = getattr(env_cfg, "commands", None)
                object_pose = getattr(commands, "object_pose", None)
                ranges = getattr(object_pose, "ranges", None)
                gr = task_over["goal_range"]
                set_any = False
                for axis, attr in (("x", "pos_x"), ("y", "pos_y"), ("z", "pos_z")):
                    if axis in gr and hasattr(ranges, attr):
                        setattr(ranges, attr, tuple(gr[axis]))
                        set_any = True
                applied.append("commands.object_pose.ranges") if set_any else skipped.append(
                    "commands.object_pose.ranges"
                )
            except Exception as exc:  # noqa: BLE001
                print("ROBOT_TASKCFG_ERR goal_range", repr(exc), flush=True)
                skipped.append("commands.object_pose.ranges")

        # (d) reward thresholds: lift minimal_height + goal-tracking std relative
        # to the (scaled) workspace, so the learning signal is robot-aware.
        if "minimal_height_m" in task_over:
            try:
                rewards = getattr(env_cfg, "rewards", None)
                mh = float(task_over["minimal_height_m"])
                touched = []
                for term_name in ("lifting_object", "object_goal_tracking",
                                  "object_goal_tracking_fine_grained"):
                    term = getattr(rewards, term_name, None)
                    params = getattr(term, "params", None)
                    if isinstance(params, dict) and "minimal_height" in params:
                        params["minimal_height"] = mh
                        touched.append(term_name)
                applied.append("rewards.minimal_height(%s)" % ",".join(touched)) if touched \
                    else skipped.append("rewards.minimal_height")
            except Exception as exc:  # noqa: BLE001
                print("ROBOT_TASKCFG_ERR minimal_height", repr(exc), flush=True)
                skipped.append("rewards.minimal_height")

        # (e) gripper close/open targets: complement the retarget command exprs
        # with the derived values (no-op when equal to the spec values).
        gripper_action = getattr(getattr(env_cfg, "actions", None), "gripper_action", None)
        if gripper_action is not None and (
            "gripper_open" in task_over or "gripper_close" in task_over
        ):
            try:
                if "gripper_open" in task_over and hasattr(gripper_action, "open_command_expr"):
                    expr = getattr(gripper_action, "open_command_expr", None) or {}
                    if isinstance(expr, dict) and expr:
                        gripper_action.open_command_expr = {
                            k: float(task_over["gripper_open"]) for k in expr
                        }
                if "gripper_close" in task_over and hasattr(gripper_action, "close_command_expr"):
                    expr = getattr(gripper_action, "close_command_expr", None) or {}
                    if isinstance(expr, dict) and expr:
                        gripper_action.close_command_expr = {
                            k: float(task_over["gripper_close"]) for k in expr
                        }
                applied.append("gripper_action.command_exprs")
            except Exception as exc:  # noqa: BLE001
                print("ROBOT_TASKCFG_ERR gripper", repr(exc), flush=True)
                skipped.append("gripper_action.command_exprs")

        # (f) dense lift-progress reward: add a continuous height-gain term so the
        # grasp->lift bootstraps past the stock step reward's flat region. Gated on
        # dense_lift_weight (>0); never set on the Franka path. Defensive — a cfg
        # shape change skips the term (recorded) instead of crashing the run.
        if "dense_lift_weight" in task_over:
            try:
                from isaaclab.managers import RewardTermCfg  # noqa: WPS433

                rewards = getattr(env_cfg, "rewards", None)
                term = RewardTermCfg(
                    func=object_lift_progress,
                    weight=float(task_over["dense_lift_weight"]),
                    params={
                        "std": float(task_over.get("dense_lift_std", 0.05)),
                        "object_name": "object",
                    },
                )
                setattr(rewards, "dense_lift_progress", term)
                applied.append("rewards.dense_lift_progress(%s)" % task_over["dense_lift_weight"])
            except Exception as exc:  # noqa: BLE001
                print("ROBOT_TASKCFG_ERR dense_lift", repr(exc), flush=True)
                skipped.append("rewards.dense_lift_progress")

        # (g) grasp-shaping reward: reward closing the gripper near the object, to
        # break the grasp<->lift chicken-and-egg (the object never moves under random
        # exploration, so no object-based reward can bootstrap the grasp). Gated on
        # grasp_shaping_weight (>0); finger joints come from the retarget plan.
        if "grasp_shaping_weight" in task_over:
            try:
                from isaaclab.managers import RewardTermCfg  # noqa: WPS433

                rewards = getattr(env_cfg, "rewards", None)
                grip = (retarget or {}).get("gripper") or {}
                term = RewardTermCfg(
                    func=grasp_shaping,
                    weight=float(task_over["grasp_shaping_weight"]),
                    params={
                        "std": float(task_over.get("grasp_shaping_std", 0.06)),
                        "object_name": "object",
                        "ee_frame_name": "ee_frame",
                        "gripper_joint_names": list(grip.get("joint_names") or []),
                        "gripper_open": float(task_over.get("gripper_open", 0.0)),
                        "gripper_close": float(task_over.get("gripper_close", 1.0)),
                    },
                )
                setattr(rewards, "grasp_shaping", term)
                applied.append("rewards.grasp_shaping(%s)" % task_over["grasp_shaping_weight"])
            except Exception as exc:  # noqa: BLE001
                print("ROBOT_TASKCFG_ERR grasp_shaping", repr(exc), flush=True)
                skipped.append("rewards.grasp_shaping")

        # (h) grasp-hold reward: reward a MAINTAINED grasp while the object is lifted
        # (closedness x height gained), so a held lift — not a transient bat-up — is
        # what pays. Gated on grasp_hold_weight (>0); finger joints from the retarget
        # plan. Never set on the Franka path. Defensive — records + skips on error.
        if "grasp_hold_weight" in task_over:
            try:
                from isaaclab.managers import RewardTermCfg  # noqa: WPS433

                rewards = getattr(env_cfg, "rewards", None)
                grip = (retarget or {}).get("gripper") or {}
                term = RewardTermCfg(
                    func=grasp_lift_hold,
                    weight=float(task_over["grasp_hold_weight"]),
                    params={
                        "std": float(task_over.get("grasp_hold_std", 0.05)),
                        "object_name": "object",
                        "ee_frame_name": "ee_frame",
                        "gripper_joint_names": list(grip.get("joint_names") or []),
                        "gripper_open": float(task_over.get("gripper_open", 0.0)),
                        "gripper_close": float(task_over.get("gripper_close", 1.0)),
                    },
                )
                setattr(rewards, "grasp_hold", term)
                applied.append("rewards.grasp_hold(%s)" % task_over["grasp_hold_weight"])
            except Exception as exc:  # noqa: BLE001
                print("ROBOT_TASKCFG_ERR grasp_hold", repr(exc), flush=True)
                skipped.append("rewards.grasp_hold")

        # (i) object scale: resize the Lift manipuland to the robot's gripper. A
        # small parallel-jaw aperture cannot force-close on the stock ~5 cm cube, so
        # the grasp->lift never completes; scaling the object down is what lets the
        # STRETCH lift succeed. Applied in BOTH training and held-out eval (register()
        # runs in each) so the policy is trained and scored on the SAME object size.
        # Gated on object_scale in the derived config; never set on the Franka path.
        if "object_scale" in task_over:
            try:
                obj = getattr(getattr(env_cfg, "scene", None), "object", None)
                spawn = getattr(obj, "spawn", None)
                if spawn is not None and hasattr(spawn, "scale"):
                    spawn.scale = tuple(task_over["object_scale"])
                    applied.append("scene.object.spawn.scale%s" % (tuple(task_over["object_scale"]),))
                else:
                    skipped.append("scene.object.spawn.scale")
            except Exception as exc:  # noqa: BLE001
                print("ROBOT_TASKCFG_ERR object_scale", repr(exc), flush=True)
                skipped.append("scene.object.spawn.scale")

        print("ROBOT_TASKCFG applied=%s skipped=%s" % (applied, skipped), flush=True)

    class _ByoRobotLiftEnvCfg(franka_lift.FrankaCubeLiftEnvCfg):  # type: ignore[name-defined]
        def __post_init__(self) -> None:  # noqa: WPS610
            super().__post_init__()
            robot_cfg = self.scene.robot
            spawn = getattr(robot_cfg, "spawn", None)
            new_spawn = sim_utils.UsdFileCfg(usd_path=usd_path)
            # Preserve articulation/rigid props from the task's spawn when present.
            for attr in ("articulation_props", "rigid_props", "activate_contact_sensors"):
                if hasattr(spawn, attr) and hasattr(new_spawn, attr):
                    setattr(new_spawn, attr, getattr(spawn, attr))
            robot_cfg.spawn = new_spawn
            if init_joint_pos and getattr(robot_cfg, "init_state", None) is not None:
                robot_cfg.init_state.joint_pos = dict(init_joint_pos)
            actuators = getattr(robot_cfg, "actuators", None)
            if isinstance(actuators, dict):
                for actuator in actuators.values():
                    # Widen joint patterns so an arbitrary arm's joints are actuated
                    # (the Franka groups key on Franka joint names).
                    if hasattr(actuator, "joint_names_expr"):
                        actuator.joint_names_expr = [".*"]
                    if stiffness is not None and hasattr(actuator, "stiffness"):
                        actuator.stiffness = stiffness
                    if damping is not None and hasattr(actuator, "damping"):
                        actuator.damping = damping
                    if effort_limit is not None and hasattr(actuator, "effort_limit"):
                        actuator.effort_limit = effort_limit
                        # Isaac 2.x ImplicitActuatorCfg forbids effort_limit !=
                        # effort_limit_sim; keep them in lockstep so the swap does
                        # not trip the actuator-cfg validation (first non-Franka
                        # break observed on-cluster).
                        if hasattr(actuator, "effort_limit_sim"):
                            actuator.effort_limit_sim = effort_limit
                # Dedicated gripper actuator group (added LAST so its higher gains win
                # for the finger joints — implicit-drive gains are written per joint,
                # last group processed wins). This is the fix that lets the fingers
                # actually CLAMP and HOLD: the arm-averaged group above leaves them
                # far too soft to exert holding force. Defensive: a cfg/import shape
                # change records + skips instead of crashing the (expensive) run. The
                # catch-all group above still covers every joint, so an unmodelled
                # extra joint (e.g. finger tips) stays actuated regardless.
                if gripper_actuator and gripper_actuator.get("joint_names"):
                    try:
                        from isaaclab.actuators import ImplicitActuatorCfg  # noqa: WPS433

                        g_eff = float(gripper_actuator["effort_limit"])
                        actuators["byo_gripper"] = ImplicitActuatorCfg(
                            joint_names_expr=list(gripper_actuator["joint_names"]),
                            stiffness=float(gripper_actuator["stiffness"]),
                            damping=float(gripper_actuator["damping"]),
                            effort_limit=g_eff,
                            effort_limit_sim=g_eff,
                        )
                        print(
                            "ROBOT_GRIPPER_ACTUATOR joints=%s stiffness=%s effort=%s"
                            % (
                                gripper_actuator["joint_names"],
                                gripper_actuator["stiffness"],
                                g_eff,
                            ),
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001 (never crash the run)
                        print("ROBOT_GRIPPER_ACTUATOR_ERR", repr(exc), flush=True)
            # Retarget the Franka-hardcoded ee_frame / actions / command body onto
            # the swapped robot's link & joint names (the seam's core gap).
            _apply_retarget(self)
            # Then drive the action scale / placement / reward thresholds from the
            # B2-derived robot-aware config so the swapped arm actually LEARNS
            # (not just runs). No-op when no derived config was supplied.
            _apply_task_config(self)

    stock_kwargs = gym.spec(STOCK_TASK_ID).kwargs
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _ByoRobotLiftEnvCfg,
            "rsl_rl_cfg_entry_point": stock_kwargs.get("rsl_rl_cfg_entry_point"),
        },
    )
    return task_id


def module_source() -> str:
    """Return this module's own source, for shipping into the Isaac container.

    The Isaac-Lab image has no ``npa`` package, so the BYO trainer writes this
    file into the job (via heredoc) and the train wrapper imports it post-boot.
    """

    from pathlib import Path

    return Path(__file__).read_text(encoding="utf-8")


# Post-boot train wrapper (runs INSIDE the Isaac-Lab container). Same hard
# constraint as the physics variant: isaaclab.sim / the franka lift cfg / gym.spec
# of the stock task pull USD ``pxr``, which only exists AFTER ``AppLauncher`` boots
# — so the variant registration MUST happen post-boot. This wrapper enforces:
#   (1) boot AppLauncher  (2) import isaaclab_tasks  (3) register the BYO variant
#   (4) ASSERT the registered env's robot USD == the customer USD (no silent stock
#   fallback)  (5) run the rsl_rl OnPolicyRunner like stock train.py.
# It reuses isaac_byo_robot_task.register() (shipped alongside) as the single
# source of truth for the articulation overrides.
TRAIN_WRAPPER_SCRIPT = r'''
import os, sys, traceback
SYS_DIR = os.environ.get("NPA_ROBOT_MODULE_DIR", "/tmp/npa_robot")
sys.path.insert(0, SYS_DIR)
NUM_ENVS = int(os.environ.get("ROBOT_NUM_ENVS", "64"))
ITERS = int(os.environ.get("ROBOT_ITERS", "2"))
SEED = int(os.environ.get("ROBOT_SEED", "0"))
OUT = os.environ.get("ROBOT_OUT_DIR", "/tmp/robotrun")
os.makedirs(OUT, exist_ok=True)
# (1) boot the sim app FIRST — everything Isaac/pxr must come after this.
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch
import gymnasium as gym
# (2) register the stock tasks, then (3) the BYO-robot variant (post-boot import).
import isaaclab_tasks  # noqa: F401
import isaac_byo_robot_task as robotmod
spec = robotmod.robot_spec_from_env()
task_cfg = robotmod.task_config_from_env()
overrides = robotmod.robot_articulation_overrides(spec)
retarget = robotmod.task_retarget_overrides(spec)
task_over = robotmod.task_config_overrides(task_cfg)
compat = robotmod.task_robot_compatibility(spec, task_kind="lift")
print("ROBOT_SPEC", spec, flush=True)
print("ROBOT_OVERRIDES", overrides, flush=True)
print("ROBOT_RETARGET_PLAN", retarget, flush=True)
print("ROBOT_TASKCFG_PLAN", task_over, flush=True)
print("ROBOT_COMPAT", compat, flush=True)
if not compat.get("task_robot_compatible", True):
    print("ROBOT_TASK_INCOMPATIBLE", compat.get("reason"), flush=True)
try:
    task = robotmod.register(spec, task_cfg)
except Exception:
    print("ROBOT_REGISTER_FAILED", flush=True); traceback.print_exc(); os._exit(42)
task = task or robotmod.STOCK_TASK_ID
print("ROBOT_TASK", task, flush=True)
# (4) build env + rsl_rl runner and train, like stock train.py.
try:
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except Exception:
        from omni.isaac.lab_rl.rsl_rl import RslRlVecEnvWrapper  # older layout
    from rsl_rl.runners import OnPolicyRunner
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=NUM_ENVS)
    if SEED:
        try:
            env_cfg.seed = SEED
        except Exception as e:
            print("could not set env_cfg.seed:", repr(e), flush=True)
        torch.manual_seed(SEED)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    acfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    acfg["max_iterations"] = ITERS
    # Guarantee a checkpoint even for a tiny probe run.
    acfg["save_interval"] = max(1, min(int(acfg.get("save_interval", 50) or 50), ITERS))
    if SEED:
        acfg["seed"] = SEED
    # Keep PPO exploring through the grasp bottleneck (same fix as the Franka
    # default path): without this the action-noise std collapses early and the
    # swapped arm locks into a reach-and-hover local optimum (the flat-reward
    # Kinova failure). ROBOT_ENTROPY_COEF="" / "stock" keeps the task default.
    ENT = os.environ.get("ROBOT_ENTROPY_COEF", "").strip()
    if ENT and ENT.lower() not in ("stock", "default", "none"):
        try:
            algo = acfg.get("algorithm")
            if isinstance(algo, dict):
                algo["entropy_coef"] = float(ENT)
                print("ROBOT_ENTROPY_COEF_SET", ENT, flush=True)
            else:
                print("ROBOT_ENTROPY_COEF_SKIP no algorithm dict", flush=True)
        except Exception as e:
            print("ROBOT_ENTROPY_COEF_ERR", repr(e), flush=True)
    print("ROBOT_AGENT_CFG_KEYS", sorted(acfg.keys()), flush=True)
    env = gym.make(task, cfg=env_cfg)
    # Definitive check that the customer robot is LIVE (not a silent stock
    # fallback): when overrides carry a usd_path, the built env's robot spawn USD
    # must equal it. If it doesn't, abort rather than train a Franka silently.
    want_usd = overrides.get("usd_path") if isinstance(overrides, dict) else None
    got_usd = None
    try:
        got_usd = getattr(env.unwrapped.scene["robot"].cfg.spawn, "usd_path", None)
    except Exception:
        try:
            got_usd = getattr(env.unwrapped.cfg.scene.robot.spawn, "usd_path", None)
        except Exception:
            got_usd = None
    print("ROBOT_USD want=%s got=%s" % (want_usd, got_usd), flush=True)
    robot_live = (want_usd is None) or (got_usd == want_usd)
    if want_usd is not None and got_usd != want_usd:
        print("ROBOT_USD_MISMATCH (refusing silent stock fallback)", flush=True)
        os._exit(44)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, acfg, log_dir=OUT, device="cuda:0")
    runner.learn(num_learning_iterations=ITERS, init_at_random_ep_len=True)
    # Defensive explicit save (learn saves at save_interval; ensure one exists).
    try:
        runner.save(os.path.join(OUT, "model_%d.pt" % ITERS))
    except Exception as e:
        print("explicit save failed (learn may have saved already):", repr(e), flush=True)
    import glob
    ckpts = sorted(glob.glob(os.path.join(OUT, "**", "model_*.pt"), recursive=True))
    # Final summary (survives any upstream log truncation): task + robot + USD live
    # + checkpoint count.
    print("ROBOT_SUMMARY task=%s robot=%s usd_live=%s ckpts=%d task_robot_compatible=%s"
          % (task, (spec or {}).get("name"), robot_live, len(ckpts),
             compat.get("task_robot_compatible", True)), flush=True)
    print("ROBOT_CKPTS", ckpts, flush=True)
    print("ROBOT_TRAIN_DONE" if (ckpts and robot_live) else "ROBOT_TRAIN_NO_CKPT", flush=True)
except Exception:
    print("ROBOT_TRAIN_FAILED", flush=True); traceback.print_exc(); os._exit(43)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
'''
