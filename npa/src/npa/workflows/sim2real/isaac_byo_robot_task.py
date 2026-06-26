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

# Bounds keep a garbage gain from producing a numerically unstable drive. The
# defaults sit well inside Franka's own range (kp up to 4500, kv up to 450).
STIFFNESS_MIN, STIFFNESS_MAX = 1.0, 100000.0
DAMPING_MIN, DAMPING_MAX = 0.1, 10000.0
EFFORT_MIN, EFFORT_MAX = 1.0, 10000.0

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


def register(spec: dict[str, Any] | None = None) -> str | None:
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
    compat = task_robot_compatibility(spec, task_kind="lift")
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
            # Retarget the Franka-hardcoded ee_frame / actions / command body onto
            # the swapped robot's link & joint names (the seam's core gap).
            _apply_retarget(self)

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
overrides = robotmod.robot_articulation_overrides(spec)
retarget = robotmod.task_retarget_overrides(spec)
compat = robotmod.task_robot_compatibility(spec, task_kind="lift")
print("ROBOT_SPEC", spec, flush=True)
print("ROBOT_OVERRIDES", overrides, flush=True)
print("ROBOT_RETARGET_PLAN", retarget, flush=True)
print("ROBOT_COMPAT", compat, flush=True)
if not compat.get("task_robot_compatible", True):
    print("ROBOT_TASK_INCOMPATIBLE", compat.get("reason"), flush=True)
try:
    task = robotmod.register(spec)
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
