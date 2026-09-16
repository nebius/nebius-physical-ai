"""Bind manipulation experiments to explicit robot articulations and measured action layouts."""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module
import re
from types import SimpleNamespace

EMBODIMENTS = ("franka", "ur10e_robotiq85", "kinova_jaco7")
_PROFILES = {
    "franka": {
        "module": "franka", "configuration": "FRANKA_PANDA_CFG",
        "arm_joints": [f"panda_joint{i}" for i in range(1, 8)],
        "gripper_joints": ["panda_finger_joint1", "panda_finger_joint2"],
        "base_body": "panda_link0", "tool_body": "panda_hand", "tool_offset_m": [0.0, 0.0, 0.1034],
        "open_position": 0.04, "closed_position": 0.0,
    },
    "ur10e_robotiq85": {
        "module": "universal_robots", "configuration": "UR10e_ROBOTIQ_2F_85_CFG",
        "arm_joints": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        "gripper_joints": ["finger_joint"], "base_body": "base_link", "tool_body": "wrist_3_link",
        # The pinned native frame view unexpectedly includes the gripper's duplicate base_link.
        # Use a unique source body. The lift reward reads the target's world position.
        "sensor_source_body": "shoulder_link",
        "tool_offset_m": [0.0, 0.0, 0.145], "open_position": 0.0, "closed_position": 0.8,
    },
    "kinova_jaco7": {
        "module": "kinova", "configuration": "KINOVA_JACO2_N7S300_CFG",
        "arm_joints": [f"j2n7s300_joint_{i}" for i in range(1, 8)],
        "gripper_joints": [f"j2n7s300_joint_{part}_{i}" for part in ("finger", "finger_tip")
                           for i in range(1, 4)],
        "base_body": "j2n7s300_link_base", "tool_body": "j2n7s300_end_effector",
        "tool_offset_m": [0.0, 0.0, 0.0], "open_position": 0.2, "closed_position": 1.2,
    },
}


def embodiment_profile(name: str) -> dict:
    """Seal a supported articulation, gripper, and grasp-frame convention.

    Args:
        name: Explicit embodiment identifier.
    Returns:
        Independent JSON-compatible profile.
    Raises:
        ValueError: The requested embodiment is unsupported.
    """
    if name not in _PROFILES:
        raise ValueError(f"Unsupported manipulation embodiment: {name}")
    return {"schema": "npa.manipulation-embodiment.v1", "name": name,
            "arm_action_scale": 0.5, "root_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            **deepcopy(_PROFILES[name])}


def recipe_embodiment(recipe: dict) -> dict:
    """Resolve legacy Franka recipes and reject mutated embodiment bindings.

    Args:
        recipe: Sealed experiment recipe.
    Returns:
        Verified supported profile.
    Raises:
        ValueError: The declared profile disagrees with its supported definition.
    """
    declared = recipe.get("embodiment")
    expected = embodiment_profile(declared["name"] if declared else "franka")
    if declared is not None and declared != expected:
        raise ValueError("Sealed embodiment bindings differ from the supported profile")
    return expected


def configure_embodiment(config, recipe: dict) -> None:
    """Replace the task template's robot, actions, and end-effector frames.

    Args:
        config: Isaac Lab lift environment configuration.
        recipe: Sealed robot profile and experiment settings.
    Returns:
        None.
    Raises:
        ValueError: The profile is invalid.
        ImportError: The pinned Isaac robot configuration is unavailable.
    """
    profile = recipe_embodiment(recipe)
    if profile["name"] == "franka":
        return
    module = import_module("isaaclab_assets.robots." + profile["module"])
    robot = getattr(module, profile["configuration"]).copy().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.rot = tuple(profile["root_rotation_xyzw"])
    if profile["name"] == "ur10e_robotiq85":
        # Face the positive-X task workspace while retaining the identity XYZW base pose.
        robot.init_state.joint_pos["shoulder_pan_joint"] = 0.0
    config.scene.robot = robot
    _configure_actions(config, profile)
    config.commands.object_pose.body_name = profile["tool_body"]
    source_body = profile.get("sensor_source_body", profile["base_body"])
    config.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/" + source_body
    target = config.scene.ee_frame.target_frames[0]
    target.prim_path = "{ENV_REGEX_NS}/Robot/" + profile["tool_body"]
    target.offset.pos = tuple(profile["tool_offset_m"])


def _configure_actions(config, profile: dict) -> None:
    from isaaclab.envs import mdp

    config.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=profile["arm_joints"], preserve_order=True,
        scale=profile["arm_action_scale"], use_default_offset=True,
    )
    config.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=profile["gripper_joints"],
        open_command_expr={name: profile["open_position"] for name in profile["gripper_joints"]},
        close_command_expr={name: profile["closed_position"] for name in profile["gripper_joints"]},
    )


def embodiment_evidence(env, recipe: dict) -> dict:
    """Verify instantiated robot/action bindings and record their actual identity.

    Args:
        env: Live Isaac environment with initialized articulation and action manager.
        recipe: Sealed embodiment definition.
    Returns:
        Measured robot identity, joint/action names, and configured physical properties.
    Raises:
        ValueError: Live body, joint, or action layout differs from the sealed profile.
    """
    profile = recipe_embodiment(recipe)
    native = env.unwrapped
    robot = native.scene["robot"]
    arm = native.action_manager.get_term("arm_action")
    gripper = native.action_manager.get_term("gripper_action")
    arm_names = list(arm.IO_descriptor.joint_names)
    gripper_names = list(gripper.IO_descriptor.joint_names)
    if arm_names != profile["arm_joints"] or set(gripper_names) != set(profile["gripper_joints"]):
        raise ValueError("Actual policy-controlled joints differ from the sealed embodiment")
    if (native.action_manager.active_terms != ["arm_action", "gripper_action"]
            or arm.action_dim != len(arm_names) or gripper.action_dim != 1):
        raise ValueError("Actual policy action layout differs from the sealed embodiment")
    required_bodies = {profile["base_body"], profile["tool_body"],
                       profile.get("sensor_source_body", profile["base_body"])}
    if not required_bodies.issubset(robot.body_names):
        raise ValueError("Actual robot lacks the sealed base, sensor source, or tool body")
    controls = _control_evidence(native, profile, arm, gripper)
    semantics = _learning_controls(native, recipe, arm, controls)
    _validate_asset(native.cfg.scene.robot, profile)
    return {"profile": profile, "robot_type": profile["name"], "state_names": list(robot.joint_names),
            "action_names": [name + "_normalized_target" for name in arm_names] + ["gripper_open_close"],
            "gripper_joint_names": gripper_names, "body_names": list(robot.body_names),
            "asset": _asset_evidence(native.cfg.scene.robot),
            "controls": controls,
            "action_semantics": semantics}


def _learning_controls(native, recipe: dict, arm, controls: dict) -> str:
    from npa.workflows.franka_rl_learning import recipe_learning

    semantics = "Isaac joint position: default offset + 0.5 * arm action; negative closes gripper"
    learning = recipe_learning(recipe)
    if learning is None:
        return semantics
    if type(arm).__name__ != "BoundedJointPositionAction" or native.cfg.npa_learning != learning:
        raise ValueError("Actual learned servo differs from the sealed learning recipe")
    controls["learning"] = learning
    limits = arm._limits.detach().cpu()
    steps = arm._step_limit.detach().cpu()
    controls["joint_target_limits_rad"] = {"minimum": limits.amin(dim=0).tolist(),
                                           "maximum": limits.amax(dim=0).tolist()}
    controls["maximum_target_step_rad"] = {"minimum": steps.amin(dim=0).tolist(),
                                            "maximum": steps.amax(dim=0).tolist()}
    return semantics + "; arm targets constrained by physical joint limits and per-step target slew"


def _validate_asset(config, profile: dict) -> None:
    from npa.workflows.sim2real.isaac_assets_compat import remap_moved_franka_usd

    module = import_module("isaaclab_assets.robots." + profile["module"])
    expected = deepcopy(getattr(module, profile["configuration"]))
    remap_moved_franka_usd(SimpleNamespace(scene=SimpleNamespace(robot=expected)))
    if (config.spawn.usd_path != expected.spawn.usd_path
            or config.spawn.variants != expected.spawn.variants):
        raise ValueError("Actual robot USD or gripper variant differs from the selected upstream asset")
    if list(config.init_state.rot) != profile["root_rotation_xyzw"]:
        raise ValueError("Actual robot root rotation differs from the sealed profile")


def _control_evidence(native, profile: dict, arm, gripper) -> dict:
    frame = native.cfg.scene.ee_frame
    target = frame.target_frames[0]
    controls = {"arm_action_scale": arm.cfg.scale, "use_default_offset": arm.cfg.use_default_offset,
                "command_body": native.cfg.commands.object_pose.body_name,
                "base_frame": frame.prim_path, "tool_frame": target.prim_path,
                "tool_offset_m": list(target.offset.pos),
                "open_positions": _joint_commands(gripper.cfg.open_command_expr, profile["gripper_joints"]),
                "closed_positions": _joint_commands(gripper.cfg.close_command_expr, profile["gripper_joints"])}
    expected = {"arm_action_scale": profile["arm_action_scale"], "use_default_offset": True,
                "command_body": profile["tool_body"],
                "base_frame": native.scene.env_regex_ns + "/Robot/" + profile.get("sensor_source_body", profile["base_body"]),
                "tool_frame": native.scene.env_regex_ns + "/Robot/" + profile["tool_body"],
                "tool_offset_m": profile["tool_offset_m"],
                "open_positions": [profile["open_position"]] * len(profile["gripper_joints"]),
                "closed_positions": [profile["closed_position"]] * len(profile["gripper_joints"])}
    if len(frame.target_frames) != 1 or controls != expected:
        raise ValueError("Actual control or grasp-frame bindings differ from the sealed profile")
    return controls


def _joint_commands(expressions: dict, joints: list[str]) -> list[float]:
    values = [[value for pattern, value in expressions.items() if re.fullmatch(pattern, name)] for name in joints]
    if any(len(matches) != 1 for matches in values):
        raise ValueError("Gripper commands must match every controlled joint exactly once")
    return [matches[0] for matches in values]


def _asset_evidence(config) -> dict:
    return {"usd_path": config.spawn.usd_path, "variants": config.spawn.variants,
            "root_position_m": list(config.init_state.pos), "root_rotation_xyzw": list(config.init_state.rot),
            "initial_joint_positions": dict(config.init_state.joint_pos),
            "disable_gravity": config.spawn.rigid_props.disable_gravity,
            "self_collisions": config.spawn.articulation_props.enabled_self_collisions,
            "actuators": {name: {key: getattr(actuator, key) for key in
                          ("joint_names_expr", "effort_limit_sim", "velocity_limit_sim", "stiffness", "damping")}
                          for name, actuator in config.actuators.items()}}


def validate_capture_embodiment(metadata: dict, recipe: dict) -> None:
    """Reject captures whose robot identity or telemetry differs from their recipe.

    Args:
        metadata: Actual capture metadata.
        recipe: Sealed experiment recipe; historical recipes may omit embodiment.
    Returns:
        None.
    Raises:
        ValueError: Embodiment or named state/action identity is inconsistent.
    """
    if "embodiment" not in recipe:
        return
    profile = recipe_embodiment(recipe)
    actual = metadata.get("embodiment", {})
    if actual.get("profile") != profile or metadata.get("robot_type") != profile["name"]:
        raise ValueError("Capture embodiment does not match the sealed recipe")
    for field in ("state_names", "action_names", "action_semantics"):
        if metadata.get(field) != actual.get(field):
            raise ValueError("Capture telemetry differs from the measured embodiment")
    expected_actions = [name + "_normalized_target" for name in profile["arm_joints"]] + ["gripper_open_close"]
    required_joints = set(profile["arm_joints"] + profile["gripper_joints"])
    if actual["action_names"] != expected_actions or not required_joints.issubset(actual["state_names"]):
        raise ValueError("Capture named joints or actions differ from the sealed embodiment")
    from npa.workflows.franka_rl_learning import recipe_learning

    if actual.get("controls", {}).get("learning") != recipe_learning(recipe):
        raise ValueError("Capture learning controls differ from the sealed recipe")
