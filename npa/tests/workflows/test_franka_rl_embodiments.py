"""Verify robot selection, actual action order, and capture embodiment provenance."""

from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest

from npa.workflows import franka_rl
from npa.workflows import franka_rl_embodiments as embodiments


class Scene(dict):
    env_regex_ns = "/World/envs/env_.*"


def _robot_config(name):
    return SimpleNamespace(
        spawn=SimpleNamespace(
            usd_path=f"{name}.usd",
            variants={"Gripper": name},
            rigid_props=SimpleNamespace(disable_gravity=False),
            articulation_props=SimpleNamespace(enabled_self_collisions=True),
        ),
        init_state=SimpleNamespace(pos=(0, 0, 0), rot=(0, 0, 0, 1), joint_pos={}),
        actuators={},
    )


@pytest.fixture(autouse=True)
def upstream_assets(monkeypatch):
    configs = {
        embodiments.embodiment_profile(name)["configuration"]: _robot_config(name)
        for name in embodiments.EMBODIMENTS
    }
    monkeypatch.setattr(
        embodiments, "import_module", lambda name: SimpleNamespace(**configs)
    )


def _live_environment(name):
    profile = embodiments.embodiment_profile(name)
    terms = {
        "arm_action": SimpleNamespace(
            action_dim=len(profile["arm_joints"]),
            cfg=SimpleNamespace(scale=0.5, use_default_offset=True),
            IO_descriptor=SimpleNamespace(joint_names=profile["arm_joints"]),
        ),
        "gripper_action": SimpleNamespace(
            action_dim=1,
            cfg=SimpleNamespace(
                open_command_expr={
                    j: profile["open_position"] for j in profile["gripper_joints"]
                },
                close_command_expr={
                    j: profile["closed_position"] for j in profile["gripper_joints"]
                },
            ),
            IO_descriptor=SimpleNamespace(joint_names=profile["gripper_joints"]),
        ),
    }
    manager = SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__)
    robot = SimpleNamespace(
        joint_names=profile["arm_joints"] + profile["gripper_joints"],
        body_names=[
            profile["base_body"],
            profile["tool_body"],
            profile.get("sensor_source_body", profile["base_body"]),
        ],
    )
    frame = SimpleNamespace(
        prim_path=Scene.env_regex_ns
        + "/Robot/"
        + profile.get("sensor_source_body", profile["base_body"]),
        target_frames=[
            SimpleNamespace(
                prim_path=Scene.env_regex_ns + "/Robot/" + profile["tool_body"],
                offset=SimpleNamespace(pos=profile["tool_offset_m"]),
            )
        ],
    )
    config = SimpleNamespace(
        scene=SimpleNamespace(robot=_robot_config(name), ee_frame=frame),
        commands=SimpleNamespace(
            object_pose=SimpleNamespace(body_name=profile["tool_body"])
        ),
    )
    env = SimpleNamespace(scene=Scene(robot=robot), action_manager=manager, cfg=config)
    return SimpleNamespace(unwrapped=env), {"embodiment": profile}


@pytest.mark.parametrize(
    "name,actions", [("franka", 8), ("ur10e_robotiq85", 7), ("kinova_jaco7", 8)]
)
def test_cli_seals_distinct_robot_and_measured_action_layout(name, actions):
    args = franka_rl._parser().parse_args(
        [
            "prepare",
            "--run-id",
            "embodiment-test",
            "--embodiment",
            name,
            "--output-path",
            "s3://example/prepared/",
        ]
    )
    recipe = franka_rl._recipe(args)
    env, _ = _live_environment(name)
    evidence = embodiments.embodiment_evidence(env, recipe)
    assert evidence["robot_type"] == name
    assert len(evidence["action_names"]) == actions
    if name != "franka":
        assert not any("panda" in joint for joint in evidence["state_names"])


@pytest.mark.parametrize(
    "mutation", ["arm_order", "gripper_joint", "action_order", "tool_body"]
)
def test_wrong_live_articulation_or_action_order_fails_before_training(mutation):
    env, recipe = _live_environment("ur10e_robotiq85")
    native = env.unwrapped
    if mutation == "arm_order":
        native.action_manager.get_term("arm_action").IO_descriptor.joint_names = [
            "finger_joint"
        ] * 6
    elif mutation == "gripper_joint":
        native.action_manager.get_term("gripper_action").IO_descriptor.joint_names = [
            "wrist_3_joint"
        ]
    elif mutation == "action_order":
        native.action_manager.active_terms.reverse()
    else:
        native.scene["robot"].body_names = ["panda_hand", "panda_link0"]
    with pytest.raises(ValueError):
        embodiments.embodiment_evidence(env, recipe)


def test_unsupported_or_mutated_profile_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        embodiments.embodiment_profile("renamed-franka")
    recipe = {"embodiment": embodiments.embodiment_profile("kinova_jaco7")}
    recipe["embodiment"]["tool_body"] = "panda_hand"
    with pytest.raises(ValueError, match="bindings differ"):
        embodiments.recipe_embodiment(recipe)


@pytest.mark.parametrize(
    "mutation",
    [
        "asset",
        "variant",
        "scale",
        "gripper",
        "root",
        "tool_offset",
        "command",
        "sensor_source",
        "missing_sensor_body",
    ],
)
def test_similar_robot_names_cannot_hide_wrong_asset_or_control(mutation):
    env, recipe = _live_environment("ur10e_robotiq85")
    native = env.unwrapped
    if mutation == "asset":
        native.cfg.scene.robot.spawn.usd_path = "another-robot.usd"
    elif mutation == "variant":
        native.cfg.scene.robot.spawn.variants = {"Gripper": "Robotiq_2f_140"}
    elif mutation == "scale":
        native.action_manager.get_term("arm_action").cfg.scale = 1.0
    elif mutation == "gripper":
        native.action_manager.get_term("gripper_action").cfg.close_command_expr = {
            "finger_joint": 0.0
        }
    elif mutation == "root":
        native.cfg.scene.robot.init_state.rot = (1, 0, 0, 0)
    elif mutation == "tool_offset":
        native.cfg.scene.ee_frame.target_frames[0].offset.pos = (0, 0, 0)
    elif mutation == "sensor_source":
        native.cfg.scene.ee_frame.prim_path = Scene.env_regex_ns + "/Robot/base_link"
    elif mutation == "missing_sensor_body":
        native.scene["robot"].body_names.remove("shoulder_link")
    else:
        native.cfg.commands.object_pose.body_name = "base_link"
    with pytest.raises(ValueError):
        embodiments.embodiment_evidence(env, recipe)


@pytest.mark.parametrize(
    "mutation", ["robot_type", "profile", "action_names", "state_names"]
)
def test_capture_cannot_relabel_a_robot_or_its_telemetry(mutation):
    env, recipe = _live_environment("kinova_jaco7")
    identity = embodiments.embodiment_evidence(env, recipe)
    metadata = {**deepcopy(identity), "embodiment": identity}
    embodiments.validate_capture_embodiment(metadata, recipe)
    if mutation == "profile":
        metadata["embodiment"]["profile"] = embodiments.embodiment_profile("franka")
    elif mutation == "robot_type":
        metadata[mutation] = "franka"
    else:
        metadata[mutation] = ["panda_joint1"]
        metadata["embodiment"][mutation] = ["panda_joint1"]
    with pytest.raises(ValueError):
        embodiments.validate_capture_embodiment(metadata, recipe)


@pytest.mark.parametrize("name", ["ur10e_robotiq85", "kinova_jaco7"])
def test_robot_configuration_preserves_upstream_and_uses_native_xyzw_identity(
    monkeypatch, name
):
    class RobotConfig(SimpleNamespace):
        def copy(self):
            return deepcopy(self)

        def replace(self, **values):
            result = self.copy()
            result.__dict__.update(values)
            return result

    profile = embodiments.embodiment_profile(name)
    initial_joints = (
        {"shoulder_pan_joint": 3.141592653589793}
        if name == "ur10e_robotiq85"
        else {"j2n7s300_joint_2": 2.76, "j2n7s300_joint_finger_[1-3]": 0.2}
    )
    source = RobotConfig(
        init_state=SimpleNamespace(rot=(0, 0, 0, 1), joint_pos=deepcopy(initial_joints))
    )
    monkeypatch.setattr(
        embodiments,
        "import_module",
        lambda module: SimpleNamespace(**{profile["configuration"]: source}),
    )
    mdp = SimpleNamespace(
        JointPositionActionCfg=SimpleNamespace,
        BinaryJointPositionActionCfg=SimpleNamespace,
    )
    monkeypatch.setitem(sys.modules, "isaaclab.envs", SimpleNamespace(mdp=mdp))
    target = SimpleNamespace(prim_path="old", offset=SimpleNamespace(pos=(0, 0, 0)))
    config = SimpleNamespace(
        scene=SimpleNamespace(ee_frame=SimpleNamespace(target_frames=[target])),
        actions=SimpleNamespace(),
        commands=SimpleNamespace(object_pose=SimpleNamespace()),
    )
    embodiments.configure_embodiment(config, {"embodiment": profile})
    assert source.init_state.rot == (0, 0, 0, 1)
    assert source.init_state.joint_pos == initial_joints
    assert config.scene.robot.init_state.rot == (0, 0, 0, 1)
    assert config.actions.arm_action.joint_names == profile["arm_joints"]
    assert config.actions.gripper_action.joint_names == profile["gripper_joints"]
    assert target.prim_path.endswith("/" + profile["tool_body"])
    if name == "ur10e_robotiq85":
        assert config.scene.robot.init_state.joint_pos["shoulder_pan_joint"] == 0
        assert config.scene.ee_frame.prim_path == "{ENV_REGEX_NS}/Robot/shoulder_link"
    else:
        assert config.scene.robot.init_state.joint_pos == initial_joints
        assert len(config.actions.gripper_action.close_command_expr) == 6


def test_historical_franka_recipe_stays_readable():
    assert embodiments.recipe_embodiment({})["name"] == "franka"
    embodiments.validate_capture_embodiment({}, {})


def test_ur_sensor_source_avoids_nested_gripper_body_name_collision():
    # The native Robotiq hierarchy repeats the arm's base_link leaf name.
    body_paths = [
        "/Robot/base_link",
        "/Robot/shoulder_link",
        "/Robot/wrist_3_link",
        "/Robot/ee_link/Robotiq_2F_85/base_link",
    ]
    profile = embodiments.embodiment_profile("ur10e_robotiq85")
    assert sum(path.endswith("/" + profile["base_body"]) for path in body_paths) == 2
    for body in (profile["sensor_source_body"], profile["tool_body"]):
        assert sum(path.endswith("/" + body) for path in body_paths) == 1


@pytest.mark.parametrize("corruption", [None, "wrong_offset", "nonfinite"])
def test_sensor_world_tcp_matches_rotated_articulation_link_pose(corruption):
    torch = pytest.importorskip("torch")
    from npa.workflows.franka_rl_environment import _tool_frame_check

    profile = embodiments.embodiment_profile("ur10e_robotiq85")
    # A 90-degree Y rotation moves the 14.5 cm local Z grasp offset to world X.
    positions = torch.tensor([[[1.0, 2.0, 3.0]]])
    quaternions = torch.tensor([[[0.0, 2**-0.5, 0.0, 2**-0.5]]])
    measured = torch.tensor([[[1.145, 2.0, 3.0]]])
    if corruption == "wrong_offset":
        measured[0, 0] = torch.tensor([1.0, 2.0, 3.145])
    elif corruption == "nonfinite":
        measured[0, 0, 0] = float("nan")
    robot = SimpleNamespace(
        body_names=["wrist_3_link"],
        data=SimpleNamespace(
            body_link_pos_w=SimpleNamespace(torch=positions),
            body_link_quat_w=SimpleNamespace(torch=quaternions),
        ),
    )
    frame = SimpleNamespace(
        data=SimpleNamespace(target_pos_w=SimpleNamespace(torch=measured))
    )
    native = SimpleNamespace(scene={"robot": robot, "ee_frame": frame})
    if corruption:
        with pytest.raises(ValueError, match="world TCP"):
            _tool_frame_check(native, profile)
    else:
        result = _tool_frame_check(native, profile)
        assert result["world_tcp_verified"] and result["maximum_error_m"] < 1e-6
