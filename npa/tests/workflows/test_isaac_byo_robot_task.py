"""Unit tests for the pure helpers of the BYO-robot task injector.

``register`` itself imports Isaac-Lab (GPU-only) and is verified by an on-cluster
probe, not here.
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import MISSING
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workflows.sim2real import isaac_byo_robot_task as rt


def _byo_spec(**over):
    spec = {
        "robot_source": "byo_usd",
        "name": "acme_arm",
        "usd_path": "/tmp/staged/acme_arm.usd",
        "ee_link": "tool0",
        "joint_names": ["j1", "j2", "j3"],
        "home_qpos": [0.0, -0.5, 0.5],
        "kp": [100.0, 200.0, 300.0],
        "kv": [10.0, 20.0, 30.0],
        "force_upper": [50.0, 60.0, 70.0],
        "force_lower": [-80.0, -60.0, -70.0],
    }
    spec.update(over)
    return spec


def test_task_id_is_gym_safe():
    assert rt._task_id("acme arm/v2") == "NPA-Lift-Cube-acme-arm-v2-v0"
    assert rt._task_id("") == "NPA-Lift-Cube-robot-v0"


def test_spec_from_env_none_when_unset_or_invalid():
    assert rt.robot_spec_from_env({}) is None
    assert rt.robot_spec_from_env({rt.ROBOT_SPEC_ENV: ""}) is None
    assert rt.robot_spec_from_env({rt.ROBOT_SPEC_ENV: "not json"}) is None
    assert rt.robot_spec_from_env({rt.ROBOT_SPEC_ENV: "[1,2,3]"}) is None  # not a dict


def test_spec_from_env_stock_franka_is_none():
    # Stock Franka routed through the BYO gate => no swap => stock fallback.
    blob = json.dumps({"robot_source": "stock_franka", "name": "franka_panda"})
    assert rt.robot_spec_from_env({rt.ROBOT_SPEC_ENV: blob}) is None


def test_spec_from_env_parses_byo():
    blob = json.dumps(_byo_spec())
    spec = rt.robot_spec_from_env({rt.ROBOT_SPEC_ENV: blob})
    assert spec is not None
    assert spec["robot_source"] == "byo_usd"
    assert spec["name"] == "acme_arm"


def test_overrides_empty_for_stock_and_for_none():
    assert rt.robot_articulation_overrides(None) == {}
    assert rt.robot_articulation_overrides({"robot_source": "stock_franka"}) == {}


def test_overrides_empty_when_no_usd():
    spec = _byo_spec(usd_path="", local_path="")
    assert rt.robot_articulation_overrides(spec) == {}


def test_overrides_full_mapping():
    ov = rt.robot_articulation_overrides(_byo_spec())
    assert ov["usd_path"] == "/tmp/staged/acme_arm.usd"
    assert ov["ee_link"] == "tool0"
    # per-joint home from joint_names + home_qpos
    assert ov["init_joint_pos"] == {"j1": 0.0, "j2": -0.5, "j3": 0.5}
    # coarse single actuator group: mean kp/kv, max |force|
    assert ov["stiffness"] == 200.0
    assert ov["damping"] == 20.0
    assert ov["effort_limit"] == 80.0
    assert ov["joint_actuators"] == [
        {"joint_name": "j1", "stiffness": 100.0, "damping": 10.0, "effort_limit": 80.0},
        {"joint_name": "j2", "stiffness": 200.0, "damping": 20.0, "effort_limit": 60.0},
        {"joint_name": "j3", "stiffness": 300.0, "damping": 30.0, "effort_limit": 70.0},
    ]


def test_overrides_falls_back_to_zero_init_when_names_missing():
    ov = rt.robot_articulation_overrides(_byo_spec(joint_names=[], home_qpos=[]))
    assert ov["init_joint_pos"] == {".*": 0.0}


def test_overrides_falls_back_when_names_qpos_mismatch():
    ov = rt.robot_articulation_overrides(
        _byo_spec(joint_names=["a", "b"], home_qpos=[0.1])
    )
    assert ov["init_joint_pos"] == {".*": 0.0}


def test_overrides_uses_local_path_when_usd_path_absent():
    spec = _byo_spec(usd_path="", local_path="/tmp/staged/from_local.usd")
    ov = rt.robot_articulation_overrides(spec)
    assert ov["usd_path"] == "/tmp/staged/from_local.usd"


def test_overrides_gains_are_bounded():
    # Garbage-huge gains are clamped, not passed through to a degenerate drive.
    spec = _byo_spec(
        kp=[1e12, 1e12], kv=[1e12, 1e12], force_upper=[1e12], force_lower=[]
    )
    ov = rt.robot_articulation_overrides(spec)
    assert ov["stiffness"] == rt.STIFFNESS_MAX
    assert ov["damping"] == rt.DAMPING_MAX
    assert ov["effort_limit"] == rt.EFFORT_MAX


def _gripper_spec(**over):
    """A 3-finger manipulator spec (Kinova-class) with a declared gripper."""

    spec = _byo_spec(
        name="kinova",
        joint_names=["a1", "a2", "f1", "f2", "f3"],
        home_qpos=[0.0, 0.5, 1.0, 1.0, 1.0],
        n_arm_joints=2,
        n_gripper_joints=3,
        gripper_joint_names=["f1", "f2", "f3"],
        finger_links=["fl1", "fl2", "fl3"],
        gripper_open=0.0,
        gripper_close=1.2,
    )
    spec.update(over)
    return spec


def test_gripper_actuator_group_uses_floors_when_no_gripper_gains():
    # A declared gripper with no per-gripper gains gets a dedicated actuator group
    # at the robot-agnostic floors — high enough to clamp and HOLD, not the too-soft
    # arm-averaged gains. This is the fix for "fingers close but cannot hold".
    ov = rt.robot_articulation_overrides(_gripper_spec())
    ga = ov["gripper_actuator"]
    assert ga["joint_names"] == ["f1", "f2", "f3"]
    assert ga["stiffness"] == rt.GRIPPER_STIFFNESS_FLOOR
    assert ga["damping"] == rt.GRIPPER_DAMPING_FLOOR
    assert ga["effort_limit"] == rt.GRIPPER_EFFORT_FLOOR


def test_gripper_actuator_group_respects_spec_gains_above_floor():
    # A spec that declares stronger gripper gains keeps them (only the floor is a
    # minimum), and they are still clamped to the global bounds.
    ov = rt.robot_articulation_overrides(
        _gripper_spec(
            gripper_kp=[900.0, 900.0], gripper_kv=[90.0], gripper_force=[500.0]
        )
    )
    ga = ov["gripper_actuator"]
    assert ga["stiffness"] == 900.0
    assert ga["damping"] == 90.0
    assert ga["effort_limit"] == 500.0
    # Below-floor spec gains are raised to the floor (never weaken the hold).
    ov2 = rt.robot_articulation_overrides(
        _gripper_spec(gripper_kp=[1.0], gripper_force=[1.0])
    )
    assert ov2["gripper_actuator"]["stiffness"] == rt.GRIPPER_STIFFNESS_FLOOR
    assert ov2["gripper_actuator"]["effort_limit"] == rt.GRIPPER_EFFORT_FLOOR


def test_no_gripper_actuator_group_for_gripperless_arm():
    # A bare (gripperless) arm gets no gripper actuator group.
    assert "gripper_actuator" not in rt.robot_articulation_overrides(_ur_spec())
    # ...and neither does the stock/no-USD path (unchanged).
    assert rt.robot_articulation_overrides({"robot_source": "stock_franka"}) == {}


def test_grasp_hold_reward_is_shipped():
    # The maintained-grasp+lift reward must ship in module_source so the in-container
    # wrapper can install it, and be robot-agnostic (no hardcoded joint names).
    assert callable(rt.grasp_lift_hold)
    src = rt.module_source()
    assert "def grasp_lift_hold" in src
    # Bootstrap-capable form: near x closed x ee-height-gain (not object height).
    assert "near * closed * height" in src


def _ur_spec(**over):
    """A gripperless UR10-class arm spec (the custom-asset test's failing case)."""

    spec = {
        "robot_source": "byo_usd",
        "name": "ur10",
        "usd_path": "/tmp/staged/ur10.usd",
        "ee_link": "tool0",
        "base_link": "base_link",
        "joint_names": [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        "n_arm_joints": 6,
        "n_gripper_joints": 0,
        "finger_links": [],
    }
    spec.update(over)
    return spec


# --------------------------------------------------------------------------- #
# task_retarget_overrides
# --------------------------------------------------------------------------- #
def test_retarget_empty_for_stock_none_and_no_usd():
    assert rt.task_retarget_overrides(None) == {}
    assert rt.task_retarget_overrides({"robot_source": "stock_franka"}) == {}
    assert rt.task_retarget_overrides(_byo_spec(usd_path="", local_path="")) == {}


def test_retarget_franka_defaults_resolve_to_panda_names():
    # A BYO spec that omits link/joint fields must resolve to the stock Franka
    # names, so the Franka path is byte-for-byte (names resolve to panda_*).
    minimal = {"robot_source": "byo_usd", "name": "franka", "usd_path": "/tmp/f.usd"}
    ov = rt.task_retarget_overrides(minimal)
    assert ov["ee_frame_source"] == "panda_link0"
    assert ov["ee_frame_target"] == "panda_hand"
    assert ov["ee_frame_name"] == "end_effector"
    assert ov["arm_joint_names"] == ["panda_joint.*"]
    assert ov["command_body_name"] == "panda_hand"
    # No gripper declared -> no gripper retarget (honesty handled separately).
    assert ov["gripper"] is None


def test_retarget_ur_like_spec_uses_robot_names_not_panda():
    ov = rt.task_retarget_overrides(_ur_spec())
    assert ov["ee_frame_source"] == "base_link"
    assert ov["ee_frame_target"] == "tool0"
    assert ov["command_body_name"] == "tool0"
    # arm joints are the UR joints (first n_arm_joints), not panda_joint.*
    assert ov["arm_joint_names"][0] == "shoulder_pan_joint"
    assert ov["arm_joint_names"][-1] == "wrist_3_joint"
    assert "panda_joint.*" not in ov["arm_joint_names"]
    assert ov["gripper"] is None


def test_retarget_arm_joints_truncated_to_n_arm_joints():
    # Extra (e.g. gripper) joints past n_arm_joints are not fed to the arm action.
    spec = _ur_spec(joint_names=["a", "b", "c", "d"], n_arm_joints=2)
    ov = rt.task_retarget_overrides(spec)
    assert ov["arm_joint_names"] == ["a", "b"]


def test_retarget_franka_gripper_resolves_to_panda_finger_pattern():
    spec = {
        "robot_source": "byo_usd",
        "name": "franka",
        "usd_path": "/tmp/f.usd",
        "ee_link": "panda_hand",
        "n_gripper_joints": 2,
        "finger_links": ["panda_leftfinger", "panda_rightfinger"],
        "gripper_open": 0.04,
        "gripper_close": 0.0,
    }
    ov = rt.task_retarget_overrides(spec)
    assert ov["gripper"]["joint_names"] == ["panda_finger.*"]
    assert ov["gripper"]["open"] == {"panda_finger_.*": 0.04}
    assert ov["gripper"]["close"] == {"panda_finger_.*": 0.0}


def test_retarget_custom_gripper_uses_declared_joint_names():
    spec = {
        "robot_source": "byo_usd",
        "name": "arm_with_robotiq",
        "usd_path": "/tmp/a.usd",
        "ee_link": "tool0",
        "n_gripper_joints": 1,
        "finger_links": ["finger"],
        "gripper_joint_names": ["robotiq_85_left_knuckle_joint"],
        "gripper_open": 0.8,
        "gripper_close": 0.0,
    }
    ov = rt.task_retarget_overrides(spec)
    assert ov["gripper"]["joint_names"] == ["robotiq_85_left_knuckle_joint"]
    assert ov["gripper"]["open"] == {"robotiq_85_left_knuckle_joint": 0.8}
    assert ov["gripper"]["close"] == {"robotiq_85_left_knuckle_joint": 0.0}


# --------------------------------------------------------------------------- #
# task_robot_compatibility (honesty: a gripperless arm cannot lift)
# --------------------------------------------------------------------------- #
def test_compat_stock_and_none_are_compatible():
    for spec in (None, {"robot_source": "stock_franka"}):
        c = rt.task_robot_compatibility(spec, task_kind="lift")
        assert c["task_robot_compatible"] is True
        assert c["has_gripper"] is True


def test_compat_gripperless_arm_cannot_lift():
    c = rt.task_robot_compatibility(_ur_spec(), task_kind="lift")
    assert c["task_robot_compatible"] is False
    assert c["has_gripper"] is False
    assert "gripper" in c["reason"].lower()
    assert c["requirements"]  # non-empty customer requirements


def test_compat_gripper_bearing_arm_can_lift():
    spec = _ur_spec(n_gripper_joints=2, finger_links=["lf", "rf"])
    c = rt.task_robot_compatibility(spec, task_kind="lift")
    assert c["task_robot_compatible"] is True
    assert c["has_gripper"] is True


def test_compat_non_grasping_task_does_not_require_gripper():
    # A reach/navigation task does not structurally need a gripper.
    c = rt.task_robot_compatibility(_ur_spec(), task_kind="reach")
    assert c["task_robot_compatible"] is True


def test_compat_explicit_has_gripper_flag_wins():
    c = rt.task_robot_compatibility(_ur_spec(has_gripper=True), task_kind="lift")
    assert c["task_robot_compatible"] is True
    assert c["has_gripper"] is True


def test_module_source_is_self_contained():
    src = rt.module_source()
    # Shipped into the Isaac container, so it must carry the helpers + register.
    assert "def robot_spec_from_env" in src
    assert "def robot_articulation_overrides" in src
    assert "def task_retarget_overrides" in src
    assert "def task_robot_compatibility" in src
    assert "def register(" in src


def test_train_wrapper_enforces_boot_before_isaac_imports():
    s = rt.TRAIN_WRAPPER_SCRIPT
    # AppLauncher boot MUST precede any isaaclab/isaac_byo_robot_task import.
    boot = s.index("app = AppLauncher(")
    assert boot < s.index("import isaaclab_tasks")
    assert boot < s.index("import isaac_byo_robot_task")
    assert s.index("import isaaclab_tasks") < s.index("robotmod.register")
    assert '"--portable-root /tmp/npa-isaac-kit"' in s
    # trains via the rsl_rl runner and emits the done/ckpt markers
    assert "OnPolicyRunner" in s and "robotmod.learn_ppo_phase" in s
    assert "ROBOT_TRAIN_DONE" in s
    assert "ROBOT_ENTROPY_ANNEALED" in s
    assert "runner.alg.entropy_coef = float(ENT_FINAL)" in s
    # Exact update counts and final/periodic checkpoint semantics are exercised
    # by the behavioral wrapper tests below, including a fresh process resume.
    assert "ROBOT_FINAL_CHECKPOINT" in s
    # refuses a silent stock fallback when a customer USD was requested
    assert "ROBOT_USD_MISMATCH" in s
    # surfaces the retarget plan + the honest task/robot compatibility verdict
    assert "task_retarget_overrides" in s
    assert "task_robot_compatibility" in s
    assert "ROBOT_COMPAT" in s
    assert "ROBOT_TASK_INCOMPATIBLE" in s
    assert "task_robot_compatible=" in s


def test_train_wrapper_applies_zero_seed_to_env_and_ppo():
    s = rt.TRAIN_WRAPPER_SCRIPT
    seed_block = s.split("# Seed 0 is valid", 1)[1].split("# Keep PPO exploring", 1)[0]
    assert "if SEED:" not in seed_block
    assert "env_cfg.seed = SEED" in seed_block
    assert "torch.manual_seed(SEED)" in seed_block
    assert 'acfg["seed"] = SEED' in seed_block
    assert 'print("ROBOT_SEED_APPLIED", SEED' in seed_block


def _wrapper_training_body():
    tree = ast.parse(rt.TRAIN_WRAPPER_SCRIPT)
    return next(
        node.body
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "runner"
                for target in statement.targets
            )
            for statement in node.body
        )
    )


def _assigns(statement, name):
    return isinstance(statement, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name
        for target in statement.targets
    )


def _exec_wrapper_statements(statements, namespace):
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(compile(module, "<actual-robot-training-wrapper>", "exec"), namespace)


@pytest.mark.parametrize(
    "model_config,expected_path",
    [
        ({"policy": {"init_noise_std": 1.0}}, ("policy", "init_noise_std")),
        (
            {"actor": {"init_noise_std": 1.0}, "policy": MISSING},
            ("actor", "init_noise_std"),
        ),
        (
            {
                "actor": {
                    "hidden_dims": [256, 128, 64],
                    "distribution_cfg": {
                        "class_name": "GaussianDistribution",
                        "init_std": 1.0,
                    },
                },
                "critic": {"hidden_dims": [256, 128, 64]},
                "policy": MISSING,
            },
            ("actor", "distribution_cfg", "init_std"),
        ),
    ],
    ids=["legacy-policy", "rsl4-actor", "rsl5-actor-distribution"],
)
def test_actual_wrapper_initial_noise_reaches_migrated_model(
    model_config, expected_path
):
    cfg = {
        "algorithm": {"learning_rate": 1e-4, "entropy_coef": 0.006},
        "save_interval": 100,
    }
    cfg.update(copy.deepcopy(model_config))
    before_algorithm = dict(cfg["algorithm"])
    body = _wrapper_training_body()
    start = next(i for i, node in enumerate(body) if _assigns(node, "algo"))
    end = next(i for i, node in enumerate(body) if _assigns(node, "env"))
    _exec_wrapper_statements(
        body[start:end],
        {
            "acfg": cfg,
            "robotmod": rt,
            "os": SimpleNamespace(environ={"ROBOT_INIT_NOISE_STD": "0.35"}),
            "json": json,
        },
    )
    actual = cfg
    for key in expected_path:
        actual = actual[key]
    assert actual == 0.35
    assert cfg["algorithm"] == before_algorithm
    if "critic" in cfg:
        assert cfg["critic"] == model_config["critic"]


def test_initial_noise_does_not_fall_back_from_invalid_actor_to_legacy_policy():
    with pytest.raises(RuntimeError, match="actor config"):
        rt.initial_action_noise_config({"actor": {}, "policy": {"init_noise_std": 1.0}})


class _UpstreamLastIndexRunner:
    """Contract double for verified RSL-RL 5.0.1 learn/save/load semantics.

    This exercises wrapper control flow, not GPU training. The actual upstream
    runner leaves its last loop index in current_learning_iteration and saves
    that value in checkpoint iter, with infos=None for periodic saves.
    """

    def __init__(self, env, cfg, log_dir, device):
        self.current_learning_iteration = 0
        self.alg = SimpleNamespace(entropy_coef=0.01)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.indices = []
        self.phase_flags = []
        self.timesteps = 0
        self.saved = {}

    def learn(self, num_learning_iterations, init_at_random_ep_len):
        self.phase_flags.append(init_at_random_ep_len)
        start = self.current_learning_iteration
        for index in range(start, start + num_learning_iterations):
            self.indices.append(index)
            self.timesteps += 1024 * 24
            self.current_learning_iteration = index
            if index % 100 == 0:
                self.save(self.log_dir / f"model_{index}.pt")
        self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")

    def save(self, path, infos=None):
        payload = {"iter": self.current_learning_iteration, "infos": infos}
        Path(path).write_text(json.dumps(payload))
        self.saved[Path(path).name] = payload

    def load(self, path):
        payload = json.loads(Path(path).read_text())
        self.current_learning_iteration = payload["iter"]
        return payload["infos"]


def _execute_training_phases(output, resume=None):
    body = _wrapper_training_body()
    start = next(i for i, node in enumerate(body) if _assigns(node, "runner"))
    end = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Import)
        and any(alias.name == "glob" for alias in node.names)
    )
    namespace = {
        "OnPolicyRunner": _UpstreamLastIndexRunner,
        "env": object(),
        "acfg": {},
        "OUT": str(output),
        "robotmod": rt,
        "os": SimpleNamespace(
            environ={"ROBOT_RESUME_CKPT_LOCAL": str(resume) if resume else ""},
            path=__import__("os").path,
        ),
        "ENT_FINAL": "0.001",
        "ENT_FRACTION": "0.6",
        "CONVERGENCE_STD": "",
        "ENT": "0.01",
        "ITERS": 2000,
        "json": json,
    }
    _exec_wrapper_statements(body[start:end], namespace)
    return namespace["runner"], Path(namespace["final_path"])


def test_actual_wrapper_executes_2000_contiguous_updates_and_consistent_final_save(
    tmp_path,
):
    runner, final_path = _execute_training_phases(tmp_path)
    assert runner.indices == list(range(2000))
    assert runner.phase_flags == [True, False]
    assert runner.timesteps == 2000 * 1024 * 24
    assert runner.saved["model_100.pt"] == {"iter": 100, "infos": None}
    assert final_path.name == "model_1999.pt"
    assert json.loads(final_path.read_text()) == {
        "iter": 1999,
        "infos": {
            "npa_ppo_iteration": {
                "value_semantics": "last_completed_zero_based",
                "completed_updates": 2000,
            }
        },
    }


@pytest.mark.parametrize(
    "checkpoint_name,next_index", [("model_100.pt", 101), ("model_1999.pt", 2000)]
)
def test_actual_wrapper_new_process_resume_adds_2000_updates(
    tmp_path, checkpoint_name, next_index
):
    _execute_training_phases(tmp_path / "first")
    runner, final_path = _execute_training_phases(
        tmp_path / "resumed", tmp_path / "first" / checkpoint_name
    )
    assert runner.indices == list(range(next_index, next_index + 2000))
    assert runner.timesteps == 2000 * 1024 * 24
    assert final_path.name == f"model_{next_index + 1999}.pt"
    payload = json.loads(final_path.read_text())
    assert payload["iter"] == next_index + 1999
    assert (
        payload["infos"]["npa_ppo_iteration"]["completed_updates"] == next_index + 2000
    )


@pytest.mark.parametrize("semantics", ["next_zero_based", "completed_updates"])
def test_actual_wrapper_honors_explicit_already_normalized_checkpoint(
    tmp_path, semantics
):
    checkpoint = tmp_path / "normalized.pt"
    checkpoint.write_text(
        json.dumps(
            {
                "iter": 2000,
                "infos": {
                    "npa_ppo_iteration": {
                        "value_semantics": semantics,
                        "completed_updates": 2000,
                    }
                },
            }
        )
    )
    runner, final_path = _execute_training_phases(tmp_path / "resumed", checkpoint)
    assert runner.indices == list(range(2000, 4000))
    assert json.loads(final_path.read_text())["iter"] == 3999


@pytest.mark.parametrize(
    "info",
    [
        [],
        {"npa_ppo_iteration": "unknown"},
        {"npa_ppo_iteration": {"value_semantics": "unknown", "completed_updates": 11}},
        {
            "npa_ppo_iteration": {
                "value_semantics": "last_completed_zero_based",
                "completed_updates": 10,
            }
        },
        {
            "npa_ppo_iteration": {
                "value_semantics": "next_zero_based",
                "completed_updates": True,
            }
        },
    ],
)
def test_checkpoint_iteration_metadata_fails_closed(info):
    with pytest.raises(RuntimeError, match="checkpoint"):
        rt.next_ppo_iteration(
            SimpleNamespace(current_learning_iteration=10),
            resumed=True,
            checkpoint_info=info,
        )


def test_phase_refuses_unverified_runner_update_count():
    runner = SimpleNamespace(current_learning_iteration=0, learn=lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="final learning iteration"):
        rt.learn_ppo_phase(
            runner,
            start_iteration=100,
            num_learning_iterations=2000,
            init_at_random_ep_len=True,
        )
