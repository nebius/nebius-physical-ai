"""Unit tests for the pure helpers of the BYO-robot task injector.

``register`` itself imports Isaac-Lab (GPU-only) and is verified by an on-cluster
probe, not here.
"""

from __future__ import annotations

import json

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


def test_overrides_falls_back_to_zero_init_when_names_missing():
    ov = rt.robot_articulation_overrides(_byo_spec(joint_names=[], home_qpos=[]))
    assert ov["init_joint_pos"] == {".*": 0.0}


def test_overrides_falls_back_when_names_qpos_mismatch():
    ov = rt.robot_articulation_overrides(_byo_spec(joint_names=["a", "b"], home_qpos=[0.1]))
    assert ov["init_joint_pos"] == {".*": 0.0}


def test_overrides_uses_local_path_when_usd_path_absent():
    spec = _byo_spec(usd_path="", local_path="/tmp/staged/from_local.usd")
    ov = rt.robot_articulation_overrides(spec)
    assert ov["usd_path"] == "/tmp/staged/from_local.usd"


def test_overrides_gains_are_bounded():
    # Garbage-huge gains are clamped, not passed through to a degenerate drive.
    spec = _byo_spec(kp=[1e12, 1e12], kv=[1e12, 1e12], force_upper=[1e12], force_lower=[])
    ov = rt.robot_articulation_overrides(spec)
    assert ov["stiffness"] == rt.STIFFNESS_MAX
    assert ov["damping"] == rt.DAMPING_MAX
    assert ov["effort_limit"] == rt.EFFORT_MAX


def test_module_source_is_self_contained():
    src = rt.module_source()
    # Shipped into the Isaac container, so it must carry the helpers + register.
    assert "def robot_spec_from_env" in src
    assert "def robot_articulation_overrides" in src
    assert "def register(" in src


def test_train_wrapper_enforces_boot_before_isaac_imports():
    s = rt.TRAIN_WRAPPER_SCRIPT
    # AppLauncher boot MUST precede any isaaclab/isaac_byo_robot_task import.
    boot = s.index("AppLauncher(headless=True).app")
    assert boot < s.index("import isaaclab_tasks")
    assert boot < s.index("import isaac_byo_robot_task")
    assert s.index("import isaaclab_tasks") < s.index("robotmod.register")
    # trains via the rsl_rl runner and emits the done/ckpt markers
    assert "OnPolicyRunner" in s and "runner.learn" in s
    assert "ROBOT_TRAIN_DONE" in s
    # refuses a silent stock fallback when a customer USD was requested
    assert "ROBOT_USD_MISMATCH" in s
