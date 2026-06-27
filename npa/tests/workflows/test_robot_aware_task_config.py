"""Unit tests for the robot-aware Lift task config (B3).

The GPU-side ``_apply_task_config`` mutates Isaac-Lab cfg objects and is exercised
on-cluster; here we test the pure surface: ``task_config_overrides`` (the mapping
the variant applies), env-var plumbing, the trainer manifest wiring, and that the
Franka path is byte-for-byte unchanged while a non-Franka arm gets a distinct,
scaled config.
"""

from __future__ import annotations

import json

from npa.workflows.sim2real import byo_isaac_trainer as trainer
from npa.workflows.sim2real import isaac_byo_robot_task as robotmod
from npa.workflows.sim2real import onboarding_derive as der
from npa.workflows.sim2real import onboarding_spec as ob


# --------------------------------------------------------------------------- #
# task_config_overrides — the pure mapping the variant applies
# --------------------------------------------------------------------------- #
def test_overrides_empty_for_none():
    assert robotmod.task_config_overrides(None) == {}
    assert robotmod.task_config_overrides("nope") == {}
    assert robotmod.task_config_overrides({}) == {}


def test_overrides_full_config():
    cfg = der.derive_task_config(
        ob.parse_onboarding_spec(
            {"robot": {"name": "kinova", "usd_path": "https://x/k.usd",
                       "n_arm_joints": 7, "n_gripper_joints": 3,
                       "gripper_joint_names": ["a", "b", "c"],
                       "gripper_open": 0.0, "gripper_close": 1.2},
             "task": {"skill": "lift"}}
        ),
        arm_joint_ranges=[(-2.0, 2.0)] * 7,
    ).to_dict()
    over = robotmod.task_config_overrides(cfg)
    assert "action_scale" in over
    assert set(over["object_init_range"]) == {"x", "y", "z"}
    assert isinstance(over["goal_range"]["x"], tuple)
    assert over["gripper_close"] == 1.2
    assert over["minimal_height_m"] == der.STOCK_MINIMAL_HEIGHT_M


def test_overrides_drops_bad_ranges():
    over = robotmod.task_config_overrides(
        {"object_init_range": {"x": [0.1], "y": "bad"}, "goal_range": "nope",
         "action_scale": "x"}
    )
    # All malformed -> nothing carried.
    assert over == {}


# --------------------------------------------------------------------------- #
# Franka byte-for-byte: stock spec -> no overrides
# --------------------------------------------------------------------------- #
def test_stock_franka_yields_no_overrides_and_no_register():
    # A stock-Franka spec produces no articulation overrides, so register() is a
    # no-op regardless of any task config (Franka path untouched).
    franka = {"robot_source": "stock_franka", "name": "franka"}
    assert robotmod.robot_articulation_overrides(franka) == {}
    # Even with a Franka-derived task config present, the stock path does not swap.
    cfg = der.derive_task_config(
        ob.parse_onboarding_spec(
            {"robot": {"name": "franka", "robot_source": "stock_franka"},
             "task": {"skill": "lift"}}
        ),
        arm_joint_ranges=list(der.FRANKA_ARM_JOINT_RANGES),
    ).to_dict()
    # The Franka-derived config equals the stock numbers, so applying it would be
    # a no-op even if it were applied.
    over = robotmod.task_config_overrides(cfg)
    assert over["action_scale"] == der.STOCK_ACTION_SCALE
    assert over["object_init_range"] == der.STOCK_OBJECT_INIT_RANGE
    assert over["goal_range"] == der.STOCK_GOAL_RANGE


# --------------------------------------------------------------------------- #
# env plumbing
# --------------------------------------------------------------------------- #
def test_task_config_from_env_roundtrip():
    payload = {"action_scale": 0.3, "goal_range": {"x": [0.4, 0.6]}}
    env = {robotmod.TASK_CONFIG_ENV: json.dumps(payload)}
    assert robotmod.task_config_from_env(env) == payload
    assert robotmod.task_config_from_env({}) is None
    assert robotmod.task_config_from_env({robotmod.TASK_CONFIG_ENV: "{bad"}) is None


# --------------------------------------------------------------------------- #
# Trainer manifest wiring: BYO branch exports the task config + entropy + log
# --------------------------------------------------------------------------- #
def _byo_manifest(task_config, entropy_coef=""):
    m = trainer.build_isaac_job_manifest(
        job_name="j", run_id="r", image="img", task="Isaac-Lift-Cube-Franka-v0",
        num_envs=64, iterations=10, s3_output_uri="s3://b/p/", s3_endpoint="https://e",
        namespace="default", service_account="agent-sa", gpu_product="P",
        robot_spec={"robot_source": "byo_usd", "name": "kinova",
                    "usd_path": "/tmp/npa_robot/robot.usd"},
        robot_usd_uri="s3://b/kinova.usd",
        task_config=task_config,
        entropy_coef=entropy_coef,
    )
    return m["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_byo_manifest_exports_task_config_and_entropy():
    script = _byo_manifest({"action_scale": 0.3, "minimal_height_m": 0.05}, entropy_coef="0.01")
    # Assert on the `export` line (the wrapper SOURCE always mentions these names).
    assert "export NPA_BYO_TASK_CONFIG_JSON=" in script
    assert "export ROBOT_ENTROPY_COEF=" in script
    # The BYO path uploads the per-iteration curve via /tmp/train_full.log.
    assert "tee /tmp/train_full.log" in script


def test_byo_manifest_omits_task_config_when_absent():
    script = _byo_manifest(None, entropy_coef="")
    assert "export NPA_BYO_TASK_CONFIG_JSON=" not in script
    assert "export ROBOT_ENTROPY_COEF=" not in script
    # The robot path is still wired and still uploads a recoverable train log.
    assert "tee /tmp/train_full.log" in script


def test_franka_default_path_unchanged_by_byo_params():
    # No robot_spec -> stock Franka default path: stock train.py, no robot exports.
    m = trainer.build_isaac_job_manifest(
        job_name="j", run_id="r", image="img", task="Isaac-Lift-Cube-Franka-v0",
        num_envs=64, iterations=10, s3_output_uri="s3://b/p/", s3_endpoint="https://e",
        namespace="default", service_account="agent-sa", gpu_product="P",
        entropy_coef="0.01",
    )
    script = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "NPA_BYO_TASK_CONFIG_JSON=" not in script
    assert "NPA_BYO_ROBOT_SPEC_JSON=" not in script
    # Franka default path uses the stock train.py with the entropy hydra override.
    assert "agent.algorithm.entropy_coef=0.01" in script
    assert "rsl_rl/train.py" in script
