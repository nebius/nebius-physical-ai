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


def test_dense_lift_weight_passthrough():
    # A positive dense_lift_weight is carried (with a default std) so the variant
    # adds the continuous lift-progress reward; a non-positive / bad value is not.
    over = robotmod.task_config_overrides({"dense_lift_weight": 2.5})
    assert over["dense_lift_weight"] == 2.5
    assert over["dense_lift_std"] == 0.05
    over2 = robotmod.task_config_overrides({"dense_lift_weight": 4.0, "dense_lift_std": 0.08})
    assert over2["dense_lift_std"] == 0.08
    assert robotmod.task_config_overrides({"dense_lift_weight": 0}) == {}
    assert robotmod.task_config_overrides({"dense_lift_weight": "nope"}) == {}


def test_dense_lift_reward_function_is_shipped():
    # The dense reward must be at module level so it travels in module_source()
    # (the Isaac image has no npa package) and be importable off-GPU (lazy torch).
    assert callable(robotmod.object_lift_progress)
    assert "def object_lift_progress" in robotmod.module_source()


def test_grasp_shaping_weight_passthrough_and_shipped():
    over = robotmod.task_config_overrides({"grasp_shaping_weight": 3.0})
    assert over["grasp_shaping_weight"] == 3.0
    assert over["grasp_shaping_std"] == 0.06
    assert robotmod.task_config_overrides({"grasp_shaping_weight": 0}) == {}
    assert robotmod.task_config_overrides({"grasp_shaping_weight": "x"}) == {}
    # Shipped at module level for the in-container wrapper.
    assert callable(robotmod.grasp_shaping)
    assert "def grasp_shaping" in robotmod.module_source()


def test_grasp_hold_weight_passthrough_and_shipped():
    over = robotmod.task_config_overrides({"grasp_hold_weight": 5.0})
    assert over["grasp_hold_weight"] == 5.0
    assert over["grasp_hold_std"] == 0.05  # default when unspecified
    over2 = robotmod.task_config_overrides({"grasp_hold_weight": 6.0, "grasp_hold_std": 0.08})
    assert over2["grasp_hold_std"] == 0.08
    # Non-positive / unparseable weights are dropped (Franka-safe gating).
    assert robotmod.task_config_overrides({"grasp_hold_weight": 0}) == {}
    assert robotmod.task_config_overrides({"grasp_hold_weight": -1}) == {}
    assert robotmod.task_config_overrides({"grasp_hold_weight": "x"}) == {}
    # Shipped at module level so the in-container wrapper can install the term.
    assert callable(robotmod.grasp_lift_hold)
    assert "def grasp_lift_hold" in robotmod.module_source()


def test_object_scale_normalizes_and_gates():
    # A scalar becomes a uniform (x, y, z) tuple; an explicit triple is preserved.
    assert robotmod.task_config_overrides({"object_scale": 0.2})["object_scale"] == (0.2, 0.2, 0.2)
    assert robotmod.task_config_overrides(
        {"object_scale": [0.2, 0.3, 0.4]}
    )["object_scale"] == (0.2, 0.3, 0.4)
    # Non-positive / malformed / wrong-arity / bool are dropped (never corrupt the
    # scene), and an absent object_scale is not carried (Franka + large-gripper safe).
    for bad in (0, -1, "big", [0.2, 0.2], [0.2, 0.2, 0.2, 0.2], True, None):
        assert "object_scale" not in robotmod.task_config_overrides({"object_scale": bad})
    assert "object_scale" not in robotmod.task_config_overrides({"action_scale": 0.5})


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
