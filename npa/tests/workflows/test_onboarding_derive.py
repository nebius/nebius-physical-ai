"""Unit tests for robot-aware config auto-derivation (B2).

Pure math — no GPU. Asserts the Franka reference reproduces the stock numbers
(so B3 keeps Franka byte-for-byte) and that a non-Franka arm (Kinova Jaco2)
derives sane, distinct values.
"""

from __future__ import annotations

import pytest

from npa.workflows.sim2real import onboarding_derive as der
from npa.workflows.sim2real import onboarding_spec as ob

# Representative Kinova J2N7S300 arm joint ranges: continuous joints 1/3/5/7 and
# limited joints 2/4/6 (radians, approximate published limits).
KINOVA_ARM_RANGES = [
    (-6.2832, 6.2832),
    (0.8203, 5.4629),
    (-6.2832, 6.2832),
    (0.5236, 5.7596),
    (-6.2832, 6.2832),
    (1.1345, 5.1487),
    (-6.2832, 6.2832),
]
KINOVA_FINGER_RANGES = [(0.0, 1.51), (0.0, 1.51), (0.0, 1.51)]


def _franka_spec() -> ob.OnboardingSpec:
    return ob.parse_onboarding_spec(
        {"robot": {"name": "franka", "robot_source": "stock_franka"},
         "task": {"skill": "lift"}}
    )


def _kinova_spec() -> ob.OnboardingSpec:
    return ob.parse_onboarding_spec(
        {
            "robot": {
                "name": "kinova_j2n7s300",
                "usd_path": "https://example.com/j2n7s300_instanceable.usd",
                "ee_link": "j2n7s300_end_effector",
                "n_arm_joints": 7,
                "joint_names": [f"j2n7s300_joint_{i}" for i in range(1, 8)],
                "n_gripper_joints": 3,
                "gripper_joint_names": [f"j2n7s300_joint_finger_{i}" for i in range(1, 4)],
                "gripper_open": 0.0,
                "gripper_close": 1.2,
                "home_qpos": [0.0, 2.9, 0.0, 1.3, 0.0, 2.07, 1.4, 0.0, 0.0, 0.0],
            },
            "task": {"skill": "lift", "goal_pos": "auto", "success_threshold": 0.4},
        }
    )


# --------------------------------------------------------------------------- #
# Franka reference reproduces the stock numbers (calibration)
# --------------------------------------------------------------------------- #
def test_franka_action_scale_reproduces_stock():
    scale, src = der.derive_action_scale(arm_joint_ranges=list(der.FRANKA_ARM_JOINT_RANGES))
    assert scale == pytest.approx(der.STOCK_ACTION_SCALE)
    assert src == "measured"


def test_franka_reach_is_stock():
    reach, src = der.derive_workspace_reach(preset="", robot_name="franka")
    assert reach == der.FRANKA_REACH_M
    assert src == "preset"


def test_franka_placement_unscaled():
    obj, goal, _ = der.derive_placement(der.FRANKA_REACH_M)
    assert obj == der.STOCK_OBJECT_INIT_RANGE
    assert goal == der.STOCK_GOAL_RANGE


def test_franka_full_config_is_stock():
    cfg = der.derive_task_config(
        _franka_spec(), arm_joint_ranges=list(der.FRANKA_ARM_JOINT_RANGES)
    )
    assert cfg.action_scale == pytest.approx(der.STOCK_ACTION_SCALE)
    assert cfg.workspace_reach_m == der.FRANKA_REACH_M
    assert cfg.object_init_range == der.STOCK_OBJECT_INIT_RANGE
    assert cfg.goal_range == der.STOCK_GOAL_RANGE
    assert cfg.minimal_height_m == der.STOCK_MINIMAL_HEIGHT_M
    assert cfg.success_distance_m == der.STOCK_SUCCESS_DISTANCE_M


# --------------------------------------------------------------------------- #
# Kinova derives sane, distinct values
# --------------------------------------------------------------------------- #
def test_kinova_reach_from_name():
    reach, src = der.derive_workspace_reach(robot_name="kinova_j2n7s300")
    assert reach == pytest.approx(0.985)
    assert src == "preset"


def test_kinova_action_scale_sane_and_capped():
    scale, _ = der.derive_action_scale(arm_joint_ranges=KINOVA_ARM_RANGES)
    # Wide (partly continuous) ranges -> capped at the stock 0.5, never above.
    assert der.ACTION_SCALE_MIN <= scale <= der.STOCK_ACTION_SCALE


def test_kinova_gripper_targets_explicit():
    g_open, g_close, src = der.derive_gripper_targets(
        explicit_open=0.0, explicit_close=1.2, finger_joint_ranges=KINOVA_FINGER_RANGES
    )
    assert (g_open, g_close) == (0.0, 1.2)
    assert src == "explicit"


def test_kinova_gripper_targets_measured_when_auto():
    g_open, g_close, src = der.derive_gripper_targets(finger_joint_ranges=KINOVA_FINGER_RANGES)
    assert g_open == pytest.approx(0.0)
    assert g_close == pytest.approx(1.51)
    assert src == "measured"


def test_kinova_placement_scaled_outward():
    obj, goal, src = der.derive_placement(0.985)
    # Longer reach than Franka -> goal box pushed outward.
    assert goal["x"][1] > der.STOCK_GOAL_RANGE["x"][1]
    assert src == "measured"


def test_kinova_full_config_distinct_from_franka():
    kin = der.derive_task_config(
        _kinova_spec(),
        arm_joint_ranges=KINOVA_ARM_RANGES,
        finger_joint_ranges=KINOVA_FINGER_RANGES,
    )
    fr = der.derive_task_config(
        _franka_spec(), arm_joint_ranges=list(der.FRANKA_ARM_JOINT_RANGES)
    )
    # The whole derived config must differ from Franka's.
    assert kin.to_dict() != fr.to_dict()
    # Specifically: distinct placement, gripper, init pose.
    assert kin.goal_range != fr.goal_range
    assert (kin.gripper_open, kin.gripper_close) == (0.0, 1.2)
    assert kin.init_joint_pos == (0.0, 2.9, 0.0, 1.3, 0.0, 2.07, 1.4, 0.0, 0.0, 0.0)
    assert kin.source["init_joint_pos"] == "explicit"
    assert kin.source["gripper"] == "explicit"


def test_explicit_task_thresholds_win():
    spec = ob.parse_onboarding_spec(
        {"robot": {"name": "kinova", "usd_path": "https://x/k.usd",
                   "n_arm_joints": 7, "n_gripper_joints": 3,
                   "gripper_joint_names": ["a", "b", "c"]},
         "task": {"skill": "lift", "lift_height_m": 0.08, "success_distance_m": 0.05}}
    )
    cfg = der.derive_task_config(spec)
    assert cfg.minimal_height_m == 0.08
    assert cfg.success_distance_m == 0.05
    assert cfg.source["minimal_height_m"] == "explicit"


def test_link_lengths_override_preset_reach():
    reach, src = der.derive_workspace_reach(
        arm_link_lengths=[0.3, 0.3, 0.3], robot_name="kinova"
    )
    assert reach == pytest.approx(0.9)
    assert src == "measured"
