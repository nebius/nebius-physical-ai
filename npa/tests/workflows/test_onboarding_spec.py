"""Unit tests for the declarative robot+task onboarding spec (B1).

Pure schema/validation — no GPU, no cluster. Validates that the documented
template and the filled Kinova example parse, that ``auto`` fields are recorded
for the derive layer, and that malformed specs fail fast with clear errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.genesis import robot_assets
from npa.workflows.sim2real import onboarding_spec as ob

ONBOARDING_DIR = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "workbench"
    / "sim2real"
    / "onboarding"
)
KINOVA_YAML = ONBOARDING_DIR / "kinova-jaco2.yaml"
TEMPLATE_YAML = ONBOARDING_DIR / "robot-onboarding.template.yaml"


def _kinova_doc() -> dict:
    return {
        "schema": ob.ONBOARDING_SCHEMA,
        "robot": {
            "name": "kinova_j2n7s300",
            "usd_path": "https://example.com/j2n7s300_instanceable.usd",
            "ee_link": "j2n7s300_end_effector",
            "base_link": "j2n7s300_link_base",
            "n_arm_joints": 7,
            "joint_names": [f"j2n7s300_joint_{i}" for i in range(1, 8)],
            "n_gripper_joints": 3,
            "gripper_joint_names": [f"j2n7s300_joint_finger_{i}" for i in range(1, 4)],
            "gripper_open": 0.0,
            "gripper_close": 1.2,
        },
        "task": {"skill": "lift", "goal_pos": "auto", "success_threshold": 0.4},
    }


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #
def test_parse_kinova_doc_explicit():
    spec = ob.parse_onboarding_spec(_kinova_doc())
    assert spec.robot.name == "kinova_j2n7s300"
    # USD asset -> byo_usd source inferred.
    assert spec.robot.robot_source == robot_assets.ROBOT_SOURCE_BYO_USD
    assert spec.robot.ee_link == "j2n7s300_end_effector"
    assert spec.robot.n_arm_joints == 7
    assert spec.robot.n_gripper_joints == 3
    assert spec.robot.gripper_close == 1.2
    # Explicit morphology -> no auto fields on the robot.
    assert spec.robot.auto_fields == set()
    assert spec.task.skill == "lift"
    assert spec.task.goal_pos_auto is True
    assert spec.task.needs_gripper is True


def test_shipped_kinova_yaml_parses():
    spec = ob.load_onboarding_spec(KINOVA_YAML)
    assert spec.robot.name == "kinova_j2n7s300"
    assert spec.robot.robot_source == robot_assets.ROBOT_SOURCE_BYO_USD
    assert len(spec.robot.joint_names) == 7
    assert len(spec.robot.gripper_joint_names) == 3
    assert len(spec.robot.home_qpos) == 10  # 7 arm + 3 finger
    assert spec.task.skill == "lift"


def test_shipped_template_yaml_parses_with_auto():
    spec = ob.load_onboarding_spec(TEMPLATE_YAML)
    # The template defers everything derivable to auto.
    for fld in ("ee_link", "base_link", "joint_names", "gripper_joint_names",
                "n_arm_joints", "n_gripper_joints", "home_qpos", "kp", "kv",
                "gripper_open", "gripper_close"):
        assert spec.robot.is_auto(fld), f"{fld} should be auto in the template"
    assert spec.task.goal_pos_auto is True


def test_auto_morphology_recorded_not_required():
    doc = {
        "robot": {
            "name": "mystery_arm",
            "usd_path": "https://example.com/arm.usd",
            "ee_link": "auto",
            "joint_names": "auto",
            "n_arm_joints": "auto",
            "n_gripper_joints": "auto",
            "gripper_joint_names": "auto",
            "home_qpos": "auto",
        },
        "task": {"skill": "lift"},
    }
    spec = ob.parse_onboarding_spec(doc)
    assert "ee_link" in spec.robot.auto_fields
    assert "joint_names" in spec.robot.auto_fields
    assert "n_arm_joints" in spec.robot.auto_fields
    # auto gripper => no false "gripperless" rejection for a lift skill.
    assert spec.task.skill == "lift"


def test_stock_franka_needs_no_asset():
    spec = ob.parse_onboarding_spec(
        {"robot": {"name": "franka", "robot_source": "stock_franka"}, "task": {"skill": "lift"}}
    )
    assert spec.robot.is_stock_franka
    assert spec.robot.robot_uri == ""


def test_preset_seeds_source():
    spec = ob.parse_onboarding_spec(
        {"robot": {"name": "ur", "preset": "ur5e", "usd_path": "https://x/ur.urdf"},
         "task": {"skill": "reach"}}
    )
    # urdf asset -> byo_urdf; reach does not require a gripper.
    assert spec.robot.robot_source == robot_assets.ROBOT_SOURCE_BYO_URDF
    assert spec.task.needs_gripper is False


# --------------------------------------------------------------------------- #
# Validation failures
# --------------------------------------------------------------------------- #
def test_reject_visual_mesh_robot():
    with pytest.raises(ob.OnboardingSpecError, match="infer robot_source"):
        ob.parse_onboarding_spec(
            {"robot": {"name": "x", "usd_path": "https://x/arm.obj"}, "task": {"skill": "lift"}}
        )


def test_reject_unknown_skill():
    with pytest.raises(ob.OnboardingSpecError, match="task.skill"):
        ob.parse_onboarding_spec(
            {"robot": {"name": "franka", "robot_source": "stock_franka"},
             "task": {"skill": "weld"}}
        )


def test_reject_gripperless_lift():
    with pytest.raises(ob.OnboardingSpecError, match="requires a gripper"):
        ob.parse_onboarding_spec(
            {"robot": {"name": "ur10", "usd_path": "https://x/ur.usd",
                       "n_arm_joints": 6, "n_gripper_joints": 0},
             "task": {"skill": "lift"}}
        )


def test_gripperless_reach_is_allowed():
    spec = ob.parse_onboarding_spec(
        {"robot": {"name": "ur10", "usd_path": "https://x/ur.usd",
                   "n_arm_joints": 6, "n_gripper_joints": 0},
         "task": {"skill": "reach"}}
    )
    assert spec.task.skill == "reach"


def test_reject_bad_goal_pos():
    with pytest.raises(ob.OnboardingSpecError, match="goal_pos"):
        ob.parse_onboarding_spec(
            {"robot": {"name": "franka", "robot_source": "stock_franka"},
             "task": {"skill": "place", "goal_pos": [1.0, 2.0]}}
        )


def test_reject_out_of_range_threshold():
    with pytest.raises(ob.OnboardingSpecError, match="success_threshold"):
        ob.parse_onboarding_spec(
            {"robot": {"name": "franka", "robot_source": "stock_franka"},
             "task": {"skill": "lift", "success_threshold": 1.5}}
        )


def test_reject_inconsistent_explicit_vectors():
    with pytest.raises(ob.OnboardingSpecError, match="share a length"):
        ob.parse_onboarding_spec(
            {"robot": {"name": "x", "usd_path": "https://x/arm.usd",
                       "n_arm_joints": 2, "n_gripper_joints": 0,
                       "kp": [1, 2], "kv": [1, 2, 3]},
             "task": {"skill": "reach"}}
        )


def test_reject_missing_blocks():
    with pytest.raises(ob.OnboardingSpecError, match="'robot' block"):
        ob.parse_onboarding_spec({"task": {"skill": "lift"}})
    with pytest.raises(ob.OnboardingSpecError, match="'task' block"):
        ob.parse_onboarding_spec({"robot": {"robot_source": "stock_franka"}})


def test_reject_bad_schema():
    with pytest.raises(ob.OnboardingSpecError, match="schema"):
        ob.parse_onboarding_spec(
            {"schema": "npa.sim2real.onboarding.v999",
             "robot": {"robot_source": "stock_franka"}, "task": {"skill": "lift"}}
        )
