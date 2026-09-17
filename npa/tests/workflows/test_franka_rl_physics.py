"""Check mimic actuator inheritance and reject conflicting composed/runtime mechanics."""

from copy import deepcopy
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows import franka_rl_physics as physics
from npa.workflows.franka_rl_embodiments import embodiment_profile


def _recipe():
    return {"embodiment": embodiment_profile("ur10e_robotiq85"),
            "stability": physics.stability_profile("ur10e-mimic-asset-v1")}


def _config():
    robot = SimpleNamespace(prim_path="/Robot", actuators={name: object() for name in
        ("shoulder", "elbow", "wrist", "gripper_drive", "gripper_finger", "gripper_passive")},
        spawn=SimpleNamespace(articulation_props=SimpleNamespace(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1)))
    return SimpleNamespace(scene=SimpleNamespace(robot=robot), actions=object(), sim=object())


def test_legacy_recipe_does_not_import_isaac_or_modify_configuration(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaaclab.actuators", None)
    config = _config()
    original = dict(config.scene.robot.actuators)
    physics.configure_stability(config, {})
    assert config.scene.robot.actuators == original
    assert not hasattr(config, "npa_stability")
    assert physics.stability_evidence(None, {}) is None


def test_inherit_all_six_joints_without_touching_policy_or_arm(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaaclab.actuators", SimpleNamespace(ImplicitActuatorCfg=SimpleNamespace))
    config = _config()
    arm = {name: config.scene.robot.actuators[name] for name in ("shoulder", "elbow", "wrist")}
    actions, simulation = config.actions, config.sim
    physics.configure_stability(config, _recipe())
    groups = config.scene.robot.actuators
    assert {name: groups[name] for name in arm} == arm
    assert set(groups) == {*arm, "gripper_mimic_asset"}
    gripper = groups["gripper_mimic_asset"]
    assert set(gripper.joint_names_expr) == {"finger_joint", "right_outer_knuckle_joint",
        "left_inner_finger_joint", "right_inner_finger_joint", "left_inner_finger_knuckle_joint",
        "right_inner_finger_knuckle_joint"}
    assert all(getattr(gripper, field) is None for field in
               ("stiffness", "damping", "effort_limit_sim", "velocity_limit_sim", "armature"))
    assert config.actions is actions and config.sim is simulation
    assert config.scene.robot.spawn.articulation_props.solver_position_iteration_count == 64


@pytest.mark.parametrize("mutation", ["unknown", "null", "solver", "source", "embodiment"])
def test_mutated_or_incompatible_profile_fails_closed(mutation):
    recipe = _recipe()
    if mutation == "null":
        recipe["stability"] = None
    elif mutation == "embodiment":
        recipe["embodiment"] = embodiment_profile("kinova_jaco7")
    elif mutation == "unknown":
        recipe["stability"]["name"] = "unknown"
    elif mutation == "solver":
        recipe["stability"]["solver_position_iterations"] = 16
    else:
        recipe["stability"]["reference"]["isaaclab_commit"] = "0" * 40
    with pytest.raises(ValueError):
        physics.recipe_stability(recipe)


def test_profile_copies_cannot_mutate_future_recipes():
    modified = _recipe()
    modified["stability"]["gripper_joints"].clear()
    assert len(_recipe()["stability"]["gripper_joints"]) == 6
    result = physics.recipe_stability(_recipe())
    result["reference"].clear()
    assert _recipe()["stability"]["reference"]


class _Array:
    def __init__(self, array):
        self.array = np.asarray(array)
        self.torch = self

    def __getitem__(self, index):
        return _Array(self.array[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def _applied_robot():
    joints = _recipe()["stability"]["gripper_joints"]
    values = {"joint_stiffness": [math.degrees(3), 0, 0, 0, 0, 0],
              "joint_damping": [math.degrees(.0002), 0, 0, 0, 0, 0],
              "joint_armature": [.0001, .0001, 0, 0, 0, 0],
              "joint_vel_limits": [math.radians(146.46), *([math.radians(10000)] * 5)],
              "joint_effort_limits": [26, 0, 0, 0, 0, 0]}
    data = SimpleNamespace(**{name: _Array(np.tile(value, (3, 1))) for name, value in values.items()})
    return SimpleNamespace(joint_names=joints, data=data, cfg=SimpleNamespace(prim_path="/Robot"))


def test_applied_properties_use_radians_and_cover_every_environment():
    evidence = physics._actuator_evidence(_applied_robot())
    assert evidence["finger_joint"]["stiffness"]["minimum"] == pytest.approx(171.8873385)
    assert evidence["right_outer_knuckle_joint"]["velocity_limit_rad_s"]["maximum"] == pytest.approx(174.5329252)


@pytest.mark.parametrize("field,joint,value", [("joint_stiffness", 2, .2), ("joint_damping", 3, .001),
    ("joint_armature", 1, 0), ("joint_vel_limits", 4, 1), ("joint_stiffness", 0, 3),
    ("joint_effort_limits", 0, 10), ("joint_armature", 0, np.nan)])
def test_inherited_loop_configuration_and_bad_unit_conversion_are_rejected(field, joint, value):
    robot = _applied_robot()
    getattr(robot.data, field).array[2, joint] = value
    with pytest.raises(ValueError, match="gripper"):
        physics._actuator_evidence(robot)


def _set(prim, name, value, sdf):
    kind = sdf.ValueTypeNames.Float if isinstance(value, float) else sdf.ValueTypeNames.Int
    prim.CreateAttribute(name, kind).Set(value)


@pytest.fixture
def stage():
    usd = pytest.importorskip("pxr.Usd")
    sdf = pytest.importorskip("pxr.Sdf")
    stage = usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Robot", "Xform")
    articulation = stage.DefinePrim("/Robot/root_joint", "PhysicsFixedJoint")
    _set(articulation, "physxArticulation:solverPositionIterationCount", 64, sdf)
    _set(articulation, "physxArticulation:solverVelocityIterationCount", 1, sdf)
    gripper = stage.DefinePrim("/Robot/ee_link", "Xform")
    variant = gripper.GetVariantSets().AddVariantSet("Physics")
    variant.AddVariant("Physx_Mimic")
    variant.SetVariantSelection("Physx_Mimic")
    driver = stage.DefinePrim("/Robot/ee_link/Robotiq_2F_85/Joints/finger_joint", "PhysicsRevoluteJoint")
    _author_driver(driver, sdf)
    _author_followers(stage, driver, sdf)
    return stage, root


def _author_driver(driver, sdf):
    driver.SetMetadata("apiSchemas", sdf.TokenListOp.CreateExplicit(["PhysicsDriveAPI:angular"]))
    for name, value in {"drive:angular:physics:stiffness": 3., "drive:angular:physics:damping": .0002,
            "drive:angular:physics:maxForce": 26., "physxJoint:maxJointVelocity": 146.46,
            "physxJoint:armature": .0001, "physics:lowerLimit": 0., "physics:upperLimit": 47.}.items():
        _set(driver, name, value, sdf)


def _author_followers(stage, driver, sdf):
    expected = {"right_outer_knuckle_joint": ("rotZ", -1.), "left_inner_finger_joint": ("rotX", 1.),
        "right_inner_finger_joint": ("rotX", -1.), "left_inner_finger_knuckle_joint": ("rotX", 1.),
        "right_inner_finger_knuckle_joint": ("rotX", 1.)}
    for name, (axis, gearing) in expected.items():
        joint = stage.DefinePrim(str(driver.GetPath().GetParentPath()) + "/" + name, "PhysicsRevoluteJoint")
        joint.SetMetadata("apiSchemas", sdf.TokenListOp.CreateExplicit([f"PhysxMimicJointAPI:{axis}"]))
        prefix = f"physxMimicJoint:{axis}:"
        for attribute, value in {"gearing": gearing, "naturalFrequency": 0., "dampingRatio": 0.}.items():
            _set(joint, prefix + attribute, value, sdf)
        _set(joint, "physxJoint:maxJointVelocity", 10000., sdf)
        joint.CreateRelationship(prefix + "referenceJoint").SetTargets([driver.GetPath()])


def test_composed_unknown_usd_schema_metadata_is_checked_without_physx_plugin(stage):
    _, root = stage
    evidence = physics._composition_evidence(root, _recipe()["stability"])
    assert len(evidence["followers"]) == 5
    assert all(not joint["independent_drive"] for joint in evidence["followers"].values())


@pytest.mark.parametrize("mutation", ["loop", "drive", "reference", "gearing", "compliance", "limit", "solver", "driver"])
def test_composed_asset_must_match_mimic_profile(stage, mutation):
    stage, root = stage
    from pxr import Sdf

    follower = stage.GetPrimAtPath("/Robot/ee_link/Robotiq_2F_85/Joints/right_outer_knuckle_joint")
    if mutation == "loop":
        root.GetChild("ee_link").GetVariantSets().GetVariantSet("Physics").ClearVariantSelection()
    elif mutation == "drive":
        follower.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(
            ["PhysxMimicJointAPI:rotZ", "PhysicsDriveAPI:angular"]))
    elif mutation == "reference":
        follower.GetRelationship("physxMimicJoint:rotZ:referenceJoint").SetTargets(["/other"])
    elif mutation == "gearing":
        follower.GetAttribute("physxMimicJoint:rotZ:gearing").Set(1.)
    elif mutation == "compliance":
        follower.GetAttribute("physxMimicJoint:rotZ:naturalFrequency").Set(5000.)
    elif mutation == "limit":
        follower.GetAttribute("physxJoint:maxJointVelocity").Set(57.2958)
    elif mutation == "solver":
        root.GetChild("root_joint").GetAttribute("physxArticulation:solverPositionIterationCount").Set(16)
    else:
        driver = stage.GetPrimAtPath("/Robot/ee_link/Robotiq_2F_85/Joints/finger_joint")
        driver.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([]))
    with pytest.raises(ValueError):
        physics._composition_evidence(root, _recipe()["stability"])


def test_full_receipt_separates_simulator_and_gearing_expected_limits(stage, monkeypatch):
    _, root = stage
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.queries",
                        SimpleNamespace(find_first_matching_prim=lambda path: root))
    recipe = _recipe()
    native = SimpleNamespace(scene={"robot": _applied_robot()}, cfg=SimpleNamespace(npa_stability=deepcopy(recipe["stability"])))
    evidence = physics.stability_evidence(native, recipe)
    assert evidence["conformant"]
    assert evidence["gearing_expected_velocity_limits_rad_s"]["left_inner_finger_joint"] == pytest.approx(math.radians(146.46))
    assert evidence["actuators"]["left_inner_finger_joint"]["velocity_limit_rad_s"]["minimum"] > 170
    native.cfg.npa_stability = None
    with pytest.raises(ValueError, match="before simulation startup"):
        physics.stability_evidence(native, recipe)
