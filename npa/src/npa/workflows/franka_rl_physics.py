"""Reconcile the pinned Robotiq mimic asset with Isaac actuator configuration."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from npa.workflows.franka_rl_embodiments import recipe_embodiment

_NAME = "ur10e-mimic-asset-v1"
_FOLLOWERS = {
    "right_outer_knuckle_joint": ("rotZ", -1.0),
    "left_inner_finger_joint": ("rotX", 1.0),
    "right_inner_finger_joint": ("rotX", -1.0),
    "left_inner_finger_knuckle_joint": ("rotX", 1.0),
    "right_inner_finger_knuckle_joint": ("rotX", 1.0),
}
_JOINTS = ["finger_joint", *_FOLLOWERS]
_REFERENCE = {
    "isaaclab_commit": "ffff603eafc6b74264a5261cc0183d6a65390d78",
    "isaac_asset_generation": "6.0",
    "reference_asset_sha256": {
        "ur10e.usd": "f38483d7ad2131ca710efd6e1bc61984293a7f0a9862615f60eb41bfcff22b32",
        "ur10e_Gripper_2F_85.usd": "acdc8ac02a715590e579d866956e814688c1dc24d82e66d8c94fee739124a0f4",
        "Robotiq_2F_85_edit.usd": "3b69fbb560dba20a77d09dcf845bedfbc2b6b1cfda9de10f8e3b22fd6ca97d43",
        "Robotiq_2F_85_phyisics_mimic.usda": "1b9625826ff31675b15c5ac117e81c69ebe5e3364efd1b9fa6d40f9dcbaac386",
    },
    "guidance": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/"
    "dev_guide/guides/gripper_tuning_example.html",
}


def stability_profile(name: str) -> dict:
    """Seal the asset-derived UR gripper compatibility settings.

    Args:
        name: Supported, versioned physics profile identifier.
    Returns:
        Independent JSON-compatible profile; hashes identify inspected references.
    Raises:
        ValueError: The profile is unsupported.
    """
    if name != _NAME:
        raise ValueError(f"Unsupported manipulation stability profile: {name}")
    return {
        "schema": "npa.manipulation-stability.v1",
        "name": name,
        "embodiment": "ur10e_robotiq85",
        "gripper_physics_variant": "Physx_Mimic",
        "actuator_properties": "inherit_composed_usd_for_all_gripper_joints",
        "gripper_joints": list(_JOINTS),
        "solver_position_iterations": 64,
        "solver_velocity_iterations": 1,
        "reference": deepcopy(_REFERENCE),
    }


def recipe_stability(recipe: dict) -> dict | None:
    """Validate an optional sealed stability profile without changing legacy recipes.

    Args:
        recipe: Prepared experiment settings.
    Returns:
        Validated profile, or None for a historical recipe without this setting.
    Raises:
        ValueError: The profile or robot differs from its sealed definition.
    """
    if "stability" not in recipe:
        return None
    declared = recipe["stability"]
    if not isinstance(declared, dict) or declared != stability_profile(
        declared.get("name")
    ):
        raise ValueError("Sealed stability settings differ from the supported profile")
    if recipe_embodiment(recipe)["name"] != declared["embodiment"]:
        raise ValueError("Stability profile requires the UR10e Robotiq embodiment")
    return deepcopy(declared)


def configure_stability(config, recipe: dict) -> None:
    """Preserve the mimic asset's drives, limits, armature and solver requirements.

    Args:
        config: Native environment after embodiment configuration, before spawning.
        recipe: Prepared experiment settings.
    Returns:
        None.
    Raises:
        ValueError: The sealed profile is invalid or the native actuator layout changed.
        ImportError: The native Isaac actuator API is unavailable.
    """
    profile = recipe_stability(recipe)
    if profile is None:
        return
    from isaaclab.actuators import ImplicitActuatorCfg

    robot = config.scene.robot
    groups = ("gripper_drive", "gripper_finger", "gripper_passive")
    if not all(group in robot.actuators for group in groups):
        raise ValueError("Native UR gripper actuator layout changed")
    for group in groups:
        del robot.actuators[group]
    # None inherits the simulator's USD values, including zero-drive followers.
    # Loop-style follower drives and tight follower limits fight mimic constraints.
    robot.actuators["gripper_mimic_asset"] = ImplicitActuatorCfg(
        joint_names_expr=list(_JOINTS),
        stiffness=None,
        damping=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
        armature=None,
    )
    robot.spawn.articulation_props.solver_position_iteration_count = profile[
        "solver_position_iterations"
    ]
    robot.spawn.articulation_props.solver_velocity_iteration_count = profile[
        "solver_velocity_iterations"
    ]
    config.npa_stability = profile


def stability_evidence(native, recipe: dict) -> dict | None:
    """Verify composed mimic mechanics and simulator-resolved actuator properties.

    Args:
        native: Initialized Isaac environment before collecting any rollout.
        recipe: Sealed experiment settings.
    Returns:
        Measured conformance evidence, or None for historical recipes.
    Raises:
        ValueError: The selected asset, mimic constraints or applied properties differ.
        ImportError: The native Isaac scene API is unavailable.
    """
    profile = recipe_stability(recipe)
    if profile is None:
        return None
    if getattr(native.cfg, "npa_stability", None) != profile:
        raise ValueError("Stability profile was not applied before simulation startup")
    from isaaclab.sim.utils.queries import find_first_matching_prim

    robot = native.scene["robot"]
    root = find_first_matching_prim(robot.cfg.prim_path)
    if root is None:
        raise ValueError("Stability profile robot prim is missing")
    composition = _composition_evidence(root, profile)
    properties = _actuator_evidence(robot)
    return {
        "profile": profile,
        "composition": composition,
        "actuators": properties,
        "gearing_expected_velocity_limits_rad_s": {
            name: math.radians(146.46) for name in _JOINTS
        },
        "reference_hash_scope": "inspected upstream files; runtime verifies composed mechanics and applied values",
        "conformant": True,
    }


def _composition_evidence(root, profile: dict) -> dict:
    gripper = root.GetChild("ee_link")
    if (
        gripper.GetVariantSets().GetVariantSet("Physics").GetVariantSelection()
        != "Physx_Mimic"
    ):
        raise ValueError("UR gripper must compose the Physx_Mimic asset variant")
    articulation = root.GetChild("root_joint")
    _require_attribute(
        articulation, "physxArticulation:solverPositionIterationCount", 64
    )
    _require_attribute(
        articulation, "physxArticulation:solverVelocityIterationCount", 1
    )
    joints = gripper.GetChild("Robotiq_2F_85").GetChild("Joints")
    driver = joints.GetChild("finger_joint")
    if "PhysicsDriveAPI:angular" not in _schemas(driver):
        raise ValueError("Mimic reference joint must retain its authored angular drive")
    for attribute, value in {
        "drive:angular:physics:stiffness": 3.0,
        "drive:angular:physics:damping": 0.0002,
        "drive:angular:physics:maxForce": 26.0,
        "physxJoint:maxJointVelocity": 146.46,
        "physxJoint:armature": 0.0001,
        "physics:lowerLimit": 0.0,
        "physics:upperLimit": 47.0,
    }.items():
        _require_attribute(driver, attribute, value)
    followers = {
        name: _mimic_evidence(joints.GetChild(name), driver, axis, gearing)
        for name, (axis, gearing) in _FOLLOWERS.items()
    }
    return {
        "gripper_physics_variant": profile["gripper_physics_variant"],
        "followers": followers,
        "articulation_solver_position_iterations": 64,
        "articulation_solver_velocity_iterations": 1,
        "solver_evidence_source": "composed USD articulation properties",
    }


def _mimic_evidence(joint, driver, axis: str, gearing: float) -> dict:
    applied = _schemas(joint)
    if (
        f"PhysxMimicJointAPI:{axis}" not in applied
        or "PhysicsDriveAPI:angular" in applied
    ):
        raise ValueError(
            f"Mimic follower must have its mimic API and no independent drive: {joint.GetName()}"
        )
    prefix = f"physxMimicJoint:{axis}:"
    for attribute, value in {
        "gearing": gearing,
        "naturalFrequency": 0.0,
        "dampingRatio": 0.0,
    }.items():
        _require_attribute(joint, prefix + attribute, value)
    if joint.GetRelationship(prefix + "referenceJoint").GetTargets() != [
        driver.GetPath()
    ]:
        raise ValueError(
            f"Mimic follower references the wrong driver: {joint.GetName()}"
        )
    _require_attribute(joint, "physxJoint:maxJointVelocity", 10000.0)
    return {
        "axis": axis,
        "gearing": gearing,
        "driver": driver.GetName(),
        "independent_drive": False,
    }


def _schemas(prim) -> list[str]:
    # Raw composed metadata includes PhysX APIs even in CPU-only USD inspection.
    schemas = prim.GetMetadata("apiSchemas")
    return schemas.ApplyOperations([]) if schemas is not None else []


def _require_attribute(prim, name: str, expected: float) -> None:
    value = prim.GetAttribute(name).Get()
    if value is None or not math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"Composed robot property differs: {prim.GetPath()} {name}")


def _actuator_evidence(robot) -> dict:
    fields = {
        "stiffness": "joint_stiffness",
        "damping": "joint_damping",
        "armature": "joint_armature",
        "velocity_limit_rad_s": "joint_vel_limits",
        "effort_limit_nm": "joint_effort_limits",
    }
    indices = [robot.joint_names.index(name) for name in _JOINTS]
    arrays = {
        name: getattr(robot.data, attribute).torch[:, indices].detach().cpu().numpy()
        for name, attribute in fields.items()
    }
    expected = {
        "stiffness": [3.0 * 180 / math.pi, 0, 0, 0, 0, 0],
        "damping": [0.0002 * 180 / math.pi, 0, 0, 0, 0, 0],
        "armature": [0.0001, 0.0001, 0, 0, 0, 0],
        "velocity_limit_rad_s": [math.radians(146.46), *([math.radians(10000)] * 5)],
    }
    for name, array in arrays.items():
        if not array.size or not np.isfinite(array).all():
            raise ValueError(f"Nonfinite or missing applied gripper property: {name}")
        if name in expected and not np.allclose(
            array, expected[name], rtol=1e-5, atol=1e-8
        ):
            raise ValueError(
                f"Applied gripper property differs from the mimic asset: {name}"
            )
    if not np.allclose(arrays["effort_limit_nm"][:, 0], 26.0, rtol=1e-6):
        raise ValueError("Applied gripper driver effort differs from the mimic asset")
    return {
        joint: {
            field: {
                "minimum": float(array[:, index].min()),
                "maximum": float(array[:, index].max()),
            }
            for field, array in arrays.items()
        }
        for index, joint in enumerate(_JOINTS)
    }
